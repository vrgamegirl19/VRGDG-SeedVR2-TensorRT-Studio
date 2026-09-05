"""Portable, observable ONNX export helpers for TensorRT engine preparation."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.onnx import symbolic_helper


@symbolic_helper.parse_args("v", "v", "is", "is", "is", "i", "b", "b", "b")
def _cudnn_convolution_symbolic(
    graph, value, weight, padding, stride, dilation, groups,
    benchmark, deterministic, allow_tf32,
):
    """Export the low-memory direct-cuDNN path as a portable ONNX Conv."""
    del benchmark, deterministic, allow_tf32
    return graph.op(
        "Conv",
        value,
        weight,
        dilations_i=dilation,
        group_i=groups,
        pads_i=[*padding, *padding],
        strides_i=stride,
    )


torch.onnx.register_custom_op_symbolic(
    "aten::cudnn_convolution", _cudnn_convolution_symbolic, 20
)


def _math_attention_context():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return nullcontext()


def _portable_export(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: Path, *, legacy: bool) -> None:
    is_encoder = args[0].ndim == 5 and args[0].shape[1] == 3
    # Keep cuDNN enabled so SeedVR2's direct-cuDNN Conv3d workaround remains
    # active. The custom symbolic above converts that CUDA-only operator into
    # a standard, portable ONNX Conv for TensorRT.
    with torch.inference_mode(), torch.backends.cudnn.flags(enabled=True), _math_attention_context():
        torch.onnx.export(
            module,
            args,
            str(output),
            input_names=["video"] if is_encoder else ["latent"],
            output_names=["latent_raw"] if is_encoder else ["sample"],
            opset_version=20,
            dynamo=not legacy,
            optimize=False,
            do_constant_folding=False,
        )


def export_portable_onnx(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output: Path,
    *,
    legacy: bool,
) -> None:
    """Export on CUDA with portable operators, falling back to CPU only on OOM."""
    print(f"Exporting ONNX (legacy tracer, portable convs) -> {output}", flush=True)
    started = time.perf_counter()
    try:
        _portable_export(module, args, output, legacy=legacy)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("CUDA ONNX export ran out of memory; falling back to CPU export. This may take a while.", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _portable_export(module.cpu(), tuple(value.cpu() for value in args), output, legacy=legacy)
    print(f"ONNX export finished in {time.perf_counter() - started:.1f}s", flush=True)
