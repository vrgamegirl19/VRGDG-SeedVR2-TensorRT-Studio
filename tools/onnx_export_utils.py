"""Portable ONNX export for TensorRT-RTX VAE graphs."""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch


def _portable_attention():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        return nullcontext()


def _to_device(
    model: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    device: torch.device,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    return model.to(device), tuple(tensor.to(device) for tensor in inputs)


def _export_on_device(
    model: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    output: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamo: bool,
    opset_version: int,
    device: torch.device,
) -> None:
    tracer = "dynamo" if dynamo else "legacy tracer"
    print(
        f"Exporting ONNX ({tracer}, portable convs) on {device} to {output.name} ...",
        flush=True,
    )
    started = time.perf_counter()
    with torch.backends.cudnn.flags(enabled=False), _portable_attention():
        torch.onnx.export(
            model,
            tuple(inputs),
            str(output),
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=opset_version,
            dynamo=dynamo,
            optimize=False,
            do_constant_folding=False,
        )
    elapsed = time.perf_counter() - started
    print(f"ONNX export finished in {elapsed:.1f}s: {output}", flush=True)


def export_onnx_fixed_shape(
    model: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    output: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamo: bool,
    opset_version: int = 20,
) -> None:
    """Trace a fixed-shape graph using portable Conv/attention ops.

    GPU export is preferred. cuDNN and fused SDPA are disabled so the
    legacy tracer records ONNX-friendly operators instead of CUDA-only
    kernels. CUDA OOM falls back to CPU, which is slower but must not hang.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parameter = next(model.parameters(), None)
    preferred = torch.device(
        "cuda" if torch.cuda.is_available() and parameter is not None and parameter.is_cuda else "cpu"
    )
    try:
        _export_on_device(
            model, inputs, output, input_names, output_names, dynamo, opset_version, preferred
        )
    except torch.cuda.OutOfMemoryError:
        if preferred.type != "cuda":
            raise
        print(
            "CUDA out of memory during ONNX export; retrying on CPU. "
            "This is slower but should complete.",
            flush=True,
        )
        torch.cuda.empty_cache()
        model, inputs = _to_device(model, inputs, torch.device("cpu"))
        _export_on_device(
            model, inputs, output, input_names, output_names, dynamo, opset_version, torch.device("cpu")
        )
