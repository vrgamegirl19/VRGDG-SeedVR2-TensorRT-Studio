"""Export a fixed-shape SeedVR2 VAE encoder graph for TensorRT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "seedvr2"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(VENDOR))

from onnx_export_utils import export_onnx_fixed_shape  # noqa: E402
from src.core.generation_utils import prepare_runner, setup_generation_context  # noqa: E402
from src.core.model_loader import materialize_model  # noqa: E402
from src.models.video_vae_v3.modules.types import MemoryState  # noqa: E402
from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d  # noqa: E402
from src.utils.debug import Debug  # noqa: E402
from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE  # noqa: E402


class Encoder(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(video, memory_state=MemoryState.DISABLED)
        if self.quant_conv is not None:
            hidden = self.quant_conv(hidden, memory_state=MemoryState.DISABLED)
        return hidden


def configure_fixed_profile(vae: torch.nn.Module) -> None:
    if hasattr(vae, "disable_slicing"):
        vae.disable_slicing()
    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(None, None)
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            module.set_memory_limit(float("inf"))
            module.set_memory_device(None)
        if hasattr(module, "slicing"):
            module.slicing = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "tensorrt_backend" / "artifacts" / "vae_encoder.onnx")
    parser.add_argument("--legacy-export", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VAE export")
    if args.frames < 1 or (args.frames - 1) % 4 or args.height % 8 or args.width % 8:
        raise ValueError("frames must be 4n+1 and spatial dimensions divisible by 8")
    device = torch.device("cuda")
    debug = Debug(enabled=True)
    ctx = setup_generation_context(dit_device=device, vae_device=device,
                                   tensor_offload_device=None, debug=debug)
    ctx["compute_dtype"] = torch.float16
    runner, _ = prepare_runner(
        dit_model=DEFAULT_DIT, vae_model=DEFAULT_VAE,
        model_dir=str(ROOT / "models" / "SEEDVR2"), debug=debug, ctx=ctx,
        block_swap_config={"blocks_to_swap": 0, "swap_io_components": False, "offload_device": None},
        attention_mode="sdpa",
    )
    materialize_model(runner, "vae", device, runner.config, debug)
    vae = runner.vae.eval()
    configure_fixed_profile(vae)
    encoder = Encoder(vae).eval()
    video = torch.randn(1, 3, args.frames, args.height, args.width, device=device, dtype=torch.float16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = encoder(video)
        # Portable GPU export: cuDNN and fused SDPA are disabled so the
        # tracer records Conv/MatMul instead of CUDA-only kernels.
        export_onnx_fixed_shape(
            encoder,
            (video,),
            args.output,
            input_names=["video"],
            output_names=["latent_raw"],
            dynamo=not args.legacy_export,
        )
    print(f"Exported: {args.output}")
    print(f"Video shape: {tuple(video.shape)}")
    print(f"Raw latent shape: {tuple(reference.shape)}")
    print(f"Reference range: {reference.min().item():.6f} .. {reference.max().item():.6f}")


if __name__ == "__main__":
    main()
