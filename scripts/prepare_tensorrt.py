"""Export and build the four fixed-shape TensorRT RTX VAE engines."""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
ARTIFACTS = ROOT / "tensorrt_backend" / "artifacts"

PROFILES = (
    ("encoder 5-frame", "vae_encoder_5f_tile512", "export_vae_encoder.py", ("--height", "512", "--width", "512", "--frames", "5", "--legacy-export")),
    ("encoder 21-frame", "vae_encoder_21f_tile512", "export_vae_encoder.py", ("--height", "512", "--width", "512", "--frames", "21", "--legacy-export")),
    ("decoder 5-frame", "vae_decoder_tile_512_5f", "export_vae_decoder.py", ("--height", "512", "--width", "512", "--latent-frames", "2", "--legacy-export")),
    ("decoder 21-frame", "vae_decoder_tile_256_21f", "export_vae_decoder.py", ("--height", "256", "--width", "256", "--latent-frames", "6", "--legacy-export")),
)


def run(command: list[str]) -> None:
    print("\n> " + shlex.join(command), flush=True)
    subprocess.run([command[0], "-u", *command[1:]], cwd=ROOT, check=True)


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for label, stem, exporter, arguments in PROFILES:
        onnx = ARTIFACTS / f"{stem}.onnx"
        engine = ARTIFACTS / f"{stem}.rtxplan"
        if engine.exists() and engine.stat().st_size > 1_000_000:
            print(f"Skipping {label}; engine already exists: {engine.name}")
            continue
        print(f"\nPreparing TensorRT {label} profile (keep this window open)...", flush=True)
        started = time.perf_counter()
        if not onnx.exists():
            run([str(PYTHON), str(ROOT / "tools" / exporter), *arguments, "--output", str(onnx)])
        run([str(PYTHON), str(ROOT / "tools" / "build_tensorrt_engine.py"), str(onnx), "--output", str(engine), "--workspace-gb", "8"])
        if not engine.exists():
            raise RuntimeError(f"TensorRT did not create {engine}")
        print(f"Completed {label} in {time.perf_counter() - started:.1f}s", flush=True)
    print("\nAll TensorRT VAE engines are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"TensorRT preparation failed with exit code {exc.returncode}. Run the installer again to resume.", file=sys.stderr)
        raise SystemExit(exc.returncode)
