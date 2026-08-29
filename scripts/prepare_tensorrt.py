"""Export and build the four fixed-shape TensorRT RTX VAE engines."""

from __future__ import annotations

import subprocess
import sys
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
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    total = len(PROFILES)
    print(
        f"Building {total} GPU-specific TensorRT VAE engines. "
        "Leave this window open; each export and engine build can take several minutes.",
        flush=True,
    )
    for index, (label, stem, exporter, arguments) in enumerate(PROFILES, start=1):
        onnx = ARTIFACTS / f"{stem}.onnx"
        engine = ARTIFACTS / f"{stem}.rtxplan"
        if engine.exists() and engine.stat().st_size > 1_000_000:
            print(f"Skipping {label}; engine already exists: {engine.name}", flush=True)
            continue
        print(f"\nPreparing TensorRT {label} profile ({index}/{total})...", flush=True)
        if not onnx.exists():
            print(f"Exporting {label} ONNX graph...", flush=True)
            run([str(PYTHON), "-u", str(ROOT / "tools" / exporter), *arguments, "--output", str(onnx)])
        else:
            print(f"Reusing existing ONNX: {onnx.name}", flush=True)
        print(f"Building TensorRT engine for {label}...", flush=True)
        run([str(PYTHON), "-u", str(ROOT / "tools" / "build_tensorrt_engine.py"), str(onnx), "--output", str(engine), "--workspace-gb", "8"])
        if not engine.exists():
            raise RuntimeError(f"TensorRT did not create {engine}")
        print(f"Finished {label}: {engine.name}", flush=True)
    print("\nAll TensorRT VAE engines are ready.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"TensorRT preparation failed with exit code {exc.returncode}. Run the installer again to resume.", file=sys.stderr)
        raise SystemExit(exc.returncode)
