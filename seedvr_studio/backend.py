from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .media import MediaError, probe
from .paths import MODELS, ROOT, SEEDVR_CLI, VENV_PYTHON


TRT_RUNNER = ROOT / "tools" / "run_tensorrt_tiled.py"
TRT_PERSISTENT_RUNNER = ROOT / "tools" / "run_tensorrt_persistent.py"
TRT_ASSEMBLER = ROOT / "tools" / "assemble_tensor_video.py"
TRT_ARTIFACTS = ROOT / "tensorrt_backend" / "artifacts"
TRT_ENGINE_21F = TRT_ARTIFACTS / "vae_decoder_tile_256_21f.rtxplan"
TRT_ENGINE_5F = TRT_ARTIFACTS / "vae_decoder_tile_512_5f.rtxplan"
TRT_POSTPROCESS = ROOT / "tools" / "postprocess_tensor_video.py"
TRT_ENCODER_5F = TRT_ARTIFACTS / "vae_encoder_5f_tile512.rtxplan"
TRT_ENCODER_21F = TRT_ARTIFACTS / "vae_encoder_21f_tile512.rtxplan"


MODEL_FILES = {
    "3B FP16 — best 3B quality": "seedvr2_ema_3b_fp16.safetensors",
    "3B FP8 — faster / less VRAM": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "7B FP8 mixed — sharper": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "7B Sharp FP16 — maximum quality": "seedvr2_ema_7b_sharp_fp16.safetensors",
}


@dataclass(frozen=True)
class RenderSettings:
    resolution: int
    max_resolution: int
    batch_size: int
    seed: int
    model_label: str
    color_correction: str
    attention_mode: str
    blocks_to_swap: int
    vae_tiling: bool
    stop_before_vae: bool
    sharpen_enabled: bool
    sharpen_strength: float
    grain_enabled: bool
    grain_intensity: float
    grain_saturation: float
    microtexture_enabled: bool = False
    microtexture_strength: float = 0.60
    skin_finishing_enabled: bool = False
    skin_evenness: float = 0.25
    skin_smoothing: float = 0.20
    skin_redness: float = 0.15
    skin_shine: float = 0.15
    blemish_mode: str = "off"
    preserve_marks: bool = True
    seam_mode: str = "match"
    seam_frames: int = 2
    decoder_mode: str = "stable"


ProgressCallback = Callable[[float, str], None]


def _safe_print(value: str, *, end: str = "\n") -> None:
    """Log child-process output without letting Windows code pages abort a render."""
    try:
        print(value, end=end, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_value, end=end, flush=True)


def backend_status() -> tuple[bool, str]:
    if not SEEDVR_CLI.exists():
        return False, "SeedVR2 engine has not been installed yet."
    if not VENV_PYTHON.exists():
        return False, "The project virtual environment is missing."
    try:
        import torch
    except ImportError:
        return False, "SeedVR2 source is present, but the CUDA runtime is not installed."
    if not torch.cuda.is_available():
        return False, "PyTorch is installed, but CUDA is not available."
    return True, f"SeedVR2 ready on {torch.cuda.get_device_name(0)}."


def tensorrt_status() -> tuple[bool, str]:
    missing = [str(p) for p in (TRT_RUNNER, TRT_PERSISTENT_RUNNER, TRT_ASSEMBLER, TRT_POSTPROCESS, TRT_ENGINE_21F, TRT_ENGINE_5F, TRT_ENCODER_5F, TRT_ENCODER_21F) if not p.exists()]
    if missing:
        return False, "TensorRT VAE artifacts missing: " + ", ".join(missing)
    return True, "TensorRT Stable and Optimized decoders ready (temporal batches 5 and 21)."


def _progress_from_log(line: str) -> tuple[float, str] | None:
    patterns = (
        (r"Validating .+", 0.03, 0.00, "Validating model files"),
        (r"Processing (?:video|image):", 0.08, 0.00, "Reading source video"),
        (r"Phase 1: VAE encoding", 0.12, 0.00, "VAE encoding"),
        (r"Encoding batch (\d+)/(\d+)", 0.12, 0.14, "VAE encoding"),
        (r"Phase 2: DiT upscaling", 0.28, 0.00, "AI restoration"),
        (r"Upscaling batch (\d+)/(\d+)", 0.28, 0.49, "AI restoration"),
        (r"Phase 3: VAE decoding", 0.78, 0.00, "VAE decoding"),
        (r"Decoding batch (\d+)/(\d+)", 0.78, 0.11, "VAE decoding"),
        (r"Phase 4: Post-processing", 0.90, 0.00, "Color correction and post-processing"),
        (r"Post-processing batch (\d+)/(\d+)", 0.90, 0.07, "Post-processing"),
        (r"Output saved to:", 0.99, 0.00, "Encoding output video"),
    )
    for pattern, base, span, label in patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if not match:
            continue
        if match.lastindex == 2:
            current, total = int(match.group(1)), max(1, int(match.group(2)))
            return min(base + span * current / total, 0.98), f"{label}: batch {current} of {total}"
        return base, label
    return None


def render(
    backend_name: str,
    source: Path,
    output: Path,
    settings: RenderSettings,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    available, reason = backend_status()
    if not available:
        raise MediaError(reason + " Run the installer for your platform again.")

    use_tensorrt = "TensorRT" in backend_name and not settings.stop_before_vae
    if use_tensorrt:
        if settings.batch_size not in (5, 21):
            raise MediaError("SeedVR2 + TensorRT currently supports temporal batch sizes 5 and 21; use SeedVR2 (Legacy) for other sizes.")
        trt_available, trt_reason = tensorrt_status()
        if not trt_available:
            raise MediaError(trt_reason)

    model = MODEL_FILES[settings.model_label]
    capture_dir = output.parent / "vae_latents"
    command = [
        str(VENV_PYTHON), str(SEEDVR_CLI), str(source),
        "--output", str(output), "--model_dir", str(MODELS),
        "--dit_model", model, "--resolution", str(settings.resolution),
        "--max_resolution", str(settings.max_resolution),
        "--batch_size", str(settings.batch_size), "--seed", str(settings.seed),
        # Keep the final temporal batch at the model's requested 4n+1 size.
        # The decoder needs temporal context; undersized end batches can
        # produce a soft/blobby final frame.
        "--uniform_batch_size",
        "--color_correction", settings.color_correction,
        "--attention_mode", settings.attention_mode, "--video_backend", "ffmpeg",
    ]
    if settings.blocks_to_swap:
        command += ["--blocks_to_swap", str(settings.blocks_to_swap),
                    "--dit_offload_device", "cpu", "--swap_io_components"]
    if settings.vae_tiling:
        command += ["--vae_encode_tiled", "--vae_decode_tiled"]

    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    if use_tensorrt:
        child_env["SEEDVR2_TRT_ENCODER"] = "1"
        if settings.decoder_mode == "stable":
            child_env["SEEDVR2_TRT_ENCODER_LEGACY"] = "1"
        else:
            child_env.pop("SEEDVR2_TRT_ENCODER_LEGACY", None)
        if settings.decoder_mode == "optimized_fast":
            child_env["SEEDVR2_TRT_ENCODER_FAST"] = "1"
        else:
            child_env.pop("SEEDVR2_TRT_ENCODER_FAST", None)
    if settings.stop_before_vae or use_tensorrt:
        capture_dir.mkdir(parents=True, exist_ok=True)
        child_env["SEEDVR2_LATENT_CAPTURE_DIR"] = str(capture_dir)
        command += ["--stop_before_vae"]
    if progress_callback:
        progress_callback(0.01, "Starting SeedVR2")

    process = subprocess.Popen(
        command, cwd=SEEDVR_CLI.parent, env=child_env, text=True,
        encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output_tail: deque[str] = deque(maxlen=60)
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if use_tensorrt and "Stop-before-VAE test mode:" in line:
            line = line.replace(
                "Stop-before-VAE test mode: DiT complete; latent capture finished. Skipping VAE decode.",
                "SeedVR2 latent capture complete; handing off to TensorRT VAE decode.",
            )
        elif use_tensorrt and "Stop-before-VAE test complete; no output video written." in line:
            line = line.replace(
                "Stop-before-VAE test complete; no output video written.",
                "SeedVR2 stage complete; TensorRT will decode latents and assemble the output video.",
            )
        if use_tensorrt and "All upscaling processes completed successfully" in line:
            line = line.replace(
                "All upscaling processes completed successfully",
                "DiT latent generation completed; TensorRT VAE decode continuing",
            )
        elif use_tensorrt and "Process " in line and "terminating" in line:
            line = line.replace("Process ", "DiT process ").replace(" terminating - VRAM will be automatically freed", " finished; TensorRT stage remains active")
        output_tail.append(line)
        _safe_print(line)
        update = _progress_from_log(line)
        if progress_callback and update:
            progress_callback(*update)
    return_code = process.wait()
    if return_code:
        raise MediaError(f"SeedVR2 render failed:\n{'\n'.join(output_tail)}")
    if settings.stop_before_vae and not use_tensorrt:
        return output
    if use_tensorrt:
        from .tensorrt_pipeline import decode_postprocess_and_assemble
        return decode_postprocess_and_assemble(
            output, source, capture_dir, settings, child_env, progress_callback
        )
    if not output.exists():
        raise MediaError("SeedVR2 finished but did not create the requested output file")
    if progress_callback:
        progress_callback(1.0, "SeedVR2 render complete")
    return output


def reprocess_tensorrt(
    job_dir: Path,
    source: Path,
    output: Path,
    *,
    sharpen_enabled: bool,
    sharpen_strength: float,
    grain_enabled: bool,
    grain_intensity: float,
    grain_saturation: float,
    microtexture_enabled: bool,
    microtexture_strength: float,
    skin_finishing_enabled: bool,
    skin_evenness: float,
    skin_smoothing: float,
    skin_redness: float,
    skin_shine: float,
    blemish_mode: str,
    preserve_marks: bool,
    seed: int,
    seam_mode: str,
    seam_frames: int,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Rebuild a TensorRT result from saved decoded batches without rerunning SeedVR2."""
    decoded_dir = job_dir / "tensorrt_decoded"
    decoded_files = sorted(decoded_dir.glob("decoded_*.pt"))
    if not decoded_files:
        raise MediaError("This result has no saved raw TensorRT decoded batches to reprocess.")
    seam_mode = seam_mode if seam_mode in {"off", "noise", "match", "blend"} else "off"
    seam_frames = max(1, min(12, int(seam_frames)))
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    selected_files: list[Path] = []
    frame_start = 0
    reprocess_dir = job_dir / f"post-{output.stem}"
    reprocess_dir.mkdir(parents=True, exist_ok=True)
    if grain_enabled:
        import torch
    for index, decoded in enumerate(decoded_files, start=1):
        frame_count = 0
        if grain_enabled:
            payload = torch.load(decoded, map_location="cpu", weights_only=False)
            video = payload["video"] if isinstance(payload, dict) and "video" in payload else payload
            frame_count = int(payload.get("original_frames", video.shape[2])) if isinstance(payload, dict) else int(video.shape[2])
            del payload, video
        if sharpen_enabled or grain_enabled or microtexture_enabled or skin_finishing_enabled:
            processed = reprocess_dir / f"processed_{index:03d}.pt"
            command = [str(VENV_PYTHON), str(TRT_POSTPROCESS), str(decoded),
                       "--output", str(processed), "--frame-start", str(frame_start),
                       "--seed", str(seed),
                       "--sharpen-strength", str(sharpen_strength if sharpen_enabled else 0.0),
                       "--microtexture-strength", str(microtexture_strength if microtexture_enabled else 0.0),
                       "--skin-evenness", str(skin_evenness if skin_finishing_enabled else 0.0),
                       "--skin-smoothing", str(skin_smoothing if skin_finishing_enabled else 0.0),
                       "--skin-redness", str(skin_redness if skin_finishing_enabled else 0.0),
                       "--skin-shine", str(skin_shine if skin_finishing_enabled else 0.0),
                       "--blemish-mode", blemish_mode if skin_finishing_enabled else "off",
                       "--grain-intensity", str(grain_intensity if grain_enabled else 0.0),
                       "--grain-saturation", str(grain_saturation)]
            if skin_finishing_enabled and preserve_marks:
                command.append("--preserve-marks")
            result = subprocess.run(command, cwd=ROOT, env=child_env, text=True,
                                    encoding="utf-8", errors="replace", capture_output=True)
            _safe_print(result.stdout, end="")
            if result.returncode:
                raise MediaError(f"TensorRT post-processing failed:\n{result.stderr[-4000:]}")
            selected_files.append(processed)
        else:
            selected_files.append(decoded)
        frame_start += frame_count
        if progress_callback:
            progress_callback(0.75 * index / len(decoded_files), f"Post-processing batch {index} of {len(decoded_files)}")
    if progress_callback:
        progress_callback(0.8, "Applying seam treatment and assembling video")
    command = [str(VENV_PYTHON), str(TRT_ASSEMBLER), *map(str, selected_files),
               "--output", str(output), "--audio", str(source), "--fps", str(probe(source).fps),
               "--seam-mode", seam_mode, "--seam-frames", str(seam_frames)]
    result = subprocess.run(command, cwd=ROOT, env=child_env, text=True,
                            encoding="utf-8", errors="replace", capture_output=True)
    _safe_print(result.stdout, end="")
    if result.returncode:
        raise MediaError(f"TensorRT reassembly failed:\n{result.stderr[-4000:]}")
    if not output.exists():
        raise MediaError("Post-processing finished but did not create an output video.")
    if progress_callback:
        progress_callback(1.0, "Post-only reprocess complete")
    return output
