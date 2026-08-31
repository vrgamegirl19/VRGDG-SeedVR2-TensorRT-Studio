"""Stable and optimized TensorRT decode paths sharing post-processing/assembly."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from .cancellation import cancellation_requested
from .media import MediaError, probe, resolve_frame_rate
from .paths import ROOT, VENV_PYTHON
from .persistent_decoder import run_persistent_decoder


TRT_RUNNER = ROOT / "tools" / "run_tensorrt_tiled.py"
TRT_ASSEMBLER = ROOT / "tools" / "assemble_tensor_video.py"
TRT_POSTPROCESS = ROOT / "tools" / "postprocess_tensor_video.py"
TRT_ARTIFACTS = ROOT / "tensorrt_backend" / "artifacts"
TRT_ENGINE_21F = TRT_ARTIFACTS / "vae_decoder_tile_256_21f.rtxplan"
TRT_ENGINE_5F = TRT_ARTIFACTS / "vae_decoder_tile_512_5f.rtxplan"
ProgressCallback = Callable[[float, str], None]


def _unpadded_output_size(source: Path, resolution: int, max_resolution: int) -> tuple[int, int]:
    """Reproduce SideResize dimensions before DivisiblePad adds black pixels."""
    info = probe(source)
    if info.width >= info.height:
        height = int(resolution)
        width = int(resolution * info.width / info.height)
    else:
        width = int(resolution)
        height = int(resolution * info.height / info.width)
    if max_resolution > 0 and max(width, height) > max_resolution:
        scale = max_resolution / max(width, height)
        width, height = round(width * scale), round(height * scale)
    return max(2, width // 2 * 2), max(2, height // 2 * 2)


def _safe_print(value: str, *, end: str = "\n") -> None:
    try:
        print(value, end=end, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(value.encode(encoding, errors="replace").decode(encoding, errors="replace"), end=end, flush=True)


def assemble_command(
    selected_files: list[Path],
    output: Path,
    source: Path,
    *,
    fps: float,
    seam_mode: str,
    seam_frames: int,
    target_width: int = 0,
    target_height: int = 0,
) -> list[str]:
    """Build an assembler command that does not exceed the Windows argument limit."""
    list_path = output.parent / "assemble-inputs.txt"
    list_path.write_text("\n".join(str(path) for path in selected_files) + "\n", encoding="utf-8")
    command = [
        str(VENV_PYTHON), str(TRT_ASSEMBLER),
        "--input-list", str(list_path),
        "--output", str(output),
        "--audio", str(source),
        "--fps", f"{fps:.9g}",
        "--seam-mode", seam_mode,
        "--seam-frames", str(seam_frames),
    ]
    if target_width > 0:
        command.extend(["--target-width", str(target_width)])
    if target_height > 0:
        command.extend(["--target-height", str(target_height)])
    return command


def _profile_latents(latent_files: list[Path]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for index, latent_file in enumerate(latent_files, start=1):
        payload = torch.load(latent_file, map_location="cpu", weights_only=False)
        latent = payload["latent"]
        latent_frames = int(latent.shape[2])
        finite = torch.isfinite(latent).all().item()
        peak = float(latent.float().abs().amax().item()) if finite else float("inf")
        if not finite or peak > 128.0:
            raise MediaError(
                f"AI restoration batch {index} produced corrupt latent values "
                f"(peak={peak:g}). The batch was stopped before TensorRT decoding."
            )
        del payload, latent
        if latent_frames == 6:
            engine, tile, overlap = TRT_ENGINE_21F, 32, 12
        elif latent_frames == 2:
            engine, tile, overlap = TRT_ENGINE_5F, 64, 24
        else:
            raise MediaError(
                f"TensorRT VAE has no fixed engine for latent temporal size {latent_frames}; "
                "use batch size 5 or 21."
            )
        profiles.append({
            "latent": latent_file,
            "output": latent_file.parent.parent / "tensorrt_decoded" / f"decoded_{index:03d}.pt",
            "engine": engine,
            "tile": tile,
            "overlap": overlap,
        })
    return profiles


def _stable_decode(
    profiles: list[dict[str, Any]], child_env: dict[str, str], progress_callback: ProgressCallback | None
) -> list[Path]:
    decoded_files: list[Path] = []
    for index, profile in enumerate(profiles, start=1):
        decoded = Path(profile["output"])
        command = [
            str(VENV_PYTHON), str(TRT_RUNNER), str(profile["engine"]), str(profile["latent"]),
            "--tile-latent", str(profile["tile"]), "--overlap-latent", str(profile["overlap"]),
            "--streams", "1", "--output", str(decoded),
        ]
        if progress_callback:
            progress_callback(
                0.78 + 0.12 * (index - 1) / len(profiles),
                f"Stable TensorRT VAE decoding batch {index} of {len(profiles)}",
            )
        result = subprocess.run(
            command, cwd=ROOT, env=child_env, text=True, encoding="utf-8", errors="replace", capture_output=True
        )
        _safe_print(result.stdout, end="")
        if result.returncode:
            raise MediaError(f"TensorRT VAE decode failed:\n{result.stderr[-4000:]}")
        decoded_files.append(decoded)
    return decoded_files


def _optimized_decode(
    profiles: list[dict[str, Any]], manifest_path: Path, child_env: dict[str, str],
    progress_callback: ProgressCallback | None,
) -> list[Path]:
    engines = {str(profile["engine"]) for profile in profiles}
    tiles = {int(profile["tile"]) for profile in profiles}
    overlaps = {int(profile["overlap"]) for profile in profiles}
    if len(engines) != 1 or len(tiles) != 1 or len(overlaps) != 1:
        raise MediaError("Optimized decoder requires one uniform TensorRT temporal profile")
    manifest = {
        "version": 1,
        "engine": next(iter(engines)),
        "tile_latent": next(iter(tiles)),
        "overlap_latent": next(iter(overlaps)),
        "jobs": [{"latent": str(profile["latent"]), "output": str(profile["output"])} for profile in profiles],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(0.78, f"Starting optimized TensorRT decoder for {len(profiles)} batches")
    run_persistent_decoder(manifest_path, len(profiles), child_env, progress_callback)
    outputs = [Path(profile["output"]) for profile in profiles]
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise MediaError("Optimized decoder did not create: " + ", ".join(missing))
    return outputs


def decode_postprocess_and_assemble(
    output: Path,
    source: Path,
    capture_dir: Path,
    settings: Any,
    child_env: dict[str, str],
    progress_callback: ProgressCallback | None,
) -> Path:
    decoded_dir = output.parent / "tensorrt_decoded"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    latent_files = sorted(capture_dir.glob("vae_latent_*.pt"))
    if not latent_files:
        raise MediaError(f"TensorRT render produced no VAE latents in {capture_dir}")
    profiles = _profile_latents(latent_files)
    requested = str(getattr(settings, "decoder_mode", "optimized_fast")).lower()
    requested = requested if requested in {"stable", "optimized", "optimized_fast"} else "optimized_fast"
    record: dict[str, Any] = {
        "version": 1,
        "requested": requested,
        "used": None,
        "fallback_reason": None,
        "source": str(source),
        "output": str(output),
        "settings": asdict(settings),
    }
    record_path = output.parent / "decoder-manifest.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if requested in {"optimized", "optimized_fast"}:
        try:
            decoded_files = _optimized_decode(
                profiles, output.parent / "persistent-decode-manifest.json", child_env, progress_callback
            )
            record["used"] = "optimized-fast" if requested == "optimized_fast" else "optimized"
        except Exception as exc:
            if cancellation_requested():
                raise MediaError("Render cancelled") from exc
            record["fallback_reason"] = str(exc)
            _safe_print(f"Optimized decoder failed; retrying saved latents with Stable decoder: {exc}")
            if progress_callback:
                progress_callback(0.78, "Optimized decoder failed; retrying saved latents with Stable decoder")
            for profile in profiles:
                Path(profile["output"]).unlink(missing_ok=True)
            decoded_files = _stable_decode(profiles, child_env, progress_callback)
            record["used"] = "stable-fallback"
    else:
        decoded_files = _stable_decode(profiles, child_env, progress_callback)
        record["used"] = "stable"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    selected_files: list[Path] = []
    for index, decoded in enumerate(decoded_files, start=1):
        if settings.sharpen_enabled or settings.grain_enabled or settings.microtexture_enabled or settings.skin_finishing_enabled:
            processed = decoded_dir / f"processed_{index:03d}.pt"
            effects = [
                str(VENV_PYTHON), str(TRT_POSTPROCESS), str(decoded), "--output", str(processed),
                "--frame-start", str((index - 1) * settings.batch_size), "--seed", str(settings.seed),
                "--sharpen-strength", str(settings.sharpen_strength if settings.sharpen_enabled else 0.0),
                "--microtexture-strength", str(settings.microtexture_strength if settings.microtexture_enabled else 0.0),
                "--skin-evenness", str(settings.skin_evenness if settings.skin_finishing_enabled else 0.0),
                "--skin-smoothing", str(settings.skin_smoothing if settings.skin_finishing_enabled else 0.0),
                "--skin-redness", str(settings.skin_redness if settings.skin_finishing_enabled else 0.0),
                "--skin-shine", str(settings.skin_shine if settings.skin_finishing_enabled else 0.0),
                "--blemish-mode", settings.blemish_mode if settings.skin_finishing_enabled else "off",
                "--grain-intensity", str(settings.grain_intensity if settings.grain_enabled else 0.0),
                "--grain-saturation", str(settings.grain_saturation),
            ]
            if settings.skin_finishing_enabled and settings.preserve_marks:
                effects.append("--preserve-marks")
            effect_result = subprocess.run(
                effects, cwd=ROOT, env=child_env, text=True, encoding="utf-8", errors="replace", capture_output=True
            )
            _safe_print(effect_result.stdout, end="")
            if effect_result.returncode:
                raise MediaError(f"TensorRT post-processing failed:\n{effect_result.stderr[-4000:]}")
            selected_files.append(processed)
        else:
            selected_files.append(decoded)

    source_fps = resolve_frame_rate(getattr(settings, "source_fps", 0.0), probe(source).fps)
    target_width, target_height = _unpadded_output_size(
        source, settings.resolution, settings.max_resolution
    )
    assemble = assemble_command(
        selected_files, output, source,
        fps=source_fps, seam_mode=settings.seam_mode, seam_frames=settings.seam_frames,
        target_width=target_width, target_height=target_height,
    )
    if progress_callback:
        progress_callback(0.96, f"Assembling {len(selected_files)} decoded batches")
    result = subprocess.run(
        assemble, cwd=ROOT, env=child_env, text=True, encoding="utf-8", errors="replace", capture_output=True
    )
    _safe_print(result.stdout, end="")
    if result.returncode:
        raise MediaError(f"TensorRT video assembly failed:\n{result.stderr[-4000:]}")
    if progress_callback:
        progress_callback(1.0, f"TensorRT render complete · decoder: {record['used']}")
    return output
