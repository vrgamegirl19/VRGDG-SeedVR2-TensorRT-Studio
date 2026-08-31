"""Small local API/static host for the next SeedVR Studio frontend.

This intentionally does not replace the Gradio app or render pipeline yet.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4
import json
import os
import signal
import shutil
import subprocess
import time
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from seedvr_studio.backend import MODEL_FILES, backend_status, render, reprocess_tensorrt, tensorrt_status
from seedvr_studio.cancellation import begin_render, cancel_current_render, cancellation_requested
from seedvr_studio.jobs import _settings
from seedvr_studio.media import FrameRateRequiredError, concat_videos, make_center_crop, make_clip, probe, resolve_frame_rate, trim_video_start
from seedvr_studio.paths import ensure_workspace
from seedvr_studio.updater import UpdateError, check_for_updates, launch_updater

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
OUTPUTS = ROOT / "outputs"

app = FastAPI(title="SeedVR Studio API", version="0.1.0")
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = Lock()

@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    response = await call_next(request)
    if request.url.path in {"/", "/index.html", "/app.js", "/styles.css"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response




def _update(job_id: str, **values: object) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(values)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "seedvr-js", "render_pipeline": "ready"}


@app.get("/api/config")
def config() -> dict[str, object]:
    installed, engine = backend_status()
    trt_installed, trt = tensorrt_status()
    return {
        "models": list(MODEL_FILES),
        "output_presets": {
            "Original / enhancement only": {"resolution": "source", "max_resolution": "source", "description": "Keep the source dimensions; enhancement only."},
            "1K / 1080p": {"resolution": 1080, "max_resolution": 1920, "description": "1080p-class output."},
            "2K / 1440p": {"resolution": 1440, "max_resolution": 2560, "description": "1440p-class output."},
            "4K / 2160p": {"resolution": 2160, "max_resolution": 3840, "description": "2160p / UHD output."},
        },
        "crop_policies": ["Preserve original aspect ratio", "Center crop to 16:9"],
        "batch_sizes": {
            "SeedVR2 + TensorRT": [5, 21],
            "SeedVR2 (Legacy)": [1, 5, 9, 13, 17, 21, 33, 45],
        },
        "seam_modes": {"off": "Off / original output", "noise": "Noise match — gentle", "match": "Color + noise match", "blend": "Boundary dissolve — strongest"},
        "features": {"skin_finishing": True, "open_output_folder": True, "saved_settings": True, "persistent_decoder": True},
        "decoder_modes": {"stable": "Stable", "optimized": "Optimized (Beta)", "optimized_fast": "Optimized Fast (Beta)"},
        "backend": {"seedvr": {"ready": installed, "message": engine}, "tensorrt": {"ready": trt_installed, "message": trt}},
        "presets": {
            "Custom / manual": None,
            "8 GB VRAM — low memory": {"model": "3B FP8 — faster / less VRAM", "resolution": 480, "max_resolution": 1920, "batch_size": 5, "attention": "sageattn_2", "blocks": 36, "vae_tiling": True},
            "12 GB VRAM — mainstream": {"model": "3B FP8 — faster / less VRAM", "resolution": 720, "max_resolution": 1920, "batch_size": 5, "attention": "sageattn_2", "blocks": 24, "vae_tiling": True},
            "16 GB VRAM — high memory": {"model": "3B FP8 — faster / less VRAM", "resolution": 1080, "max_resolution": 2560, "batch_size": 5, "attention": "sageattn_2", "blocks": 12, "vae_tiling": True},
            "24 GB VRAM — enthusiast": {"model": "3B FP8 — faster / less VRAM", "resolution": 1080, "max_resolution": 3840, "batch_size": 21, "attention": "sageattn_2", "blocks": 0, "vae_tiling": False},
            "32 GB+ VRAM — workstation": {"model": "3B FP16 — best 3B quality", "resolution": 1440, "max_resolution": 3840, "batch_size": 21, "attention": "sageattn_2", "blocks": 0, "vae_tiling": False},
        },
    }


@app.get("/api/outputs")
def outputs() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    fps_by_directory: dict[Path, float] = {}
    for path in sorted(OUTPUTS.rglob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True):
        relative = path.relative_to(OUTPUTS).as_posix()
        if path.parent not in fps_by_directory:
            source_candidates = sorted(candidate for candidate in path.parent.glob("source.*") if candidate.is_file())
            try:
                fps_by_directory[path.parent] = probe(source_candidates[0] if source_candidates else path).fps
            except Exception:
                fps_by_directory[path.parent] = 30.0
        result.append({
            "name": path.name,
            "path": relative,
            "url": f"/media/{relative}",
            "bytes": path.stat().st_size,
            "modified": path.stat().st_mtime,
            "fps": fps_by_directory[path.parent],
            "reprocessable": any((path.parent / "tensorrt_decoded").glob("decoded_*.pt")),
        })
    return result


def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes", "on"}


# The launcher (scripts/launch_js.ps1) redirects the uvicorn process' standard
# streams to these files next to the rendered outputs.
SERVER_LOGS = {
    "stdout": OUTPUTS / "js_server.log",
    "stderr": OUTPUTS / "js_server_error.log",
}


def _tail_lines(path: Path, count: int, small_file_bytes: int = 4 * 1024 * 1024) -> tuple[list[str], bool]:
    """Return the final ``count`` lines of ``path`` and whether more lines exist above them."""
    size = path.stat().st_size
    if size <= small_file_bytes:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text.splitlines()[-count:], False
    # Large file: walk backwards in 256 KiB chunks so we never load the whole log.
    chunk = 256 * 1024
    collected: list[str] = []
    offset = size
    with path.open("rb") as handle:
        while offset > 0 and len(collected) < count:
            start = max(0, offset - chunk)
            handle.seek(start)
            data = handle.read(offset - start)
            offset = start
            text = data.decode("utf-8", errors="replace")
            if start > 0:
                cut = text.find("\n")
                if cut == -1:
                    continue  # whole chunk was one truncated line; keep walking back
                text = text[cut + 1:]
            collected = text.splitlines() + collected
    return collected[-count:], True


@app.get("/api/logs/server")
def server_log(source: str = "stdout", lines: int = 300) -> dict[str, object]:
    """Tail the local server log shown in the UI's log pane."""
    name = source if source in SERVER_LOGS else "stdout"
    path = SERVER_LOGS[name]
    count = max(1, min(lines, 2000))
    if not path.is_file():
        return {"source": name, "exists": False, "size": 0, "lines": [], "truncated": False}
    tail, truncated = _tail_lines(path, count)
    return {"source": name, "exists": True, "size": path.stat().st_size, "lines": tail, "truncated": truncated}


@app.get("/api/logs/job")
def job_log(job_id: str = "", lines: int = 300) -> dict[str, object]:
    """Tail a render job's render.log for the UI's log pane.

    The job id is resolved through the in-memory registry first, then by
    scanning outputs/js-*-<id> folders so logs of jobs from before a server
    restart are still reachable.
    """
    count = max(1, min(lines, 2000))
    job_dir = None
    if job_id:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is not None:
            job_dir = Path(str(job.get("_job_dir", "")))
        elif OUTPUTS.is_dir():
            for directory in sorted(OUTPUTS.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
                if directory.is_dir() and directory.name.endswith(f"-{job_id}"):
                    job_dir = directory
                    break
    log_path = None
    if job_dir is not None:
        for candidate in (job_dir / "render.log", job_dir / f"reprocess-{job_id}.log"):
            if candidate.is_file():
                log_path = candidate
                break
    if log_path is None:
        return {"source": job_id, "name": "", "exists": False, "size": 0, "lines": [], "truncated": False}
    tail, truncated = _tail_lines(log_path, count)
    return {"source": job_id, "name": log_path.parent.name, "exists": True, "size": log_path.stat().st_size, "lines": tail, "truncated": truncated}


CHUNK_LENGTH_OPTIONS = (30, 60, 120, 180, 300, 600, 900, 1800)


def _automatic_chunk_seconds(info, batch_size: int) -> int:
    """Choose about ten checkpoints using frame count and temporal-batch alignment."""
    fps = max(float(info.fps), 1.0)
    total_frames = max(1, round(float(info.duration) * fps))
    temporal_batch = max(1, int(batch_size))
    target_frames = max(temporal_batch, round(total_frames / 10 / temporal_batch) * temporal_batch)
    desired_seconds = target_frames / fps
    return min(CHUNK_LENGTH_OPTIONS, key=lambda seconds: abs(seconds - desired_seconds))


def _render_chunked(job_id: str, source: Path, job_dir: Path, output: Path, settings, info, values: dict[str, object], report) -> None:
    """Render long videos in checkpointed temporal chunks, resuming completed work."""
    if settings.stop_before_vae:
        raise RuntimeError("Chunked rendering is unavailable when Stop before VAE is enabled.")
    requested_chunk_seconds = float(values.get("chunk_seconds", 0))
    auto_chunking = requested_chunk_seconds <= 0
    chunk_seconds = float(_automatic_chunk_seconds(info, settings.batch_size) if auto_chunking else max(5.0, requested_chunk_seconds))
    if auto_chunking:
        frames_per_chunk = round(chunk_seconds * max(float(info.fps), 1.0))
        report(0.001, f"Auto chunk length selected: {chunk_seconds / 60:g} minutes ({frames_per_chunk} frames per chunk)")
    overlap_frames = max(0, min(20, int(settings.batch_size) - 1))
    overlap_seconds = overlap_frames / max(info.fps, 1.0)
    checkpoint_path = job_dir / "chunk-manifest.json"
    manifest: dict[str, object] = {"chunk_seconds": chunk_seconds, "chunk_mode": "auto" if auto_chunking else "manual", "source_frames": round(info.duration * info.fps), "overlap_frames": overlap_frames, "duration": info.duration, "completed": [], "chunks": []}
    if checkpoint_path.exists():
        try:
            existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if existing.get("chunk_seconds") == chunk_seconds and existing.get("overlap_frames") == overlap_frames:
                manifest.update(existing)
        except (OSError, ValueError, TypeError):
            pass
    total = max(1, int((info.duration + chunk_seconds - 1e-6) // chunk_seconds))
    completed = {int(index) for index in manifest.get("completed", [])}
    rendered: list[Path] = []
    for index in range(total):
        start = index * chunk_seconds
        if start >= info.duration:
            break
        chunk_end = min(info.duration, start + chunk_seconds)
        chunk_path = job_dir / "chunks" / f"chunk-{index:04d}.mp4"
        final_chunk = job_dir / "chunks" / f"final-{index:04d}.mp4"
        if index in completed and final_chunk.exists():
            rendered.append(final_chunk)
            report((index + 1) / total, f"Resuming: chunk {index + 1} of {total} already complete")
            continue
        render_start = max(0.0, start - overlap_seconds) if index else 0.0
        render_length = min(info.duration - render_start, (chunk_end - start) + (start - render_start))
        report(index / total, f"Rendering chunk {index + 1} of {total}")
        make_clip(source, job_dir / "chunks" / f"source-{index:04d}.mp4", render_start, render_length)
        # The backend name is supplied through the values map; this keeps chunk rendering identical to normal rendering.
        from seedvr_studio.backend import render as render_backend
        render_backend(str(values["backend_name"]), job_dir / "chunks" / f"source-{index:04d}.mp4", chunk_path, settings,
                       lambda progress, message: report((index + min(progress, 0.99)) / total, f"Chunk {index + 1}/{total}: {message}"))
        if index and overlap_seconds > 0:
            trim_video_start(chunk_path, final_chunk, overlap_seconds)
        else:
            final_chunk = chunk_path
        rendered.append(final_chunk)
        completed.add(index)
        manifest["completed"] = sorted(completed)
        manifest["chunks"] = [str(path.relative_to(job_dir)) for path in rendered]
        checkpoint_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _update(job_id, chunk_index=index + 1, chunk_total=total, resumable=True, checkpoint=str(checkpoint_path), log_file=str(job_dir / "render.log"))
    report(0.99, "Assembling completed chunks")
    concat_videos(rendered, output)
    manifest["assembled"] = str(output.relative_to(job_dir))
    checkpoint_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _render_job(job_id: str, source: Path, job_dir: Path, values: dict[str, object]) -> None:
    begin_render()
    started = time.perf_counter()
    try:
        source_info = probe(source)
        crop_policy = str(values.get("crop_policy", "Preserve original aspect ratio"))
        if crop_policy == "Center crop to 16:9":
            source = make_center_crop(source, job_dir / f"cropped{source.suffix or '.mp4'}")
        info = probe(source)
        resolved_fps = resolve_frame_rate(info.fps, values.get("source_fps", 0.0))
        info = replace(info, fps=resolved_fps, frames=info.frames or round(info.duration * resolved_fps))
        output_preset = str(values.get("output_preset", "1K / 1080p"))
        if output_preset in {"Original / enhancement only", "Original enhancement only"}:
            values["resolution"] = min(info.width, info.height)
            values["max_resolution"] = max(info.width, info.height)
        else:
            preset_sizes = {"1K / 1080p": (1080, 1920), "2K / 1440p": (1440, 2560), "4K / 2160p": (2160, 3840)}
            values["resolution"], values["max_resolution"] = preset_sizes.get(output_preset, (1080, 1920))
        job_type = str(values["job_type"])
        if job_type == "preview":
            start = min(max(float(values["preview_start"]), 0.0), max(info.duration - 0.1, 0.0))
            length = min(float(values["preview_seconds"]), max(info.duration - start, 0.1))
            source_for_render = make_clip(source, job_dir / "preview-source.mp4", start, length)
            output = job_dir / "restored.mp4"
        else:
            source_for_render = source
            output = job_dir / f"{source.stem}-restored.mp4"
        settings = _settings(values["resolution"], values["max_resolution"], values["batch_size"], values["seed"], values["model_label"], values["color_correction"], values["attention_mode"], values["blocks_to_swap"], values["vae_tiling"], values["stop_before_vae"], values["sharpen_enabled"], values["sharpen_strength"], values["grain_enabled"], values["grain_intensity"], values["grain_saturation"], values.get("microtexture_enabled", False), values.get("microtexture_strength", .60), values.get("skin_finishing_enabled", False), values.get("skin_evenness", .25), values.get("skin_smoothing", .20), values.get("skin_redness", .15), values.get("skin_shine", .15), values.get("blemish_mode", "off"), values.get("preserve_marks", True), values.get("seam_mode", "match"), values.get("seam_frames", 2), values.get("decoder_mode", "optimized_fast"), info.fps)
        (job_dir / "job-manifest.json").write_text(json.dumps({
            "version": 1, "job_id": job_id, "job_type": job_type,
            "backend": str(values["backend_name"]), "source": str(source_for_render),
            "output": str(output), "settings": settings.__dict__,
        }, indent=2), encoding="utf-8")
        _update(job_id, status="running", progress=0.02, message="Starting SeedVR2", fps=info.fps, duration=length if job_type == "preview" else info.duration, started_at=time.time())
        log_path = job_dir / "render.log"
        def report(progress: float, message: str) -> None:
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{progress:.3f}] {message}\n")
            except OSError:
                pass
            _update(job_id, progress=float(progress), message=message)
        if _bool(values.get("chunked_render", False)) and job_type == "full":
            _render_chunked(job_id, source_for_render, job_dir, output, settings, info, values, report)
        else:
            render(str(values["backend_name"]), source_for_render, output, settings, report)
        if settings.stop_before_vae:
            _update(job_id, status="complete", progress=1.0, message="Latents captured; stopped before VAE decode", elapsed_seconds=time.perf_counter() - started, original_url=f"/media/{source_for_render.relative_to(OUTPUTS).as_posix()}", restored_url=None, output=None)
            return
        _update(job_id, status="complete", progress=1.0, message="Render complete", elapsed_seconds=time.perf_counter() - started, original_url=f"/media/{source_for_render.relative_to(OUTPUTS).as_posix()}", restored_url=f"/media/{output.relative_to(OUTPUTS).as_posix()}", output=str(output), output_relative=output.relative_to(OUTPUTS).as_posix(), reprocessable=any((job_dir / "tensorrt_decoded").glob("decoded_*.pt")))
    except Exception as exc:
        log_path = job_dir / "render.log"
        details = "\n".join([f"SeedVR Studio render failure at {time.strftime('%Y-%m-%d %H:%M:%S')}", repr(exc), traceback.format_exc()])
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(details + "\n")
        except OSError:
            pass
        final_status = "cancelled" if cancellation_requested() else "error"
        _update(job_id, status=final_status, message=str(exc), elapsed_seconds=time.perf_counter() - started, error=str(exc), failure_reason=str(exc), failure_code="fps_required" if isinstance(exc, FrameRateRequiredError) else None, log_file=str(log_path), resumable=_bool(values.get("chunked_render", False)))


def _reprocess_job(job_id: str, source_output: Path, values: dict[str, object]) -> None:
    started = time.perf_counter()
    job_dir = source_output.parent
    try:
        source_candidates = sorted(path for path in job_dir.glob("source.*") if "restored" not in path.stem and path.is_file())
        if not source_candidates:
            raise RuntimeError("The original source video for this result is missing.")
        source = source_candidates[0]
        output = job_dir / f"{source_output.stem}-post-{job_id}.mp4"
        log_path = job_dir / f"reprocess-{job_id}.log"
        _update(job_id, status="running", progress=0.01, message="Loading saved TensorRT batches", started_at=time.time(), log_file=str(log_path))
        def report(progress: float, message: str) -> None:
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{progress:.3f}] {message}\n")
            except OSError:
                pass
            _update(job_id, progress=float(progress), message=message)
        reprocess_tensorrt(job_dir, source, output,
                           sharpen_enabled=_bool(values.get("sharpen_enabled", False)), sharpen_strength=float(values.get("sharpen_strength", .25)),
                           grain_enabled=_bool(values.get("grain_enabled", False)), grain_intensity=float(values.get("grain_intensity", .02)), grain_saturation=float(values.get("grain_saturation", .5)),
                           microtexture_enabled=_bool(values.get("microtexture_enabled", False)), microtexture_strength=float(values.get("microtexture_strength", .60)),
                           skin_finishing_enabled=_bool(values.get("skin_finishing_enabled", False)), skin_evenness=float(values.get("skin_evenness", .25)), skin_smoothing=float(values.get("skin_smoothing", .20)),
                           skin_redness=float(values.get("skin_redness", .15)), skin_shine=float(values.get("skin_shine", .15)), blemish_mode=str(values.get("blemish_mode", "off")), preserve_marks=_bool(values.get("preserve_marks", True)),
                           seed=int(values.get("seed", 42)), seam_mode=str(values.get("seam_mode", "match")), seam_frames=int(values.get("seam_frames", 2)), progress_callback=report)
        _update(job_id, status="complete", progress=1.0, message="Post-only reprocess complete", elapsed_seconds=time.perf_counter() - started,
                original_url=f"/media/{source.relative_to(OUTPUTS).as_posix()}", restored_url=f"/media/{output.relative_to(OUTPUTS).as_posix()}",
                output=str(output), output_relative=output.relative_to(OUTPUTS).as_posix(), reprocessable=True, fps=probe(source).fps)
    except Exception as exc:
        log_path = job_dir / f"reprocess-{job_id}.log"
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{repr(exc)}\n{traceback.format_exc()}\n")
        except OSError:
            pass
        _update(job_id, status="error", message=str(exc), error=str(exc), failure_reason=str(exc), elapsed_seconds=time.perf_counter() - started, log_file=str(log_path), resumable=False)


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    job_type: str = Form("preview"), backend_name: str = Form("SeedVR2 + TensorRT"),
    output_preset: str = Form("1K / 1080p"), crop_policy: str = Form("Preserve original aspect ratio"),
    preview_start: float = Form(0), preview_seconds: float = Form(3), resolution: int = Form(1080), max_resolution: int = Form(3840), batch_size: int = Form(21), seed: int = Form(42),
    model_label: str = Form("3B FP16 — best 3B quality"), color_correction: str = Form("none"), attention_mode: str = Form("sageattn_2"), blocks_to_swap: int = Form(0), vae_tiling: str = Form("false"), stop_before_vae: str = Form("false"),
    sharpen_enabled: str = Form("false"), sharpen_strength: float = Form(.25), grain_enabled: str = Form("false"), grain_intensity: float = Form(.02), grain_saturation: float = Form(.5),
    microtexture_enabled: str = Form("false"), microtexture_strength: float = Form(.60),
    skin_finishing_enabled: str = Form("false"), skin_evenness: float = Form(.25), skin_smoothing: float = Form(.20),
    skin_redness: float = Form(.15), skin_shine: float = Form(.15), blemish_mode: str = Form("off"), preserve_marks: str = Form("true"),
    seam_mode: str = Form("match"), seam_frames: int = Form(2),
    chunked_render: str = Form("false"), chunk_seconds: float = Form(0), decoder_mode: str = Form("optimized_fast"), source_fps: float = Form(0),
) -> dict[str, str]:
    ensure_workspace()
    job_id = uuid4().hex[:10]
    job_dir = OUTPUTS / f"js-{job_type}-{job_id}"
    job_dir.mkdir(parents=True)
    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    source = job_dir / f"source{suffix}"
    with source.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    values = locals().copy()
    values.update({"model_label": model_label, "vae_tiling": _bool(vae_tiling), "stop_before_vae": _bool(stop_before_vae), "sharpen_enabled": _bool(sharpen_enabled), "grain_enabled": _bool(grain_enabled), "microtexture_enabled": _bool(microtexture_enabled), "skin_finishing_enabled": _bool(skin_finishing_enabled), "preserve_marks": _bool(preserve_marks), "chunked_render": _bool(chunked_render), "chunk_seconds": float(chunk_seconds) if float(chunk_seconds) > 0 else 0.0})
    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "message": "Queued", "job_type": job_type, "resumable": _bool(chunked_render) and job_type == "full", "_source": str(source), "_job_dir": str(job_dir), "_values": values}
    Thread(target=_render_job, args=(job_id, source, job_dir, values), daemon=True).start()
    return {"id": job_id}


@app.post("/api/reprocess")
def create_reprocess_job(
    output_path: str = Form(...), seed: int = Form(42),
    sharpen_enabled: str = Form("false"), sharpen_strength: float = Form(.25),
    grain_enabled: str = Form("false"), grain_intensity: float = Form(.02), grain_saturation: float = Form(.5),
    microtexture_enabled: str = Form("false"), microtexture_strength: float = Form(.60),
    skin_finishing_enabled: str = Form("false"), skin_evenness: float = Form(.25), skin_smoothing: float = Form(.20),
    skin_redness: float = Form(.15), skin_shine: float = Form(.15), blemish_mode: str = Form("off"), preserve_marks: str = Form("true"),
    seam_mode: str = Form("match"), seam_frames: int = Form(2),
) -> dict[str, str]:
    relative = output_path.removeprefix("/media/").lstrip("/\\")
    source_output = (OUTPUTS / relative).resolve()
    if OUTPUTS.resolve() not in source_output.parents or not source_output.is_file():
        raise HTTPException(status_code=404, detail="Selected output was not found")
    if not any((source_output.parent / "tensorrt_decoded").glob("decoded_*.pt")):
        raise HTTPException(status_code=400, detail="This result has no reusable TensorRT decoded batches")
    job_id = uuid4().hex[:10]
    values = {"seed": seed, "sharpen_enabled": _bool(sharpen_enabled), "sharpen_strength": sharpen_strength,
              "grain_enabled": _bool(grain_enabled), "grain_intensity": grain_intensity, "grain_saturation": grain_saturation,
              "microtexture_enabled": _bool(microtexture_enabled), "microtexture_strength": microtexture_strength,
              "skin_finishing_enabled": _bool(skin_finishing_enabled), "skin_evenness": skin_evenness, "skin_smoothing": skin_smoothing,
              "skin_redness": skin_redness, "skin_shine": skin_shine, "blemish_mode": blemish_mode, "preserve_marks": _bool(preserve_marks),
              "seam_mode": seam_mode, "seam_frames": seam_frames}
    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "message": "Queued for post-only reprocess", "job_type": "reprocess", "resumable": False,
                        "_source": str(source_output), "_job_dir": str(source_output.parent), "_values": values}
    Thread(target=_reprocess_job, args=(job_id, source_output, values), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {key: value for key, value in job.items() if not key.startswith("_")}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.get("resumable"):
            raise HTTPException(status_code=400, detail="This job was not configured for resumable chunk rendering")
        if job.get("status") not in {"error", "cancelled"}:
            raise HTTPException(status_code=409, detail="Only a failed or cancelled job can be resumed")
        source = Path(str(job["_source"]))
        job_dir = Path(str(job["_job_dir"]))
        values = dict(job["_values"])
        job["status"] = "queued"
        job["message"] = "Resuming from the last completed chunk"
        job["error"] = None
        job["failure_reason"] = None
    Thread(target=_render_job, args=(job_id, source, job_dir, values), daemon=True).start()
    return {"ok": True, "id": job_id}


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = Path(str(job.get("_job_dir", ""))) / "render.log"
    if not log_path.is_file():
        raise HTTPException(status_code=404, detail="No render log has been saved yet")
    return FileResponse(log_path, media_type="text/plain", filename=f"seedvr-{job_id}.log")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
    message = cancel_current_render()
    with JOBS_LOCK:
        started_at = JOBS[job_id].get("started_at")
    elapsed = time.time() - float(started_at) if started_at else 0.0
    _update(job_id, status="cancelled", message=message, elapsed_seconds=elapsed)
    return {"ok": True, "message": message}


@app.get("/api/update/check")
def update_check() -> dict[str, object]:
    return check_for_updates(ROOT)


@app.post("/api/update/apply")
def apply_update() -> dict[str, object]:
    with JOBS_LOCK:
        active_render = any(job.get("status") in {"queued", "running"} for job in JOBS.values())
    if active_render:
        raise HTTPException(status_code=409, detail="Finish or stop the active render before updating.")
    status = check_for_updates(ROOT)
    if not status["supported"]:
        raise HTTPException(status_code=409, detail=str(status["message"]))
    if not status["update_available"]:
        raise HTTPException(status_code=409, detail="SeedVR Studio is already up to date.")
    try:
        launch_updater(ROOT, os.getpid())
    except UpdateError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    def stop_for_update() -> None:
        time.sleep(0.75)
        os.kill(os.getpid(), signal.SIGTERM)
    Thread(target=stop_for_update, daemon=True).start()
    return {"ok": True, "message": "Updater started. SeedVR Studio will close and restart automatically."}


@app.post("/api/shutdown")
def shutdown() -> dict[str, object]:
    """Release active inference resources and stop the local JS Studio server."""
    message = cancel_current_render()
    def stop_server() -> None:
        time.sleep(0.35)
        os.kill(os.getpid(), signal.SIGTERM)
    Thread(target=stop_server, daemon=True).start()
    return {"ok": True, "message": message}


@app.post("/api/open-folder")
def open_output_folder(output_path: str = Form(...)) -> dict[str, object]:
    """Open Explorer with a workspace video selected."""
    relative = output_path.removeprefix("/media/").lstrip("/\\")
    candidate = (OUTPUTS / relative).resolve()
    if OUTPUTS.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Selected output was not found")
    subprocess.Popen(["explorer.exe", f"/select,{candidate}"], close_fds=True)
    return {"ok": True}


@app.get("/media/{relative_path:path}")
def media(relative_path: str) -> FileResponse:
    candidate = (OUTPUTS / relative_path).resolve()
    if OUTPUTS.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(candidate)


@app.get("/app.js", include_in_schema=False)
def serve_app_js() -> FileResponse:
    # Windows can register .js as text/plain. Module scripts require a
    # JavaScript MIME type, so do not leave this asset to MIME guessing.
    return FileResponse(WEB / "app.js", media_type="application/javascript")


@app.get("/styles.css", include_in_schema=False)
def serve_styles_css() -> FileResponse:
    return FileResponse(WEB / "styles.css", media_type="text/css")


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
