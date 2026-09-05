from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


class FrameRateRequiredError(MediaError):
    pass


def resolve_frame_rate(detected_fps: float, override_fps: float = 0.0) -> float:
    """Return a usable source FPS or ask the UI to collect one from the user."""
    for candidate in (detected_fps, override_fps):
        try:
            fps = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fps) and 1.0 <= fps <= 240.0:
            return fps
    raise FrameRateRequiredError(
        "Frame rate could not be detected. Enter the source frame rate to continue."
    )


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    frames: int


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MediaError(f"{name} was not found in PATH")
    return found


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.returncode:
        tail = "\n".join(process.stdout.splitlines()[-30:])
        raise MediaError(f"Command failed ({process.returncode}):\n{tail}")
    return process


def probe(path: str | Path) -> VideoInfo:
    command = [
        _tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ]
    payload = json.loads(run(command).stdout)
    stream = payload["streams"][0]
    try:
        numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
        fps = float(numerator) / max(float(denominator), 1.0)
    except (AttributeError, TypeError, ValueError):
        fps = 0.0
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    try:
        frames = int(stream.get("nb_frames") or round(duration * fps))
    except (TypeError, ValueError):
        frames = round(duration * fps)
    return VideoInfo(duration, int(stream["width"]), int(stream["height"]), fps, frames)


def make_clip(source: Path, target: Path, start: float, duration: float) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _tool("ffmpeg"), "-y", "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(target),
    ]
    run(command)
    return target


def trim_video_start(source: Path, target: Path, start: float) -> Path:
    """Drop a temporal overlap from the front of a rendered chunk."""
    target.parent.mkdir(parents=True, exist_ok=True)
    offset = max(0.0, float(start))
    run([
        _tool("ffmpeg"), "-y", "-i", str(source),
        "-vf", f"trim=start={offset:.6f},setpts=PTS-STARTPTS",
        "-af", f"atrim=start={offset:.6f},asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-preset", "fast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(target),
    ])
    return target


def concat_videos(chunks: list[Path], target: Path) -> Path:
    """Join rendered chunks with one continuous timeline.

    Stream-copy concatenation preserves per-chunk timestamps. Re-encoding the
    assembled timeline gives FFmpeg one timestamp domain for both streams and
    prevents audio from outliving the final video frame.
    """
    if not chunks:
        raise MediaError("No rendered chunks are available to assemble")
    target.parent.mkdir(parents=True, exist_ok=True)
    listing = target.parent / "concat-list.txt"
    listing.write_text("\n".join(f"file '{path.resolve().as_posix().replace("'", "'\\''")}'" for path in chunks), encoding="utf-8")
    try:
        run([
            _tool("ffmpeg"), "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
            "-map", "0:v:0", "-map", "0:a?",
            "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
            "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "fast", "-crf", "15",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(target),
        ])
    finally:
        listing.unlink(missing_ok=True)
    return target


def make_center_crop(source: Path, target: Path, aspect_ratio: float = 16 / 9) -> Path:
    """Create a centered crop without changing the source's frame rate or audio."""
    target.parent.mkdir(parents=True, exist_ok=True)
    info = probe(source)
    source_ratio = info.width / max(info.height, 1)
    if abs(source_ratio - aspect_ratio) < 1e-6:
        shutil.copy2(source, target)
        return target
    if source_ratio > aspect_ratio:
        crop_width = max(2, int(info.height * aspect_ratio) // 2 * 2)
        crop_height = info.height // 2 * 2
    else:
        crop_width = info.width // 2 * 2
        crop_height = max(2, int(info.width / aspect_ratio) // 2 * 2)
    crop_x = max(0, (info.width - crop_width) // 2)
    crop_y = max(0, (info.height - crop_height) // 2)
    vf = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"
    run([
        _tool("ffmpeg"), "-y", "-i", str(source), "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(target),
    ])
    return target


def make_frame_clip(source: Path, target: Path, start: float, frame_count: int) -> Path:
    """Create a silent preview clip containing exactly ``frame_count`` video frames."""
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _tool("ffmpeg"), "-y", "-ss", f"{start:.3f}", "-i", str(source),
        "-map", "0:v:0", "-frames:v", str(frame_count),
        "-c:v", "libx264", "-preset", "fast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(target),
    ]
    run(command)
    return target


def make_thumbnail(source: Path, target: Path) -> Path:
    """Create a small JPEG preview from the first video frame."""
    target.parent.mkdir(parents=True, exist_ok=True)
    run([
        _tool("ffmpeg"), "-y", "-ss", "0", "-i", str(source), "-frames:v", "1",
        "-vf", "scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:-1:-1:color=black",
        "-q:v", "5", str(target),
    ])
    return target


def make_demo_restore(source: Path, target: Path, resolution: int) -> Path:
    """Fast non-AI renderer used to test the complete UI before models are installed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale='if(gt(iw,ih),-2,{resolution})':'if(gt(iw,ih),{resolution},-2)':flags=lanczos,"
        "hqdn3d=0.8:0.8:2:2,unsharp=5:5:0.65:3:3:0.25"
    )
    command = [
        _tool("ffmpeg"), "-y", "-i", str(source), "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
        str(target),
    ]
    run(command)
    return target


def frame_at(source: Path, target: Path, timestamp: float) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    run([
        _tool("ffmpeg"), "-y", "-ss", f"{timestamp:.3f}", "-i", str(source),
        "-frames:v", "1", "-q:v", "2", str(target),
    ])
    return target
