"""Assemble decoded VAE tensors into a video, optionally preserving audio."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F


def load_video_tensor(path: Path) -> tuple[torch.Tensor, int | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "video" in payload:
        tensor = payload["video"]
        keep = payload.get("original_frames")
        return tensor, int(keep) if keep is not None else None
    return payload, None


def _spatial_lowpass(video: torch.Tensor) -> torch.Tensor:
    """Blur each frame independently while retaining the tensor layout."""
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    blurred = F.avg_pool2d(frames, kernel_size=5, stride=1, padding=2)
    return blurred.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def smooth_boundary(previous: torch.Tensor, current: torch.Tensor, mode: str, frame_count: int) -> torch.Tensor:
    """Reduce decoded-batch statistic jumps without changing frame count."""
    count = min(max(1, frame_count), previous.shape[2], current.shape[2])
    if mode == "off" or count < 1:
        return current
    result = current.float().clone()
    reference = previous[:, :, -count:].float()
    head = result[:, :, :count]
    fade = torch.linspace(1.0, 0.0, count + 1, dtype=head.dtype)[0:count].view(1, 1, count, 1, 1)

    if mode in {"noise", "match", "blend"}:
        reference_high = reference - _spatial_lowpass(reference)
        head_low = _spatial_lowpass(head)
        head_high = head - head_low
        ref_noise = reference_high.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        head_noise = head_high.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        noise_matched = head_low + head_high * (ref_noise / head_noise).clamp(0.5, 2.0)
        head = head.lerp(noise_matched, fade)

    if mode in {"match", "blend"}:
        ref_mean = reference.mean(dim=(0, 2, 3, 4), keepdim=True)
        ref_std = reference.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        head_mean = head.mean(dim=(0, 2, 3, 4), keepdim=True)
        head_std = head.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        matched = (head - head_mean) * (ref_std / head_std).clamp(0.75, 1.33) + ref_mean
        head = head.lerp(matched, fade)

    if mode == "blend":
        anchor = previous[:, :, -1:].float().expand(-1, -1, count, -1, -1)
        blend = torch.linspace(0.35, 0.0, count + 1, dtype=head.dtype)[0:count].view(1, 1, count, 1, 1)
        head = head.lerp(anchor, blend)

    result[:, :, :count] = head.clamp(-1, 1)
    return result.to(dtype=current.dtype)


def crop_decoder_padding(
    frames, reference: Path | None, target_width: int = 0, target_height: int = 0, *, log: bool = True
):
    """Remove TensorRT's bottom/right alignment padding using the source aspect ratio.

    TensorRT alignment rows are not guaranteed to remain exactly black after
    sharpening, grain, or other post effects, so geometry is the authoritative
    signal. The decoder pads on the bottom/right; retain the largest top-left
    rectangle matching the source aspect ratio.
    """
    frame_height, frame_width = frames.shape[1:3]
    if target_width > 0 and target_height > 0:
        if target_width > frame_width or target_height > frame_height:
            raise ValueError(
                f"Requested unpadded size {target_width}x{target_height} exceeds "
                f"decoded canvas {frame_width}x{frame_height}"
            )
        if log and (target_width, target_height) != (frame_width, frame_height):
            print(
                f"Removing known resize padding: {frame_width}x{frame_height} "
                f"-> {target_width}x{target_height}"
            )
        return frames[:, :target_height, :target_width, :], (target_width, target_height)
    if not reference:
        return frames, (frames.shape[2], frames.shape[1])
    capture = cv2.VideoCapture(str(reference))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if source_width <= 0 or source_height <= 0:
        return frames, (frames.shape[2], frames.shape[1])
    source_ratio = source_width / source_height
    if source_width >= source_height:
        target_width = frame_width
        target_height = max(2, round(target_width / source_ratio) // 2 * 2)
        if target_height <= frame_height:
            if log and target_height != frame_height:
                print(f"Removing TensorRT alignment padding: {frame_width}x{frame_height} -> {target_width}x{target_height}")
            frames = frames[:, :target_height, :target_width, :]
            return frames, (target_width, target_height)
    else:
        target_height = frame_height
        target_width = max(2, round(target_height * source_ratio) // 2 * 2)
        if target_width <= frame_width:
            if log and target_width != frame_width:
                print(f"Removing TensorRT alignment padding: {frame_width}x{frame_height} -> {target_width}x{target_height}")
            frames = frames[:, :target_height, :target_width, :]
            return frames, (target_width, target_height)
    if log and (target_width, target_height) != (frame_width, frame_height):
        print(f"Restoring source aspect: {frame_width}x{frame_height} -> {target_width}x{target_height}")
    return frames, (target_width, target_height)


def _input_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.inputs)
    if args.input_list:
        for line in args.input_list.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                paths.append(Path(text))
    if not paths:
        raise ValueError("Provide decoded tensors as arguments or --input-list")
    return paths


def _uint8_frames(tensor: torch.Tensor):
    frames = tensor[0].permute(1, 2, 3, 0)
    return ((frames.float().clamp(-1, 1) + 1) * 127.5).byte().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument("--input-list", type=Path, help="Text file with one decoded tensor path per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seam-mode", choices=("off", "noise", "match", "blend"), default="match")
    parser.add_argument("--seam-frames", type=int, default=2)
    parser.add_argument("--target-width", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=0)
    args = parser.parse_args()
    inputs = _input_paths(args)
    temp = args.output.with_name(args.output.stem + "_noaudio.mp4")
    encoder = None
    encode_command: list[str] = []
    output_size = None
    previous = None
    total_frames = 0
    write_error = None
    try:
        for index, path in enumerate(inputs, start=1):
            tensor, keep = load_video_tensor(path)
            if keep is not None:
                tensor = tensor[:, :, :keep]
            if previous is not None:
                tensor = smooth_boundary(previous, tensor, args.seam_mode, args.seam_frames)
            frames = _uint8_frames(tensor)
            frames, batch_size = crop_decoder_padding(
                frames, args.audio, args.target_width, args.target_height, log=index == 1
            )
            if encoder is None:
                output_size = batch_size
                width, height = output_size
                encode_command = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s:v", f"{width}x{height}", "-r", f"{args.fps:.9g}", "-i", "pipe:0",
                    "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp),
                ]
                encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)
                print(f"Assembling {len(inputs)} decoded batches...", flush=True)
            assert encoder.stdin is not None
            for frame in frames:
                if (frame.shape[1], frame.shape[0]) != output_size:
                    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
                encoder.stdin.write(frame.tobytes())
            total_frames += int(frames.shape[0])
            previous = tensor
            if index == 1 or index == len(inputs) or index % 50 == 0:
                print(f"Assembled batch {index} of {len(inputs)} ({total_frames} frames)", flush=True)
        if encoder is None:
            raise ValueError("At least one decoded tensor is required")
    except BrokenPipeError as exc:
        write_error = exc
    finally:
        if encoder is not None and encoder.stdin is not None:
            encoder.stdin.close()
    if encoder is None:
        raise ValueError("At least one decoded tensor is required")
    return_code = encoder.wait()
    if return_code:
        temp.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(return_code, encode_command) from write_error
    if write_error is not None:
        temp.unlink(missing_ok=True)
        raise write_error
    if args.audio:
        video_duration = total_frames / max(args.fps, 1e-6)
        subprocess.run(["ffmpeg", "-y", "-i", str(temp), "-i", str(args.audio),
                        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                        "-c:a", "aac", "-t", f"{video_duration:.9f}", str(args.output)], check=True)
        temp.unlink(missing_ok=True)
    else:
        temp.replace(args.output)
    print(f"Frames: {total_frames}")
    print(f"Output: {output_size[0]}x{output_size[1]} @ {args.fps:g}fps")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
