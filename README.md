# VRGDG SeedVR2 TensorRT Studio

Local, GPU-accelerated video restoration and upscaling with SeedVR2, TensorRT, and a purpose-built browser interface.

VRGDG SeedVR2 TensorRT Studio turns the SeedVR2 pipeline into a practical Linux workflow: load a video, test a short preview, compare the result frame by frame, and complete long renders with resumable checkpoints. Processing stays on your machine.

> **Linux branch:** This branch targets native Linux. For the supported Windows package, use the `main` branch.

![VRGDG SeedVR2 TensorRT Studio full interface](images/seedvr2-studio-screenshot.png)

## Highlights

- **Fast local restoration** — SeedVR2 inference with TensorRT-accelerated VAE decoding on supported NVIDIA RTX GPUs.
- **Fast 2K upscaling** — As a real-world example, an 8-second clip took approximately 8 minutes to upscale to 2K on an NVIDIA RTX 5090 using the largest **7B Sharp FP16** model. Render times vary with source resolution, frame rate, settings, and available VRAM.
- **Preview before committing** — render a short segment, then inspect Original, Restored, Compare, or Side by side views.
- **Long-render recovery** — save completed chunks and continue from the first unfinished chunk after an interruption.
- **Practical output controls** — choose resolution, aspect policy, model precision, temporal batch, seed, and color correction.
- **Non-destructive finishing** — reprocess sharpening, grain, seam smoothing, and optional skin finishing without rerunning restoration.
- **Project-based history** — reopen previous outputs and keep media, manifests, and logs together under `outputs/`.

## Before and after examples

Each recording demonstrates the Studio's interactive comparison view, with the original footage on one side of the wipe and the restored result on the other.

- [Watch comparison sample 1 in your browser (MP4, 6.6 MB)](https://cdn.jsdelivr.net/gh/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio@eebba66e6c73cb420f2f24a581961802b6177f65/videos/comparisons/comparison-sample-01.mp4)
- [Watch comparison sample 2 in your browser (MP4, 13 MB)](https://cdn.jsdelivr.net/gh/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio@eebba66e6c73cb420f2f24a581961802b6177f65/videos/comparisons/comparison-sample-02.mp4)

## Requirements

| | Recommended setup |
|---|---|
| Operating system | Ubuntu 24.04 LTS or a modern x86-64 Linux distribution |
| GPU | NVIDIA RTX with a current driver |
| Storage | At least 35 GB free |
| Network | Internet connection for the first installation |
| CUDA | CUDA Toolkit 13.4 with `nvcc` for TensorRT |
| Python | Python 3.12+ with `venv` support |

TensorRT engines are built locally for the installed GPU, Linux runtime, and CUDA/TensorRT versions. Do not copy `.rtxplan` files from Windows, another GPU, or a different runtime.

## Quick start

1. Clone the Linux branch: `git clone --branch linux https://github.com/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio.git`.
2. Enter the repository and run `chmod +x "Install SeedVR Studio.sh" "Launch SeedVR Studio Pro.sh" scripts/*.sh`.
3. Run `./Install\ SeedVR\ Studio.sh` and leave the terminal open until setup completes.
4. Run `./Launch\ SeedVR\ Studio\ Pro.sh`.
5. Drop a source video into the Media panel, render a short preview, inspect it, and then render the full video.

The installer prepares a private Linux Python environment, checks FFmpeg and CUDA, installs CUDA PyTorch, SeedVR2, SageAttention, TensorRT RTX, model weights, and builds Linux/GPU-specific engines. Downloads and engine preparation are resumable. The launcher detects an incomplete full installation and starts the installer automatically.

For prerequisites, headless use, repair options, and troubleshooting, see the [Linux installation guide](docs/INSTALLATION_LINUX.md).

## Workflow

### 1. Load and inspect media

Drop a video onto the Media panel or browse for a file. After a preview or full render completes, use the viewer tabs to switch between the source and restored result.

| Load a source | Choose a viewer mode |
|---|---|
| ![Media panel with the original-video drop zone](images/ui-guide/controls/media-input.png) | ![Original, Restored, Compare, and Side by side viewer tabs](images/ui-guide/controls/viewer-modes.png) |

- **Original** shows the uploaded source.
- **Restored** shows the latest rendered result.
- **Compare** overlays the restored video with a draggable wipe divider.
- **Side by side** places both videos next to each other.

The shared viewer also supports timeline scrubbing, frame stepping, synchronized playback, fullscreen, zoom, and pan.

### 2. Choose the render path and target

Start with **SeedVR2 + TensorRT** for the accelerated path, then select a hardware preset and output target. Presets provide sensible starting values; **Custom / manual** leaves every setting under your control.

| Render engine | Hardware preset | Output and aspect |
|---|---|---|
| ![SeedVR2 and TensorRT render-engine selector](images/ui-guide/controls/render-engine.png) | ![Hardware preset selector](images/ui-guide/controls/hardware-preset.png) | ![Output-size and crop-policy selectors](images/ui-guide/controls/output-and-crop.png) |

Output presets include original-size enhancement, 1K/1080p, 2K/1440p, and 4K/2160p. Preserve the source aspect ratio or center-crop it to 16:9.

### 3. Tune restoration

Choose the model, temporal batch, seed, and color-matching strategy. FP8 models use less VRAM; FP16 models prioritize quality and require more memory. Temporal batches follow SeedVR2's `4n+1` rule, while the TensorRT path currently supports batches **5** and **21**.

| Model, batch, and seed | Color correction | Resumable rendering |
|---|---|---|
| ![Model, temporal-batch, and seed controls](images/ui-guide/controls/model-batch-seed.png) | ![Color-correction selector and available modes](images/ui-guide/controls/color-correction.png) | ![Resumable long-video render and chunk-length controls](images/ui-guide/controls/resumable-render.png) |

Color correction is applied during the main restoration. Use `none` to retain source colors; `lab` is a useful general-purpose matching mode. The other available modes are `wavelet`, `wavelet_adaptive`, `hsv`, and `adain`.

For long videos, enable resumable rendering. **Auto** selects a chunk length from the duration, frame count, frame rate, and temporal batch; manual choices range from 30 seconds to 30 minutes. Completed chunks remain available if a job stops, and **Retry from last chunk** continues at the first unfinished section.

### 4. Balance speed, memory, and finishing

TensorRT uses **Optimized Fast** decoding automatically. If it fails, the Studio retries the saved latents with its internal Stable decoder, avoiding another restoration pass.

| Performance | Post-processing | Skin finishing |
|---|---|---|
| ![Attention, block swapping, and VAE-tiling controls](images/ui-guide/controls/performance.png) | ![Sharpening, film grain, and seam-smoothing controls](images/ui-guide/controls/post-processing.png) | ![Optional skin-finishing controls](images/ui-guide/controls/skin-finishing.png) |

- **Attention:** SageAttention 2 is the preferred default. Missing accelerated backends fall through to another installed option and finally SDPA.
- **Blocks to swap:** higher values reduce VRAM demand but cost speed.
- **VAE tiling:** reduces memory use at a speed cost; leave it off unless VRAM is insufficient.
- **Smooth TensorRT batch seams:** matches color and noise around decoded batch boundaries and is enabled by default.
- **Skin finishing:** uses a stabilized skin mask to refine existing pixels. It is non-generative and cannot recreate missing facial identity detail; preview stronger settings first.

Post-processing can be changed later with **Reprocess post only**, without rerunning SeedVR2.

### 5. Preview, render, and reopen results

Set a short preview range, render it, and compare the result before starting the complete video. The status bar reports readiness, elapsed time, and ETA while a job runs.

| Preview range | Render actions |
|---|---|
| ![Preview start and length fields](images/ui-guide/controls/preview-range.png) | ![Reprocess, preview, and full-render buttons](images/ui-guide/controls/render-actions.png) |

| Render status | Previous results |
|---|---|
| ![Ready state, elapsed time, ETA, power, and fullscreen controls](images/ui-guide/controls/render-status.png) | ![Previous-results selector and load button](images/ui-guide/controls/previous-results.png) |

Every job receives its own folder under `outputs/` with rendered media, manifests, and logs. Use **Load selected output** to reopen a result from the current workspace.

## Render engines

| Engine | Best for | Temporal batches | Notes |
|---|---|---:|---|
| **SeedVR2 + TensorRT** | Recommended accelerated workflow | 5, 21 | Uses TensorRT VAE decoding and automatic fallback from saved latents. |
| **SeedVR2 (Legacy)** | Compatibility and additional batch choices | SeedVR2-compatible values | Standard PyTorch path; can be substantially slower on long videos. |

The temporal-batch menu updates automatically when the render engine changes.

## Settings that matter most

- Begin with the hardware preset closest to your VRAM capacity.
- Use a short representative preview before every long render.
- Increase temporal batch only when the selected engine and available VRAM support it.
- Use block swapping or VAE tiling when memory is the limiting factor, accepting the speed tradeoff.
- Keep skin finishing subtle and confirm faces, freckles, moles, and other permanent marks in the preview.
- Use post-only reprocessing for finishing changes; it is much faster than repeating restoration.

## Files and local data

Model weights live in `models/` and are not committed to GitHub. Outputs, virtual environments, TensorRT engines and caches, temporary ONNX exports, and optional runtime files also remain local.

| Path | Purpose |
|---|---|
| `web/` | JavaScript Studio interface |
| `api_server.py` | FastAPI service |
| `seedvr_studio/` | Rendering pipeline and legacy Gradio integration |
| `tools/` | TensorRT, assembly, post-processing, and diagnostic utilities |
| `tensorrt_backend/` | Optional native TensorRT sources |
| `scripts/` | Setup, launcher, download, verification, and engine-preparation scripts |
| `vendor/seedvr2/` | Compatible Apache-2.0 SeedVR2 integration required by the Studio |

## Documentation

- [Linux installation and troubleshooting](docs/INSTALLATION_LINUX.md)
- [SageAttention setup and verification](docs/SAGEATTENTION.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

The Studio source is released under the [MIT License](LICENSE). SeedVR2 and bundled runtime components retain their upstream licenses; see the [third-party notices](THIRD_PARTY_NOTICES.md).
