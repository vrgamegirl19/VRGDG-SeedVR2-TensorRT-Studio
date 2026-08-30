# VRGDG SeedVR2 TensorRT Studio

A local video restoration and upscale studio for SeedVR2 with TensorRT acceleration for faster VAE processing, a custom JavaScript interface, and FastAPI backend.

## Screenshot

![VRGDG SeedVR2 TensorRT Studio](images/seedvr2-studio-screenshot.png)

## Features

- Original, Restored, Compare, and synchronized Side by side viewing modes.
- One shared play button, timeline scrubbing, frame stepping, fullscreen, zoom, pan, and wipe comparison.
- Preview renders and full-video renders with elapsed time and ETA.
- Output presets for original-size enhancement, 1K/1080p, 2K/1440p, and 4K/2160p.
- Preserve the original aspect ratio or center-crop to 16:9.
- Resumable long-video rendering with automatic frame-aware chunk sizing, manual 30-second to 30-minute choices, retry from the last completed chunk, failure reasons, and saved logs.
- Post-only reprocessing for sharpening, film grain, TensorRT seam smoothing, and optional skin finishing.
- TensorRT decoding uses Optimized Fast automatically and falls back to the internal Stable decoder from saved latents if needed, without rerunning AI restoration.
- Saved settings, previous-output loading, and an Open project folder action.

## Install and start

This package targets Windows with an NVIDIA RTX GPU. Allow at least 35 GB of free disk space for dependencies, model weights, temporary ONNX exports, and locally built TensorRT engines.

1. Double-click **Install SeedVR Studio.bat** once after downloading or cloning the repository.
2. Leave it open while it installs Python/FFmpeg if needed, creates the private environment, installs CUDA PyTorch, SeedVR2, SageAttention 2, and TensorRT RTX, downloads the default model and VAE, and builds engines for your GPU.
3. Double-click **Launch SeedVR Studio Pro.bat**.

The Pro launcher also detects an incomplete installation and opens the installer automatically. Downloads and TensorRT preparation are resumable. Detailed instructions and troubleshooting are in the [installation guide](docs/INSTALLATION.md).

Model weights are stored locally in `models\` and are not committed to GitHub. Outputs, virtual environments, TensorRT engines/caches, and optional runtime files are also kept local.

## Render engines

- **SeedVR2 + TensorRT** — the recommended fast path. It runs SeedVR2 restoration while using TensorRT to accelerate VAE decoding. Fixed TensorRT engines currently support temporal batch sizes **5** and **21**.
- **SeedVR2 (Legacy)** — the standard PyTorch SeedVR2 path. It supports more temporal batch sizes, but it can be substantially slower, especially for long videos.

The batch-size dropdown automatically shows only values supported by the selected engine.

## Workflow

1. Start the Studio and drop an original video into the Media panel.
2. Choose the render engine, output preset, crop policy, model, batch size, and settings.
3. Render a short preview and inspect it in the viewer.
4. Adjust settings as needed, then render the full video.
5. Use **Reprocess post only** to change post-processing without rerunning the AI restoration.

Each job is stored in its own directory under `outputs\`, including output media and logs.

## Settings notes

- Temporal batches follow SeedVR2's `4n+1` rule. Larger batches can improve consistency but use more VRAM.
- Color correction is part of the main SeedVR2 render. `none` leaves colors unchanged; `lab` is the general-purpose matching mode.
- SageAttention 2 is installed and preferred by default. Missing accelerated attention backends fall through to another available option and finally SDPA. See the [SageAttention guide](docs/SAGEATTENTION.md) for verification and repair steps.
- Leave VAE tiling off unless VRAM is insufficient; it saves memory but reduces speed.
- TensorRT uses **Optimized Fast** automatically; every job records the requested and actual decoder in its manifests and log.
- Skin finishing is a non-generative post effect that enhances existing skin pixels; it cannot recreate missing facial identity detail.

## Project layout

- `web/` — JavaScript Studio interface.
- `api_server.py` — FastAPI service.
- `seedvr_studio/` — rendering and legacy Gradio integration.
- `tools/` — TensorRT, assembly, post-processing, and diagnostics.
- `tensorrt_backend/` — optional native TensorRT sources.
- `scripts/` — one-click setup, launcher, model download, verification, and engine preparation scripts.
- `vendor/seedvr2/` — the compatible Apache-2.0 SeedVR2 integration required by the Studio.

## License

The Studio source is released under the MIT License. SeedVR2 and runtime components retain their upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
## UI guide

The Studio is organized around two areas: the settings panel on the left and the video viewer on the right. The screenshots below are cropped to one UI section at a time.

### Viewer

#### Viewer modes

![Viewer modes](<images/ui-guide/controls/viewer-modes.png>)

- **Original** shows the source video.
- **Restored** shows the rendered result.
- **Compare** provides a draggable wipe between source and result.
- **Side by side** displays both videos together.

Use the timeline, frame field, navigation buttons, and play button to review frames. Zoom with the zoom control or mouse wheel; drag to pan and double-click to reset.

#### Render status

![Render status](<images/ui-guide/controls/render-status.png>)

The top status bar shows whether the Studio is ready or rendering, elapsed time, ETA, power/stop control, fullscreen control, and the current viewer zoom.

### Source and previous results

#### Media upload

![Media upload](<images/ui-guide/controls/media.png>)

Drop a source video into **Original video**, or click the upload area to browse. The panel also indicates when no source is loaded.

#### Previous results

![Previous results](<images/ui-guide/controls/previous-results.png>)

Choose a previous workspace output from the dropdown and click **Load selected output** to reopen it in the viewer.

### Render configuration

#### Render engine, hardware, output, and aspect

![Render engine and output settings](<images/ui-guide/controls/render-engine-hardware-output.png>)

- **Render engine** selects **SeedVR2 + TensorRT** or **SeedVR2 (Legacy)**.
- **Hardware preset** provides a starting configuration for the available VRAM. **Custom / manual** enables manual tuning.
- **Output size** offers the original enhancement target, 1K / 1080p, 2K / 1440p, and 4K / 2160p options.
- **Crop / aspect policy** preserves the source aspect ratio or center-crops to 16:9.
- **Resumable long-video render** stores completed chunks so an interrupted long render can continue.

#### Model and temporal batch

![Model and temporal batch](<images/ui-guide/controls/model-temporal.png>)

- **Model** selects the installed SeedVR2 checkpoint. FP8 models use less VRAM; FP16 models require more memory.
- **Temporal batch** controls how many frames are processed together. Larger values can improve temporal consistency but require more VRAM.
- **Seed** makes a render repeatable when the source and other settings are unchanged.

#### Color correction

![Color correction](<images/ui-guide/controls/color-correction.png>)

Color correction controls how restored colors are matched. Start with `none` to preserve the source color character; `lab` is a general-purpose matching option.

### Performance and memory

#### TensorRT decoder, attention, and memory

![Performance, attention, and memory](<images/ui-guide/controls/performance.png>)

TensorRT uses **Optimized Fast** decoding automatically. If optimized decoding fails, saved latents can retry with the internal Stable decoder without repeating AI restoration.

**Attention** selects an installed accelerated backend. Missing backends fall through automatically; `sdpa` is the safest compatibility option. **Blocks to swap** trades speed for VRAM, and **VAE tiling** lowers VAE memory use at a speed cost.

### Post-processing

#### Sharpen, grain, and batch seams

![Post-processing](<images/ui-guide/controls/post-processing.png>)

- **Extra sharpen** adds controlled edge definition.
- **Film grain** adds adjustable grain intensity and saturation.
- **Smooth TensorRT batch seams** reduces visible color or noise changes at temporal batch boundaries.

#### Skin finishing

![Skin finishing](<images/ui-guide/controls/skin-finishing.png>)

Skin finishing is a non-generative pass for existing skin pixels. It includes tone, smoothing, redness, shine, blemish cleanup, mark preservation, and face-aware microtexture controls. Preview before using stronger values; it cannot recreate missing identity detail.

### Recommended workflow

1. Upload a source video.
2. Choose a hardware preset or configure settings manually.
3. Render a short preview with **Preview start** and **Length**.
4. Check the result in **Original**, **Restored**, **Compare**, and **Side by side** modes.
5. Render the full video once the preview looks right.
6. Use **Reprocess post only** to adjust finishing without rerunning restoration.
7. Use **Load selected output** to reopen a previous result.

Use **Save current settings** to keep a configuration for later. Each job stores its media, manifest, and logs in the project’s `outputs\\` directory.