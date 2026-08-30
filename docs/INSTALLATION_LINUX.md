# Linux installation guide

This branch targets a native Linux installation with an NVIDIA RTX GPU. It does not use Wine, the Windows portable Python runtime, or Windows-built TensorRT engines.

## Supported baseline

- Ubuntu 24.04 LTS or another modern x86-64 Linux distribution
- Python 3.12 or newer with `venv` support
- A supported NVIDIA RTX GPU and current proprietary NVIDIA driver
- CUDA Toolkit 13.4 when using the TensorRT path
- At least 35 GB of free disk space
- A graphical browser and `xdg-open` for automatic launch

Ubuntu 22.04 can work after Python 3.12 is installed separately, but its default Python 3.10 is too old for this package. Other distributions are usable when the equivalent system packages are installed manually.

TensorRT RTX engines are tied to the GPU, operating system, CUDA/TensorRT versions, and build profile. Always build them on the Linux computer that will run the Studio. Never copy `.rtxplan` files from Windows or another GPU.

## Before installation

Verify the NVIDIA driver:

```bash
nvidia-smi
```

For the accelerated TensorRT engine, also verify the CUDA compiler:

```bash
nvcc --version
```

The Linux installer requires CUDA 13.4 for the pinned CUDA 13 environment. Install it using the [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and review the [TensorRT RTX prerequisites](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/prerequisites.html) before running the full setup. A driver-reported CUDA compatibility version from `nvidia-smi` is not a replacement for the toolkit; `nvcc` must exist.

## Install

From the repository root:

```bash
chmod +x "Install SeedVR Studio.sh" "Launch SeedVR Studio Pro.sh" scripts/*.sh
./Install\ SeedVR\ Studio.sh
```

The installer:

1. Checks Linux, the NVIDIA driver, Python, FFmpeg, build tools, and CUDA.
2. Creates `.venv` with the Linux interpreter at `.venv/bin/python`.
3. Installs the pinned CUDA PyTorch stack and Linux Triton dependency.
4. Installs SeedVR2, SageAttention, TensorRT RTX, ONNX tooling, and the Studio.
5. Downloads the default model and VAE.
6. Builds four Linux/GPU-specific TensorRT engines.
7. Runs the non-rendering readiness check.

Downloads and engine creation are resumable. Rerun the installer after an interruption.

## Launch

```bash
./Launch\ SeedVR\ Studio\ Pro.sh
```

The launcher opens `http://127.0.0.1:7870/` with `xdg-open`. Use **Exit Studio** in the interface or press **Ctrl+C** in the terminal to stop the server and release GPU memory.

For a headless or remote system:

```bash
./scripts/launch_js.sh --no-browser
```

The service binds only to `127.0.0.1`. Use an SSH tunnel rather than exposing it directly:

```bash
ssh -L 7870:127.0.0.1:7870 user@linux-host
```

Then open `http://127.0.0.1:7870/` on the local computer.

## Installer options

```bash
./scripts/install.sh --repair
./scripts/install.sh --skip-models
./scripts/install.sh --skip-tensorrt
```

`--skip-tensorrt` leaves the **SeedVR2 (Legacy)** engine available. Start a deliberately partial installation with:

```bash
./scripts/launch_js.sh --skip-install-check
```

## Troubleshooting

Installation logs are saved to `outputs/install-linux.log`. Server logs are saved to `outputs/js_server.log` and `outputs/js_server_error.log`.

- **`nvidia-smi` fails:** install a current proprietary NVIDIA driver and reboot.
- **`nvcc` is missing:** install the CUDA 13.4 toolkit, or use `--skip-tensorrt` for the Legacy path.
- **Python is too old:** use Python 3.12+. Ubuntu 24.04 includes Python 3.12; Ubuntu 22.04 needs a separately installed newer Python.
- **SageAttention does not load:** confirm the CUDA toolkit is available during installation and review the package build output in `install-linux.log`.
- **TensorRT import or engine build fails:** confirm CUDA major versions match and rerun `./scripts/install.sh --repair`.
- **The browser does not open:** visit `http://127.0.0.1:7870/` manually.
- **A Wayland file manager does not open:** install `xdg-utils`; the output path is still shown in the Studio and logs.

## Updating the Linux branch

After pulling branch updates, rerun:

```bash
./scripts/install.sh --repair
```

Engine files remain local under `tensorrt_backend/artifacts/` and are rebuilt only when missing.
