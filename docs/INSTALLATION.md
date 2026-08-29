# Installation guide

## Supported setup

The packaged installer currently targets:

- Windows 11
- An NVIDIA RTX GPU with a current driver
- At least 35 GB of free disk space for the environment, models, temporary ONNX files, and TensorRT engines
- An internet connection for the first installation

TensorRT RTX engines are built for the GPU in the computer running the installer. Do not copy RTX plan files between different GPU models or TensorRT versions.

## One-click installation

1. Download or clone the repository.
2. Double-click **Install SeedVR Studio.bat**.
3. Leave the installer window open. It installs or checks Python 3.12, FFmpeg, CUDA PyTorch, the included compatible SeedVR2 integration, SageAttention 2, Triton for Windows, TensorRT RTX, ONNX tools, the default model and VAE, and four TensorRT engines.
4. When it reports success, double-click **Launch SeedVR Studio Pro.bat**.

The Pro launcher also detects an incomplete setup and starts the installer. Setup is resumable: completed model downloads and valid engines are skipped.

## Optional installer switches

Run these from PowerShell only if needed:

~~~powershell
.\scripts\install.ps1 -Repair
.\scripts\install.ps1 -SkipTensorRT
.\scripts\install.ps1 -SkipModels
~~~

**SkipTensorRT** leaves only SeedVR2 Legacy usable. **SkipModels** delays model downloads until a model is first used. A complete normal installation uses neither switch.

## Why engines are built locally

TensorRT plan files are tied to the GPU architecture and runtime version. The repository includes the exporter and builder, not another computer's plans. The installer creates:

- vae_encoder_5f_tile512.rtxplan
- vae_encoder_21f_tile512.rtxplan
- vae_decoder_tile_512_5f.rtxplan
- vae_decoder_tile_256_21f.rtxplan

If setup stops during this stage, run it again. Existing valid plans are retained. ONNX export and TensorRT builds should keep printing progress; several minutes per engine is normal.

## Logs and troubleshooting

The installer log is saved to **outputs\install.log**. Render logs are saved inside project directories under **outputs**.

Common failures:

- **No NVIDIA driver detected:** install or update the NVIDIA driver, restart Windows, and rerun setup.
- **Not enough disk space:** clear space and rerun. Partial model downloads resume.
- **Installer sits on the legacy ONNX deprecation warning with no `Exported:` line:** this was a hang in CPU TorchScript export of the 512-tile VAE. Current builds export on the GPU with portable convolution ops. Close the stuck window, update to this version, and rerun setup. Completed engines are skipped. To finish install without TensorRT, run `.\scripts\install.ps1 -SkipTensorRT` and use SeedVR2 Legacy until you rerun a full setup.
- **TensorRT build failure:** close GPU-heavy programs, confirm the NVIDIA driver is current, then rerun.
- **SageAttention verification failure:** follow the [SageAttention guide](SAGEATTENTION.md).
- **FFmpeg still missing after winget:** restart Windows so the system PATH refreshes, then rerun.

The installer does not require a separate full CUDA Toolkit. The Python CUDA and TensorRT packages provide the runtime used by the Studio.
