# SeedVR2 TensorRT backend

This is an independent optimization path for SeedVR2. It uses the public
SeedVR2 model/code and NVIDIA TensorRT-RTX; it does not depend on Topaz
executables, model files, or private libraries.

## Current stage

The workspace currently has the Python TensorRT-RTX bindings and ONNX export
tools installed in the project virtual environment. Run the smoke test from
the repository root:

```bash
./.venv/bin/python tools/tensorrt_smoke.py
```

The first implementation target is a fixed-shape VAE graph. Once its TensorRT
output matches the PyTorch VAE within an agreed tolerance, the DiT graph and
custom temporal operators can be tackled. The existing PyTorch backend remains
the fallback during development.
