#!/usr/bin/env bash
set -Eeuo pipefail

SKIP_MODELS=0
SKIP_TENSORRT=0
REPAIR=0

usage() {
    cat <<'EOF'
Usage: ./scripts/install.sh [options]

Options:
  --skip-models     Do not download the default SeedVR2 model and VAE.
  --skip-tensorrt   Do not install/build the TensorRT RTX path.
  --repair          Reinstall packages in the existing virtual environment.
  -h, --help        Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --skip-models) SKIP_MODELS=1 ;;
        --skip-tensorrt) SKIP_TENSORRT=1 ;;
        --repair) REPAIR=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTPUTS="$STUDIO_ROOT/outputs"
VENV="$STUDIO_ROOT/.venv"
STUDIO_PYTHON="$VENV/bin/python"
MARKER="$STUDIO_ROOT/.seedvr-studio-installed"
LOG="$OUTPUTS/install-linux.log"

mkdir -p "$OUTPUTS"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$STUDIO_ROOT"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

step() {
    printf '\n== %s ==\n' "$1"
}

fail() {
    echo "INSTALLATION FAILED: $1" >&2
    echo "Detailed log: $LOG" >&2
    exit 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
    fail "This installer is for Linux. Use the Windows installer from the main branch on Windows."
fi

echo "SeedVR Studio Linux installer"
echo "Installs the UI, CUDA PyTorch, SeedVR2, SageAttention, TensorRT RTX, models, and GPU-specific engines."
echo "Interrupted downloads and engine builds can be resumed by running this installer again."

install_apt_packages() {
    local packages=("$@")
    ((${#packages[@]})) || return 0
    command -v apt-get >/dev/null 2>&1 || fail "Missing system packages: ${packages[*]}. Install them with your distribution package manager."
    local elevate=()
    if [[ "$EUID" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || fail "sudo is required to install system packages: ${packages[*]}"
        elevate=(sudo)
    fi
    step "Installing Linux system packages"
    "${elevate[@]}" apt-get update
    "${elevate[@]}" apt-get install -y "${packages[@]}"
}

base_packages=()
command -v ffmpeg >/dev/null 2>&1 || base_packages+=(ffmpeg)
command -v ffprobe >/dev/null 2>&1 || base_packages+=(ffmpeg)
command -v git >/dev/null 2>&1 || base_packages+=(git)
command -v curl >/dev/null 2>&1 || base_packages+=(curl)
command -v gcc >/dev/null 2>&1 || base_packages+=(build-essential)
command -v g++ >/dev/null 2>&1 || base_packages+=(build-essential)
command -v make >/dev/null 2>&1 || base_packages+=(build-essential)
command -v xdg-open >/dev/null 2>&1 || base_packages+=(xdg-utils)
if ((${#base_packages[@]})); then
    mapfile -t base_packages < <(printf '%s\n' "${base_packages[@]}" | sort -u)
    install_apt_packages "${base_packages[@]}"
fi

command -v nvidia-smi >/dev/null 2>&1 || fail "No NVIDIA driver was detected. Install a current proprietary NVIDIA driver and reboot."
nvidia-smi >/dev/null || fail "nvidia-smi exists but cannot communicate with the NVIDIA driver."

find_python() {
    local candidate
    for candidate in python3.12 python3; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

BASE_PYTHON="$(find_python || true)"
if [[ -z "$BASE_PYTHON" ]]; then
    install_apt_packages python3.12 python3.12-venv
    BASE_PYTHON="$(find_python || true)"
fi
[[ -n "$BASE_PYTHON" ]] || fail "Python 3.12 or newer is required. Ubuntu 24.04 provides python3.12 and python3.12-venv."

if ! "$BASE_PYTHON" -c 'import ensurepip' >/dev/null 2>&1; then
    python_version="$($BASE_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    install_apt_packages "python${python_version}-venv"
fi

if ((SKIP_TENSORRT == 0)); then
    command -v nvcc >/dev/null 2>&1 || fail "TensorRT setup requires the CUDA 13.4 toolkit with nvcc on PATH. Install the matching NVIDIA CUDA toolkit or rerun with --skip-tensorrt."
    CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
    [[ "$CUDA_VERSION" == 13.4* ]] || fail "The Linux TensorRT profile requires CUDA 13.4; nvcc reported ${CUDA_VERSION:-an unknown version}."
    if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
        export CUDA_HOME=/usr/local/cuda
    fi
fi

step "Creating the private Python environment"
if ((REPAIR)); then
    echo "Repair mode will reinstall packages in the existing environment."
fi
if [[ ! -x "$STUDIO_PYTHON" ]]; then
    if ! "$BASE_PYTHON" -m venv "$VENV"; then
        fail "Could not create the virtual environment. Install the venv package for $BASE_PYTHON and retry."
    fi
fi

pip_install() {
    "$STUDIO_PYTHON" -m pip --isolated install "$@"
}

pip_install --upgrade pip setuptools wheel --index-url https://pypi.org/simple

step "Installing the tested CUDA 13 PyTorch build"
if ! "$STUDIO_PYTHON" -m pip --isolated install --pre \
    'torch==2.15.0.dev20260824+cu130' \
    'torchvision==0.30.0.dev20260824+cu130' \
    'torchaudio==2.11.0.dev20260824+cu130' \
    --index-url https://download.pytorch.org/whl/nightly/cu130; then
    echo "The exact tested nightly is unavailable; installing the newest mutually compatible CUDA 13 nightly."
    pip_install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
fi

step "Installing SeedVR Studio and SeedVR2 dependencies"
pip_install --editable "$STUDIO_ROOT" --index-url https://pypi.org/simple
pip_install --requirement "$STUDIO_ROOT/vendor/seedvr2/requirements.txt" --index-url https://pypi.org/simple

step "Installing Linux attention and TensorRT dependencies"
if ((SKIP_TENSORRT)); then
    pip_install 'sageattention==1.0.6' --index-url https://pypi.org/simple
else
    pip_install --requirement "$STUDIO_ROOT/requirements-linux-cu130.txt" --index-url https://pypi.org/simple
fi

step "Checking CUDA and attention support"
VERIFY_CODE="import sys, torch, triton; sys.path.insert(0, 'vendor/seedvr2'); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); print('Triton:', triton.__version__); from src.optimization.compatibility import SAGE_ATTN_2_AVAILABLE; assert SAGE_ATTN_2_AVAILABLE, 'SageAttention 2 did not load'; print('SageAttention 2: ready')"
if ((SKIP_TENSORRT == 0)); then
    VERIFY_CODE+="; import tensorrt_rtx; print('TensorRT RTX:', tensorrt_rtx.__version__)"
fi
if ! "$STUDIO_PYTHON" -c "$VERIFY_CODE"; then
    fail "CUDA or SageAttention verification failed. Review docs/INSTALLATION_LINUX.md."
fi

if ((SKIP_MODELS)); then
    echo "Model download skipped. Models will download when first selected."
else
    step "Downloading the default SeedVR2 3B FP8 model and VAE"
    "$STUDIO_PYTHON" "$SCRIPT_DIR/download_models.py" || fail "The model download failed. Run this installer again to resume it."
fi

if ((SKIP_TENSORRT)); then
    echo "TensorRT engine preparation skipped. Only SeedVR2 Legacy will be available."
else
    step "Building GPU-specific TensorRT VAE engines"
    echo "This is a one-time operation. Do not copy these engines to a different GPU, operating system, or TensorRT version."
    "$STUDIO_PYTHON" "$SCRIPT_DIR/prepare_tensorrt.py" || fail "TensorRT engine preparation failed. Run this installer again to resume it."
fi

if ((SKIP_MODELS == 0 && SKIP_TENSORRT == 0)); then
    step "Final readiness check"
    "$STUDIO_PYTHON" "$SCRIPT_DIR/verify_install.py" || fail "The final installation check failed."
    date --iso-8601=seconds > "$MARKER"
fi

echo
echo "SeedVR Studio is installed. Start it with:"
echo '  ./Launch\ SeedVR\ Studio\ Pro.sh'
