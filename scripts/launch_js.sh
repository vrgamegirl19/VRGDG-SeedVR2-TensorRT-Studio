#!/usr/bin/env bash
set -Eeuo pipefail

NO_BROWSER=0
SKIP_INSTALL_CHECK=0

usage() {
    cat <<'EOF'
Usage: ./scripts/launch_js.sh [options]

Options:
  --no-browser          Start the server without opening a browser.
  --skip-install-check  Start a deliberately partial/Legacy installation.
  -h, --help            Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --no-browser) NO_BROWSER=1 ;;
        --skip-install-check) SKIP_INSTALL_CHECK=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STUDIO_PYTHON="$STUDIO_ROOT/.venv/bin/python"
STUDIO_URL="http://127.0.0.1:7870/"
HEALTH_URL="${STUDIO_URL}api/health"
SHUTDOWN_URL="${STUDIO_URL}api/shutdown"
OUTPUTS="$STUDIO_ROOT/outputs"
LOG_OUT="$OUTPUTS/js_server.log"
LOG_ERR="$OUTPUTS/js_server_error.log"
MARKER="$STUDIO_ROOT/.seedvr-studio-installed"
SERVER_PID=""
OWNS_SERVER=0

mkdir -p "$OUTPUTS"

health_check() {
    if command -v curl >/dev/null 2>&1; then
        curl --fail --silent --max-time 2 "$HEALTH_URL" >/dev/null 2>&1
    elif [[ -x "$STUDIO_PYTHON" ]]; then
        "$STUDIO_PYTHON" - "$HEALTH_URL" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
    else
        return 1
    fi
}

required_files=(
    "$STUDIO_PYTHON"
    "$STUDIO_ROOT/vendor/seedvr2/inference_cli.py"
    "$STUDIO_ROOT/models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    "$STUDIO_ROOT/models/SEEDVR2/ema_vae_fp16.safetensors"
    "$STUDIO_ROOT/tensorrt_backend/artifacts/vae_decoder_tile_256_21f.rtxplan"
    "$STUDIO_ROOT/tensorrt_backend/artifacts/vae_decoder_tile_512_5f.rtxplan"
    "$STUDIO_ROOT/tensorrt_backend/artifacts/vae_encoder_5f_tile512.rtxplan"
    "$STUDIO_ROOT/tensorrt_backend/artifacts/vae_encoder_21f_tile512.rtxplan"
)

if ((SKIP_INSTALL_CHECK == 0)); then
    installation_needed=0
    [[ -f "$MARKER" ]] || installation_needed=1
    for required in "${required_files[@]}"; do
        [[ -e "$required" ]] || installation_needed=1
    done
    if ((installation_needed)); then
        echo "SeedVR Studio setup is incomplete. Starting the Linux installer..."
        "$SCRIPT_DIR/install.sh"
    fi
fi

[[ -x "$STUDIO_PYTHON" ]] || {
    echo "The Linux environment is incomplete. Run ./Install SeedVR Studio.sh first." >&2
    exit 1
}

open_browser() {
    ((NO_BROWSER)) && return 0
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$STUDIO_URL" >/dev/null 2>&1 &
    else
        echo "Open $STUDIO_URL in your browser."
    fi
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if ((OWNS_SERVER)) && [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        curl --silent --max-time 2 -X POST "$SHUTDOWN_URL" >/dev/null 2>&1 || true
        sleep 0.5
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if health_check; then
    echo "SeedVR Studio is already running at $STUDIO_URL"
    open_browser
    OWNS_SERVER=0
    exit 0
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
cd "$STUDIO_ROOT"
"$STUDIO_PYTHON" -m uvicorn api_server:app --host 127.0.0.1 --port 7870 >"$LOG_OUT" 2>"$LOG_ERR" &
SERVER_PID=$!
OWNS_SERVER=1

for _ in {1..60}; do
    if health_check; then
        echo "SeedVR Studio is ready at $STUDIO_URL"
        open_browser
        echo "Use Exit Studio in the UI or press Ctrl+C here to release GPU and RAM."
        wait "$SERVER_PID"
        exit $?
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "SeedVR Studio exited during startup. See $LOG_ERR" >&2
        wait "$SERVER_PID" || true
        exit 1
    fi
    sleep 0.5
done

echo "SeedVR Studio did not become ready. See $LOG_ERR" >&2
exit 1
