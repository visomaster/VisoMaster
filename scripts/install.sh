#!/usr/bin/env bash
# =============================================================================
# VisoMaster — Cross-platform Install Script
# Supports: Linux, macOS, Windows (Git Bash / MSYS2 / WSL)
#
# Usage:
#   bash scripts/install.sh [--dev | --full] [--cuda 124 | --cuda 118]
#
# Modes:
#   --dev   Download only the default models needed for development (default)
#   --full  Download all available models
#
# CUDA versions:
#   --cuda 124  Use CUDA 12.4 requirements (default)
#   --cuda 118  Use CUDA 11.8 requirements (older GPUs)
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL_MODE="dev"
CUDA_VER="124"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)   MODEL_MODE="dev";  shift ;;
        --full)  MODEL_MODE="full"; shift ;;
        --cuda)  CUDA_VER="$2";     shift 2 ;;
        --cuda=*) CUDA_VER="${1#*=}"; shift ;;
        -h|--help)
            echo "Usage: bash scripts/install.sh [--dev|--full] [--cuda 124|--cuda 118]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Detect OS ─────────────────────────────────────────────────────────────────
OS="linux"
case "$(uname -s)" in
    Darwin*)  OS="macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
esac

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Detect Python ─────────────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "[ERROR] Python not found. Install Python 3.10+ or activate your conda environment first."
    exit 1
fi

# ── Detect pip flags ──────────────────────────────────────────────────────────
PIP_FLAGS=""
if [[ "$EUID" -eq 0 ]] && [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    PIP_FLAGS="--break-system-packages"
fi

echo ""
echo "============================================================"
echo "  VisoMaster Installer"
echo "  OS: $OS | CUDA: $CUDA_VER | Models: $MODEL_MODE"
echo "============================================================"
echo ""

# ── Step 1: Submodules ────────────────────────────────────────────────────────
echo "[1/5] Initialising git submodules..."
if command -v git &>/dev/null; then
    git submodule update --init --recursive
    echo "      Done."
else
    echo "      [WARN] git not found — skipping submodule init."
fi

# ── Step 2: System dependencies (Linux only) ──────────────────────────────────
if [[ "$OS" == "linux" ]]; then
    echo "[2/5] Installing system dependencies..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq
        apt-get install -y -qq \
            python3-pip python3-dev ffmpeg \
            libgl1-mesa-glx libglib2.0-0 \
            libxkbcommon0 libdbus-1-3 \
            > /dev/null 2>&1 || true
    elif command -v yum &>/dev/null; then
        yum install -y python3-pip python3-devel ffmpeg mesa-libGL glib2 \
            > /dev/null 2>&1 || true
    elif command -v brew &>/dev/null; then
        : # macOS — handled below
    fi
    echo "      Done."
elif [[ "$OS" == "macos" ]]; then
    echo "[2/5] Installing system dependencies (macOS)..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg || true
    else
        echo "      [WARN] Homebrew not found. Install ffmpeg manually: https://ffmpeg.org"
    fi
    echo "      Done."
else
    echo "[2/5] Windows detected — skipping system package install."
    echo "      Ensure ffmpeg is available in PATH or in dependencies/."
fi

# ── Step 3: Python dependencies ───────────────────────────────────────────────
REQUIREMENTS="requirements_cu${CUDA_VER}.txt"
if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "[ERROR] Requirements file not found: $REQUIREMENTS"
    echo "        Valid options: requirements_cu124.txt, requirements_cu118.txt"
    exit 1
fi

echo "[3/5] Installing Python dependencies from $REQUIREMENTS..."
$PYTHON -m pip install $PIP_FLAGS --upgrade pip --quiet
$PYTHON -m pip install $PIP_FLAGS -r "$REQUIREMENTS"
echo "      Done."

# ── Helper: install bun ───────────────────────────────────────────────────────
install_bun() {
    echo "      bun not found — installing bun..."

    if [[ "$OS" == "macos" ]] && command -v brew &>/dev/null; then
        # Homebrew is the cleanest path on macOS
        brew tap oven-sh/bun
        brew install bun
    elif command -v curl &>/dev/null; then
        # Official install script (Linux, macOS, Git Bash / MSYS2 / WSL on Windows)
        # Requires: unzip
        if [[ "$OS" == "linux" ]] && command -v apt-get &>/dev/null; then
            apt-get install -y -qq unzip > /dev/null 2>&1 || true
        elif [[ "$OS" == "linux" ]] && command -v yum &>/dev/null; then
            yum install -y unzip > /dev/null 2>&1 || true
        fi
        curl -fsSL https://bun.sh/install | bash
        # The installer drops bun into ~/.bun/bin — add it to PATH for the rest of this script
        export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
        export PATH="$BUN_INSTALL/bin:$PATH"
    else
        echo "      [ERROR] curl is required to install bun automatically."
        echo "              Install bun manually: https://bun.sh/docs/installation"
        return 1
    fi

    if command -v bun &>/dev/null; then
        echo "      bun $(bun --version) installed successfully."
    else
        echo "      [ERROR] bun installation failed. Install manually: https://bun.sh/docs/installation"
        return 1
    fi
}

# ── Step 4: Frontend dependencies ─────────────────────────────────────────────
echo "[4/5] Installing frontend dependencies (visomaster-ui)..."
if [[ -d "visomaster-ui" ]]; then
    # Ensure bun is available, installing it if necessary
    if ! command -v bun &>/dev/null; then
        install_bun
    fi

    if command -v bun &>/dev/null; then
        (cd visomaster-ui && bun install)
        echo "      Done (bun)."
    elif command -v npm &>/dev/null; then
        echo "      [WARN] Falling back to npm. Install bun for faster installs: https://bun.sh"
        (cd visomaster-ui && npm install)
        echo "      Done (npm)."
    else
        echo "      [ERROR] Neither bun nor npm is available. Install bun: https://bun.sh/docs/installation"
        exit 1
    fi
else
    echo "      [SKIP] visomaster-ui directory not found."
fi

# ── Step 5: Download models ───────────────────────────────────────────────────
echo "[5/5] Downloading models (mode: $MODEL_MODE)..."
$PYTHON download_models.py --mode "$MODEL_MODE"
echo "      Done."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  Copy .env.example to .env and fill in your credentials:"
echo "    cp .env.example .env"
echo ""
echo "  Launch VisoMaster:"
echo "    bash scripts/launch.sh --mode qt          # Native Qt UI"
echo "    bash scripts/launch.sh --mode webview     # Qt + embedded web UI"
echo "    bash scripts/launch.sh --mode web         # Web-only (API + browser)"
echo "============================================================"
echo ""
