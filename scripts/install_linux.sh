#!/bin/bash
# VisoMaster Linux Installation Script
# Designed for RunPod and similar GPU cloud environments
# Requires: NVIDIA GPU with CUDA support

set -e

echo "============================================"
echo "  VisoMaster Linux Installer"
echo "============================================"
echo ""

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then
    PIP_BREAK="--break-system-packages"
else
    PIP_BREAK=""
fi

# Detect if we're in a conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "[INFO] Conda environment detected: $CONDA_DEFAULT_ENV"
    PIP_BREAK=""
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[1/5] Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq python3-pip python3-dev ffmpeg libgl1-mesa-glx libglib2.0-0 libxkbcommon0 libdbus-1-3 > /dev/null 2>&1 || true
elif command -v yum &> /dev/null; then
    yum install -y python3-pip python3-devel ffmpeg mesa-libGL glib2 > /dev/null 2>&1 || true
fi
echo "    Done."

echo "[2/5] Installing Python dependencies..."
pip install $PIP_BREAK --upgrade pip > /dev/null 2>&1
pip install $PIP_BREAK -r requirements_cu124.txt 2>&1 | tail -5
echo "    Done."

echo "[3/5] Downloading models..."
python3 download_models.py
echo "    Done."

echo "[4/5] Downloading additional dependencies..."
# Create dependencies directory if it doesn't exist
mkdir -p dependencies
echo "    Note: Download dependency files from:"
echo "    https://github.com/visomaster/visomaster-assets/releases/tag/v0.1.0_dp"
echo "    and place them in the 'dependencies/' folder."
echo ""

echo "[5/5] Installation complete!"
echo ""
echo "============================================"
echo "  To run VisoMaster:"
echo "    python3 main.py"
echo ""
echo "  For headless/WebRTC mode (RunPod):"
echo "    python3 main.py --headless"
echo "    or use the run script:"
echo "    bash scripts/run_linux.sh"
echo "============================================"
