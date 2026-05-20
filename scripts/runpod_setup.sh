#!/bin/bash
# VisoMaster RunPod Quick Setup
# Run this script on a fresh RunPod instance with an NVIDIA GPU
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/crazidev/VisoMaster/main/scripts/runpod_setup.sh | bash
#   OR
#   bash scripts/runpod_setup.sh

set -e

echo "============================================"
echo "  VisoMaster - RunPod Setup"
echo "============================================"
echo ""

# Clone if not already in the repo
if [ ! -f "main.py" ]; then
    echo "[1/4] Cloning VisoMaster..."
    git clone https://github.com/crazidev/VisoMaster.git
    cd VisoMaster
else
    echo "[1/4] Already in VisoMaster directory."
fi

echo "[2/4] Installing system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg libgl1-mesa-glx libglib2.0-0 libxkbcommon0 libdbus-1-3 2>/dev/null || true
echo "    Done."

echo "[3/4] Installing Python dependencies..."
pip install --break-system-packages --upgrade pip > /dev/null 2>&1 || pip install --upgrade pip > /dev/null 2>&1
pip install --break-system-packages -r requirements_cu124.txt 2>&1 | tail -5 || pip install -r requirements_cu124.txt 2>&1 | tail -5
echo "    Done."

echo "[4/4] Downloading models..."
python3 download_models.py
echo "    Done."

echo ""
echo "============================================"
echo "  Setup Complete!"
echo ""
echo "  Start the WebRTC server:"
echo "    python3 main.py"
echo ""
echo "  WHIP endpoint (for Larix/OBS):"
echo "    http://<pod-ip>:9091/whip"
echo ""
echo "  Web client:"
echo "    http://<pod-ip>:9091/"
echo ""
echo "  Note: Enable WebRTC in Settings tab,"
echo "  then switch to WebRTC in the dropdown."
echo "============================================"
