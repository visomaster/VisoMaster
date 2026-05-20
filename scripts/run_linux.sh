#!/bin/bash
# VisoMaster Linux Run Script
# For RunPod and headless GPU environments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Set environment variables for headless Qt
export QT_QPA_PLATFORM=offscreen
export DISPLAY=:0

# Parse arguments
HEADLESS=false
WEBRTC_PORT=9091
WEBRTC_HTTPS_PORT=9090

while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS=true
            shift
            ;;
        --http-port)
            WEBRTC_PORT="$2"
            shift 2
            ;;
        --https-port)
            WEBRTC_HTTPS_PORT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================"
echo "  VisoMaster - Starting"
echo "============================================"
echo ""

if [ "$HEADLESS" = true ]; then
    echo "[Mode] Headless (WebRTC only)"
    echo "[WebRTC] HTTP port: $WEBRTC_PORT"
    echo "[WebRTC] HTTPS port: $WEBRTC_HTTPS_PORT"
    echo ""
    echo "Connect via WHIP: http://<your-ip>:${WEBRTC_PORT}/whip"
    echo "Web client: http://<your-ip>:${WEBRTC_PORT}/"
    echo ""
fi

python3 main.py "$@"
