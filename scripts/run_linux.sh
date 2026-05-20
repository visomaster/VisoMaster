#!/bin/bash
# VisoMaster Linux Run Script
# For RunPod and similar GPU environments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================"
echo "  VisoMaster - Starting"
echo "============================================"
echo ""
echo "  WHIP endpoint (for Larix/OBS):"
echo "    http://<your-ip>:9091/whip"
echo ""
echo "  Web client:"
echo "    http://<your-ip>:9091/"
echo ""

python3 main.py "$@"
