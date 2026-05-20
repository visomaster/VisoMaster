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

# ── TURN Server Configuration (Required for RunPod/proxy environments) ──
# WebRTC media (UDP) cannot traverse HTTP proxies. A TURN relay is needed.
# Uncomment and set these to enable TURN relay:
#
# export TURN_URL="turn:your-turn-server.com:3478,turns:your-turn-server.com:5349"
# export TURN_USERNAME="your-username"
# export TURN_PASSWORD="your-password"
#
# Free options for testing:
#   - Open Relay Project: https://www.metered.ca/tools/openrelay/
#   - Self-host coturn: https://github.com/coturn/coturn
#
# Example with Open Relay (metered.ca free TURN):
# export TURN_URL="turn:a.relay.metered.ca:80,turn:a.relay.metered.ca:443,turns:a.relay.metered.ca:443"
# export TURN_USERNAME="your-api-key"
# export TURN_PASSWORD="your-api-key"

if [ -z "$TURN_URL" ]; then
    echo "  ⚠️  WARNING: No TURN server configured!"
    echo "  WebRTC streaming will NOT work through RunPod proxy."
    echo "  Set TURN_URL, TURN_USERNAME, TURN_PASSWORD env vars."
    echo ""
fi

python3 main.py "$@"
