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

# ── Tailscale VPN (optional, for direct WebRTC without TURN) ──────────────
# Set TAILSCALE_AUTHKEY in your RunPod template env vars to auto-connect.
# Get a key from: https://login.tailscale.com/admin/settings/keys
if [ -n "$TAILSCALE_AUTHKEY" ]; then
    echo ""
    echo "  Setting up Tailscale VPN tunnel..."
    bash "$SCRIPT_DIR/setup_tailscale.sh"
fi

echo ""
echo "  WHIP endpoint (for Larix/OBS):"
echo "    http://<your-ip>:9091/whip"
echo ""
echo "  Web client:"
echo "    http://<your-ip>:9091/"
echo ""

# ── TURN Server Configuration (fallback if no VPN) ───────────────────────
# Only needed if NOT using Tailscale/WireGuard.
# WebRTC media (UDP) cannot traverse HTTP proxies without a TURN relay.
#
# export TURN_URL="turn:your-turn-server.com:3478,turns:your-turn-server.com:5349"
# export TURN_USERNAME="your-username"
# export TURN_PASSWORD="your-password"

if [ -z "$TAILSCALE_AUTHKEY" ] && [ -z "$TURN_URL" ]; then
    echo "  ⚠️  WARNING: No Tailscale or TURN server configured!"
    echo "  WebRTC streaming will NOT work through RunPod proxy."
    echo "  Either set TAILSCALE_AUTHKEY or TURN_URL/TURN_USERNAME/TURN_PASSWORD."
    echo ""
fi

python3 main.py "$@"
