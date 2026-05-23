#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Tailscale Setup for RunPod
# ─────────────────────────────────────────────────────────────────────────────
# This script installs and connects Tailscale on a RunPod instance.
# Once connected, your PC (also on Tailscale) can reach the pod directly
# via its Tailscale IP — including UDP for WebRTC.
#
# Prerequisites:
#   1. Create a free Tailscale account at https://tailscale.com
#   2. Generate an auth key at: https://login.tailscale.com/admin/settings/keys
#      - Check "Reusable" and "Ephemeral" (auto-removes when pod stops)
#   3. Set the auth key as environment variable TAILSCALE_AUTHKEY
#      (either in RunPod template env vars, or export it before running this script)
#
# Usage:
#   export TAILSCALE_AUTHKEY="tskey-auth-xxxxx"
#   bash scripts/setup_tailscale.sh
#
# After running:
#   - The pod will appear in your Tailscale admin panel
#   - Access VisoMaster at http://<tailscale-ip>:9091
#   - WebRTC will work directly (no TURN needed!)
# ─────────────────────────────────────────────────────────────────────────────

set -e

# Check for auth key
if [ -z "$TAILSCALE_AUTHKEY" ]; then
    echo "ERROR: TAILSCALE_AUTHKEY environment variable not set."
    echo ""
    echo "Get one from: https://login.tailscale.com/admin/settings/keys"
    echo "Then: export TAILSCALE_AUTHKEY=\"tskey-auth-xxxxx\""
    exit 1
fi

echo "── Installing Tailscale ──────────────────────────────────────────"

# Install Tailscale (works on Ubuntu/Debian-based RunPod images)
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

echo "── Starting Tailscale daemon ─────────────────────────────────────"

# Start tailscaled in background (userspace networking for containers)
# RunPod containers don't have /dev/net/tun, so we use userspace mode
tailscaled --tun=userspace-networking --state=/tmp/tailscale-state &
sleep 2

echo "── Connecting to Tailscale network ───────────────────────────────"

# Connect with the auth key
# --hostname: gives the pod a recognizable name in your Tailscale panel
# --accept-routes: allows routing through the network
HOSTNAME="runpod-visomaster-$(hostname | tail -c 8)"
tailscale up --authkey="$TAILSCALE_AUTHKEY" --hostname="$HOSTNAME" --accept-routes

# Get and display the Tailscale IP
TAILSCALE_IP=$(tailscale ip -4)
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Tailscale connected!"
echo "  Tailscale IP: $TAILSCALE_IP"
echo ""
echo "  Access VisoMaster from your PC at:"
echo "    http://$TAILSCALE_IP:9091"
echo ""
echo "  WebRTC will work directly over the tunnel (no TURN needed)"
echo "══════════════════════════════════════════════════════════════════"
echo ""
