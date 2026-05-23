#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Tailscale Setup for RunPod
# ─────────────────────────────────────────────────────────────────────────────
# Installs and connects Tailscale. Tries kernel mode first (full UDP support
# for WebRTC), falls back to userspace mode (TCP/WebSocket only).
#
# Prerequisites:
#   1. Create a free Tailscale account at https://tailscale.com
#   2. Generate an auth key at: https://login.tailscale.com/admin/settings/keys
#      - Check "Reusable" and "Ephemeral"
#   3. Set TAILSCALE_AUTHKEY environment variable
#
# Usage:
#   export TAILSCALE_AUTHKEY="tskey-auth-xxxxx"
#   bash scripts/setup_tailscale.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

if [ -z "$TAILSCALE_AUTHKEY" ]; then
    echo "ERROR: TAILSCALE_AUTHKEY not set."
    echo "Get one from: https://login.tailscale.com/admin/settings/keys"
    exit 1
fi

echo "── Installing Tailscale ──────────────────────────────────────────"
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

echo "── Starting Tailscale daemon ─────────────────────────────────────"

# Kill any existing tailscaled
pkill tailscaled 2>/dev/null || true
sleep 1

# Try kernel mode first (requires /dev/net/tun — gives full UDP support)
USE_USERSPACE=false
if [ -e /dev/net/tun ]; then
    echo "  /dev/net/tun found — using kernel networking (full UDP/WebRTC support)"
    tailscaled --state=/tmp/tailscale-state --socket=/tmp/tailscale.sock &
    sleep 2
else
    echo "  /dev/net/tun NOT found — using userspace networking (WebSocket only)"
    echo "  To enable WebRTC: run pod with --cap-add=NET_ADMIN and create /dev/net/tun"
    USE_USERSPACE=true
    
    # Create TUN device if we have permissions
    if [ -w /dev ]; then
        mkdir -p /dev/net
        mknod /dev/net/tun c 10 200 2>/dev/null || true
        chmod 600 /dev/net/tun 2>/dev/null || true
        
        if [ -e /dev/net/tun ]; then
            echo "  Created /dev/net/tun — retrying kernel mode"
            USE_USERSPACE=false
            tailscaled --state=/tmp/tailscale-state --socket=/tmp/tailscale.sock &
            sleep 2
        fi
    fi
    
    if [ "$USE_USERSPACE" = true ]; then
        tailscaled --tun=userspace-networking --state=/tmp/tailscale-state --socket=/tmp/tailscale.sock &
        sleep 2
    fi
fi

echo "── Connecting to Tailscale network ───────────────────────────────"

HOSTNAME="runpod-visomaster-$(hostname | tail -c 8)"
tailscale --socket=/tmp/tailscale.sock up --authkey="$TAILSCALE_AUTHKEY" --hostname="$HOSTNAME" --accept-routes

# Get Tailscale IP
TAILSCALE_IP=$(tailscale --socket=/tmp/tailscale.sock ip -4)

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Tailscale connected!"
echo "  Tailscale IP: $TAILSCALE_IP"
echo ""
if [ "$USE_USERSPACE" = true ]; then
    echo "  Mode: USERSPACE (WebSocket streaming only)"
    echo "  WebRTC will NOT work — using WebSocket fallback"
    echo ""
    echo "  To enable WebRTC, recreate pod with:"
    echo "    docker run --cap-add=NET_ADMIN --device=/dev/net/tun ..."
else
    echo "  Mode: KERNEL (full UDP support — WebRTC will work!)"
fi
echo ""
echo "  Access VisoMaster from your PC at:"
echo "    http://$TAILSCALE_IP:9091"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# Export for other scripts to use
export TAILSCALE_IP
