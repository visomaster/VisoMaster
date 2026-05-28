#!/usr/bin/env bash
# =============================================================================
# VisoMaster — VNC + noVNC startup script
# Starts: TigerVNC → Xfce4 → noVNC websockify → all services
# =============================================================================
set -euo pipefail

STARTUPDIR=${STARTUPDIR:-/dockerstartup}
NO_VNC_HOME=${NO_VNC_HOME:-/opt/noVNC}
VNC_PORT=${VNC_PORT:-5901}
NO_VNC_PORT=${NO_VNC_PORT:-6901}
VNC_RESOLUTION=${VNC_RESOLUTION:-1280x800}
VNC_COL_DEPTH=${VNC_COL_DEPTH:-24}
VNC_PW=${VNC_PW:-visomaster}
VNC_PASSWORDLESS=${VNC_PASSWORDLESS:-false}
DISPLAY=${DISPLAY:-:1}

LOG=/workspace/logs/vnc_startup.log
mkdir -p /workspace/logs

echo "============================================================" | tee -a $LOG
echo "  VisoMaster VNC Container — $(date)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# ── VNC password ──────────────────────────────────────────────────────────────
mkdir -p ~/.vnc
if [[ "$VNC_PASSWORDLESS" == "true" ]]; then
    echo "" | vncpasswd -f > ~/.vnc/passwd
else
    echo "$VNC_PW" | vncpasswd -f > ~/.vnc/passwd
fi
chmod 600 ~/.vnc/passwd

# ── Kill any stale VNC server ─────────────────────────────────────────────────
vncserver -kill $DISPLAY 2>/dev/null || true
rm -f /tmp/.X*-lock /tmp/.X11-unix/X${DISPLAY#:} 2>/dev/null || true

# ── Start TigerVNC ────────────────────────────────────────────────────────────
echo "[VNC] Starting TigerVNC on display $DISPLAY (${VNC_RESOLUTION})..." | tee -a $LOG
vncserver $DISPLAY \
    -depth $VNC_COL_DEPTH \
    -geometry $VNC_RESOLUTION \
    -rfbport $VNC_PORT \
    -rfbauth ~/.vnc/passwd \
    -SecurityTypes VncAuth \
    -fg &
VNC_PID=$!

# Wait for VNC to be ready
sleep 2

# ── Start Xfce4 desktop ───────────────────────────────────────────────────────
echo "[VNC] Starting Xfce4 desktop..." | tee -a $LOG
DISPLAY=$DISPLAY startxfce4 &>/workspace/logs/xfce4.log &

sleep 2

# ── Start noVNC websockify ────────────────────────────────────────────────────
echo "[VNC] Starting noVNC on port $NO_VNC_PORT..." | tee -a $LOG
$NO_VNC_HOME/utils/novnc_proxy \
    --vnc localhost:$VNC_PORT \
    --listen $NO_VNC_PORT \
    &>/workspace/logs/novnc.log &

echo "" | tee -a $LOG
echo "  VNC:   vnc://localhost:$VNC_PORT  (password: $VNC_PW)" | tee -a $LOG
echo "  noVNC: http://localhost:$NO_VNC_PORT/vnc.html" | tee -a $LOG
echo "" | tee -a $LOG

# ── Tailscale (optional — set TAILSCALE_AUTHKEY to enable) ───────────────────
#
# Mode selection (automatic):
#   kernel    — /dev/net/tun available + NET_ADMIN cap (--privileged pods)
#               Full UDP forwarding. WebRTC works natively.
#   userspace — no TUN device. All traffic relayed over TCP/DERP.
#               WebRTC UDP will NOT work without a TURN server.
#
# In both cases Tailscale is installed at image build time, so this block
# only needs to start the daemon and authenticate.
if [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
    echo "[tailscale] Starting Tailscale..." | tee -a $LOG

    # ── Ensure tailscale binary is present ───────────────────────────────────
    # (It's baked into the image, but if somehow missing, install it now)
    if ! command -v tailscale &>/dev/null; then
        echo "[tailscale] Binary not found — installing now..." | tee -a $LOG
        curl -fsSL https://tailscale.com/install.sh | sh >> /workspace/logs/tailscale.log 2>&1
    fi

    # Kill any stale daemon from a previous run
    pkill tailscaled 2>/dev/null || true
    sleep 1

    # ── Probe for a usable TUN device ────────────────────────────────────────
    # Step 1: create the device node if it doesn't exist yet.
    #         This succeeds in --privileged containers; silently fails otherwise.
    if [ ! -e /dev/net/tun ]; then
        mkdir -p /dev/net
        mknod /dev/net/tun c 10 200 2>/dev/null || true
        chmod 600 /dev/net/tun 2>/dev/null || true
    fi

    # Step 2: verify the device is actually readable, not just present.
    #         mknod can create the file even without privilege, but the kernel
    #         will refuse open() on it — tailscaled would crash immediately.
    #         A successful read(0 bytes) from the char device confirms it works.
    TUN_USABLE=false
    if [ -e /dev/net/tun ]; then
        if cat /dev/net/tun > /dev/null 2>&1 || \
           dd if=/dev/net/tun bs=1 count=0 2>/dev/null; then
            TUN_USABLE=true
        elif [ -r /dev/net/tun ] && [ -w /dev/net/tun ]; then
            # Some kernels allow open() but block read — check open permission
            # by trying a subshell redirect (fails fast with EPERM if blocked)
            ( exec 3<>/dev/net/tun ) 2>/dev/null && TUN_USABLE=true || true
        fi
    fi

    # ── Start daemon in the right mode ───────────────────────────────────────
    mkdir -p /workspace/data  # ensure state dir exists before daemon starts
    if [ "$TUN_USABLE" = "true" ]; then
        TAILSCALE_MODE="kernel (full UDP — WebRTC works)"
        echo "[tailscale] /dev/net/tun is usable — starting in kernel mode." | tee -a $LOG
        tailscaled \
            --state=/workspace/data/tailscale-state \
            --socket=/tmp/tailscale.sock \
            >> /workspace/logs/tailscale.log 2>&1 &
    else
        TAILSCALE_MODE="userspace (TCP/DERP only — WebRTC UDP blocked)"
        echo "[tailscale] /dev/net/tun not usable — falling back to userspace mode." | tee -a $LOG
        echo "[tailscale] For kernel mode (full UDP): run with --privileged or --device=/dev/net/tun." | tee -a $LOG
        tailscaled \
            --tun=userspace-networking \
            --state=/workspace/data/tailscale-state \
            --socket=/tmp/tailscale.sock \
            >> /workspace/logs/tailscale.log 2>&1 &
    fi

    # Wait for daemon socket to be ready (up to 10s)
    for i in $(seq 1 10); do
        [ -S /tmp/tailscale.sock ] && break
        sleep 1
    done

    # ── Authenticate ─────────────────────────────────────────────────────────
    TS_HOSTNAME="visomaster-$(cat /etc/hostname 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9' | tail -c 8)"
    tailscale --socket=/tmp/tailscale.sock up \
        --authkey="$TAILSCALE_AUTHKEY" \
        --hostname="$TS_HOSTNAME" \
        --accept-routes \
        >> /workspace/logs/tailscale.log 2>&1

    # ── Wait for IP assignment (up to 15s) ───────────────────────────────────
    TS_IP=""
    for i in $(seq 1 15); do
        TS_IP=$(tailscale --socket=/tmp/tailscale.sock ip -4 2>/dev/null || true)
        [ -n "$TS_IP" ] && break
        sleep 1
    done
    TS_IP="${TS_IP:-unavailable}"

    # ── Print connection info ─────────────────────────────────────────────────
    echo "" | tee -a $LOG
    echo "╔══════════════════════════════════════════════════════════╗" | tee -a $LOG
    echo "  Tailscale connected" | tee -a $LOG
    echo "  Mode:          $TAILSCALE_MODE" | tee -a $LOG
    echo "  Tailscale IP:  $TS_IP" | tee -a $LOG
    echo "  VisoMaster:    http://${TS_IP}:8000" | tee -a $LOG
    echo "  WebRTC WHIP:   http://${TS_IP}:9091/whip" | tee -a $LOG
    echo "  filebrowser:   http://${TS_IP}:8585" | tee -a $LOG
    echo "  noVNC:         http://${TS_IP}:6901/vnc.html" | tee -a $LOG
    echo "╚══════════════════════════════════════════════════════════╝" | tee -a $LOG
    echo "" | tee -a $LOG

    # Export so child scripts can use it
    export TAILSCALE_IP="$TS_IP"
else
    echo "[tailscale] TAILSCALE_AUTHKEY not set — skipping Tailscale." | tee -a $LOG
fi

# ── Trust desktop launchers (Xfce4 requires this to show icons) ──────────────
if [ -d /root/Desktop ]; then
    for f in /root/Desktop/*.desktop; do
        [ -f "$f" ] && gio set "$f" metadata::trusted true 2>/dev/null || \
            chmod +x "$f"
    done
fi

# ── Start all VisoMaster services ─────────────────────────────────────────────
$STARTUPDIR/start_services.sh &

# ── Keep container alive ──────────────────────────────────────────────────────
if [[ "${1:-}" == "--wait" ]]; then
    echo "[VNC] Container running. Logs in /workspace/logs/" | tee -a $LOG
    wait $VNC_PID
else
    exec "$@"
fi
