#!/usr/bin/env bash
# =============================================================================
# VisoMaster — Service launcher
# Starts: VisoMaster API, filebrowser
# All services log to /workspace/logs/
# =============================================================================
set -euo pipefail

VISOMASTER_HOME=${VISOMASTER_HOME:-/workspace/VisoMaster}
CONDA_ENV=${CONDA_ENV:-visomaster}
CONDA_DIR=${CONDA_DIR:-/opt/conda}
PYTHON=$CONDA_DIR/envs/$CONDA_ENV/bin/python3

API_PORT=${API_PORT:-8000}
FILEBROWSER_PORT=${FILEBROWSER_PORT:-8585}
STREAMRELAY_PORT=${STREAMRELAY_PORT:-9091}

LOG_DIR=/workspace/logs
mkdir -p $LOG_DIR

# ── Workspace data dirs ───────────────────────────────────────────────────────
# Symlink model_assets, tensorrt-engines and output into the persistent volume
# so they survive container restarts when /workspace/data is a network volume.
DATA_DIR=/workspace/data
mkdir -p $DATA_DIR/model_assets $DATA_DIR/output $DATA_DIR/tensorrt-engines

if [ ! -L $VISOMASTER_HOME/model_assets ]; then
    rm -rf $VISOMASTER_HOME/model_assets
    ln -s $DATA_DIR/model_assets $VISOMASTER_HOME/model_assets
fi
if [ ! -L $VISOMASTER_HOME/tensorrt-engines ]; then
    rm -rf $VISOMASTER_HOME/tensorrt-engines
    ln -s $DATA_DIR/tensorrt-engines $VISOMASTER_HOME/tensorrt-engines
fi
if [ ! -L $VISOMASTER_HOME/output ]; then
    rm -rf $VISOMASTER_HOME/output
    ln -s $DATA_DIR/output $VISOMASTER_HOME/output
fi

# Restore last workspace state from volume if present
if [ -f $DATA_DIR/last_workspace.json ] && [ ! -f $VISOMASTER_HOME/last_workspace.json ]; then
    cp $DATA_DIR/last_workspace.json $VISOMASTER_HOME/last_workspace.json
fi

cd $VISOMASTER_HOME

# ── Download models if not present ───────────────────────────────────────────
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    if [ ! "$(ls -A $DATA_DIR/model_assets 2>/dev/null)" ]; then
        echo "[services] Downloading VisoMaster models (first run)..."
        $PYTHON download_models.py >> $LOG_DIR/model_download.log 2>&1 &
        echo "[services] Model download running in background — tail $LOG_DIR/model_download.log"
    fi
fi

# ── VisoMaster FastAPI server ─────────────────────────────────────────────────
echo "[services] Starting VisoMaster API on port $API_PORT..."
$PYTHON -m app.api.server >> $LOG_DIR/visomaster_api.log 2>&1 &
API_PID=$!
echo "[services] API PID=$API_PID"

# Wait for API to be ready (up to 30s)
for i in $(seq 1 30); do
    if curl -sf http://localhost:$API_PORT/api/system > /dev/null 2>&1; then
        echo "[services] VisoMaster API is ready."
        break
    fi
    sleep 1
done

# ── filebrowser ───────────────────────────────────────────────────────────────
echo "[services] Starting filebrowser on port $FILEBROWSER_PORT..."
filebrowser \
    --address 0.0.0.0 \
    --port $FILEBROWSER_PORT \
    --root /workspace \
    --noauth \
    >> $LOG_DIR/filebrowser.log 2>&1 &
echo "[services] filebrowser PID=$!"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  VisoMaster Services"
echo "============================================================"
echo "  VisoMaster Web UI:  http://localhost:$API_PORT"
echo "  filebrowser:        http://localhost:$FILEBROWSER_PORT"
echo "  noVNC desktop:      http://localhost:${NO_VNC_PORT:-6901}/vnc.html"
echo "  streamrelay WebRTC: http://localhost:$STREAMRELAY_PORT"
echo "============================================================"
echo "  Logs: /workspace/logs/"
echo ""

# Keep alive — exit when API exits
wait $API_PID
