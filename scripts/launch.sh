#!/usr/bin/env bash
# =============================================================================
# VisoMaster — Cross-platform Launch Script
# Supports: Linux, macOS, Windows (Git Bash / MSYS2 / WSL)
#
# Usage:
#   bash scripts/launch.sh [--mode <mode>] [-- <extra args>]
#
# Modes:
#   qt        Native Qt desktop UI (main.py)                        [default]
#   webview   Native Qt window with embedded web UI (web_main.py)
#   web       Headless API server + React frontend in browser
#
# Extra args are forwarded to the Python entry point (webview mode only):
#   bash scripts/launch.sh --mode webview -- --skip-workspace
#   bash scripts/launch.sh --mode webview -- --workspace path/to/ws.json
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="qt"
EXTRA_ARGS=()

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)   MODE="$2"; shift 2 ;;
        --mode=*) MODE="${1#*=}"; shift ;;
        --)       shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            echo "Usage: bash scripts/launch.sh [--mode qt|webview|web] [-- <extra args>]"
            echo ""
            echo "Modes:"
            echo "  qt        Native Qt desktop UI (default)"
            echo "  webview   Qt window with embedded React web UI"
            echo "  web       Headless API server + React frontend in browser"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Detect Python ─────────────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "[ERROR] Python not found. Activate your conda environment first."
    exit 1
fi

# ── Load .env if present ──────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# ── Detect OS for PATH additions ──────────────────────────────────────────────
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        # Add bundled dependencies to PATH on Windows
        if [[ -d "dependencies" ]]; then
            export PATH="$PROJECT_ROOT/dependencies:$PATH"
        fi ;;
esac

echo ""
echo "============================================================"
echo "  VisoMaster — mode: $MODE"
echo "============================================================"
echo ""

case "$MODE" in
    # ── Mode 1: Native Qt UI ─────────────────────────────────────────────────
    qt)
        echo "  Starting native Qt UI..."
        exec $PYTHON main.py "${EXTRA_ARGS[@]}"
        ;;

    # ── Mode 2: Qt + embedded web UI ─────────────────────────────────────────
    webview)
        echo "  Starting Qt + WebView UI..."
        echo ""
        echo "  NOTE: The Vite dev server must be running."
        echo "  In a separate terminal: cd visomaster-ui && bun run dev"
        echo ""

        # Check if Vite dev server is already up
        VITE_PORT="${VITE_PORT:-5173}"
        if command -v curl &>/dev/null; then
            if ! curl -sf "http://localhost:${VITE_PORT}" > /dev/null 2>&1; then
                echo "  [WARN] Vite dev server not detected on port ${VITE_PORT}."
                echo "         Start it first or the webview will show a blank page."
                echo ""
            fi
        fi

        exec $PYTHON web_main.py "${EXTRA_ARGS[@]}"
        ;;

    # ── Mode 3: Web-only (API server + React frontend) ────────────────────────
    web)
        echo "  Starting web-only mode (API server + React frontend)..."
        echo ""

        # Trap to kill background processes on exit
        cleanup() {
            echo ""
            echo "  Shutting down..."
            [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
            [[ -n "${UI_PID:-}" ]]  && kill "$UI_PID"  2>/dev/null || true
            wait 2>/dev/null || true
        }
        trap cleanup EXIT INT TERM

        # Start API server in background
        echo "  [1/2] Starting FastAPI server on http://localhost:8000 ..."
        $PYTHON -m app.api.server &
        API_PID=$!

        # Wait briefly for the server to be ready
        sleep 2

        # Start Vite dev server in background
        echo "  [2/2] Starting Vite dev server (visomaster-ui)..."
        if command -v bun &>/dev/null; then
            (cd visomaster-ui && bun run dev) &
        elif command -v npm &>/dev/null; then
            (cd visomaster-ui && npm run dev) &
        else
            echo "  [ERROR] bun or npm required for web mode. Install bun: https://bun.sh"
            exit 1
        fi
        UI_PID=$!

        echo ""
        echo "  ✓ API server:  http://localhost:8000"
        echo "  ✓ Web UI:      http://localhost:5173"
        echo ""
        echo "  Press Ctrl+C to stop both servers."
        echo ""

        # Wait for either process to exit
        wait -n "$API_PID" "$UI_PID" 2>/dev/null || wait
        ;;

    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "        Valid modes: qt, webview, web"
        exit 1
        ;;
esac
