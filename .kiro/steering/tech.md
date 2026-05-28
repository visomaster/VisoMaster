# VisoMaster — Tech Stack & Build System

## Runtime Environments

The project has three UI modes that share the same Python backend:

1. **Qt desktop app** — `main.py` → PySide6 `QApplication` + `MainWindow`
2. **Qt WebEngine app** — `web_main.py` → `QWebEngineView` + `BackendBridge` (QWebChannel) + React UI
3. **Headless API server + React frontend** — `app/api/server.py` (FastAPI/uvicorn) + `visomaster-ui/` (React/Vite)

All modes use the same `app/processors/` inference pipeline and `app/core/state.py` (`AppState`) as the single source of truth.

`Start.bat` presents a menu to choose between all three modes and manages background process lifecycle (Vite dev server + API server PIDs saved to `logs/`).

## Python Backend

| Concern | Library / Version |
|---|---|
| UI framework | PySide6 6.8 |
| Web engine | PySide6.QtWebEngineWidgets + QWebChannel |
| Inference engine | onnxruntime-gpu 1.20 |
| GPU acceleration | PyTorch 2.4 + CUDA 12.4 (or 11.8) |
| TensorRT | tensorrt 10.6 (optional, auto-detected) |
| Computer vision | opencv-python 4.10 |
| Image processing | kornia, scikit-image, Pillow 9.5 |
| API server | FastAPI 0.115 + uvicorn 0.32 |
| WebRTC | aiortc + aiohttp (in `packages/streamrelay`) |
| Virtual camera | pyvirtualcam 0.11 |
| Video muxing | FFmpeg (bundled at `dependencies/ffmpeg.exe` on Windows) |
| UI theming | qdarktheme, qdarkstyle, pyqt-toast-notification |
| Python version | 3.10.13 (conda environment `visomaster`) |

Two requirements files exist for different CUDA versions:
- `requirements_cu124.txt` — CUDA 12.4 (primary)
- `requirements_cu118.txt` — CUDA 11.8 (older GPUs)

## Frontend (`visomaster-ui/`)

| Concern | Library / Version |
|---|---|
| Framework | React 19 + TanStack Start |
| Router | TanStack Router (file-based) |
| Language | TypeScript 5 |
| Build tool | Vite 7 |
| Package manager | Bun |
| Styling | Tailwind CSS 4 + shadcn/ui (Radix primitives) |
| State management | Zustand 5 |
| Server state / caching | TanStack React Query 5 |
| WebSocket | react-use-websocket 4 |
| Drag and drop | @dnd-kit/core + sortable |
| Icons | @tabler/icons-react + lucide-react |
| Charts | Recharts |
| Path alias | `@/` → `visomaster-ui/src/` |
| Dev proxy | Vite proxies `/api` and `/ws` to `http://localhost:8000` |

## Key Architecture Patterns

- **AppState** (`app/core/state.py`) is the single source of truth — a typed Python dataclass shared by all three modes. Never duplicate state between them.
- **ModelsProcessor** lazily loads ONNX/TensorRT/DFM models on first use. Models are cached in `self.models` dict; `None` means not yet loaded.
- **FrameWorker** (`app/processors/workers/frame_worker.py`) is a `threading.Thread` that runs the full detect → align → swap → restore → enhance pipeline for one frame.
- **VideoProcessor** owns the play loop, frame queue, and ffmpeg subprocess. In Qt mode it's a `QObject`; in API mode it's a headless class with the same interface.
- **streamrelay** (`packages/streamrelay/`) is a git submodule running as a separate process. It writes BGR frames into a named shared memory segment (`visomaster_webrtc_frame`). The main process reads from it.
- **FastAPI routes** are organized by domain under `app/api/routes/`. Each router uses `Depends(get_app_state)`, `Depends(get_models_processor)`, and `Depends(get_video_processor)` for dependency injection.
- **EventBus** (`app/api/events.py`) bridges sync worker threads to async WebSocket clients. Three channels: JSON events queue (`/ws/events`), latest-frame-wins JPEG slot (`/ws/preview`), and latest-wins playback state slot (`/ws/playback`).
- **BackendBridge** (`app/ui/bridge.py`) is the Qt WebEngine equivalent of the FastAPI layer — exposes `@Slot` methods and `Signal`s to the React UI via `QWebChannel`. Mirrors the REST API 1-to-1 so the frontend transport layer can call either path identically.
- **Frontend transport** (`visomaster-ui/src/transport/`) abstracts over both the WebSocket/REST path (headless mode) and the QWebChannel bridge path (Qt WebEngine mode).
- **Frontend stores** (`visomaster-ui/src/store/`) are Zustand stores. WebSocket events from `/ws/events` (or bridge signals) update them in real time.

## Common Commands

### Python / Backend

```bash
# Create and activate conda environment
conda create -n visomaster python=3.10.13 -y
conda activate visomaster

# Install CUDA + cuDNN (CUDA 12.4)
conda install -c nvidia/label/cuda-12.4.1 cuda-runtime
conda install -c conda-forge cudnn

# Install Python dependencies (CUDA 12.4)
pip install -r requirements_cu124.txt

# Download ONNX models
python download_models.py

# Run via launcher menu (Windows) — choose Qt / WebView / Web
Start.bat

# Run Qt desktop app directly
python main.py

# Run Qt WebEngine app directly (Vite must be running first)
python web_main.py
python web_main.py --skip-workspace
python web_main.py --auto-last-workspace
python web_main.py --workspace path/to/workspace.json

# Run headless API server
python -m app.api.server
# or:
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload

# Linux / RunPod install
bash scripts/install_linux.sh
python3 main.py
```

### Frontend

```bash
# Install dependencies
cd visomaster-ui && bun install

# Start dev server (port 5173)
bun run dev

# Build for production
bun run build

# Add a shadcn/ui component
bunx shadcn add <component>
```

### Model / Asset Management

```bash
# Download all ONNX models
python download_models.py

# Check dependency integrity
python scripts/check_dependencies.py

# Fix onnxruntime GPU provider issues
bash scripts/fix_onnxruntime.sh
```

## Inference Providers

Models run on one of three ONNX Runtime execution providers, switchable at runtime:
- `CUDA` — CUDAExecutionProvider (default)
- `TensorRT` — TensorrtExecutionProvider (fastest; engines cached in `tensorrt-engines/`)
- `CPU` — CPUExecutionProvider (fallback)

TensorRT engines are auto-built on first use and cached. The `trt_engine_cache_path` and `trt_ep_context_file_path` both point to `tensorrt-engines/`.

## Model Storage

- ONNX models: `model_assets/` (downloaded via `download_models.py`)
- DFM models: `model_assets/dfm_models/` (user-supplied DeepFaceLab files)
- TensorRT engines: `tensorrt-engines/` (auto-generated)
- Workspace state: `last_workspace.json` (auto-saved on close, auto-loaded on launch)
- Thumbnails: `.thumbnails/` (auto-generated cache)
- Server logs: `logs/` (vite.log, api.log, vite.pid, api.pid — created by Start.bat)
