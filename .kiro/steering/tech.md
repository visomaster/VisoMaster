# VisoMaster — Tech Stack & Build System

## Runtime Environments

The project has two parallel UI modes that share the same Python backend:

1. **Legacy Qt desktop app** — `main.py` → PySide6 `QApplication` + `MainWindow`
2. **Headless API server + React frontend** — `app/api/server.py` (FastAPI/uvicorn) + `frontend/` (React/Vite)

Both modes use the same `app/processors/` inference pipeline and `app/core/state.py` as the single source of truth.

## Python Backend

| Concern | Library / Version |
|---|---|
| UI framework | PySide6 6.8 |
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

## Frontend

| Concern | Library / Version |
|---|---|
| Framework | React 19 |
| Language | TypeScript 6 |
| Build tool | Vite 8 |
| Package manager | Bun |
| Styling | Tailwind CSS 3 + PostCSS |
| State management | Zustand 5 |
| Server state / caching | TanStack React Query 5 |
| WebSocket | react-use-websocket 4 |
| Drag and drop | @dnd-kit/core + sortable |
| Icons | lucide-react |
| Path alias | `@/` → `frontend/src/` |
| Dev proxy | Vite proxies `/api` and `/ws` to `http://localhost:8000` |

## Key Architecture Patterns

- **AppState** (`app/core/state.py`) is the single source of truth — a plain Python dataclass shared by both the Qt layer and the FastAPI layer. Never duplicate state between them.
- **ModelsProcessor** lazily loads ONNX/TensorRT/DFM models on first use. Models are cached in `self.models` dict; `None` means not yet loaded.
- **FrameWorker** (`app/processors/workers/frame_worker.py`) is a `threading.Thread` that runs the full detect → align → swap → restore → enhance pipeline for one frame.
- **VideoProcessor** owns the play loop, frame queue, and ffmpeg subprocess. In Qt mode it's a `QObject`; in API mode it's a headless class with the same interface.
- **streamrelay** (`packages/streamrelay/`) is a git submodule running as a separate process. It writes BGR frames into a named shared memory segment (`visomaster_webrtc_frame`). The main process reads from it.
- **FastAPI routes** are organized by domain under `app/api/routes/`. Each router uses `Depends(get_app_state)` and `Depends(get_video_processor)` for dependency injection.
- **Frontend API client** (`frontend/src/api/client.ts`) is a thin typed wrapper around `fetch`. All API calls go through `api.*` methods — no raw `fetch` calls in components.
- **Frontend store** (`frontend/src/store/appStore.ts`) is a single Zustand store. WebSocket events from `/ws/events` update it in real time.

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

# Run Qt desktop app (Windows)
Start.bat
# or directly:
python main.py

# Run headless API server
python -m app.api.server
# or:
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload

# Linux / RunPod install
bash scripts/install_linux.sh
python3 main.py
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
