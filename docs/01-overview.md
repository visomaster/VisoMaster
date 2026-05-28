# 01 · Overview

## Repo layout

```
VisoMaster/
├── main.py                    # Qt desktop entry point: builds QApplication + MainWindow
├── web_main.py                # Qt WebEngine entry point: wraps React UI in a native window
├── download_models.py         # CLI to fetch ONNX models from visomaster-assets
├── Start.bat / Start_Portable.bat   # Windows launchers
├── requirements_cu118.txt
├── requirements_cu124.txt
├── uvicorn.ini                # uvicorn config for headless API server
├── last_workspace.json        # auto-saved on close, auto-loaded on launch
│
├── app/
│   ├── helpers/               # downloader, integrity checks, file IO, types
│   ├── processors/            # ★ THE BACKEND
│   │   ├── face_detectors.py        # RetinaFace / SCRFD / Yolov8 / Yunet
│   │   ├── face_landmark_detectors.py # 5/68/3d68/98/106/203/478 landmarks
│   │   ├── face_swappers.py         # Inswapper / InStyleSwapper / SimSwap / Ghost / CSCS
│   │   ├── face_restorers.py        # GFPGAN / CodeFormer / GPEN / VQFR / RestoreFormer
│   │   ├── face_masks.py            # Occluder / DFL XSeg / FaceParser / CLIPSeg / mouth+eye restore
│   │   ├── face_editors.py          # LivePortrait motion + makeup
│   │   ├── frame_enhancers.py       # RealEsrGAN / BSRGAN / DDColor / DeOldify
│   │   ├── models_data.py           # model registry: name → local path + hash + URL
│   │   ├── models_processor.py      # ★ loads & owns all model sessions
│   │   ├── video_processor.py       # ★ orchestrates playback, recording, frame queue
│   │   ├── workers/
│   │   │   └── frame_worker.py      # ★ per-frame worker thread (the actual swap)
│   │   ├── utils/
│   │   │   ├── dfm_model.py         # DeepFaceLab DFM file loader
│   │   │   ├── engine_builder.py    # ONNX → TensorRT engine builder
│   │   │   ├── faceutil.py          # alignment / landmark math
│   │   │   └── tensorrt_predictor.py # TensorRT inference wrapper
│   │   └── external/
│   │       ├── cliplib/             # vendored OpenAI CLIP for text-driven masks
│   │       ├── clipseg.py
│   │       └── resnet.py
│   │
│   ├── api/                   # ★ FASTAPI HEADLESS BACKEND
│   │   ├── server.py          # app factory, lifespan, CORS
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── deps.py            # dependency injectors (get_app_state, get_video_processor)
│   │   ├── events.py          # EventBus: async broadcast to WebSocket subscribers
│   │   ├── ws.py              # WebSocket endpoints (/ws/events, /ws/preview)
│   │   └── routes/            # one file per domain (system, schema, state, workspace…)
│   │
│   ├── core/
│   │   └── state.py           # AppState dataclass — single source of truth
│   │
│   └── ui/                    # ★ THE QT FRONTEND (legacy desktop mode)
│       ├── main_ui.py               # MainWindow: holds ALL runtime state
│       ├── core/
│       │   ├── MainWindow.ui        # Qt Designer XML
│       │   ├── main_window.py       # generated from .ui
│       │   └── proxy_style.py
│       ├── styles/                  # qss stylesheets (dark/light)
│       └── widgets/
│           ├── widget_components.py # CardButton, ToggleButton, sliders, dialogs
│           ├── ui_workers.py        # QThreads for media loading + WebRTC server spawn
│           ├── output_window.py     # borderless OBS-friendly output window
│           ├── event_filters.py
│           ├── *_layout_data.py     # ★ declarative parameter schemas (see 03-ui-layer.md)
│           └── actions/             # ★ functions that mutate MainWindow state
│               ├── card_actions.py
│               ├── common_actions.py
│               ├── control_actions.py
│               ├── filter_actions.py
│               ├── graphics_view_actions.py
│               ├── layout_actions.py
│               ├── list_view_actions.py
│               ├── save_load_actions.py
│               └── video_control_actions.py
│
├── visomaster-ui/             # ★ REACT WEB FRONTEND (TanStack Start + shadcn/ui)
│   └── src/
│       ├── api/client.ts      # typed fetch wrapper
│       ├── store/             # Zustand stores
│       ├── hooks/             # useEvents, usePreviewStream
│       ├── components/        # faces/, layout/, output/, parameters/, shared/, source/, ui/
│       ├── routes/            # TanStack Router file-based routes
│       └── transport/         # WebSocket channel abstraction
│
├── packages/
│   └── streamrelay/           # git submodule: WebStreamer WHIP/WebSocket server
│       └── src/streamrelay/
│           ├── server.py            # StreamServer + run_server entry point
│           ├── protocol.py          # shared-memory frame layout
│           ├── reader.py
│           └── client/              # vanilla JS browser camera client
│
├── docker/                    # Docker build files and compose configs
│   ├── Dockerfile
│   ├── Dockerfile.cuda118
│   ├── docker-compose.yml
│   └── README.md
│
├── model_assets/              # downloaded ONNX models live here
│   └── dfm_models/            # user-supplied DeepFaceLab DFM/ONNX files
├── tensorrt-engines/          # auto-built .trt engines + ORT EP cache
├── dependencies/              # bundled FFmpeg/etc. for portable Windows build
├── streamrelay-certs/         # self-signed cert/key for HTTPS WebRTC
├── tools/                     # misc dev tools
├── scripts/                   # install / setup scripts (Linux, RunPod, Windows)
├── docs/                      # architecture documentation (this folder)
└── .thumbnails/               # cached thumbnails for media + face cards
```

## Runtime modes

VisoMaster has three runtime modes that share the same inference pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│  MODE 1 — Qt Desktop (main.py)                                      │
│  PySide6 QApplication → MainWindow → VideoProcessor + ModelsProcessor│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  MODE 2 — Headless API + React (web_main.py / uvicorn)              │
│  FastAPI server → AppState → VideoProcessor + ModelsProcessor       │
│  React frontend (visomaster-ui) ← REST + WebSocket                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  MODE 3 — Qt WebEngine Bridge (web_main.py Qt mode)                 │
│  QWebEngineView loads React UI → JS↔Python bridge for native calls  │
└─────────────────────────────────────────────────────────────────────┘

All modes share:
  app/processors/*   — inference pipeline (GPU-heavy, UI-free)
  app/core/state.py  — AppState single source of truth
```

## Runtime layers (Qt desktop mode)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Qt event loop (PySide6)                                             │
│  ┌──────────────┐  signals/slots   ┌────────────────────────────┐   │
│  │  MainWindow  │ ◄─────────────► │  Action functions           │   │
│  │  + widgets   │                  │  (card_actions, etc.)       │   │
│  └──────┬───────┘                  └─────────┬──────────────────┘   │
│         │ owns                               │ mutate state on       │
│         ▼                                    ▼                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  MainWindow.parameters / .control / .markers                 │   │
│  │  .target_faces / .input_faces / .merged_embeddings           │   │
│  │  .video_processor / .models_processor                        │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │ drives                                                      │
│         ▼                                                            │
│  ┌──────────────┐  QTimer ticks   ┌────────────────────────────┐   │
│  │ VideoProcessor│ ─────────────► │ FrameWorker (Thread)        │   │
│  │ (QObject)    │  spawns          │ — runs the pipeline         │   │
│  └──────┬───────┘                  └─────────┬──────────────────┘   │
│         │ reads frames from                  │ uses                  │
│         ▼                                    ▼                       │
│  ┌──────────────┐                  ┌────────────────────────────┐   │
│  │ cv2.VideoCapture│              │ ModelsProcessor              │   │
│  │  OR shm (WebRTC)│              │ (ONNX/TensorRT/DFM)         │   │
│  └──────────────┘                  └────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘

           ┌───────────────────────────────────────┐
           │  separate process: streamrelay        │
           │  aiohttp + WebStreamer → /whip, /ws   │
           │  writes BGR frames into shared memory │
           │  (named "visomaster_webrtc_frame")    │
           └───────────────────────────────────────┘
```

## Key external dependencies

From `requirements_cu124.txt`:

- **PySide6** — Qt for Python
- **PyTorch 2.4 + CUDA 12.4 + Torchvision** — tensor ops, image transforms
- **onnxruntime-gpu 1.20** — primary inference engine
- **tensorrt 10.6** — accelerated path; engines cached under `tensorrt-engines/`
- **opencv-python** — capture, codecs, color conversion
- **aiortc + aiohttp** — WebRTC server (in streamrelay)
- **pyvirtualcam** — write to OBS / Unity virtual camera
- **kornia, scikit-image** — image processing
- **FastAPI 0.115 + uvicorn 0.32** — headless API server
- **qdarktheme, qdarkstyle, pyqt-toast-notification** — UI chrome

## What each top-level file does

| File | Role |
|---|---|
| `main.py` | Qt desktop entry point. Creates `QApplication`, sets stylesheet, builds `MainWindow`, runs event loop. |
| `web_main.py` | Qt WebEngine entry point. Starts FastAPI server, loads React UI in a `QWebEngineView`, bridges native capabilities via JS↔Python bridge. |
| `download_models.py` | Iterates `models_list` from `app/processors/models_data.py` and downloads each ONNX file with hash verification. |
| `app/ui/main_ui.py` | `MainWindow` class: holds runtime state, wires up signals, dispatches keyboard shortcuts. |
| `app/api/server.py` | FastAPI app factory with lifespan startup/shutdown, CORS, and route registration. |
| `app/core/state.py` | `AppState` dataclass: single source of truth shared by both Qt and API modes. |
| `app/processors/video_processor.py` | `VideoProcessor`: timers, frame queues, ffmpeg subprocess for recording, virtual cam, WebRTC shm reader. |
| `app/processors/models_processor.py` | `ModelsProcessor`: lazy-loads ONNX/TensorRT/DFM models, exposes a method per model family. |
| `app/processors/workers/frame_worker.py` | `FrameWorker(threading.Thread)`: detect → align → swap → restore → enhance for one frame. |
| `last_workspace.json` | Workspace snapshot: target media list, target/source/embedding cards, parameters, markers, control. |
