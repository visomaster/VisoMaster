# 01 · Overview

## Repo layout

```
VisoMaster/
├── main.py                    # entry point: builds QApplication + MainWindow
├── download_models.py         # CLI to fetch ONNX models from visomaster-assets
├── Start.bat / Start_Portable.bat   # Windows launchers
├── requirements_cu118.txt
├── requirements_cu124.txt
├── last_workspace.json        # auto-saved on close, auto-loaded on launch
│
├── app/
│   ├── helpers/               # downloader, integrity checks, file IO, types
│   ├── onnxmodels/            # placeholder dir (models live in model_assets/)
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
│   └── ui/                    # ★ THE QT FRONTEND
│       ├── main_ui.py               # MainWindow: holds ALL runtime state
│       ├── core/
│       │   ├── MainWindow.ui        # Qt Designer XML
│       │   ├── main_window.py       # generated from .ui
│       │   ├── proxy_style.py
│       │   └── media/               # icons
│       ├── styles/                  # qss stylesheets (dark/light)
│       ├── external/
│       │   └── certificates/        # self-signed cert/key for HTTPS WebRTC
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
├── packages/
│   └── streamrelay/           # git submodule: aiortc-based WHIP/WebRTC server
│       └── src/streamrelay/
│           ├── server.py            # StreamServer + run_server entry point
│           ├── protocol.py          # shared-memory frame layout
│           ├── reader.py
│           └── client/              # vanilla JS browser camera client
│
├── model_assets/              # downloaded ONNX models live here
│   └── dfm_models/            # user-supplied DeepFaceLab DFM/ONNX files
├── tensorrt-engines/          # auto-built .trt engines + ORT EP cache
├── dependencies/              # bundled FFmpeg/etc. for portable Windows build
├── streamrelay-certs/         # alt cert location for HTTPS WebRTC
├── tools/                     # misc dev tools
├── scripts/                   # install / setup scripts (Linux, RunPod, Windows)
└── .thumbnails/               # cached thumbnails for media + face cards
```

## Runtime layers

```
┌─────────────────────────────────────────────────────────────────┐
│  Qt event loop (PySide6)                                        │
│  ┌──────────────┐  signals/slots   ┌────────────────────────┐   │
│  │  MainWindow  │ ◄─────────────► │  Action functions      │   │
│  │  + widgets   │                  │  (card_actions, etc.)  │   │
│  └──────┬───────┘                  └─────────┬──────────────┘   │
│         │ owns                               │ mutate state on  │
│         ▼                                    ▼                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  MainWindow.parameters / .control / .markers         │       │
│  │  .target_faces / .input_faces / .merged_embeddings   │       │
│  │  .video_processor / .models_processor                │       │
│  └──────┬───────────────────────────────────────────────┘       │
│         │ drives                                                 │
│         ▼                                                        │
│  ┌──────────────┐  QTimer ticks   ┌────────────────────────┐   │
│  │ VideoProcessor │ ─────────────► │ FrameWorker (Thread)  │   │
│  │ (QObject)    │  spawns          │ — runs the pipeline   │   │
│  └──────┬───────┘                  └─────────┬──────────────┘   │
│         │                                    │                  │
│         │ reads frames from                  │ uses             │
│         ▼                                    ▼                  │
│  ┌──────────────┐                  ┌────────────────────────┐   │
│  │ cv2.VideoCapture │              │ ModelsProcessor        │   │
│  │  OR shm (WebRTC) │              │ (ONNX/TensorRT/DFM)    │   │
│  └──────────────┘                  └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

           ┌───────────────────────────────────────┐
           │  separate process: streamrelay        │
           │  aiohttp + aiortc → /whip, /ws/stream │
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
- **qdarktheme, qdarkstyle, pyqt-toast-notification** — UI chrome

## What each top-level file does

| File | Role |
|---|---|
| `main.py` | Entry point. Creates `QApplication`, sets stylesheet, builds `MainWindow`, runs event loop. |
| `download_models.py` | Iterates `models_list` from `app/processors/models_data.py` and downloads each ONNX file with hash verification. |
| `app/ui/main_ui.py` | `MainWindow` class: holds runtime state, wires up signals, dispatches keyboard shortcuts. |
| `app/processors/video_processor.py` | `VideoProcessor` (`QObject`): timers, frame queues, ffmpeg subprocess for recording, virtual cam, WebRTC shm reader. |
| `app/processors/models_processor.py` | `ModelsProcessor`: lazy-loads ONNX/TensorRT/DFM models, exposes a method per model family. |
| `app/processors/workers/frame_worker.py` | `FrameWorker(threading.Thread)`: detect → align → swap → restore → enhance for one frame. |
| `last_workspace.json` | Workspace snapshot: target media list, target/source/embedding cards, parameters, markers, control. |
