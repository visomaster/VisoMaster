# VisoMaster — Project Structure

## Top-Level Layout

```
VisoMaster/
├── main.py                    # Qt desktop app entry point
├── download_models.py         # CLI model downloader
├── last_workspace.json        # auto-saved session state (gitignored)
├── requirements_cu124.txt     # Python deps (CUDA 12.4)
├── requirements_cu118.txt     # Python deps (CUDA 11.8)
├── Start.bat / Start_Portable.bat  # Windows launchers
│
├── app/                       # Python application code
├── frontend/                  # React/TypeScript UI
├── packages/streamrelay/      # git submodule: WebRTC WHIP server
├── model_assets/              # downloaded ONNX models (not in git)
├── tensorrt-engines/          # auto-built TRT engines (not in git)
├── dependencies/              # bundled FFmpeg for portable Windows build
├── scripts/                   # install/setup scripts (Linux, RunPod, Windows)
├── docs/                      # architecture documentation
├── tools/                     # misc dev utilities
└── .thumbnails/               # auto-generated media/face thumbnail cache
```

## Python App (`app/`)

```
app/
├── core/
│   └── state.py               # AppState dataclass — single source of truth for all session data
│
├── api/                       # FastAPI headless backend
│   ├── server.py              # app factory, lifespan (startup/shutdown), CORS
│   ├── schemas.py             # Pydantic request/response models
│   ├── deps.py                # FastAPI dependency injectors (get_app_state, get_video_processor)
│   ├── events.py              # EventBus: async broadcast to WebSocket subscribers
│   ├── ws.py                  # WebSocket endpoints (/ws/events, /ws/preview)
│   └── routes/
│       ├── system.py          # GET /api/system, provider switching, GPU memory
│       ├── schema.py          # GET /api/schema — parameter/control schema for UI
│       ├── state.py           # GET/PUT /api/state — control + per-face parameters
│       ├── workspace.py       # save/load/reset workspace
│       ├── target_media.py    # media list, folder scan, media selection
│       ├── faces.py           # target faces, input faces, face assignment
│       ├── embeddings.py      # merged embeddings CRUD
│       ├── playback.py        # play/stop/seek/record/markers/snapshot
│       └── sources.py         # webcam list, WebRTC start/stop, transform
│
├── processors/                # Inference pipeline (GPU-heavy, UI-free)
│   ├── models_processor.py    # Owns all model sessions; lazy-loads on first use
│   ├── video_processor.py     # Play loop, frame queue, ffmpeg recording, virtual cam
│   ├── face_detectors.py      # RetinaFace, SCRFD, YOLOv8, YuNet
│   ├── face_landmark_detectors.py  # 5/68/3d68/98/106/203/478-point landmark models
│   ├── face_swappers.py       # Inswapper, InStyleSwapper, SimSwap, GhostFace, CSCS
│   ├── face_restorers.py      # GFPGAN, CodeFormer, GPEN, VQFR, RestoreFormer
│   ├── face_masks.py          # Occluder, DFL XSeg, FaceParser, CLIPSeg, mouth/eye restore
│   ├── face_editors.py        # LivePortrait motion extraction + makeup
│   ├── frame_enhancers.py     # RealESRGAN, BSRGAN, DDColor, DeOldify
│   ├── models_data.py         # Model registry: name → local path + hash + download URL
│   ├── workers/
│   │   └── frame_worker.py    # threading.Thread: detect→align→swap→restore→enhance per frame
│   ├── utils/
│   │   ├── faceutil.py        # Face alignment, landmark math, affine transforms
│   │   ├── dfm_model.py       # DeepFaceLab DFM/ONNX file loader
│   │   ├── engine_builder.py  # ONNX → TensorRT engine builder
│   │   └── tensorrt_predictor.py  # TensorRT inference wrapper with thread pool
│   └── external/
│       ├── cliplib/           # Vendored OpenAI CLIP (text-driven masking)
│       ├── clipseg.py
│       └── resnet.py
│
├── helpers/
│   ├── miscellaneous.py       # File I/O, ParametersDict, output path helpers
│   ├── downloader.py          # Model file download with hash verification
│   ├── integrity_checker.py   # File hash checking
│   ├── recording.py           # Recording helpers
│   └── typing_helper.py       # Shared type aliases
│
└── ui/                        # PySide6 Qt desktop UI (legacy mode)
    ├── main_ui.py             # MainWindow: holds runtime state, wires signals
    ├── core/
    │   ├── MainWindow.ui      # Qt Designer XML layout
    │   ├── main_window.py     # Auto-generated from .ui
    │   └── proxy_style.py     # Custom Qt style proxy
    ├── styles/
    │   ├── dark_styles.qss    # Dark theme stylesheet
    │   └── light_styles.qss   # Light theme stylesheet
    └── widgets/
        ├── widget_components.py      # Reusable Qt widgets (CardButton, sliders, dialogs)
        ├── ui_workers.py             # QThreads for media loading, WebRTC server spawn
        ├── output_window.py          # Borderless OBS-friendly output window
        ├── event_filters.py          # Qt event filter helpers
        ├── common_layout_data.py     # Declarative parameter schema (shared controls)
        ├── swapper_layout_data.py    # Declarative parameter schema (face swap)
        ├── face_editor_layout_data.py # Declarative parameter schema (face editor)
        ├── settings_layout_data.py   # Declarative parameter schema (settings/control)
        └── actions/                  # Functions that mutate MainWindow state
            ├── card_actions.py       # Face/media card interactions
            ├── common_actions.py     # Shared UI actions
            ├── control_actions.py    # Control parameter mutations
            ├── filter_actions.py     # Face filter/search
            ├── graphics_view_actions.py  # Preview canvas interactions
            ├── layout_actions.py     # Panel layout management
            ├── list_view_actions.py  # List widget interactions
            ├── save_load_actions.py  # Workspace save/load
            └── video_control_actions.py  # Playback controls
```

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts          # Typed fetch wrapper — all API calls go through api.*
│   ├── store/
│   │   └── appStore.ts        # Single Zustand store for all UI state
│   ├── hooks/
│   │   ├── useEvents.ts       # WebSocket /ws/events → store updates
│   │   └── usePreviewStream.ts # WebSocket /ws/preview → video frame display
│   ├── components/
│   │   ├── layout/            # TopBar, shell layout
│   │   ├── source/            # SourcePanel, MediaSource, WebcamSource, StreamingSource
│   │   ├── faces/             # FaceSwapPanel, FacePairRow, EmbeddingsSection, TargetFaceDialog
│   │   ├── output/            # OutputPanel (preview canvas + playback controls)
│   │   ├── parameters/        # FaceOptionsPanel, ParameterBlock (dynamic controls)
│   │   └── shared/            # IconButton, ResourceBar, SectionHeader
│   ├── lib/
│   │   └── utils.ts           # clsx/tailwind-merge helpers
│   ├── App.tsx                # Root component, layout composition
│   └── main.tsx               # React entry point
├── vite.config.ts             # Vite config: React plugin, @ alias, dev proxy
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

## WebRTC Submodule (`packages/streamrelay/`)

A separate Python process (git submodule). Receives video via WHIP/WebSocket, writes raw BGR frames into a named shared memory segment (`visomaster_webrtc_frame`). The main process polls this shared memory to read frames.

```
packages/streamrelay/src/streamrelay/
├── server.py      # StreamServer + run_server entry point (aiohttp + aiortc)
├── protocol.py    # Shared memory layout constants (SHM_HEADER_BYTES, frame format)
├── reader.py      # Shared memory reader helper
└── client/        # Vanilla JS browser camera client (served over HTTP)
```

## Conventions

### Python
- All processor classes (`FaceDetectors`, `FaceSwappers`, etc.) are instantiated once inside `ModelsProcessor` and accessed via `self.face_detectors`, etc.
- Model loading is always done inside `with self.model_lock:` (reentrant lock).
- Frame data flows as **BGR numpy arrays** internally; conversion to RGB happens only at the FrameWorker boundary.
- `ParametersDict` (in `app/helpers/miscellaneous.py`) is a dict subclass that falls back to `default_parameters` for missing keys — always use it for per-face parameters.
- Layout data files (`*_layout_data.py`) are the authoritative schema for all parameters and controls. The `default` key in each widget config is the canonical default value.
- API route files contain only HTTP handler functions. Business logic belongs in processors or helpers.

### TypeScript / React
- Use the `@/` path alias for all imports within `frontend/src/`.
- All backend communication goes through `api.*` in `frontend/src/api/client.ts` — no raw `fetch` in components.
- Global UI state lives in the Zustand store (`useAppStore`). Server-fetched data that needs caching uses TanStack Query.
- WebSocket events from `/ws/events` drive real-time store updates via `useEvents` hook.
- Tailwind utility classes are composed with `cn()` from `frontend/src/lib/utils.ts` (clsx + tailwind-merge).

### File Naming
- Python: `snake_case` for files and functions, `PascalCase` for classes.
- TypeScript/React: `PascalCase` for component files and classes, `camelCase` for hooks and utilities.
- New API routes go in `app/api/routes/<domain>.py` and must be registered in `app/api/server.py`.
