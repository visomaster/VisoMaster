# VisoMaster — Project Structure

## Top-Level Layout

```
VisoMaster/
├── main.py                    # Qt desktop app entry point
├── web_main.py                # Qt WebEngine + FastAPI entry point (React UI mode)
├── download_models.py         # CLI model downloader
├── uvicorn.ini                # uvicorn config for headless API server
├── last_workspace.json        # auto-saved session state (gitignored)
├── requirements_cu124.txt     # Python deps (CUDA 12.4)
├── requirements_cu118.txt     # Python deps (CUDA 11.8)
├── Start.bat / Start_Portable.bat  # Windows launchers
│
├── app/                       # Python application code
├── visomaster-ui/             # React/TypeScript web UI (TanStack Start + shadcn/ui)
├── packages/streamrelay/      # git submodule: WebStreamer WHIP/WebSocket server
├── docker/                    # Docker build files and compose configs
├── model_assets/              # downloaded ONNX models (not in git)
├── tensorrt-engines/          # auto-built TRT engines (not in git)
├── dependencies/              # bundled FFmpeg for portable Windows build
├── scripts/                   # install/setup scripts (Linux, RunPod, Windows)
├── docs/                      # architecture documentation
├── tools/                     # misc dev utilities
├── assets/                    # static assets (icons, images)
├── streamrelay-certs/         # self-signed certs for HTTPS WebRTC
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
│   ├── ws.py                  # WebSocket endpoints (/ws/events, /ws/preview, /ws/playback)
│   └── routes/
│       ├── system.py          # GET /api/system, provider switching, GPU memory
│       ├── schema.py          # GET /api/schema — parameter/control schema for UI
│       ├── state.py           # GET/PUT /api/state — control + per-face parameters
│       ├── workspace.py       # save/load/reset workspace
│       ├── target_media.py    # media list, folder scan, media selection
│       ├── faces.py           # target faces, input faces, face assignment
│       ├── embeddings.py      # merged embeddings CRUD
│       ├── playback.py        # play/stop/seek/record/markers/snapshot
│       ├── sources.py         # webcam list, WebRTC start/stop, transform
│       └── models.py          # model list, download status
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
    ├── web_main.py            # WebMainWindow: Qt + QWebEngineView + BackendBridge
    ├── bridge.py              # BackendBridge: QWebChannel @Slot/@Signal bridge to React
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
        ├── preview_window.py         # Native Qt preview window with playback controls
        ├── headless_preview.py       # Standalone preview window for headless API mode
        ├── preview_controls.html     # HTML playback controls embedded in PreviewWindow
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

## React Web Frontend (`visomaster-ui/`)

TanStack Start application with shadcn/ui components. Communicates with the FastAPI backend over REST and WebSocket.

```
visomaster-ui/
├── src/
│   ├── api/
│   │   └── client.ts          # Typed fetch wrapper — all API calls go through api.*
│   ├── store/                 # Zustand stores for all UI state
│   ├── hooks/
│   │   ├── useEvents.ts       # WebSocket /ws/events → store updates
│   │   └── usePreviewStream.ts # WebSocket /ws/preview → video frame display
│   ├── components/
│   │   ├── faces/             # FaceSwapPanel, FacePairRow, EmbeddingsSection, TargetFaceDialog
│   │   ├── layout/            # TopBar, shell layout
│   │   ├── output/            # OutputPanel (preview canvas + playback controls)
│   │   ├── parameters/        # FaceOptionsPanel, ParameterBlock (dynamic controls)
│   │   ├── shared/            # IconButton, ResourceBar, SectionHeader
│   │   ├── source/            # SourcePanel, MediaSource, WebcamSource, StreamingSource
│   │   └── ui/                # shadcn/ui base components (Button, Dialog, Slider, etc.)
│   ├── routes/
│   │   ├── __root.tsx         # Root layout with providers
│   │   └── index.tsx          # Main app route
│   ├── transport/
│   │   ├── channel.ts         # WebSocket channel abstraction
│   │   ├── http.ts            # Base HTTP helpers
│   │   ├── types.ts           # Shared transport types
│   │   └── index.ts
│   ├── lib/
│   │   └── utils.ts           # cn() helper (clsx + tailwind-merge)
│   ├── main.tsx               # React entry point
│   ├── router.tsx             # TanStack Router setup
│   ├── routeTree.gen.ts       # Auto-generated route tree
│   └── styles.css / theme.css # Global styles and CSS variables
├── vite.config.ts             # Vite config: React plugin, @ alias, dev proxy
├── tailwind.config.js
├── tsconfig.json
├── components.json            # shadcn/ui config
└── package.json
```

### Web UI Tech Stack

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

## API Documentation (`docs/api/`)

Full REST + WebSocket reference. 74 endpoints across 12 domain files:

```
docs/api/
├── README.md          # API reference index
├── 01-system.md       # GET/POST /api/system/* (provider, GPU memory)
├── 02-schema.md       # GET /api/schema/* (widget descriptor format, rendering guide)
├── 03-state.md        # GET/PUT /api/state/* (control, parameters, copy/paste/reset)
├── 04-workspace.md    # GET/POST /api/workspace/*
├── 05-target-media.md # CRUD + thumbnail /api/target-media/*
├── 06-target-faces.md # find, list, select, assign/unassign, clear
├── 07-input-faces.md  # scan-folder, list, delete, clear, thumbnail
├── 08-embeddings.md   # merge, export, import, delete, clear
├── 09-playback.md     # play, stop, seek, step, record, markers, snapshot
├── 10-sources.md      # webcam list/select, webrtc start/stop/status, transform
├── 11-websockets.md   # /ws/events (all server events + client commands), /ws/preview
└── 12-workflows.md    # 10 end-to-end call sequences
```

## Architecture Documentation (`docs/`)

```
docs/
├── README.md                    # Doc index and architecture summary
├── 01-overview.md               # Full repo layout, runtime modes, layer diagrams
├── 02-startup-and-bootstrap.md  # Entry sequences, model bootstrap, provider detection
├── 03-ui-layer.md               # Qt desktop UI + React web UI
├── 04-backend-pipeline.md       # VideoProcessor, FrameWorker, ModelsProcessor, threading
├── 05-models-and-inference.md   # ONNX registry, TensorRT, DFM, ArcFace/landmark mappings
├── 06-data-flows.md             # End-to-end flows: load, swap, record, webcam, WebRTC
├── 07-state-and-persistence.md  # parameters/control/markers schema, workspace JSON
├── 08-streamrelay-webrtc.md     # streamrelay package: WHIP, WebSocket, shared memory
├── 12-glossary.md               # Codebase terminology
├── ui-design.md                 # React UI design reference
└── api/                         # Full API reference (see above)
```

## Docker (`docker/`)

```
docker/
├── Dockerfile             # CUDA 12.4 image
├── Dockerfile.cuda118     # CUDA 11.8 image
├── docker-compose.yml     # Compose config (GPU passthrough, volume mounts)
├── README.md              # Docker usage guide
├── config/                # Runtime config files
├── desktop/               # Desktop/display config for headless GPU
└── scripts/               # Container entrypoint and setup scripts
```

## WebRTC Submodule (`packages/streamrelay/`)

A separate Python process (git submodule). Receives video via WHIP/WebSocket, writes raw BGR frames into a named shared memory segment (`visomaster_webrtc_frame`). The main process polls this shared memory to read frames.

```
packages/streamrelay/src/streamrelay/
├── server.py      # StreamServer + run_server entry point (aiohttp + WebStreamer)
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
- Use the `@/` path alias for all imports within `visomaster-ui/src/`.
- All backend communication goes through `api.*` in `visomaster-ui/src/api/client.ts` — no raw `fetch` in components.
- Global UI state lives in Zustand stores. Server-fetched data that needs caching uses TanStack Query.
- WebSocket events from `/ws/events` drive real-time store updates via `useEvents` hook.
- Tailwind utility classes are composed with `cn()` from `visomaster-ui/src/lib/utils.ts` (clsx + tailwind-merge).
- shadcn/ui components live in `visomaster-ui/src/components/ui/` — add new ones with `bunx shadcn add <component>`.

### File Naming
- Python: `snake_case` for files and functions, `PascalCase` for classes.
- TypeScript/React: `PascalCase` for component files and classes, `camelCase` for hooks and utilities.
- New API routes go in `app/api/routes/<domain>.py` and must be registered in `app/api/server.py`.
