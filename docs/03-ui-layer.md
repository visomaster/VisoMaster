# 03 · UI Layer

VisoMaster has two UI layers that share the same backend pipeline:

1. **Qt desktop UI** (`app/ui/`) — PySide6 `QMainWindow`, parameter widgets generated from `*_layout_data.py` files.
2. **React web UI** (`visomaster-ui/`) — TanStack Start + shadcn/ui, communicates over REST and WebSocket.

---

## Qt Desktop UI

### MainWindow

`app/ui/main_ui.py::MainWindow` extends both `QMainWindow` and the generated `Ui_MainWindow`. It owns:

- Two backend objects: `video_processor`, `models_processor`.
- Six dicts that hold the user's working set (target media, target faces, source faces, embeddings, parameters, markers).
- A `control` dict for app-level settings (provider, threads, detector, restorers, virtual cam, output folder…).
- Lazy-spawned `QThread` workers (`TargetMediaLoaderWorker`, `InputFacesLoaderWorker`, `FilterWorker`).
- A `multiprocessing.Process` for the WebRTC server (`webrtc_server_process`).

### Custom signals on MainWindow

```python
placeholder_update_signal  = Signal(QListWidget, bool)   # show/hide list placeholder
gpu_memory_update_signal   = Signal(int, int)            # (used, total) MB
model_loading_signal       = Signal()                    # show modal "loading models"
model_loaded_signal        = Signal()                    # hide modal
display_messagebox_signal  = Signal(str, str, QWidget)   # (title, message, parent)
```

These let worker threads request UI updates from the GUI thread without blocking.

### Layout-data driven widgets

Parameter widgets are not declared in the `.ui` file. They are **generated at runtime** by `layout_actions.add_widgets_to_tab_layout()` reading one of:

- `COMMON_LAYOUT_DATA`   — restorers, expression restorer (per-face, applied for any swap).
- `SWAPPER_LAYOUT_DATA`  — swapper model, landmarks correction, color/blend, face mask sub-controls (per-face).
- `SETTINGS_LAYOUT_DATA` — global app `control` (provider, detector, threads, recording, WebRTC ports, virtual cam…).
- `FACE_EDITOR_LAYOUT_DATA` — LivePortrait expression sliders (per-face).

Each layout file is a `LayoutDictTypes` (typed in `app/helpers/typing_helper.py`):

```python
LayoutDictTypes = Dict[str, Dict[str, Dict[str, int|str|list|float|bool|Callable]]]
#                  ^section ^widget_name  ^attribute
```

A widget descriptor looks like:

```python
'FaceRestorerEnableToggle': {
    'level': 1,                    # indent depth in the panel
    'label': 'Enable Face Restorer',
    'default': False,
    'help': '...',
    'exec_function': control_actions.toggle_virtualcam,   # optional callback
    'exec_function_args': [],
    'parentToggle': 'SomeOtherToggle',                    # show only when parent is on
    'requiredToggleValue': True,
    'parentSelection': 'SwapModelSelection',              # show only for specific model
    'requiredSelectionValue': 'DeepFaceLive (DFM)',
    'options': ['Inswapper128', 'SimSwap512', ...],       # for SelectionBox widgets
    'min_value': 0, 'max_value': 100, 'step': 1,          # for sliders
    'decimals': 2,                                         # for ParameterDecimalSlider
}
```

The factory `add_widgets_to_tab_layout` instantiates one of these widget classes (in `widget_components.py`):

| Descriptor shape | Widget class |
|---|---|
| has `options` | `SelectionBox` (combo box) |
| has `min_value` + `max_value` (int) | `ParameterSlider` + `ParameterLineEdit` |
| has `min_value` + `max_value` + `decimals` | `ParameterDecimalSlider` + `ParameterLineDecimalEdit` |
| `default` is bool | `ToggleButton` |
| has `default` string only | `ParameterText` |

`data_type='parameter'` writes into `main_window.parameters[face_id][name]`. `data_type='control'` writes into `main_window.control[name]`.

Every widget gets:

- A `set_value(value)` method.
- A `reset_to_default_value()` method.
- A debounced change handler that calls `common_actions.update_parameter()` or `update_control()`.

### Action modules (`app/ui/widgets/actions/`)

This is the de-facto API surface — the place where UI events translate to "do something to the model state".

| Module | Responsibility |
|---|---|
| `common_actions.py` | Generic helpers: `create_control`, `update_parameter`, `refresh_frame`, `set_widgets_values_using_face_id_parameters`, GPU memory probe, message boxes, pixmap conversion. |
| `card_actions.py` | Target/source face card management: `find_target_faces`, `clear_target_faces`, `clear_input_faces`, `clear_merged_embeddings`. |
| `control_actions.py` | Settings tab callbacks: `change_execution_provider`, `change_threads_number`, `change_theme`, `set_video_playback_fps`, `toggle_virtualcam`, `toggle_output_window`. |
| `filter_actions.py` | Wires the search boxes for target media / input faces / embeddings (delegates to `FilterWorker`). |
| `graphics_view_actions.py` | Updates the central `QGraphicsView`, handles fit-to-view and image overlays. |
| `layout_actions.py` | Builds parameter panels from layout-data; sets up the menu bar; show/hide panels; enable/disable while recording. |
| `list_view_actions.py` | Adds thumbnail card buttons to list widgets; triggers `TargetMediaLoaderWorker` / `InputFacesLoaderWorker`. |
| `save_load_actions.py` | Workspace save/load, embedding JSON import/export, per-face parameters JSON. |
| `video_control_actions.py` | Play/record buttons, seek slider, frame markers, fullscreen, save current frame. |

A pattern repeats throughout:

```python
# in main_ui.py
self.swapfacesButton.clicked.connect(partial(video_control_actions.process_swap_faces, self))

# in video_control_actions.py
def process_swap_faces(main_window: 'MainWindow'):
    main_window.video_processor.process_current_frame()
```

This means **`MainWindow` is the single argument every action takes**.

### Card buttons

Cards are checkable `QPushButton`s with thumbnails. There are four flavors:

- **`TargetMediaCardButton`** — a video, image, webcam, or WebRTC source.
  - `load_media()` opens a `cv2.VideoCapture` (or attaches WebRTC `SharedMemory`), assigns it to `video_processor.media_capture`, and refreshes the first frame.
- **`TargetFaceCardButton`** — a face detected in the target media.
  - Holds an `embedding_store: Dict[recognition_model, np.ndarray]` plus `assigned_input_faces` and `assigned_merged_embeddings`.
  - `calculate_assigned_input_embedding()` averages/medians assigned source embeddings according to `EmbMergeMethodSelection`.
- **`InputFaceCardButton`** — a face image used as a swap source.
  - Selecting + assigning to a target face is the user's main interaction.
- **`EmbeddingCardButton`** — a saved/merged embedding (multiple input faces aggregated).

### Target medias model

There are three sources, exposed via the input-source `QTabWidget`:

- **Tab 0 — Media:** `targetVideosList` populated from a folder (`load_videos_and_images_from_folder`) or via drag-and-drop.
- **Tab 1 — Streaming:** a sub-tab widget with:
  - **Webcam:** `webcamList`, populated by enumerating `WebcamMaxNoSelection` indexes.
  - **WebRTC:** `webrtcList`, populated by spawning the StreamRelay subprocess and adding a placeholder card.

`MainWindow.on_input_source_tab_changed` and `on_streaming_sub_tab_changed` orchestrate the cleanup when the user switches modes (release webcam, kill WebRTC subprocess, clear scene, reset FPS label).

### Keyboard shortcuts

From `MainWindow.keyPressEvent`:

| Key | Action |
|---|---|
| Space | Toggle play/pause |
| R | Toggle record |
| S | Toggle "Swap Faces" |
| F | Add marker (Alt+F removes) |
| Q / W | Previous / next marker |
| A / D | Rewind / advance 30 frames |
| C / V | Rewind / advance 1 frame |
| Z | Seek to start |
| F11 | Fullscreen toggle |

### Output window

`app/ui/widgets/output_window.py` exposes `OutputWindow`: a borderless `QWidget` that displays the latest processed frame. It's intended for OBS "Window Capture" so users don't need a virtual camera. `VideoProcessor._send_frame_to_output_window` pushes frames into it whenever `OutputWindowEnableToggle` is on.

## Qt WebEngine UI (`app/ui/web_main.py` + `app/ui/bridge.py`)

`WebMainWindow` is the entry point for mode 2 (Qt + embedded React). It hosts a `QWebEngineView` that fills the entire window and loads the Vite dev server at `http://localhost:5173`.

### Architecture

```
WebMainWindow (QMainWindow)
├── QWebEngineView          — React UI (TopBar + panels)
│   └── QWebChannel         — JS ↔ Python bridge
│       └── BackendBridge   — @Slot methods + Signal emissions
├── QGraphicsView (hidden)  — stub for VideoProcessor compatibility
└── Hidden stub widgets     — list widgets, buttons, sliders
    (needed by existing action helpers that reference main_window.*)
```

### `BackendBridge` (`app/ui/bridge.py`)

Exposes Python `@Slot` methods to JavaScript and emits `Signal`s back to the React UI. All slots mirror the REST API 1-to-1 so the frontend transport adapter can call them identically.

**Signals (Python → JS):**

| Signal | Payload | Description |
|---|---|---|
| `playbackStateChanged` | JSON str | Full playback state snapshot |
| `framePositionChanged` | JSON str | High-frequency per-frame position |
| `gpuMemoryChanged` | JSON str | `{ used_mb, total_mb }` every 2 s |
| `stateUpdated` | JSON str | Control/parameter mutation echo |
| `fpsUpdated` | JSON str | `{ fps }` for streaming sources |
| `recordingFinished` | JSON str | `{ output_path }` |
| `modelLoading` / `modelLoaded` | — | Spinner control |
| `facesFound` | JSON str | After face detection completes |
| `workspaceLoaded` | JSON str | Frontend should re-pull state |
| `virtcamStateChanged` | JSON str | `{ enabled }` actual state |
| `errorOccurred` | JSON str | `{ message }` |
| `previewWindowOpened` / `previewWindowClosed` | — | Native preview window state |

**Key slots (JS → Python):**

| Slot | Signature | Description |
|---|---|---|
| `play()` | — | Start processing loop |
| `stop()` | — | Stop processing loop |
| `seek(frame)` | `int` | Seek to frame |
| `step(n)` | `int` | Step N frames |
| `getPlayback()` | → `str` | Current playback state JSON |
| `getState()` | → `str` | Full session state JSON |
| `setControl(name, value_json)` | `str, str` | Set one control value |
| `setParameter(face_id, name, value_json)` | `str, str, str` | Set one per-face parameter |
| `pickFolder()` | → `str` | Native folder picker |
| `pickFolderAt(initial_dir)` | `str` → `str` | Folder picker at specific path |
| `scanFolder(path, recursive)` | `str, bool` → `str` | Scan folder for media files |
| `selectMedia(media_id)` | `str` → `str` | Load and start playing a media item |
| `deleteMedia(media_id)` | `str` → `str` | Remove a media item |
| `getThumbnail(thumb_type, item_id)` | `str, str` → `str` | Base64 JPEG data URI |

Frame rendering is done off the main thread via `_process_frame_async()` (a `QThread` wrapper) so the UI stays responsive during GPU inference.

### Native preview window

`app/ui/widgets/preview_window.py::PreviewWindow` — a native Qt window that displays processed frames. Opened automatically when media is selected via `_open_preview_window()`. Supports playback controls (play/pause/seek/markers) via `preview_controls.html` embedded in a `QWebEngineView`.

`app/ui/widgets/headless_preview.py` — a singleton that manages a standalone `PreviewWindow` in headless API mode (no `QApplication` running). Opens a Qt event loop on a background thread when requested via the `open_preview_window` WebSocket command.

---

The React frontend is a TanStack Start application using shadcn/ui components and Tailwind CSS 4.

### Stack

| Concern | Library |
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

### Directory layout

```
visomaster-ui/src/
├── api/
│   └── client.ts          # typed fetch wrapper — all API calls go through api.*
├── store/                 # Zustand stores for UI state
├── hooks/
│   ├── useEvents.ts       # WebSocket /ws/events → store updates
│   └── usePreviewStream.ts # WebSocket /ws/preview → video frame display
├── components/
│   ├── faces/             # FaceSwapPanel, FacePairRow, EmbeddingsSection, TargetFaceDialog
│   ├── layout/            # TopBar, shell layout
│   ├── output/            # OutputPanel (preview canvas + playback controls)
│   ├── parameters/        # FaceOptionsPanel, ParameterBlock (dynamic controls)
│   ├── shared/            # IconButton, ResourceBar, SectionHeader
│   ├── source/            # SourcePanel, MediaSource, WebcamSource, StreamingSource
│   └── ui/                # shadcn/ui base components
├── routes/
│   ├── __root.tsx         # root layout
│   └── index.tsx          # main app route
├── transport/
│   ├── channel.ts         # WebSocket channel abstraction
│   ├── http.ts            # base HTTP helpers
│   ├── types.ts           # shared transport types
│   └── index.ts
├── lib/
│   └── utils.ts           # cn() helper (clsx + tailwind-merge)
├── main.tsx               # React entry point
├── router.tsx             # TanStack Router setup
└── styles.css / theme.css # global styles and CSS variables
```

### Key conventions

- All backend communication goes through `api.*` in `visomaster-ui/src/api/client.ts` — no raw `fetch` in components.
- Global UI state lives in Zustand stores. Server-fetched data that needs caching uses TanStack Query.
- WebSocket events from `/ws/events` drive real-time store updates via `useEvents` hook.
- Tailwind utility classes are composed with `cn()` from `visomaster-ui/src/lib/utils.ts`.
- The parameter UI is built dynamically from the `/api/schema` response, mirroring the Qt layout-data approach.
