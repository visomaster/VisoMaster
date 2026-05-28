# 02 · Startup & Bootstrap

## Entry sequences

### `Start.bat` — unified launcher (Windows)

`Start.bat` is the primary Windows entry point. It presents a menu and launches one of three modes:

```
  1. Qt Desktop   — native Qt UI (main.py)
  2. WebView      — Qt + embedded React UI (web_main.py)
  3. Web          — API server + browser UI (headless)
```

**Python detection order:**
1. Bundled portable Python at `dependencies/Python/python.exe` (calls `scripts/setenv.bat`)
2. Falls back to `conda activate visomaster`

**Common setup steps (all modes):**
- Loads `.env` file if present (sets environment variables)
- Adds `dependencies/` to `PATH` (FFmpeg etc.)
- Detects `bun` or `npm` for frontend commands

**Mode 2 (WebView) startup sequence:**
1. Starts Vite dev server hidden (`bun run dev` in `visomaster-ui/`), PID saved to `logs/vite.pid`
2. Waits 4 s for Vite to bind
3. Starts FastAPI server hidden (`python -m app.api.server`), PID saved to `logs/api.pid`
4. Waits 2 s for API to bind
5. Launches `python web_main.py` (foreground)
6. On exit: kills both background servers via PowerShell `Stop-Process`

**Mode 3 (Web) startup sequence:**
1. Starts FastAPI server hidden, PID saved to `logs/api.pid`
2. Waits 3 s
3. Starts Vite dev server hidden, PID saved to `logs/vite.pid`
4. Prints URLs and waits for keypress
5. On exit: kills both background servers

Logs are written to `logs/vite.log`, `logs/vite.err.log`, `logs/api.log`, `logs/api.err.log`.

---

### Qt desktop mode (`main.py`)

1. **`Start.bat` → mode 1** or `python main.py` directly:
   - Activates the `visomaster` conda env (or uses bundled Python).
   - Adds bundled `dependencies/` (e.g. ffmpeg) to `PATH`.
   - Calls `python main.py`.

2. **`main.py`**:
   ```python
   _streamrelay_src = Path(__file__).parent / "packages" / "streamrelay" / "src"
   if _streamrelay_src.is_dir() and str(_streamrelay_src) not in sys.path:
       sys.path.insert(0, str(_streamrelay_src))   # bootstrap submodule

   app = QtWidgets.QApplication(sys.argv)
   signal.signal(signal.SIGINT, lambda *a: app.closeAllWindows())   # Ctrl+C handling
   _signal_timer = QtCore.QTimer(); _signal_timer.start(1000)        # let Python signals fire

   app.setStyle(ProxyStyle())
   _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"})
   _style += open("app/ui/styles/dark_styles.qss").read()
   app.setStyleSheet(_style)

   window = main_ui.MainWindow()
   window.show()
   sys.exit(app.exec())
   ```

3. **`MainWindow.__init__`** (in `app/ui/main_ui.py`):
   ```python
   self.setupUi(self)              # generated from MainWindow.ui
   self.initialize_variables()     # creates self.video_processor, self.models_processor,
                                   # plus all dicts: target_faces, input_faces, parameters, control...
   self.initialize_widgets()       # wires every signal/slot, builds parameter widgets
                                   # from COMMON/SWAPPER/SETTINGS/FACE_EDITOR _LAYOUT_DATA
   self.load_last_workspace()      # if last_workspace.json exists, prompt to load
   ```

4. **Closing** (`closeEvent`):
   - Calls `video_processor.stop_processing()`.
   - Stops media loader workers.
   - Terminates the WebRTC subprocess (if running).
   - Closes the borderless output window.
   - **Auto-saves `last_workspace.json`** via `save_load_actions.save_current_workspace`.

### Qt WebEngine mode (`web_main.py`)

Entry point for the React-in-Qt mode. Launched by `Start.bat` mode 2, or directly:

```
python web_main.py [--skip-workspace | --auto-last-workspace | --workspace <path>]
```

Flags (mutually exclusive):
- `--skip-workspace` — start with an empty session, no dialog
- `--auto-last-workspace` — silently load `last_workspace.json`
- `--workspace <path>` — load a specific workspace JSON

Startup sequence:
1. Bootstraps streamrelay path (`packages/streamrelay/src`).
2. Creates `QApplication`, applies dark theme + `dark_styles.qss`.
3. Instantiates `WebMainWindow` (`app/ui/web_main.py`):
   - Builds hidden stub Qt widgets (list widgets, buttons, sliders) for compatibility with existing action helpers.
   - Initialises `ModelsProcessor` and `VideoProcessor`.
   - Creates `BackendBridge` (`app/ui/bridge.py`) and registers it on a `QWebChannel`.
   - Injects `qwebchannel.js` into the page.
   - Loads `http://localhost:5173` (Vite dev server) in `QWebEngineView`.
   - Loads workspace per the CLI flags.

The `BackendBridge` exposes `@Slot` methods to JavaScript that mirror the REST API 1-to-1 (play, stop, seek, setControl, setParameter, scanFolder, selectMedia, getThumbnail, etc.) and emits `Signal`s back to the React UI (playbackStateChanged, framePositionChanged, gpuMemoryChanged, stateUpdated, fpsUpdated, recordingFinished, workspaceLoaded, virtcamStateChanged, errorOccurred, previewWindowOpened/Closed).

### Headless API + React mode (mode 3 / uvicorn)

1. **`Start.bat` → mode 3**, `python -m app.api.server`, or `uvicorn app.api.server:app`:
   - Starts the FastAPI application defined in `app/api/server.py`.
   - The lifespan handler creates a shared `AppState` and headless `VideoProcessor` instance.
   - Routes are registered from `app/api/routes/`.

2. **React frontend** (`visomaster-ui/`):
   - Run via `bun run dev` for hot reload (Vite dev server on port 5173).
   - Vite dev proxy forwards `/api` and `/ws` to `http://localhost:8000`.
   - On load, fetches `/api/schema` to build the parameter UI dynamically.
   - Connects to `/ws/events` for real-time state updates, `/ws/preview` for the JPEG frame stream, and `/ws/playback` for high-frequency position updates.

---

## What `initialize_variables` constructs (Qt mode)

```python
self.video_processor   = VideoProcessor(self)        # owns frame queue + threads
self.models_processor  = ModelsProcessor(self)       # owns ONNX/TRT/DFM model sessions

# media loader QThreads (re-spawned on demand)
self.video_loader_worker        = False
self.input_faces_loader_worker  = False
self.target_videos_filter_worker = FilterWorker(...)
self.input_faces_filter_worker  = FilterWorker(...)
self.merged_embeddings_filter_worker = FilterWorker(...)
self.webrtc_server_process      = None  # multiprocessing.Process

# UI card dictionaries (id → button widget)
self.target_videos      : Dict[int, TargetMediaCardButton]
self.target_faces       : Dict[int, TargetFaceCardButton]
self.input_faces        : Dict[int, InputFaceCardButton]
self.merged_embeddings  : Dict[int, EmbeddingCardButton]

# parameter & control state
self.parameters             : {face_id: ParametersDict}    # per-face swap settings
self.default_parameters     : ParametersDict               # built from layout_data files
self.copied_parameters      : ParametersDict               # for "copy parameters" UX
self.current_widget_parameters: ParametersDict
self.control                : {control_name: value}        # global app settings
self.parameter_widgets      : {name: widget}               # widget bindings
self.markers                : {frame_number: {parameters, control}}

# live source state
self.webcam_rotation, self.webcam_flip_h, self.webcam_flip_v
self.webrtc_rotation, self.webrtc_flip_h, self.webrtc_flip_v
self._output_window  # borderless capture window, lazy
```

---

## Model bootstrap

Models are **not loaded at startup**. The `ModelsProcessor.__init__` only:

1. Detects whether `onnxruntime` has the CUDA EP (falls back to CPU if not).
2. Detects whether `torch.cuda.is_available()` (falls back to CPU if not).
3. Builds an empty registry:
   ```python
   self.models[model_name] = None
   self.models_path[model_name] = './model_assets/inswapper_128.fp16.onnx'  # for example
   ```
4. Stores TensorRT engine paths under `tensorrt-engines/`.
5. Constructs the per-family helpers (`FaceDetectors`, `FaceSwappers`, …), each of which holds a back-reference to the `ModelsProcessor`.

**Lazy loading.** A model is loaded the first time something calls `ModelsProcessor.load_model('Inswapper128')` (or the TRT/DFM equivalent). In Qt mode the first call shows a "Loading models, please wait" modal via `model_loading_signal` / `model_loaded_signal`.

---

## Provider detection (the warning paths)

```
device='cuda' requested
  ↓
'CUDAExecutionProvider' in onnxruntime.get_available_providers() ?
  ├── no  → print install hint, fall back to device='cpu'
  └── yes → torch.cuda.is_available() ?
            ├── no  → fall back to device='cpu'
            └── yes → keep device='cuda'
```

Then `provider_name` is `'TensorRT'` if `device == 'cuda'` else `'CPU'`.

---

## Where models come from

- `download_models.py` reads `models_list` from `app/processors/models_data.py` and downloads each entry's URL into `model_assets/` (or sub-dirs like `model_assets/liveportrait_onnx/`).
- TensorRT engines are **built on first use** via `engine_builder.onnx_to_trt`. The path includes the trt version: `motion_extractor.10.6.0.trt`. Already-built engines in `tensorrt-engines/` are reused.
- DFM files are user-supplied. `app/helpers/miscellaneous.py::get_dfm_models_data` scans `model_assets/dfm_models/` for `.dfm` and `.onnx` files and registers them.
