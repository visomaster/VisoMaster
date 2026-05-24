# 09 · Coupling & Natural Seams

This is a candid map of where the UI is tangled with the backend, and where the code is already set up for a clean split. If you skip everything else, read this — it tells you the exact refactor work the new architecture demands.

## Where UI and backend are tangled

### 1. `MainWindow` *is* the application state

Every backend operation takes `main_window` and reads:

- `main_window.control` (global settings)
- `main_window.parameters[face_id]` (per-face swap settings)
- `main_window.target_faces`, `main_window.input_faces`, `main_window.merged_embeddings` (working set)
- `main_window.markers` (per-frame parameter overrides)

…and writes them as side effects. Examples:

- `card_actions.find_target_faces(main_window)` mutates `main_window.target_faces` directly.
- `FrameWorker` reads `main_window.swapfacesButton.isChecked()` and `main_window.faceMaskCheckBox.isChecked()` from inside a worker thread.
- `models_processor.load_model` calls `main_window.model_loading_signal.emit()`.

**Refactor needed.** Introduce a plain Python `AppState` class (or pydantic model) and pass that to actions instead of `main_window`. The Qt `MainWindow` becomes a *view* over `AppState`. The same `AppState` becomes the request/response payload for the new API.

### 2. The backend imports the UI

`video_processor.py` imports:

```python
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
```

…to call `update_graphics_view`, `update_gpu_memory_progressbar`, `reset_media_buttons`, `set_play_button_icon_to_play`, `disable_all_parameters_and_control_widget`, etc.

`frame_worker.py` imports:

```python
from app.ui.widgets.actions import common_actions
from app.ui.widgets.actions import video_control_actions
```

…to call `get_pixmap_from_frame` (depends on Qt) and `update_parameters_and_control_from_marker`.

`models_processor.py` calls `self.main_window.model_loading_signal.emit()`.

**Refactor needed.** Replace these direct calls with **events** the UI subscribes to:

```python
class FrameProcessor:
    on_frame_done   : Callable[[int, np.ndarray], None]
    on_progress     : Callable[[float], None]      # 0..1 for recording
    on_state_change : Callable[[str], None]        # "loading_model", "playing", "stopped", ...
```

Then the Qt layer (or the API layer) wires those callbacks. The processors stop knowing about Qt or pixmaps.

### 3. Pixmap conversion happens inside `FrameWorker`

`FrameWorker.run` calls `common_widget_actions.get_pixmap_from_frame(main_window, frame)` and emits a `(frame_number, QPixmap, np.ndarray)` tuple. This is convenient for Qt but useless for an HTTP API.

**Refactor needed.** `FrameWorker` should emit only `(frame_number, np.ndarray BGR)`. The Qt UI converts to QPixmap on receipt. The HTTP API encodes to JPEG/WebP. The WebRTC re-publisher hands the ndarray to aiortc.

### 4. Recording control is split across UI and backend

`record_video` (in `video_control_actions`) sets `video_processor.recording = True` then triggers a Play. `process_video` checks that flag and decides whether to spawn ffmpeg. `stop_processing` checks the flag again to do the audio mux step.

**Refactor needed.** Move recording ownership into `VideoProcessor` as a single method:

```python
video_processor.start_recording(output_path: str, codec='libx264', crf=18, mux_audio=True)
video_processor.stop_recording() -> output_path  # returns final muxed file
```

…and have the UI call those instead of toggling a boolean on a shared object.

### 5. Workspace persistence is intertwined with widget state

`save_load_actions.save_current_workspace` reads `target_face.cropped_face` (a numpy array stored on the widget) and `target_face.embedding_store`. `load_saved_workspace` clicks buttons in `main_window` to trigger downstream loads.

**Refactor needed.** Workspace becomes pure state I/O:

```python
state.to_json() -> dict
AppState.from_json(d: dict) -> AppState
```

Cards become render artifacts of the state, not the state itself.

## Where the code already has clean seams

### 1. `app/processors/face_*.py` and `app/processors/utils/*.py`

These family files already only touch `models_processor` and torch/numpy. They have **no Qt imports**. They can be lifted as-is into a service.

### 2. `ModelsProcessor` (mostly)

Aside from `main_window.model_loading_signal.emit()` (4 callsites) and `main_window.dfm_models_data` (1 read), `ModelsProcessor` is independent. Replace those with a callback like `on_model_loading_changed: Callable[[bool], None]` and it's clean.

### 3. `models_data.py`

Pure data. No coupling.

### 4. `app/helpers/*`

Pure helpers. `downloader`, `integrity_checker`, `miscellaneous` — all fine to reuse.

### 5. `streamrelay/`

Already a separate process with a defined SHM protocol. **Zero changes needed** to use it from a new UI.

### 6. `*_layout_data.py`

These are declarative dictionaries describing parameters. The same files can drive a React UI — either ship them as JSON, or expose them via `GET /api/schema/parameters`.

## Coupling summary

| Component | UI dependency | Effort to decouple |
|---|---|---|
| `face_detectors.py` and friends | None | None — already standalone |
| `models_data.py` | None | None |
| `ModelsProcessor` | 5 calls to MainWindow signals | Low — replace with callback |
| `DFMModel`, `TensorRTPredictor`, `EngineBuilder` | None | None |
| `FrameWorker` | reads buttons + parameters off MainWindow, calls `get_pixmap_from_frame` | **High** — central refactor target |
| `VideoProcessor` | imports 4 UI action modules; calls `update_graphics_view`, `set_play_button_icon_to_play`, etc. | **High** — but mechanical |
| `streamrelay/` | None | None |
| `helpers/` | None (mostly; `recording.py` is empty stub) | None |
| `widget_components.py` | All Qt | Replace entirely with React components |
| `widgets/actions/*.py` | Qt + MainWindow | Convert to API endpoints |
| `*_layout_data.py` | Imports a few action callbacks | **Low** — replace `exec_function: Callable` with `exec_function: str` (event name) |

## The refactor in one paragraph

The right first move is to introduce **`AppState`** as a normal Python dataclass tree, separate it from `MainWindow`, and rewrite `FrameWorker` to operate on `AppState` and emit raw ndarrays via callbacks. Once `FrameWorker` no longer touches Qt, `VideoProcessor` reduces to a thin orchestrator and can be moved into a Python service. The Qt UI keeps existing as one consumer; a new FastAPI server becomes a second consumer; the React app talks to that server.
