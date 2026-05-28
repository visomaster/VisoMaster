# 07 · State & Persistence

## `AppState` — single source of truth

`app/core/state.py::AppState` is a Python dataclass that holds all session data. Both the Qt `MainWindow`/`WebMainWindow` and the FastAPI server hold a reference to the **same** instance. No other shared state channel exists.

```python
@dataclass
class AppState:
    # ── Global settings ────────────────────────────────────────────────
    control: Dict[str, Any]                    # mirrors main_window.control
    default_parameters: Dict[str, Any]         # built from layout-data 'default' keys
    current_widget_parameters: Dict[str, Any]  # what the panel currently shows

    # ── Per-face parameters ────────────────────────────────────────────
    parameters: Dict[str, Any]                 # face_id → ParametersDict

    # ── Working set ────────────────────────────────────────────────────
    target_media:  Dict[str, MediaRef]
    target_faces:  Dict[str, TargetFace]
    input_faces:   Dict[str, InputFace]
    embeddings:    Dict[str, MergedEmbedding]

    # ── Selection ──────────────────────────────────────────────────────
    selected_media_id: Optional[str]
    selected_face_id:  Optional[str]

    # ── Markers ────────────────────────────────────────────────────────
    markers: Dict[int, Marker]                 # frame_number → Marker

    # ── Streaming transforms ───────────────────────────────────────────
    webcam_transform: StreamTransform          # rotation, flip_h, flip_v
    webrtc_transform: StreamTransform
    media_transform:  StreamTransform

    # ── Folder memory ──────────────────────────────────────────────────
    last_target_media_folder: str
    last_input_media_folder:  str
    loaded_embedding_filename: str
    output_media_folder: str

    # ── Playback options ───────────────────────────────────────────────
    loop_enabled: bool
```

### Dataclass types

| Class | Fields | Notes |
|---|---|---|
| `EmbeddingStore` | `store: Dict[str, np.ndarray]` | Per-recognition-model 512-dim ArcFace vectors |
| `TargetFace` | `face_id`, `cropped_face`, `embedding_store`, `assigned_input_face_ids`, `assigned_embedding_ids`, `assigned_input_embedding` | One detected face in the target media |
| `InputFace` | `face_id`, `media_path`, `embedding_store`, `cropped_face` | One source face image |
| `MergedEmbedding` | `embedding_id`, `name`, `embedding_store` | Saved/merged embedding from multiple input faces |
| `MediaRef` | `media_id`, `media_path`, `file_type` | `file_type` ∈ `video\|image\|webcam\|webrtc` |
| `Marker` | `frame_number`, `parameters`, `control` | Per-frame parameter override |
| `StreamTransform` | `rotation` (0/90/180/270), `flip_h`, `flip_v` | Applied to webcam/webrtc/media frames |

### Key methods

```python
state.get_parameters(face_id)          # → ParametersDict (creates if absent)
state.set_parameter(face_id, name, value)
state.set_control(name, value)         # also syncs loop_enabled field
state.new_face_id()                    # → str (uuid1-based)
state.to_json()                        # → dict for last_workspace.json
AppState.from_json(d, default_parameters)  # reconstruct from workspace dict
```

---

## Qt mode runtime state on `MainWindow` / `WebMainWindow`

In Qt modes the window object holds **card widget dictionaries** that mirror the `AppState` working set. These are kept in sync with `AppState` by the action helpers.

```python
# Card collections (id → button widget)
target_videos        : Dict[media_id, TargetMediaCardButton]
target_faces         : Dict[face_id,  TargetFaceCardButton]
input_faces          : Dict[face_id,  InputFaceCardButton]
merged_embeddings    : Dict[embedding_id, EmbeddingCardButton]
selected_video_button       : TargetMediaCardButton | False
cur_selected_target_face_button : TargetFaceCardButton | False
selected_target_face_id     : str | False

# Parameters (per-face) — mirrors AppState.parameters
parameters           : Dict[face_id, ParametersDict]
default_parameters   : ParametersDict
copied_parameters    : ParametersDict
current_widget_parameters : ParametersDict

# Control (global) — mirrors AppState.control
control              : Dict[str, bool|int|float|str]

# Markers — mirrors AppState.markers
markers              : Dict[frame_number, {'parameters': ..., 'control': ...}]

# Streaming transforms (per source kind)
webcam_rotation, webcam_flip_h, webcam_flip_v
webrtc_rotation, webrtc_flip_h, webrtc_flip_v
media_rotation, media_flip_h, media_flip_v

# Misc
loaded_embedding_filename       : str
last_target_media_folder_path   : str
last_input_media_folder_path    : str
loading_new_media               : bool
is_full_screen                  : bool
dfm_models_data                 : Dict[name, path]

# Backend
video_processor      : VideoProcessor
models_processor     : ModelsProcessor
webrtc_server_process: multiprocessing.Process | None
_output_window       : OutputWindow | None
_preview_window      : PreviewWindow | None   # WebMainWindow only
```

---

## `ParametersDict` semantics

`app/helpers/miscellaneous.py::ParametersDict` is a `UserDict` subclass that **falls back to the default** when a key is missing:

```python
class ParametersDict(UserDict):
    def __init__(self, parameters, default_parameters):
        super().__init__(parameters)
        self._default_parameters = default_parameters

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            self.__setitem__(key, self._default_parameters[key])
            return self._default_parameters[key]
```

Why: when a workspace JSON saved with version N is loaded on version N+1 that introduced new parameters, accessing those new keys returns the defaults instead of crashing. New entries are also persisted on read.

---

## Default values

`default_parameters` is filled at startup by collecting the `'default'` key from every widget descriptor in the layout-data files:

```python
# In server.py lifespan and WebMainWindow._init_processors():
_collect_defaults(COMMON_LAYOUT_DATA,      default_parameters)
_collect_defaults(SWAPPER_LAYOUT_DATA,     default_parameters)
_collect_defaults(FACE_EDITOR_LAYOUT_DATA, default_parameters)
_collect_defaults(SETTINGS_LAYOUT_DATA,    default_control)
```

String defaults that look like numbers are coerced to `int`/`float` so numeric comparisons work correctly.

---

## Per-face parameters

When a target face card is added:

```python
state.get_parameters(face_id)
# → ParametersDict(copy.deepcopy(default_parameters), default_parameters)
```

When the user edits a slider, `update_parameter` writes to whichever face is currently selected:

```python
update_parameter(main_window, name, value):
    if main_window.selected_target_face_id:
        main_window.parameters[main_window.selected_target_face_id][name] = value
    else:
        main_window.current_widget_parameters[name] = value
    if exec_function: exec_function(main_window, value, *args)
    refresh_frame(main_window)
```

---

## Workspace JSON schema (`last_workspace.json`)

Written by `save_load_actions.save_current_workspace` (Qt modes) or `AppState.to_json()` (API mode):

```jsonc
{
  "selected_media_id": "247013112649665177041422535211796140790",   // or null
  "target_medias_data": [
    { "media_id": "...", "media_path": "C:/.../Woman1.mp4", "file_type": "video" },
    ...
  ],

  "target_faces_data": {
    "<face_id>": {
      "face_id": "...",
      "cropped_face": [[[r,g,b], ...], ...],   // ndarray as nested list (uint8 BGR)
      "embedding_store": {                      // recognition_model → 512-vec
        "Inswapper128ArcFace": [0.123, ...],
        "SimSwapArcFace":      [...],
        "GhostArcFace":        [...],
        "CSCSArcFace":         [...]
      },
      "assigned_input_face_ids":       ["<face_id>", ...],
      "assigned_embedding_ids":        ["<embedding_id>", ...],
      "assigned_input_embedding":      { "recognition_model": [floats] },
      "parameters":  { "<param>": value, ... },
      "control":     { "<control>": value, ... }
    }
  },

  "input_faces_data": {
    "<face_id>": {
      "face_id": "...",
      "media_path": "C:/.../source.png",
      "embedding_store": { ... }
    }
  },

  "embeddings_data": {
    "<embedding_id>": {
      "embedding_id": "...",
      "name": "Alice merged",
      "embedding_store": { "recognition_model": [floats] }
    }
  },

  "markers": {
    "<frame_number>": {
      "frame_number": N,
      "parameters": { "<face_id>": { "<param>": value } },
      "control":    { ... }
    }
  },

  "control": { "<control>": value, ... },
  "current_widget_parameters": { ... },
  "last_target_media_folder_path": "...",
  "last_input_media_folder_path":  "...",
  "loaded_embedding_filename":     "...",
  "webcam_transform": { "rotation": 0, "flip_h": false, "flip_v": false },
  "webrtc_transform": { "rotation": 0, "flip_h": false, "flip_v": false }
}
```

---

## Embeddings JSON (separate, exportable)

```jsonc
[
  {
    "name": "Alice (10 faces averaged)",
    "embedding_store": {
      "Inswapper128ArcFace": [0.0123, -0.4567, ...],
      "SimSwapArcFace":      [...],
      "GhostArcFace":        [...],
      "CSCSArcFace":         [...]
    }
  },
  ...
]
```

---

## Per-face parameters JSON (export from a face)

```jsonc
{
  "parameters": { "<param>": value, ... },
  "control":    { "<control>": value, ... }
}
```

---

## Control reference (selected examples from `SETTINGS_LAYOUT_DATA`)

| Control key | Type | Default | Notes |
|---|---|---|---|
| `ThemeSelection` | enum | `Dark` | Dark/Dark-Blue/Light |
| `ProvidersPrioritySelection` | enum | `CUDA` | CUDA/TensorRT/TensorRT-Engine/CPU |
| `nThreadsSlider` | int | `2` | Worker threads (and frame_queue size) |
| `VideoPlaybackCustomFpsToggle` + `VideoPlaybackCustomFpsSlider` | bool + int | off, 30 | Override video playback FPS |
| `AutoSwapToggle` | bool | `false` | Auto-find faces on media load |
| `DetectorModelSelection` | enum | `RetinaFace` | RetinaFace/Yolov8/SCRFD/Yunet |
| `DetectorScoreSlider` | int | `50` | %-confidence threshold |
| `MaxFacesToDetectSlider` | int | `20` | |
| `LandmarkDetectToggle` | bool | `false` | + `LandmarkDetectModelSelection` (5/68/3d68/98/106/203/478) |
| `RecognitionModelSelection` | enum | `Inswapper128ArcFace` | |
| `SimilarityTypeSelection` | enum | `Opal` | Opal/Pearl/Optimal |
| `MaxDFMModelsSlider` | int | `1` | LRU cap for DFM sessions |
| `FrameEnhancerEnableToggle` + `FrameEnhancerTypeSelection` | bool + enum | off, RealEsrgan-x2-Plus | |
| `WebcamMaxNoSelection` | enum | `1` | How many webcams to enumerate |
| `WebcamBackendSelection` | enum | `Default` | Default/DirectShow/MSMF/V4L/V4L2/GSTREAMER |
| `WebcamMaxResSelection` | enum | `1280x720` | |
| `SendVirtCamFramesEnableToggle` + `VirtCamBackendSelection` | bool + enum | off, obs | |
| `OutputWindowEnableToggle` | bool | `false` | borderless OBS-friendly window |
| `OutputMediaFolder` | str | `''` | Where recordings/snapshots land |
| `WebRTCHttpPortText` | str | `9091` | |
| `WebRTCHttpsPortText` | str | `9090` | |
| `WebRTCBindAddressText` | str | `0.0.0.0` | |
| `loop_enabled` | bool | `false` | Loop video playback (also synced to `AppState.loop_enabled`) |

---

## Key per-face parameters (selected from `SWAPPER_LAYOUT_DATA` and `COMMON_LAYOUT_DATA`)

| Parameter | Type | Notes |
|---|---|---|
| `SwapModelSelection` | enum | Inswapper128 / InStyleSwapper256 v{A,B,C} / DeepFaceLive (DFM) / SimSwap512 / GhostFace-v{1,2,3} / CSCS |
| `SwapperResSelection` | enum | 128/256/384/512 (depends on swapper) |
| `DFMModelSelection` | dynamic | Populated from `model_assets/dfm_models/*.dfm,*.onnx` |
| `DFMAmpMorphSlider` | int 1..100 | For AMP DFM models |
| `DFMRCTColorToggle` | bool | Reinhard color transfer |
| `SimilarityThresholdSlider` | int | Match threshold for this target face |
| `FaceAdjEnableToggle` + `KpsXSlider`/`KpsYSlider`/`KpsScaleSlider`/`FaceScaleAmountSlider` | bool + int sliders | Global keypoint nudge |
| `LandmarksPositionAdjEnableToggle` + `EyeLeft{X,Y}` ... `MouthRight{X,Y}` | bool + int sliders | Per-keypoint nudge |
| `FaceRestorerEnableToggle` + chain selectors / fidelity / blend | bool + ... | First restorer pass |
| `FaceRestorerEnable2Toggle` + ... | bool + ... | Second restorer pass |
| `FaceExpressionEnableToggle` + LivePortrait params | bool + ... | Expression restorer |
