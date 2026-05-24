# 07 · State & Persistence

All "session state" lives on the `MainWindow` instance. There is **no** dedicated state store class — the window object is the store.

## Runtime state on `MainWindow`

```python
# Card collections (id → button widget)
target_videos        : Dict[media_id, TargetMediaCardButton]
target_faces         : Dict[face_id,  TargetFaceCardButton]
input_faces          : Dict[face_id,  InputFaceCardButton]
merged_embeddings    : Dict[embedding_id, EmbeddingCardButton]
selected_video_button       : TargetMediaCardButton | False
cur_selected_target_face_button : TargetFaceCardButton | False
selected_target_face_id     : str | False

# Parameters (per-face)
parameters           : Dict[face_id, ParametersDict]   # current values
default_parameters   : ParametersDict                  # built from layout-data 'default'
copied_parameters    : ParametersDict                  # for copy/paste UX
current_widget_parameters : ParametersDict             # what the panel currently shows

# Control (global)
control              : Dict[str, bool|int|float|str]   # provider, threads, output folder, …

# Markers (per-frame parameter overrides)
markers              : Dict[frame_number, {'parameters': ..., 'control': ...}]

# Streaming transforms (per source kind)
webcam_rotation, webcam_flip_h, webcam_flip_v
webrtc_rotation, webrtc_flip_h, webrtc_flip_v

# Misc
loaded_embedding_filename       : str
last_target_media_folder_path   : str
last_input_media_folder_path    : str
loading_new_media               : bool
is_full_screen                  : bool
dfm_models_data                 : Dict[name, path]   # via misc_helpers.DFM_MODELS_DATA

# Backend
video_processor      : VideoProcessor
models_processor     : ModelsProcessor
webrtc_server_process: multiprocessing.Process | None
_output_window       : OutputWindow | None
```

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

## Default values

`default_parameters` is filled by `add_widgets_to_tab_layout`:

```python
default = layout_data[section][widget_name]['default']
common_actions.create_default_parameter(main_window, widget_name, default)
```

Which appends to `main_window.default_parameters[widget_name] = default`. Same for `control` via `create_control`.

## Per-face parameters

When a target face card is added (`list_view_actions.add_media_thumbnail_to_target_faces_list`):

```python
common_actions.create_parameter_dict_for_face_id(main_window, face_id)
  → main_window.parameters[face_id] = ParametersDict(default_parameters.copy(), default_parameters)
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

## Workspace JSON schema (`last_workspace.json`)

Written by `save_load_actions.save_current_workspace`:

```jsonc
{
  "selected_media_id": "247013112649665177041422535211796140790",   // or false
  "target_medias_data": [
    { "media_id": "...", "media_path": "C:/.../Woman1.mp4" },
    ...
  ],

  "target_faces_data": {
    "<face_id>": {
      "cropped_face": [[[r,g,b], ...], ...],   // ndarray as nested list (uint8 RGB)
      "embedding_store": {                      // recognition_model → 512-vec
        "Inswapper128ArcFace": [0.123, ...],
        "SimSwapArcFace":      [...],
        "GhostArcFace":        [...],
        "CSCSArcFace":         [...]
      },
      "parameters":  { "<param>": value, ... }, // dict (NOT ParametersDict)
      "control":     { "<control>": value, ... },
      "assigned_input_faces":       ["<face_id>", ...],
      "assigned_merged_embeddings": ["<embedding_id>", ...],
      "assigned_input_embedding":   { recognition_model: [floats] }
    }
  },

  "input_faces_data": {
    "<face_id>": { "media_path": "C:/.../source.png" }
  },

  "embeddings_data": {
    "<embedding_id>": {
      "embedding_name":  "Alice merged",
      "embedding_store": { recognition_model: [floats] }
    }
  },

  "markers": {
    "<frame_number>": {
      "parameters": { "<face_id>": { "<param>": value } },
      "control":    { ... }
    }
  },

  "control": { "<control>": value, ... },     // current global control snapshot
  "current_widget_parameters": { ... },        // for when no face is selected
  "last_target_media_folder_path": "...",
  "last_input_media_folder_path":  "...",
  "loaded_embedding_filename":     "..."
}
```

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

## Per-face parameters JSON (export from a face)

```jsonc
{
  "parameters": { "<param>": value, ... },
  "control":    { "<control>": value, ... }
}
```

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
