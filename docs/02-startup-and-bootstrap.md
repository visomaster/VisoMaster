# 02 · Startup & Bootstrap

## Entry sequence

1. **`Start.bat`** (Windows) or `python main.py` (Linux):
   - Activates the `visomaster` conda env.
   - Runs `app/ui/core/convert_ui_to_py.bat` to regenerate `main_window.py` from `MainWindow.ui` (only if Qt Designer is around).
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

## What `initialize_variables` constructs

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
self.default_parameters     : ParametersDict              # built from layout_data files
self.copied_parameters      : ParametersDict              # for "copy parameters" UX
self.current_widget_parameters: ParametersDict
self.control                : {control_name: value}       # global app settings
self.parameter_widgets      : {name: widget}              # widget bindings
self.markers                : {frame_number: {parameters, control}}

# live source state
self.webcam_rotation, self.webcam_flip_h, self.webcam_flip_v
self.webrtc_rotation, self.webrtc_flip_h, self.webrtc_flip_v
self._output_window  # borderless capture window, lazy
```

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

**Lazy loading.** A model is loaded the first time something calls `ModelsProcessor.load_model('Inswapper128')` (or the TRT/DFM equivalent). The first call shows a "Loading models, please wait" modal via `model_loading_signal` / `model_loaded_signal`.

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

## Where models come from

- `download_models.py` reads `models_list` from `app/processors/models_data.py` and downloads each entry's URL into `model_assets/` (or sub-dirs like `model_assets/liveportrait_onnx/`).
- TensorRT engines are **built on first use** via `engine_builder.onnx_to_trt`. The path includes the trt version: `motion_extractor.10.6.0.trt`. Already-built engines in `tensorrt-engines/` are reused.
- DFM files are user-supplied. `app/helpers/miscellaneous.py::get_dfm_models_data` scans `model_assets/dfm_models/` for `.dfm` and `.onnx` files and registers them.
