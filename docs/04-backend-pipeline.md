# 04 · Backend Pipeline

The processing core lives in three classes:

| Class | File | Lifetime |
|---|---|---|
| `VideoProcessor` | `app/processors/video_processor.py` | One per `MainWindow`. Owns the frame queue, timers, ffmpeg subprocess. |
| `ModelsProcessor` | `app/processors/models_processor.py` | One per `MainWindow`. Owns all model sessions. |
| `FrameWorker` | `app/processors/workers/frame_worker.py` | One per processed frame. Short-lived `threading.Thread`. |

## VideoProcessor (the orchestrator)

`VideoProcessor(QObject)` exposes signals consumed by the UI:

```python
frame_processed_signal        = Signal(int, QPixmap, numpy.ndarray)  # for video playback
webcam_frame_processed_signal = Signal(QPixmap, numpy.ndarray)       # for webcam/webrtc
single_frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)  # for single-frame preview
fps_update_signal             = Signal(float)                        # for the FPS overlay
```

State on a `VideoProcessor` instance:

```python
self.media_capture    : cv2.VideoCapture | None   # video / webcam
self.webrtc_shm       : SharedMemory | None       # WebRTC frame buffer
self.file_type        : 'video' | 'image' | 'webcam' | 'webrtc' | None
self.fps, self.current_frame_number, self.max_frame_number
self.media_path       : str
self.processing       : bool
self.recording        : bool
self.recording_sp     : subprocess.Popen | None   # ffmpeg encoder
self.virtcam          : pyvirtualcam.Camera | None
self.frame_queue      : queue.Queue (maxsize = num_threads)
self.threads          : Dict[int, threading.Thread]
self.frames_to_display: Dict[int, (QPixmap, numpy.ndarray)]
self.webcam_frames_to_display: queue.Queue
self.frame_read_timer    : QTimer  # triggers process_next_frame
self.frame_display_timer : QTimer  # triggers display_next_frame
self.gpu_memory_update_timer : QTimer  # 5s tick
```

### The play loop (video)

```
process_video()                                     # entered when user clicks Play
  ├── if recording: create_ffmpeg_subprocess()
  ├── frame_read_timer.start(interval)              # ~ 1000/fps * 0.8 ms
  ├── frame_display_timer.start()
  └── gpu_memory_update_timer.start(5000)

frame_read_timer tick → process_next_frame()
  ├── if frame_queue full → return (back-pressure)
  ├── ret, frame = misc_helpers.read_frame(media_capture)
  ├── frame = frame[..., ::-1]                      # BGR → RGB
  ├── frame_queue.put(current_frame_number)
  ├── start_frame_worker(current_frame_number, frame)   # spawns FrameWorker thread
  └── current_frame_number += 1

FrameWorker.run() (in a worker thread)
  ├── parameters = main_window.parameters.copy()
  ├── frame = self.process_frame()                  # detect → swap → restore → enhance
  ├── pixmap = get_pixmap_from_frame(frame)
  └── emit frame_processed_signal(frame_number, pixmap, frame)

frame_processed_signal slot (UI thread): store_frame_to_display
  └── frames_to_display[frame_number] = (pixmap, frame)

frame_display_timer tick → display_next_frame()
  ├── if next_frame_to_display not in frames_to_display: return
  ├── pixmap, frame = frames_to_display.pop(next_frame_to_display)
  ├── send_frame_to_virtualcam(frame)
  ├── _send_frame_to_output_window(frame)
  ├── if recording: recording_sp.stdin.write(frame.tobytes())
  └── update_graphics_view(pixmap)
      next_frame_to_display += 1
```

The pattern is **producer (frame_read_timer) + consumer (frame_display_timer)** with a fixed-size queue acting as back-pressure. `num_threads` limits parallelism; the queue caps at the same value.

### Webcam loop

Same shape as video, but:

- `process_next_webcam_frame` reads from the open `cv2.VideoCapture`.
- Frames go into `webcam_frames_to_display` (a `queue.Queue`, order matters less because it's live).
- `_apply_streaming_transforms(frame)` applies user-set rotation + horizontal/vertical flip and updates the FPS counter.
- Recording is **disabled** for webcam.

### WebRTC loop

WebRTC frames originate in a separate process (`streamrelay.run_server`) and arrive via shared memory:

```
process_video()  (file_type == 'webrtc')
  ├── try SharedMemory(name="visomaster_webrtc_frame")
  ├── if not found yet → poll every 500ms via _try_attach_webrtc_shm
  └── once attached, switch to process_next_webrtc_frame()

process_next_webrtc_frame()
  ├── counter = struct.unpack_from("<I", shm.buf, 0)[0]
  ├── if counter == self._last_webrtc_counter: return  (no new frame)
  ├── self._last_webrtc_counter = counter
  ├── w, h = struct.unpack_from("<I", shm.buf, 4), struct.unpack_from("<I", shm.buf, 8)
  ├── frame = bytes(shm.buf[12 : 12 + w*h*3]).reshape(h, w, 3)  # BGR
  ├── frame = frame[..., ::-1]                       # → RGB
  ├── frame = _apply_streaming_transforms(frame)
  └── start_frame_worker(...)
```

Header layout matches `streamrelay.protocol`:

```
0..3   : counter (uint32 LE, increments per write)
4..7   : width   (uint32 LE)
8..11  : height  (uint32 LE)
12..N  : raw BGR bytes (W*H*3)
```

### Recording (video → mp4)

```
record_video(checked=True)
  └── video_processor.recording = True
      buttonMediaPlay.setChecked(True)   → triggers play_video → process_video

create_ffmpeg_subprocess()
  └── subprocess.Popen([
        "ffmpeg",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(self.fps),
        "-i", "pipe:",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p",
        "-c:v", "libx264", "-crf", "18",
        "temp_output.mp4"
      ], stdin=subprocess.PIPE)

display_next_frame() → recording_sp.stdin.write(frame.tobytes())

stop_processing()
  ├── recording_sp.stdin.close(); recording_sp.wait()
  ├── final_path = misc_helpers.get_output_file_path(media_path, OutputMediaFolder)
  └── ffmpeg -i temp_output.mp4 -ss <start> -to <end> -i <orig> \
             -c copy -map 0:v:0 -map 1:a:0? -shortest <final>
      # mux audio from the original file over the processed video
      os.remove(temp_output.mp4)
```

### Virtual camera

`pyvirtualcam.Camera(width, height, fps, backend='obs'|'unitycapture', fmt=BGR)` is reinitialized whenever the frame dimensions change. `send_frame_to_virtualcam` calls `virtcam.send(frame)` then `sleep_until_next_frame()`.

## FrameWorker (the actual swap)

`FrameWorker(threading.Thread)` is constructed per frame:

```python
worker = FrameWorker(frame, main_window, frame_number, frame_queue, is_single_frame=False)
worker.start()    # or worker.run() for synchronous single-frame previews
```

Inside `run()`:

```python
with self.main_window.models_processor.model_lock:
    update_parameters_and_control_from_marker(main_window, frame_number)
self.parameters = main_window.parameters.copy()

if swapfacesButton.isChecked() or editFacesButton.isChecked() or FrameEnhancerEnableToggle:
    frame = self.process_frame()
else:
    frame = frame[..., ::-1]   # RGB → BGR (display always wants BGR)

pixmap = common_widget_actions.get_pixmap_from_frame(main_window, frame)

# Emit the appropriate signal based on context
if file_type in ('webcam', 'webrtc') and not is_single_frame:
    webcam_frame_processed_signal.emit(pixmap, frame)
elif not is_single_frame:
    frame_processed_signal.emit(frame_number, pixmap, frame)
else:
    single_frame_processed_signal.emit(frame_number, pixmap, frame)

# Always drain the queue
frame_queue.task_done()
```

### `process_frame()` — the per-frame pipeline

1. **Move to GPU.** `img = torch.from_numpy(frame).to(device).permute(2,0,1)` (CHW).
2. **Upscale to 512.** Detection assumes 512+ on both axes.
3. **Manual rotation** if `ManualRotationEnableToggle`.
4. **Detect.**
   ```python
   bboxes, kpss_5, kpss = models_processor.run_detect(
       img, control['DetectorModelSelection'],
       max_num=control['MaxFacesToDetectSlider'],
       score=control['DetectorScoreSlider']/100,
       use_landmark_detection=control['LandmarkDetectToggle'],
       landmark_detect_mode=control['LandmarkDetectModelSelection'],
       rotation_angles=[0,90,180,270] if control['AutoRotationToggle'] else [0])
   ```
5. **Recognize.** For each detected face, run `run_recognize_direct` with the user's `RecognitionModelSelection` (Inswapper128ArcFace / SimSwapArcFace / GhostArcFace / CSCSArcFace) to get a 512-dim embedding.
6. **Match.** For each detection, walk through `main_window.target_faces`. If cosine distance `≥ SimilarityThresholdSlider`, this detection should be processed for this target.
7. **Adjust keypoints.** `keypoints_adjustments` applies per-face X/Y/scale offsets from `KpsXSlider` etc.
8. **Pick the embedding.** ArcFace family is selected via `arcface_mapping_model_dict[SwapModelSelection]`; the source embedding is read from `target_face.assigned_input_embedding[arcface_model]`.
9. **Swap.** `swap_core(img, kps_5, s_e=source_emb, t_e=target_emb, parameters, control, dfm_model=…)` — calls one of:
   - `run_inswapper`
   - `run_iss_swapper` (variants A/B/C)
   - `run_swapper_simswap512`
   - `run_swapper_ghostface` (v1/v2/v3)
   - `run_swapper_cscs`
   - DFM convert (via `DFMModel.convert`)
   Then optional `apply_face_expression_restorer` via LivePortrait.
10. **Restore.** If `FaceRestorerEnableToggle`, calls `apply_facerestorer` with one or two restorer chains (GFPGAN/CodeFormer/GPEN/VQFR/RestoreFormer++).
11. **Mask & blend.** `apply_occlusion`, `apply_dfl_xseg`, `apply_face_parser`, `restore_mouth`, `restore_eyes`, `apply_face_makeup`, `apply_fake_diff`.
12. **Edit.** If `editFacesButton.isChecked`, run `swap_edit_face_core` (LivePortrait pose/expression).
13. **Reverse rotation.** If manual rotation was applied.
14. **Overlays.** Bounding boxes, landmarks, face-compare panel.
15. **Frame enhance.** `enhance_core` does tile-based RealEsrGAN/BSRGAN/UltraSharp/UltraMix/DeOldify/DDColor.
16. **Return BGR.** Whatever the path, the worker returns BGR uint8 ready for cv2/QPixmap.

## ModelsProcessor (the model owner)

```python
ModelsProcessor.__init__(main_window, device='cuda')
  ├── detect actually-available providers; fall back to CPU if needed
  ├── self.providers = [('CUDAExecutionProvider'), ('CPUExecutionProvider')]
  ├── self.trt_ep_options = { engine_cache_path: 'tensorrt-engines', builder_optimization_level: 5, … }
  ├── populate self.models[name] = None and self.models_path[name] from models_data.models_list
  ├── populate self.models_trt[name] = None from models_data.models_trt_list (LivePortrait engines)
  └── construct family helpers:
      face_detectors / face_landmark_detectors / face_swappers / face_restorers
      face_masks / face_editors / frame_enhancers
```

Lazy load:

```python
ModelsProcessor.load_model('Inswapper128')
  ├── with self.model_lock:
  │     if not exists(path): raise FileNotFoundError(...)
  │     self.main_window.model_loading_signal.emit()
  │     session = onnxruntime.InferenceSession(path, providers=self.providers)
  │     active = session.get_providers()
  │     self._model_devices[name] = 'cuda' if any('CUDA' or 'Tensorrt' in active) else 'cpu'
  │     # double-check: another thread may have loaded it
  │     if self.models[name]: del session; return self.models[name]
  │     self.main_window.model_loaded_signal.emit()
  │     return session
```

DFM models are loaded with an LRU cap controlled by `MaxDFMModelsSlider`.

TensorRT engines are built on first use:

```python
load_model_trt('LivePortraitWarpingSpadeFix', ...)
  ├── if not engine file → onnx2trt(onnx_path, trt_path, precision='fp16')
  └── return TensorRTPredictor(trt_path, pool_size=nThreads, device=self.device)
```

## Family helpers (pure inference)

Each family file in `app/processors/` declares one class that takes a `models_processor` reference and exposes the inference functions used by `FrameWorker`. They lazy-call `models_processor.load_model(name)` when needed.

| File | Class | Public methods |
|---|---|---|
| `face_detectors.py` | `FaceDetectors` | `run_detect`, `detect_retinaface`, `detect_scrdf`, `detect_yoloface`, `detect_yunet` |
| `face_landmark_detectors.py` | `FaceLandmarkDetectors` | `run_detect_landmark`, `detect_face_landmark_5/68/3d68/98/106/203/478` |
| `face_swappers.py` | `FaceSwappers` | `run_recognize_direct`, `recognize`, `calc_inswapper_latent`, `run_inswapper`, `calc_swapper_latent_iss`, `run_iss_swapper`, `calc_swapper_latent_simswap512`, `run_swapper_simswap512`, `calc_swapper_latent_ghost`, `run_swapper_ghostface`, `calc_swapper_latent_cscs`, `run_swapper_cscs` |
| `face_restorers.py` | `FaceRestorers` | `apply_facerestorer`, `run_GFPGAN`, `run_GPEN_256/512/1024/2048`, `run_codeformer`, `run_VQFR_v2`, `run_RestoreFormerPlusPlus` |
| `face_masks.py` | `FaceMasks` | `apply_occlusion`, `apply_dfl_xseg`, `apply_face_parser`, `run_occluder`, `run_dfl_xseg`, `run_faceparser`, `run_CLIPs`, `restore_mouth`, `restore_eyes`, `apply_fake_diff`, `soft_oval_mask` |
| `face_editors.py` | `FaceEditors` | `lp_motion_extractor`, `lp_appearance_feature_extractor`, `lp_retarget_eye/lip`, `lp_stitch`, `lp_stitching`, `lp_warp_decode`, `apply_face_makeup`, `face_parser_makeup_direct_rgb` |
| `frame_enhancers.py` | `FrameEnhancers` | `run_enhance_frame_tile_process`, `run_realesrganx2/x4`, `run_realesrx4v3`, `run_bsrganx2/x4`, `run_ultrasharpx4`, `run_ultramixx4`, `run_deoldify_artistic/stable/video`, `run_ddcolor`, `run_ddcolor_artistic` |

## Threading model

- **Qt main thread (UI).** All widget I/O. Reads `parameters`/`control` (mutated only by widget callbacks). Spawns timers.
- **Frame worker threads (`threading.Thread`).** One per in-flight frame, capped by `num_threads`. They read input via shared memory or cv2, do GPU inference, emit Qt signals (cross-thread safe).
- **Model lock.** `ModelsProcessor.model_lock` is a `threading.RLock` taken inside `load_model` (and by `FrameWorker.run` while it pulls parameters/markers, since markers may mutate during seek).
- **Loader QThreads.** `TargetMediaLoaderWorker`, `InputFacesLoaderWorker`, `FilterWorker` are `QThread`s used to scan folders and decode thumbnails without blocking the UI.
- **WebRTC subprocess.** A full `multiprocessing.Process` running `streamrelay.run_server` with its own asyncio loop. Talks to the rest of the app exclusively via shared memory.
