# 12 · Glossary

Terms used throughout the VisoMaster codebase.

| Term | Meaning |
|---|---|
| **Target media** | The video / image / live source you want to **process**. Lives in `targetVideosList`. Stored as `target_videos` on `MainWindow`. |
| **Target face** | A face **detected in the target media** that you want to swap. Each target face has its own parameters. Stored as `target_faces` on `MainWindow`. |
| **Source face** (also "input face") | A face **image you supply** to swap onto a target face. Stored as `input_faces` on `MainWindow`. |
| **Embedding** | A 512-dim vector produced by an ArcFace model from a cropped face. Used for both **identity matching** (comparing target faces against new detections) and **swap conditioning** (passed to the swapper as `s_e`). |
| **Embedding store** | A dict `{recognition_model_name: embedding_array}` cached on every face card. Stored for all 4 ArcFace variants (`Inswapper128ArcFace`, `SimSwapArcFace`, `GhostArcFace`, `CSCSArcFace`) so switching recognition model doesn't require re-running detection. |
| **Merged embedding** | An embedding made by combining multiple source embeddings (mean or median per `EmbMergeMethodSelection`). Useful when you have many photos of one person. |
| **Assigned input embedding** | The final per-target embedding computed from `assigned_input_faces ∪ assigned_merged_embeddings`. This is what gets fed into the swapper. |
| **DFM** | DeepFaceLab Model. A pre-trained face-swap model bundled as `.dfm` (zipped ONNX). Trained per-identity by the user. Lives in `model_assets/dfm_models/`. |
| **ArcFace** | A face recognition model that outputs 512-dim embeddings. Different swappers were trained against different ArcFace variants — `arcface_mapping_model_dict` selects the right one. |
| **Inswapper / InStyleSwapper / SimSwap / GhostFace / CSCS** | Different face-swap model architectures, each with different inputs/strengths. |
| **kps_5 / kpss_5** | 5-point keypoints (left eye, right eye, nose, left mouth, right mouth). The standard alignment template. |
| **kps / kpss / kps_all** | N-point landmarks (5/68/3d68/98/106/203/478) depending on `LandmarkDetectModelSelection`. |
| **arcface_dst** | Reference 5-point template at 112×112 used for ArcFace alignment. |
| **FFHQ_kps** | Reference 5-point template at 512×512 used for swap models trained on the FFHQ dataset. |
| **Similarity Type** | How embedding distance is computed. Options: `Opal`, `Pearl`, `Optimal` — different normalization conventions. |
| **Similarity Threshold** | Per-target setting. A new detection has to match the target face's stored embedding above this cosine similarity to be processed. |
| **Marker** | A frame number with a snapshot of `parameters` and `control`. While playing, marker values temporarily override the live values. Used to vary swap settings across a video. |
| **Restorer** | A face-image-to-face-image super-resolution / detail model run **after** the swap. GFPGAN, CodeFormer, GPEN, VQFR, RestoreFormer++. Two chained passes are supported. |
| **Frame Enhancer** | A whole-frame super-resolution / colorization model run **after** restoration. RealEsrGAN, BSRGAN, UltraSharp, UltraMix, DDColor, DeOldify. |
| **LivePortrait** | A motion-driven face animation model. Used in two ways here: (a) **expression restoration** (preserve the original face's expression on the swapped face); (b) **face editor** (manual sliders for expressions/poses). |
| **Mask** | An alpha map used to blend the swapped face back into the original frame. Sources: face-parser (per-region BiSeNet), occluder (XSeg), DFL XSeg, CLIPSeg (text-driven), elliptical mouth/eye masks, `restore_mouth` / `restore_eyes` blends. |
| **Provider** | An ONNX Runtime execution provider. CUDA, TensorRT, CPU. Switching provider re-creates all sessions. |
| **TensorRT engine** | A compiled `.trt` file derived from an ONNX model. Faster than ONNX Runtime CUDA for static-shape models. Built lazily by `engine_builder.onnx_to_trt`. |
| **TensorRT-Engine** (provider option) | An ORT mode where TensorRT EP context models are emitted to `tensorrt-engines/`. Combines the convenience of ONNX Runtime with TRT performance. |
| **VirtualCam** | A virtual webcam device (OBS Virtual Camera or Unity Capture) that other apps see as a normal webcam. VisoMaster pushes processed frames into it via `pyvirtualcam`. |
| **Output Window** | A borderless `QWidget` that displays only the processed frame, designed for OBS "Window Capture" so users can stream without a virtual camera driver. |
| **Streamrelay** | The bundled aiortc-based WebRTC server (`packages/streamrelay/`) that ingests video from a phone or browser. Writes BGR frames into a shared-memory block named `visomaster_webrtc_frame`. |
| **WHIP** | "WebRTC-HTTP Ingestion Protocol". A simple POST-an-SDP-offer / receive-an-SDP-answer flow used by Larix Broadcaster, OBS, and other streaming apps. Supported at `/whip`. |
| **Workspace** | A complete snapshot of the user's working set (target media list, target faces, source faces, embeddings, markers, parameters, control). Persisted as `last_workspace.json` automatically; can be exported/imported. |
| **Action functions** | The functions in `app/ui/widgets/actions/*.py`. They take `MainWindow` and mutate state. They're the de-facto API for the UI today and the model for the future REST endpoints. |
| **Parameters vs Control** | **Parameters** are per-target-face (one slider value per face: e.g. blend, restorer fidelity). **Control** is global app state (provider, threads, detector, output folder, virtual cam, WebRTC ports). |
| **`ParametersDict`** | A `UserDict` subclass that returns a default value (and persists it) when an unknown key is read. Lets older workspaces survive newer parameter additions. |
| **Frame Worker** | A short-lived `threading.Thread` that runs the full inference pipeline for one frame. One worker per frame; capped at `nThreadsSlider` workers in flight at once. |
| **Two-timer loop** | The play loop: a `frame_read_timer` enqueues frames; a `frame_display_timer` drains processed frames in order. The frame queue between them is a back-pressure channel sized at `nThreads`. |
