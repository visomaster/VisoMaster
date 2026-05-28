# 05 · Models & Inference

## Model registry (`app/processors/models_data.py`)

Two lists drive everything:

### `models_list` (ONNX)

Each entry:

```python
{
    "model_name":  "Inswapper128",
    "local_path":  "./model_assets/inswapper_128.fp16.onnx",
    "hash":        "<sha256>",
    "url":         "https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/inswapper_128.fp16.onnx",
}
```

Model families covered (full list in `models_data.py`):

| Family | Models |
|---|---|
| **Swappers** | Inswapper128, InStyleSwapper256 v{A,B,C}, SimSwap512, GhostFace-v{1,2,3}, CSCS |
| **DFM ArcFace** | Inswapper128ArcFace, SimSwapArcFace, GhostArcFace, CSCSArcFace |
| **Detectors** | RetinaFace, SCRFD2.5g, YoloFace8n, YunetN |
| **Landmarks** | FaceLandmark{5,68,3d68,98,106,203,478} |
| **Restorers** | GFPGAN-v1.4, CodeFormer, GPEN-{256,512,1024,2048}, VQFR-v2, RestoreFormer++ |
| **Masks** | Occluder, DFLXSeg, FaceParser |
| **Frame enhancers** | RealEsrGAN-x{2,4}-Plus, RealEsr-General-x4v3, BSRGAN-x{2,4}, UltraSharp-x4, UltraMix-x4, DeOldify-{Artistic,Stable,Video}, DDColor, DDColor-Artistic |
| **LivePortrait** | MotionExtractor, AppearanceFeatureExtractor, StitchingEye, StitchingLip, Stitching, WarpingSpadeFix (also as TRT engines) |
| **Other** | CLIP / CLIPSeg (for text-driven masks) |

### `models_trt_list` (TensorRT engines)

Built only when `tensorrt` is importable. Engines are versioned by trt version:

```python
'local_path': f'{models_dir}/liveportrait_onnx/motion_extractor.{trt.__version__}.trt'
```

The mapping `arcface_mapping_model_dict` selects which ArcFace model corresponds to each swapper:

```python
{
    'Inswapper128':                  'Inswapper128ArcFace',
    'InStyleSwapper256 Version A':   'Inswapper128ArcFace',
    'InStyleSwapper256 Version B':   'Inswapper128ArcFace',
    'InStyleSwapper256 Version C':   'Inswapper128ArcFace',
    'DeepFaceLive (DFM)':            'Inswapper128ArcFace',
    'SimSwap512':                    'SimSwapArcFace',
    'GhostFace-v1':                  'GhostArcFace',
    'GhostFace-v2':                  'GhostArcFace',
    'GhostFace-v3':                  'GhostArcFace',
    'CSCS':                          'CSCSArcFace',
}
```

## Inference path: ONNX

Models are run via `onnxruntime.InferenceSession` with `IOBinding` to keep tensors on the GPU when available:

```python
session = onnxruntime.InferenceSession(model_path, providers=self.providers)
io_binding = session.io_binding()
io_binding.bind_input(name, device_type='cuda', device_id=0,
                      element_type=np.float16, shape=tensor.shape,
                      buffer_ptr=tensor.data_ptr())
io_binding.bind_output(name, device_type='cuda', element_type=np.float16,
                       shape=output.shape, buffer_ptr=output.data_ptr())
session.run_with_iobinding(io_binding)
```

This avoids a CPU round-trip when the input is already a CUDA `torch.Tensor`. Most family helper methods accept torch tensors and write into pre-allocated output tensors.

## Inference path: TensorRT

`TensorRTPredictor` (in `app/processors/utils/tensorrt_predictor.py`) wraps a serialized engine:

- Allocates I/O buffers (`OrderedDict[str, torch.Tensor]`) sized to engine specs.
- `predict(feed_dict)` copies inputs into bound buffers and runs the engine.
- `predict_async(feed_dict, stream)` does the same on an explicit `torch.cuda.Stream` for overlap.
- Uses a `pool_size` matching `nThreads` so multiple worker threads can predict in parallel without contending on a single context.

`engine_builder.onnx_to_trt(onnx_model_path, trt_model_path, precision='fp16')`:

- Creates a `trt.Builder`, sets workspace pool to 3 GB, optimization level 5, FP16 if supported.
- Parses the ONNX file, builds a serialized engine, writes it to disk.
- Used on first call to `load_model_trt(...)` if the `.trt` file doesn't exist yet.

When the user picks **Providers Priority = "TensorRT-Engine"**, `switch_providers_priority` swaps the providers list so ONNX Runtime emits cached TensorRT EP context models from `tensorrt-engines/` (much faster cold start than building engines manually).

## DFM (DeepFaceLab) models

`app/processors/utils/dfm_model.py::DFMModel` loads `.dfm` (which are zipped ONNX) and `.onnx` files from `model_assets/dfm_models/`. Public surface:

```python
DFMModel(model_path, providers, device='cuda')
  .get_model_path() → str
  .get_input_res()  → int                         # e.g. 224, 256, 320
  .has_morph_value() → bool                       # AMP variant?
  .convert(img: torch.Tensor, morph_factor=0.75, rct=False)
       → (predicted_face: torch.Tensor,
          predicted_mask: torch.Tensor)
  .rct(img, like, mask=None, like_mask=None, mask_cutoff=0.5)
       → torch.Tensor  # reinhard color transfer
```

`MaxDFMModelsSlider` (default 1) limits how many DFM sessions are kept in memory. When the cap is hit, the oldest entry is evicted before loading a new one.

## ArcFace embedding flow

```
img (torch.Tensor, RGB, on GPU)
  ├── crop & align using kps_5 + arcface_dst (5 reference points)
  ├── normalize to [-1, 1] (or model-specific normalization)
  ├── ONNX session → 512-dim embedding (numpy.ndarray)
  └── (cached on TargetFaceCardButton.embedding_store[recognition_model])
```

Cosine similarity is computed via `models_processor.findCosineDistance(v1, v2)`. The `SimilarityTypeSelection` ('Opal', 'Pearl', 'Optimal') tweaks the similarity computation (e.g. norm vs raw), defined in `face_swappers.recognize`.

## Swap latents

Each swapper has a "latent calc" + "run" pair:

```python
latent = models_processor.calc_inswapper_latent(source_embedding)
models_processor.run_inswapper(image_tensor, latent, output_tensor)
# → output_tensor (filled in-place)
```

`calc_*_latent` is a small linear projection (typically `embedding @ emap`) that converts the ArcFace 512-vector into the swap model's input space. `emap` is loaded from the inswapper ONNX file via `load_inswapper_iss_emap`.

## Mask flow

`apply_face_parser(img, parameters)` runs the BiSeNet face-parser ONNX, producing a per-pixel class map (eyes, brows, nose, mouth, hair, etc.). The user's per-region opacity sliders blend each class mask back into the swapped face.

`apply_occlusion` and `apply_dfl_xseg` run XSeg-trained ONNX models that segment occluders (hands, hair) so the original (non-swapped) pixels show through.

`run_CLIPs(img, CLIPText, CLIPAmount)` uses CLIP + CLIPSeg to mask out regions matching a free-text prompt — for example "glasses", "hat".

`restore_mouth` and `restore_eyes` blend the original face's mouth/eyes back into the swapped face using elliptical alpha masks (computed from `kps_5`), preserving the source's expression and gaze.

## LivePortrait expression restoration

When `FaceExpressionEnableToggle` is on, `FrameWorker.apply_face_expression_restorer(driving, target, parameters)` runs:

1. `lp_motion_extractor(driving)` → driving keypoints + transform parameters.
2. `lp_appearance_feature_extractor(target)` → 3D feature volume.
3. `lp_motion_extractor(target)` → source keypoints.
4. Optional `lp_retarget_eye(kp_source, eye_close_ratio)` and `lp_retarget_lip(...)`.
5. `lp_stitching(kp_source, kp_driving)` → final keypoints.
6. `lp_warp_decode(feature_3d, kp_source, kp_driving)` → warped face image.

The face is then alpha-blended over the swap result with `FaceExpressionFriendlyFactorDecimalSlider` controlling intensity.

## Model dependency graph

Models are loaded lazily on first use. The table below shows which models depend on which others at runtime.

### Pipeline order (per frame)

```
Frame input
  │
  ▼
[1] Face Detector  (one of)
      RetinaFace · SCRFD2.5g · YoloFace8n · YunetN
  │
  ▼  optional — LandmarkDetectToggle or Face Editor active
[2] Landmark Detector  (one of)
      FaceLandmark5 (default/fallback)
      FaceLandmark68 · FaceLandmark3d68 · FaceLandmark98
      FaceLandmark106 · FaceLandmark203 (forced when Face Editor on)
      FaceLandmark478
  │
  ▼  always — identity matching
[3] ArcFace Backbone  (determined by chosen swapper)
      Inswapper128ArcFace  ← Inswapper128, InStyleSwapper256 A/B/C, DFM
      SimSwapArcFace       ← SimSwap512
      GhostArcFace         ← GhostFace-v1/v2/v3
      CSCSArcFace          ← CSCS
  │
  ▼  optional — swap enabled
[4] Face Swapper  (one of)
      Inswapper128 · InStyleSwapper256 A/B/C
      SimSwap512 · GhostFace-v1/v2/v3 · CSCS
      DeepFaceLive (DFM)  ← user-supplied .dfm file
  │
  ▼  optional — post-swap, up to two slots
[5] Face Restorer  (one of per slot)
      GFPGANv1.4 · CodeFormer · GPENBFR256/512/1024/2048
      VQFRv2 · RestoreFormerPlusPlus
      (restorer "Reference" alignment also calls FaceLandmark5 internally)
  │
  ▼  optional — masking
[6] Face Masks  (any combination)
      Occluder · XSeg · FaceParser
      CLIPSeg (RD64ClipText — PyTorch .pth, not ONNX)
  │
  ▼  optional — edit mode (replaces swap step)
[7] Face Editor — LivePortrait  (all load together)
      LivePortraitMotionExtractor
      LivePortraitAppearanceFeatureExtractor
      LivePortraitWarpingSpade (ONNX) / LivePortraitWarpingSpadeFix (TRT)
      LivePortraitStitching
      LivePortraitStitchingEye   ← only when eye retargeting enabled
      LivePortraitStitchingLip   ← only when lip retargeting enabled
      (makeup sub-feature also requires FaceParser)
  │
  ▼  optional — full-frame post-processing
[8] Frame Enhancer  (one of)
      RealEsrganx2Plus · RealEsrganx4Plus · RealEsrx4v3
      BSRGANx2 · BSRGANx4 · UltraSharpx4 · UltraMixx4
      DeoldifyArt · DeoldifyStable · DeoldifyVideo
      DDColorArt · DDcolor
```

### Hard dependencies

| Model / feature | Requires |
|---|---|
| Any face swapper | Matching ArcFace backbone (see `arcface_mapping_model_dict`) |
| Any face swapper | A face detector (step 1) |
| Face Editor (LivePortrait) | `FaceLandmark203` — forced on when edit mode activates |
| Face Editor (LivePortrait) | MotionExtractor + AppearanceFeatureExtractor + WarpingSpade + Stitching (all 4 always load together) |
| Face Restorer with "Reference" alignment | `FaceLandmark5` (called internally to re-align) |
| Face Editor makeup | `FaceParser` |
| CLIPSeg mask | `RD64ClipText` (.pth) |

### Independent models

All restorers, all frame enhancers, `Occluder`, and `XSeg` are fully self-contained — they load on demand and do not trigger any other model.

---

## Auto-offload

`ModelsProcessor.offload_models_for_parameter_change(param_name, old_value, new_value, all_parameters)` is called by the UI layer every time a parameter or control value changes. It unloads models that are no longer needed, freeing VRAM immediately rather than waiting for a manual "Clear GPU Memory" action.

### How it is triggered

- **`update_parameter`** (`common_actions.py`) — covers all per-face parameter changes, including reset-to-default (which flows through the same `set_value` → signal → `update_parameter` path).
- **`update_control`** (`common_actions.py`) — covers control-level parameters: detector model, landmark model, frame enhancer, recognition model.

### Offload rules

| Parameter changed | Models offloaded |
|---|---|
| `SwapModelSelection` | Old swapper model + old ArcFace backbone (if the new swapper uses a different one) |
| `FaceRestorerTypeSelection` | Old restorer model (only if neither slot 1 nor slot 2 still references it) |
| `FaceRestorerEnableToggle` → off | Slot 1 restorer model (if slot 2 doesn't also use it) |
| `FaceRestorerEnable2Toggle` → off | Slot 2 restorer model (if slot 1 doesn't also use it) |
| `OccluderEnableToggle` → off | `Occluder` |
| `DFLXSegEnableToggle` → off | `XSeg` |
| `FaceParserEnableToggle` → off | `FaceParser` (only if all makeup toggles are also off) |
| Any makeup toggle → off | `FaceParser` (only if parser toggle and all other makeup toggles are also off) |
| `ClipEnableToggle` → off | CLIPSeg session |
| `FaceEditorEnableToggle` → off | All 7 LivePortrait models (if `FaceExpressionEnableToggle` is also off) |
| `FaceExpressionEnableToggle` → off | All 7 LivePortrait models (if `FaceEditorEnableToggle` is also off) |
| `FrameEnhancerEnableToggle` → off | Current frame enhancer model |
| `FrameEnhancerTypeSelection` changes | Old frame enhancer model |
| `DetectorModelSelection` changes | Old detector model |
| `LandmarkDetectToggle` → off | Current landmark model |
| `LandmarkDetectModelSelection` changes | Old landmark model |
| `RecognitionModelSelection` changes | Old ArcFace recognition model |

### Safety rules

- `unload_model` is a no-op if the model is not currently loaded — no guard needed at call sites.
- Restorer offload checks both slots before unloading, so switching slot 1 from GFPGAN to CodeFormer while slot 2 is also using GFPGAN will not unload GFPGAN.
- `FaceParser` is shared by the parser mask and all makeup features; it is only unloaded when every consumer is off.
- LivePortrait models are shared by Face Editor and Face Expression Restorer; they are only unloaded when both are off.

---

## Notable global state

`ModelsProcessor` keeps:

- `arcface_dst` — 5-point reference template at 112×112 used for ArcFace alignment.
- `FFHQ_kps` — 5-point reference at 512×512 for swap models.
- `LandmarksSubsetIdxs` — indices into the 478-point Mediapipe face mesh for the relevant subset.
- `mean_lmk`, `anchors`, `emap` — populated lazily by detection / swap helpers.
