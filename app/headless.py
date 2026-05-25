"""Headless driver for VisoMaster's face-swap pipeline.

The desktop app drives the GPU compute (``FrameWorker.process_frame`` ->
``ModelsProcessor`` -> ONNX models) entirely off Qt widget state held on the
``MainWindow``. This module provides a lightweight stand-in for that window so
the *exact same* compute path can run without a display — e.g. on a Modal GPU.

Nothing here re-implements the swap logic: it reconstructs the GUI state the
pipeline reads (``control`` dict, per-face ``parameters``, ``target_faces`` with
embeddings) and then calls the real ``FrameWorker.process_frame``.
"""

import gc
import queue
import subprocess
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from app.helpers.miscellaneous import ParametersDict
from app.processors.models_processor import ModelsProcessor
from app.processors.workers.frame_worker import FrameWorker
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA

# The four ArcFace models double as both the recognition (matching) keyspace and
# the swapper-input embedding keyspace, so computing all of them covers any swapper.
RECOGNITION_OPTIONS = list(
    SETTINGS_LAYOUT_DATA["Face Recognition"]["RecognitionModelSelection"]["options"]
)


# --------------------------------------------------------------------------- #
# Default control/parameter construction (mirrors layout_actions widget setup) #
# --------------------------------------------------------------------------- #
def _typed_default(widget_name: str, widget_data: dict):
    """Coerce a layout-data ``default`` to the type the running app stores.

    Mirrors ``layout_actions.add_widgets_to_tab_layout``: toggles -> bool,
    selections -> str, decimal sliders -> float, plain sliders -> int.
    """
    default = widget_data["default"]
    if callable(default):
        default = default()
    if "Toggle" in widget_name:
        return bool(default)
    if "Selection" in widget_name:
        return default
    if "DecimalSlider" in widget_name:
        return float(default)
    if "Slider" in widget_name:  # checked after DecimalSlider on purpose
        return int(default)
    return default  # Text and anything else


def _build_defaults(layout: dict) -> dict:
    out = {}
    for _group, widgets in layout.items():
        for widget_name, widget_data in widgets.items():
            out[widget_name] = _typed_default(widget_name, widget_data)
    return out


class _NoopSignal:
    def emit(self, *args, **kwargs):
        pass


class _Button:
    """Stand-in for a checkable Qt button."""

    def __init__(self, checked: bool):
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class HeadlessTargetFace:
    """Minimal stand-in for ``TargetFaceCardButton`` used during matching/swap."""

    def __init__(self, face_id: str, embedding_store: dict, assigned_input_embedding: dict):
        self.face_id = face_id
        self.embedding_store = embedding_store
        self.assigned_input_embedding = assigned_input_embedding

    def get_embedding(self, embedding_swap_model: str) -> np.ndarray:
        return self.embedding_store.get(embedding_swap_model, np.array([]))


class HeadlessMainWindow:
    """Provides exactly the attributes the compute path reads off ``MainWindow``."""

    def __init__(self, device: str = "cuda", provider: str = "CUDA"):
        # Signals the pipeline emits for the (now absent) loading dialog.
        self.model_loading_signal = _NoopSignal()
        self.model_loaded_signal = _NoopSignal()

        # Layout-derived defaults, exactly as the GUI would initialise them.
        self.default_parameters = {}
        for layout in (COMMON_LAYOUT_DATA, SWAPPER_LAYOUT_DATA, FACE_EDITOR_LAYOUT_DATA):
            self.default_parameters.update(_build_defaults(layout))
        self.control = _build_defaults(SETTINGS_LAYOUT_DATA)
        self.control["OutputMediaFolder"] = ""

        self.parameters = {}  # face_id -> ParametersDict
        self.current_widget_parameters = ParametersDict(
            dict(self.default_parameters), self.default_parameters
        )
        self.selected_target_face_id = False
        self.target_faces = {}
        self.dfm_models_data = {}
        self.markers = {}

        # Button states: swap on, everything else off.
        self.swapfacesButton = _Button(True)
        self.editFacesButton = _Button(False)
        self.faceCompareCheckBox = _Button(False)
        self.faceMaskCheckBox = _Button(False)

        # FrameWorker only stores this reference; the swap path never touches it.
        self.video_processor = types.SimpleNamespace(file_type="video", processing=False)

        self.models_processor = ModelsProcessor(self, device=device)
        # Force a plain CUDA/ONNX provider (no TensorRT engine building for the
        # batch path); also sets device and moves the LivePortrait mask to GPU.
        self.models_processor.switch_providers_priority(provider)


# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #
def compute_embedding_store(models_processor, bgr_image: np.ndarray, control: dict,
                            max_num: int = 1) -> dict:
    """Detect the most prominent face in a BGR image and return its embeddings
    across every recognition model (the same store the GUI builds per face)."""
    rgb = np.ascontiguousarray(bgr_image[..., ::-1])
    img = torch.from_numpy(rgb.astype("uint8")).to(models_processor.device).permute(2, 0, 1)

    _bboxes, kpss_5, _kpss = models_processor.run_detect(
        img,
        control["DetectorModelSelection"],
        max_num=max_num,
        score=control["DetectorScoreSlider"] / 100.0,
        input_size=(512, 512),
        use_landmark_detection=control["LandmarkDetectToggle"],
        landmark_detect_mode=control["LandmarkDetectModelSelection"],
        landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
        from_points=control["DetectFromPointsToggle"],
    )
    if len(kpss_5) == 0:
        raise ValueError("No face detected in the provided face image.")

    kps_5 = kpss_5[0]
    store = {}
    for model in RECOGNITION_OPTIONS:
        embedding, _crop = models_processor.run_recognize_direct(
            img, kps_5, control["SimilarityTypeSelection"], model
        )
        store[model] = embedding
    return store


def merge_embedding_stores(stores: list, method: str = "Mean") -> dict:
    """Average several per-face embedding stores (matches TargetFaceCardButton
    .calculate_assigned_input_embedding)."""
    if not stores:
        return {}
    reducer = np.median if method == "Median" else np.mean
    models = set().union(*[s.keys() for s in stores])
    return {
        model: reducer([s[model] for s in stores if model in s], axis=0)
        for model in models
    }


# --------------------------------------------------------------------------- #
# Settings -> control / parameters                                            #
# --------------------------------------------------------------------------- #
_CONTROL_SHORTCUTS = {
    "detector_model": "DetectorModelSelection",
    "detector_score": "DetectorScoreSlider",
    "recognition_model": "RecognitionModelSelection",
    "max_faces": "MaxFacesToDetectSlider",
    "landmark_detect": "LandmarkDetectToggle",
}


def apply_control_settings(mw: HeadlessMainWindow, settings: dict):
    settings = settings or {}
    for key, control_key in _CONTROL_SHORTCUTS.items():
        if key in settings:
            mw.control[control_key] = settings[key]
    for key, value in (settings.get("control") or {}).items():
        mw.control[key] = value


def make_face_parameters(mw: HeadlessMainWindow, settings: dict,
                         similarity_threshold, extra_overrides: dict = None) -> ParametersDict:
    """Build a per-face ParametersDict from defaults + global shortcuts + global
    ``parameters`` + this face's ``extra_overrides``. Missing keys fall back to
    defaults. Per-face overrides win over global; the threshold arg always wins."""
    settings = settings or {}
    overrides = dict(settings.get("parameters") or {})

    if "swap_model" in settings:
        overrides["SwapModelSelection"] = settings["swap_model"]

    restorer = settings.get("face_restorer")
    if restorer:
        overrides["FaceRestorerEnableToggle"] = bool(restorer.get("enabled", True))
        if "type" in restorer:
            overrides["FaceRestorerTypeSelection"] = restorer["type"]
        if "det_type" in restorer:
            overrides["FaceRestorerDetTypeSelection"] = restorer["det_type"]
        if "blend" in restorer:
            overrides["FaceRestorerBlendSlider"] = restorer["blend"]
        if "fidelity" in restorer:
            overrides["FaceFidelityWeightDecimalSlider"] = restorer["fidelity"]

    if extra_overrides:
        overrides.update(extra_overrides)
    overrides["SimilarityThresholdSlider"] = similarity_threshold

    return ParametersDict(overrides, mw.default_parameters)


def _register_target_face(mw: HeadlessMainWindow, face_id: str, target_img_bgr,
                          source_imgs_bgr: list, threshold, settings: dict,
                          extra_overrides: dict = None):
    """Build one target face: source identity to swap in (averaged over the
    given source images) matched against a reference (or itself, for swap-all)."""
    source_stores = [
        compute_embedding_store(mw.models_processor, img, mw.control)
        for img in source_imgs_bgr
    ]
    assigned = merge_embedding_stores(
        source_stores, mw.control.get("EmbMergeMethodSelection", "Mean")
    )
    if not assigned:
        raise ValueError("Could not build a source embedding from the given face image(s).")

    if target_img_bgr is not None:
        target_store = compute_embedding_store(mw.models_processor, target_img_bgr, mw.control)
    else:
        # Swap-all: a non-degenerate target embedding (reuse source) + threshold 0
        # makes every detected face match.
        target_store = assigned

    mw.target_faces[face_id] = HeadlessTargetFace(face_id, target_store, assigned)
    mw.parameters[face_id] = make_face_parameters(mw, settings, threshold, extra_overrides)


def setup(mw: HeadlessMainWindow, entries: list, settings: dict):
    """Configure one or more face swaps. This is the headless equivalent of the
    GUI's target-face cards + assigned source faces.

    ``entries`` is a list of dicts:
      - ``source_imgs``: list[BGR ndarray] — identity to swap *in* (averaged).
      - ``target_img``:  BGR ndarray | None — reference of who to *replace*.
                         May be None only for a single swap-all entry.
      - ``parameters``:  dict (optional) — per-face parameter overrides.
      - ``swap_model``:  str  (optional) — per-face swapper model.
      - ``threshold``:   int  (optional) — per-face match threshold.

    Each detected face in a frame is matched (by ArcFace embedding) against every
    entry's reference; the first whose similarity >= its threshold is swapped to
    that entry's source identity. Faces matching no entry are left untouched.
    """
    apply_control_settings(mw, settings)
    if not entries:
        raise ValueError("No swap entries provided.")

    global_threshold = settings.get("similarity_threshold", 60)
    swap_all = len(entries) == 1 and entries[0].get("target_img") is None

    for i, entry in enumerate(entries):
        target_img = entry.get("target_img")
        if target_img is None and not swap_all:
            raise ValueError(
                "Multi-face mode requires a target_face reference for every swap entry."
            )
        extra = dict(entry.get("parameters") or {})
        if entry.get("swap_model"):
            extra["SwapModelSelection"] = entry["swap_model"]
        threshold = 0 if target_img is None else entry.get("threshold", global_threshold)
        _register_target_face(mw, str(i), target_img, entry["source_imgs"],
                              threshold, settings, extra)


def detect_face_crops(mw: HeadlessMainWindow, bgr_image: np.ndarray,
                      pad: float = 0.4, dedup_threshold=None) -> list:
    """Detect faces in a BGR image and return padded crops in reading order.

    Used by the ``find-faces`` helper so the distinct people in a group shot can
    be saved as numbered references and paired with source identities. Returns a
    list of dicts: ``{"crop": BGR ndarray, "bbox": (x1, y1, x2, y2)}``.
    """
    rgb = np.ascontiguousarray(bgr_image[..., ::-1])
    img = torch.from_numpy(rgb.astype("uint8")).to(mw.models_processor.device).permute(2, 0, 1)
    control = mw.control

    bboxes, kpss_5, _kpss = mw.models_processor.run_detect(
        img,
        control["DetectorModelSelection"],
        max_num=control["MaxFacesToDetectSlider"],
        score=control["DetectorScoreSlider"] / 100.0,
        input_size=(512, 512),
        use_landmark_detection=control["LandmarkDetectToggle"],
        landmark_detect_mode=control["LandmarkDetectModelSelection"],
        landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
        from_points=control["DetectFromPointsToggle"],
    )

    height, width = bgr_image.shape[:2]
    seen_embeddings = []
    faces = []
    for i in range(len(kpss_5)):
        if dedup_threshold is not None:
            emb, _ = mw.models_processor.run_recognize_direct(
                img, kpss_5[i], control["SimilarityTypeSelection"],
                control["RecognitionModelSelection"],
            )
            if any(mw.models_processor.findCosineDistance(emb, e) >= dedup_threshold
                   for e in seen_embeddings):
                continue
            seen_embeddings.append(emb)

        x1, y1, x2, y2 = (float(v) for v in bboxes[i][:4])
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(width, int(x2 + px)), min(height, int(y2 + py))
        crop = np.ascontiguousarray(bgr_image[cy1:cy2, cx1:cx2])
        faces.append({"crop": crop, "bbox": (cx1, cy1, cx2, cy2)})

    # Reading order: group into rough rows, then left-to-right.
    faces.sort(key=lambda f: (f["bbox"][1] // 50, f["bbox"][0]))
    return faces


# --------------------------------------------------------------------------- #
# Frame processing                                                             #
# --------------------------------------------------------------------------- #
_DUMMY_QUEUE = queue.Queue()


def process_rgb_frame(mw: HeadlessMainWindow, rgb_frame: np.ndarray, frame_number: int) -> np.ndarray:
    """Run the real per-frame swap pipeline on one RGB frame; returns BGR."""
    worker = FrameWorker(
        np.ascontiguousarray(rgb_frame.astype("uint8")),
        mw, frame_number, _DUMMY_QUEUE, is_single_frame=True,
    )
    worker.parameters = mw.parameters
    worker.is_view_face_compare = False
    worker.is_view_face_mask = False
    return np.ascontiguousarray(worker.process_frame())


def _ffmpeg_writer(width: int, height: int, fps: float, out_path: str) -> subprocess.Popen:
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p",
        "-c:v", "libx264", "-crf", "18",
        out_path,
    ]
    return subprocess.Popen(args, stdin=subprocess.PIPE)


def _mux_audio(video_only_path: str, source_media_path: str, out_path: str):
    """Copy the source's audio track onto the swapped video (audio optional)."""
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_only_path,
        "-i", source_media_path,
        "-c", "copy",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(args, check=False)


def swap_image(mw: HeadlessMainWindow, in_path: str, out_path: str):
    bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {in_path}")
    out_bgr = process_rgb_frame(mw, bgr[..., ::-1], 0)
    cv2.imwrite(out_path, out_bgr)


def swap_video(mw: HeadlessMainWindow, in_path: str, out_path: str,
               progress_every: int = 30):
    """Swap every frame of a video and mux the original audio back in."""
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    temp_path = str(Path(out_path).with_suffix(".video_only.mp4"))
    writer = None
    out_w = out_h = None
    frame_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            out_bgr = process_rgb_frame(mw, frame_bgr[..., ::-1], frame_idx)

            if writer is None:
                out_h, out_w = out_bgr.shape[:2]
                writer = _ffmpeg_writer(out_w, out_h, fps, temp_path)
            elif out_bgr.shape[0] != out_h or out_bgr.shape[1] != out_w:
                # Keep a constant raw-video size (e.g. if a restorer/enhancer
                # changes dimensions on some frames).
                out_bgr = np.ascontiguousarray(cv2.resize(out_bgr, (out_w, out_h)))

            writer.stdin.write(out_bgr.tobytes())
            frame_idx += 1

            if frame_idx % progress_every == 0:
                print(f"Processed {frame_idx}/{total or '?'} frames", flush=True)
                torch.cuda.empty_cache()
    finally:
        cap.release()
        if writer is not None:
            writer.stdin.close()
            writer.wait()
        gc.collect()
        torch.cuda.empty_cache()

    if writer is None:
        raise ValueError("No frames were read from the input video.")

    _mux_audio(temp_path, in_path, out_path)
    Path(temp_path).unlink(missing_ok=True)
    print(f"Done: {frame_idx} frames -> {out_path}", flush=True)
    return out_path
