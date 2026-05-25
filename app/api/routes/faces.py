"""
POST   /api/target-faces/find
GET    /api/target-faces
POST   /api/target-faces/{face_id}/select
POST   /api/target-faces/{face_id}/assign-input/{input_face_id}
DELETE /api/target-faces/{face_id}/assign-input/{input_face_id}
POST   /api/target-faces/{face_id}/assign-embedding/{embedding_id}
DELETE /api/target-faces/{face_id}/assign-embedding/{embedding_id}
DELETE /api/target-faces/{face_id}
POST   /api/target-faces/clear
GET    /api/target-faces/{face_id}/thumbnail

GET    /api/input-faces
POST   /api/input-faces/scan-folder
DELETE /api/input-faces/{face_id}
GET    /api/input-faces/{face_id}/thumbnail
POST   /api/input-faces/clear
"""
from __future__ import annotations

import struct
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from app.api.deps import get_app_state, get_models_processor, get_video_processor
from app.api.schemas import (
    AssignEmbeddingRequest,
    AssignFaceRequest,
    FaceCard,
    FindFacesResponse,
    InputFaceCard,
    OkResponse,
    ScanFolderRequest,
    ScanInputFolderResponse,
)
from app.core.state import AppState, EmbeddingStore, InputFace, TargetFace
from app.helpers.miscellaneous import (
    ensure_thumbnail_dir,
    get_hash_from_filename,
    get_image_files,
    get_thumbnail_path,
    read_image_file,
    save_thumbnail,
)


router = APIRouter(tags=["faces"])

# ── helpers ───────────────────────────────────────────────────────────────────

def _face_thumbnail_url(face_id: str, kind: str = "target") -> str:
    return f"/api/{kind}-faces/{face_id}/thumbnail"


def _save_face_thumbnail(face_id: str, cropped_bgr: np.ndarray, kind: str = "target") -> str:
    """Save a face crop as a thumbnail and return the path."""
    ensure_thumbnail_dir()
    thumb_dir = Path(".thumbnails")
    thumb_path = str(thumb_dir / f"face_{kind}_{face_id}.png")
    save_thumbnail(cropped_bgr, thumb_path)
    return thumb_path


def _recognition_model_options() -> List[str]:
    """Return the list of recognition model names from settings layout data."""
    try:
        from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
        return SETTINGS_LAYOUT_DATA["Face Recognition"]["RecognitionModelSelection"]["options"]
    except Exception:
        return ["Inswapper128ArcFace", "SimSwapArcFace", "GhostArcFace", "CSCSArcFace"]


def _compute_assigned_embedding(
    state: AppState,
    target_face: TargetFace,
    merge_method: str = "Mean",
) -> EmbeddingStore:
    """Average/median all assigned source embeddings into one EmbeddingStore."""
    all_stores: List[EmbeddingStore] = []
    for fid in target_face.assigned_input_face_ids:
        if fid in state.input_faces:
            all_stores.append(state.input_faces[fid].embedding_store)
    for eid in target_face.assigned_embedding_ids:
        if eid in state.embeddings:
            all_stores.append(state.embeddings[eid].embedding_store)

    if not all_stores:
        return EmbeddingStore()

    result: dict[str, np.ndarray] = {}
    all_models = set()
    for s in all_stores:
        all_models.update(s.store.keys())

    for model in all_models:
        vecs = [s.store[model] for s in all_stores if model in s.store]
        if not vecs:
            continue
        stacked = np.stack(vecs, axis=0)
        if merge_method == "Median":
            result[model] = np.median(stacked, axis=0).astype(np.float32)
        else:
            result[model] = np.mean(stacked, axis=0).astype(np.float32)

    return EmbeddingStore(store=result)


# ── Target faces ──────────────────────────────────────────────────────────────

@router.post("/api/target-faces/find", response_model=FindFacesResponse)
def find_target_faces(
    state: AppState = Depends(get_app_state),
    mp=Depends(get_models_processor),
    vp=Depends(get_video_processor),
):
    """
    Run face detection on the current frame and register new target faces.
    Skips faces already registered (cosine similarity above threshold).

    When a video is actively playing we use the last processed frame stored in
    vp.current_frame rather than calling media_capture.read() — the capture is
    not thread-safe and concurrent reads trigger the FFmpeg async_lock assertion.
    """
    from app.helpers.miscellaneous import read_frame as _read_frame_locked

    control = state.control

    # ── Read current frame ────────────────────────────────────────────────
    frame: np.ndarray | None = None

    if vp.file_type == "image" and vp.media_path:
        frame = read_image_file(vp.media_path)

    elif vp.file_type == "video":
        # If the play loop is running, grab the last delivered frame instead of
        # touching the capture object from this thread.
        if vp.processing and isinstance(vp.current_frame, np.ndarray) and vp.current_frame.size > 0:
            frame = vp.current_frame  # already BGR
        elif vp.media_capture:
            ret, frame = _read_frame_locked(vp.media_capture)
            if ret:
                vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, vp.current_frame_number)
            else:
                frame = None

    elif vp.file_type == "webcam" and vp.media_capture:
        ret, frame = _read_frame_locked(vp.media_capture)
        if not ret:
            frame = None

    elif vp.file_type == "webrtc" and vp.webrtc_shm is not None:
        from streamrelay.protocol import SHM_HEADER_BYTES
        w = struct.unpack_from("<I", vp.webrtc_shm.buf, 4)[0]
        h = struct.unpack_from("<I", vp.webrtc_shm.buf, 8)[0]
        if w > 0 and h > 0:
            raw = bytes(vp.webrtc_shm.buf[SHM_HEADER_BYTES: SHM_HEADER_BYTES + w * h * 3])
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()

    # Fall back to last known frame if nothing else is available
    if frame is None and isinstance(getattr(vp, "current_frame", None), np.ndarray) and vp.current_frame.size > 0:
        frame = vp.current_frame

    if frame is None:
        raise HTTPException(status_code=400, detail="No frame available. Select a media source and play or seek first.")

    # BGR → RGB → CHW tensor
    frame_rgb = frame[..., ::-1]
    img = torch.from_numpy(frame_rgb.astype("uint8")).to(mp.device).permute(2, 0, 1)

    # ── Detect ────────────────────────────────────────────────────────────
    detector = control.get("DetectorModelSelection", "RetinaFace")
    max_faces = int(control.get("MaxFacesToDetectSlider", 20))
    det_score = float(control.get("DetectorScoreSlider", 50)) / 100.0
    use_lmk = bool(control.get("LandmarkDetectToggle", False))
    lmk_mode = control.get("LandmarkDetectModelSelection", "203")
    lmk_score = float(control.get("LandmarkDetectScoreSlider", 50)) / 100.0
    from_pts = bool(control.get("DetectFromPointsToggle", False))
    auto_rot = bool(control.get("AutoRotationToggle", False))
    rotation_angles = [0, 90, 180, 270] if auto_rot else [0]

    _, kpss_5, _ = mp.run_detect(
        img, detector, max_num=max_faces, score=det_score,
        input_size=(512, 512),
        use_landmark_detection=use_lmk,
        landmark_detect_mode=lmk_mode,
        landmark_score=lmk_score,
        from_points=from_pts,
        rotation_angles=rotation_angles,
    )

    recognition_model = control.get("RecognitionModelSelection", "Inswapper128ArcFace")
    similarity_type = control.get("SimilarityTypeSelection", "Opal")

    # Clear existing target faces and parameters so Find always gives a fresh
    # result from the current frame rather than skipping already-registered faces.
    state.target_faces.clear()
    state.parameters.clear()
    state.selected_face_id = None

    new_faces: List[FaceCard] = []

    for face_kps in kpss_5:
        face_emb, cropped_img = mp.run_recognize_direct(img, face_kps, similarity_type, recognition_model)

        # Only store the embedding for the currently selected recognition model.
        # Other model embeddings are computed on-demand during swapping.
        emb_store: dict[str, np.ndarray] = {recognition_model: face_emb}

        face_id = str(uuid.uuid1().int)
        cropped_bgr = cropped_img.cpu().numpy()[..., ::-1]
        cropped_bgr = np.ascontiguousarray(cropped_bgr)

        tf = TargetFace(
            face_id=face_id,
            cropped_face=cropped_bgr,
            embedding_store=EmbeddingStore(store=emb_store),
        )
        state.target_faces[face_id] = tf
        _save_face_thumbnail(face_id, cropped_bgr, "target")

        new_faces.append(FaceCard(
            face_id=face_id,
            thumbnail_url=_face_thumbnail_url(face_id, "target"),
        ))

    # Auto-select first face if none selected
    if state.target_faces and not state.selected_face_id:
        state.selected_face_id = next(iter(state.target_faces))

    return FindFacesResponse(found=len(new_faces), faces=new_faces)


@router.get("/api/target-faces", response_model=List[FaceCard])
def list_target_faces(state: AppState = Depends(get_app_state)):
    return [
        FaceCard(
            face_id=fid,
            thumbnail_url=_face_thumbnail_url(fid, "target"),
            assigned_input_face_ids=tf.assigned_input_face_ids,
            assigned_embedding_ids=tf.assigned_embedding_ids,
        )
        for fid, tf in state.target_faces.items()
    ]


@router.post("/api/target-faces/{face_id}/select", response_model=OkResponse)
def select_target_face(face_id: str, state: AppState = Depends(get_app_state)):
    if face_id not in state.target_faces:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    state.selected_face_id = face_id
    return OkResponse(message=f"Selected face {face_id}")


@router.post("/api/target-faces/{face_id}/assign-input/{input_face_id}", response_model=OkResponse)
def assign_input_face(
    face_id: str,
    input_face_id: str,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    tf = state.target_faces.get(face_id)
    if tf is None:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    if input_face_id not in state.input_faces:
        raise HTTPException(status_code=404, detail=f"Input face '{input_face_id}' not found")
    if input_face_id not in tf.assigned_input_face_ids:
        tf.assigned_input_face_ids.append(input_face_id)
    merge_method = state.control.get("EmbMergeMethodSelection", "Mean")
    tf.assigned_input_embedding = _compute_assigned_embedding(state, tf, merge_method)
    vp.process_current_frame()
    return OkResponse(message=f"Assigned input face {input_face_id} to target {face_id}")


@router.delete("/api/target-faces/{face_id}/assign-input/{input_face_id}", response_model=OkResponse)
def unassign_input_face(
    face_id: str,
    input_face_id: str,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    tf = state.target_faces.get(face_id)
    if tf is None:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    tf.assigned_input_face_ids = [x for x in tf.assigned_input_face_ids if x != input_face_id]
    merge_method = state.control.get("EmbMergeMethodSelection", "Mean")
    tf.assigned_input_embedding = _compute_assigned_embedding(state, tf, merge_method)
    vp.process_current_frame()
    return OkResponse(message=f"Unassigned input face {input_face_id} from target {face_id}")


@router.post("/api/target-faces/{face_id}/assign-embedding/{embedding_id}", response_model=OkResponse)
def assign_embedding(
    face_id: str,
    embedding_id: str,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    tf = state.target_faces.get(face_id)
    if tf is None:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    if embedding_id not in state.embeddings:
        raise HTTPException(status_code=404, detail=f"Embedding '{embedding_id}' not found")
    if embedding_id not in tf.assigned_embedding_ids:
        tf.assigned_embedding_ids.append(embedding_id)
    merge_method = state.control.get("EmbMergeMethodSelection", "Mean")
    tf.assigned_input_embedding = _compute_assigned_embedding(state, tf, merge_method)
    vp.process_current_frame()
    return OkResponse(message=f"Assigned embedding {embedding_id} to target {face_id}")


@router.delete("/api/target-faces/{face_id}/assign-embedding/{embedding_id}", response_model=OkResponse)
def unassign_embedding(
    face_id: str,
    embedding_id: str,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    tf = state.target_faces.get(face_id)
    if tf is None:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    tf.assigned_embedding_ids = [x for x in tf.assigned_embedding_ids if x != embedding_id]
    merge_method = state.control.get("EmbMergeMethodSelection", "Mean")
    tf.assigned_input_embedding = _compute_assigned_embedding(state, tf, merge_method)
    vp.process_current_frame()
    return OkResponse(message=f"Unassigned embedding {embedding_id} from target {face_id}")


@router.delete("/api/target-faces/{face_id}", response_model=OkResponse)
def delete_target_face(face_id: str, state: AppState = Depends(get_app_state)):
    if face_id not in state.target_faces:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    del state.target_faces[face_id]
    state.parameters.pop(face_id, None)
    if state.selected_face_id == face_id:
        state.selected_face_id = next(iter(state.target_faces), None)
    return OkResponse(message=f"Removed target face {face_id}")


@router.post("/api/target-faces/clear", response_model=OkResponse)
def clear_target_faces(state: AppState = Depends(get_app_state)):
    state.target_faces.clear()
    state.parameters.clear()
    state.selected_face_id = None
    return OkResponse(message="All target faces cleared")


@router.get("/api/target-faces/{face_id}/thumbnail")
def target_face_thumbnail(face_id: str, state: AppState = Depends(get_app_state)):
    tf = state.target_faces.get(face_id)
    if tf is None:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    thumb_path = Path(".thumbnails") / f"face_target_{face_id}.png"
    if not thumb_path.is_file():
        # Regenerate from stored crop
        _save_face_thumbnail(face_id, tf.cropped_face, "target")
    if not thumb_path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    return FileResponse(str(thumb_path), media_type="image/png")


# ── Input faces ───────────────────────────────────────────────────────────────

@router.get("/api/input-faces", response_model=List[InputFaceCard])
def list_input_faces(state: AppState = Depends(get_app_state)):
    return [
        InputFaceCard(
            face_id=fid,
            media_path=f.media_path,
            thumbnail_url=_face_thumbnail_url(fid, "input"),
        )
        for fid, f in state.input_faces.items()
    ]


@router.post("/api/input-faces/scan-folder", response_model=ScanInputFolderResponse)
def scan_input_folder(
    body: ScanFolderRequest,
    state: AppState = Depends(get_app_state),
    mp=Depends(get_models_processor),
):
    """
    Scan a folder for face images, run detection + recognition on each,
    and register them as input faces.
    """
    folder = Path(body.path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")

    image_files = get_image_files(str(folder), body.recursive)
    control = state.control
    recognition_model = control.get("RecognitionModelSelection", "Inswapper128ArcFace")
    similarity_type = control.get("SimilarityTypeSelection", "Opal")

    items: List[InputFaceCard] = []
    for file_path in image_files:
        frame = read_image_file(file_path)
        if frame is None:
            continue
        frame_rgb = frame[..., ::-1]
        img = torch.from_numpy(frame_rgb.astype("uint8")).to(mp.device).permute(2, 0, 1)

        try:
            _, kpss_5, _ = mp.run_detect(
                img, control.get("DetectorModelSelection", "RetinaFace"),
                max_num=1, score=0.5, input_size=(512, 512),
            )
        except Exception:
            continue

        if len(kpss_5) == 0:
            continue

        face_kps = kpss_5[0]
        # Only embed with the currently selected model; others are loaded on-demand during swapping.
        emb, cropped = mp.run_recognize_direct(img, face_kps, similarity_type, recognition_model)
        emb_store: dict[str, np.ndarray] = {recognition_model: emb}

        face_id = str(uuid.uuid1().int)
        cropped_bgr = cropped.cpu().numpy()[..., ::-1]
        cropped_bgr = np.ascontiguousarray(cropped_bgr)

        state.input_faces[face_id] = InputFace(
            face_id=face_id,
            media_path=file_path,
            embedding_store=EmbeddingStore(store=emb_store),
            cropped_face=cropped_bgr,
        )
        _save_face_thumbnail(face_id, cropped_bgr, "input")
        items.append(InputFaceCard(
            face_id=face_id,
            media_path=file_path,
            thumbnail_url=_face_thumbnail_url(face_id, "input"),
        ))

    state.last_input_media_folder = str(folder)
    return ScanInputFolderResponse(items=items)


@router.delete("/api/input-faces/{face_id}", response_model=OkResponse)
def delete_input_face(face_id: str, state: AppState = Depends(get_app_state)):
    if face_id not in state.input_faces:
        raise HTTPException(status_code=404, detail=f"Input face '{face_id}' not found")
    del state.input_faces[face_id]
    # Remove from any target face assignments
    for tf in state.target_faces.values():
        if face_id in tf.assigned_input_face_ids:
            tf.assigned_input_face_ids.remove(face_id)
            merge_method = state.control.get("EmbMergeMethodSelection", "Mean")
            tf.assigned_input_embedding = _compute_assigned_embedding(state, tf, merge_method)
    return OkResponse(message=f"Removed input face {face_id}")


@router.post("/api/input-faces/clear", response_model=OkResponse)
def clear_input_faces(state: AppState = Depends(get_app_state)):
    state.input_faces.clear()
    for tf in state.target_faces.values():
        tf.assigned_input_face_ids.clear()
        tf.assigned_input_embedding = EmbeddingStore()
    return OkResponse(message="All input faces cleared")


@router.get("/api/input-faces/{face_id}/thumbnail")
def input_face_thumbnail(face_id: str, state: AppState = Depends(get_app_state)):
    face = state.input_faces.get(face_id)
    if face is None:
        raise HTTPException(status_code=404, detail=f"Input face '{face_id}' not found")
    thumb_path = Path(".thumbnails") / f"face_input_{face_id}.png"
    if not thumb_path.is_file() and face.cropped_face is not None:
        _save_face_thumbnail(face_id, face.cropped_face, "input")
    if not thumb_path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    return FileResponse(str(thumb_path), media_type="image/png")
