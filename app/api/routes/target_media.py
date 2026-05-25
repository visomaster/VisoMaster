"""
GET    /api/target-media
POST   /api/target-media/scan-folder
POST   /api/target-media/add-files
POST   /api/target-media/{media_id}/select
DELETE /api/target-media/{media_id}
GET    /api/target-media/{media_id}/thumbnail
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List

import cv2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import get_app_state, get_video_processor
from app.api.schemas import MediaCard, OkResponse, ScanFolderRequest, ScanFolderResponse, AddFilesRequest
from app.core.state import AppState, MediaRef
from app.helpers.miscellaneous import (
    get_image_files,
    get_video_files,
    get_file_type,
    get_hash_from_filename,
    get_thumbnail_path,
    save_thumbnail,
    ensure_thumbnail_dir,
    read_image_file,
)

router = APIRouter(prefix="/api/target-media", tags=["target-media"])


def _thumbnail_url(media_id: str) -> str:
    return f"/api/target-media/{media_id}/thumbnail"


def _ensure_thumbnail(media_path: str, file_type: str) -> str | None:
    """Generate and cache a thumbnail; return the file hash (used as cache key)."""
    try:
        file_hash = get_hash_from_filename(media_path)
        thumb_path = get_thumbnail_path(file_hash)
        if Path(thumb_path).is_file():
            return file_hash
        ensure_thumbnail_dir()
        frame = None
        if file_type == "image":
            frame = read_image_file(media_path)
        elif file_type == "video":
            cap = cv2.VideoCapture(media_path)
            ret, frame = cap.read()
            cap.release()
        if frame is not None:
            save_thumbnail(frame, thumb_path)
            return file_hash
    except Exception as exc:
        print(f"[thumbnail] Failed for {media_path}: {exc}")
    return None


@router.get("", response_model=List[MediaCard])
def list_target_media(state: AppState = Depends(get_app_state)):
    """Return all registered target media cards."""
    return [
        MediaCard(
            media_id=ref.media_id,
            media_path=ref.media_path,
            file_type=ref.file_type,
            thumbnail_url=_thumbnail_url(ref.media_id),
        )
        for ref in state.target_media.values()
    ]


@router.post("/scan-folder", response_model=ScanFolderResponse)
def scan_folder(
    body: ScanFolderRequest,
    state: AppState = Depends(get_app_state),
):
    """Scan a folder for video/image files and register them."""
    folder = Path(body.path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")

    video_files = get_video_files(str(folder), body.recursive)
    image_files = get_image_files(str(folder), body.recursive)
    all_files = video_files + image_files

    items: List[MediaCard] = []
    for file_path in all_files:
        file_type = get_file_type(file_path)
        if not file_type:
            continue
        media_id = str(uuid.uuid1().int)
        ref = MediaRef(media_id=media_id, media_path=file_path, file_type=file_type)
        state.target_media[media_id] = ref
        _ensure_thumbnail(file_path, file_type)
        items.append(MediaCard(
            media_id=media_id,
            media_path=file_path,
            file_type=file_type,
            thumbnail_url=_thumbnail_url(media_id),
        ))

    state.last_target_media_folder = str(folder)
    return ScanFolderResponse(items=items)


@router.post("/add-files", response_model=ScanFolderResponse)
def add_files(
    body: AddFilesRequest,
    state: AppState = Depends(get_app_state),
):
    """Register one or more individual file paths as target media."""
    items: List[MediaCard] = []
    for file_path in body.paths:
        p = Path(file_path)
        if not p.is_file():
            continue
        file_type = get_file_type(str(p))
        if not file_type:
            continue
        media_id = str(uuid.uuid1().int)
        ref = MediaRef(media_id=media_id, media_path=str(p), file_type=file_type)
        state.target_media[media_id] = ref
        _ensure_thumbnail(str(p), file_type)
        items.append(MediaCard(
            media_id=media_id,
            media_path=str(p),
            file_type=file_type,
            thumbnail_url=_thumbnail_url(media_id),
        ))
    return ScanFolderResponse(items=items)


@router.post("/{media_id}/select", response_model=OkResponse)
def select_media(
    media_id: str,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """
    Make a registered media item the active source.
    Opens the cv2.VideoCapture (or attaches WebRTC shm) on the VideoProcessor.
    """
    ref = state.target_media.get(media_id)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"Media '{media_id}' not found")

    # Stop any current processing
    vp.stop_processing()

    if ref.file_type == "video":
        cap = cv2.VideoCapture(ref.media_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Cannot open video: {ref.media_path}")
        vp.media_capture = cap
        vp.fps = cap.get(cv2.CAP_PROP_FPS)
        vp.max_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        vp.current_frame_number = 0

    elif ref.file_type == "image":
        vp.media_capture = None
        vp.fps = 0
        vp.max_frame_number = 0
        vp.current_frame_number = 0

    vp.media_path = ref.media_path
    vp.file_type = ref.file_type
    state.selected_media_id = media_id

    # For images, render immediately — there's no play loop, just a single frame.
    if ref.file_type == "image":
        vp.process_current_frame()

    return OkResponse(message=f"Selected media {media_id} ({ref.file_type})")


@router.delete("/{media_id}", response_model=OkResponse)
def delete_media(
    media_id: str,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Remove a media card from the list."""
    if media_id not in state.target_media:
        raise HTTPException(status_code=404, detail=f"Media '{media_id}' not found")
    if state.selected_media_id == media_id:
        vp.stop_processing()
        state.selected_media_id = None
    del state.target_media[media_id]
    return OkResponse(message=f"Removed media {media_id}")


@router.get("/{media_id}/thumbnail")
def get_thumbnail(
    media_id: str,
    state: AppState = Depends(get_app_state),
):
    """Return the cached thumbnail image for a media card."""
    ref = state.target_media.get(media_id)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"Media '{media_id}' not found")

    file_hash = get_hash_from_filename(ref.media_path)
    thumb_path = get_thumbnail_path(file_hash)

    if not Path(thumb_path).is_file():
        _ensure_thumbnail(ref.media_path, ref.file_type)

    if not Path(thumb_path).is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not available")

    media_type = "image/png" if thumb_path.endswith(".png") else "image/jpeg"
    return FileResponse(thumb_path, media_type=media_type)
