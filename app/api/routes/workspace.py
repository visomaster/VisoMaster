"""
GET  /api/workspace
POST /api/workspace/save
POST /api/workspace/load
POST /api/workspace/reset
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state
from app.api.schemas import OkResponse, WorkspaceLoadRequest, WorkspaceSaveRequest
from app.core.state import AppState

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


@router.get("")
def get_workspace(state: AppState = Depends(get_app_state)):
    """Return the current workspace as a JSON-serialisable dict."""
    return state.to_json()


@router.post("/save", response_model=OkResponse)
def save_workspace(
    body: WorkspaceSaveRequest,
    state: AppState = Depends(get_app_state),
):
    """Save the current workspace to a JSON file."""
    path = Path(body.filename)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_json(), f, indent=4)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return OkResponse(message=f"Workspace saved to {path}")


@router.post("/load", response_model=OkResponse)
def load_workspace(
    body: WorkspaceLoadRequest,
    state: AppState = Depends(get_app_state),
):
    """
    Load a workspace JSON file into the current state.
    Media files are registered but NOT opened — the client must call
    POST /api/target-media/{media_id}/select to activate one.
    """
    path = Path(body.filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    loaded = AppState.from_json(data, state.default_parameters)

    # Merge loaded state into the live state in-place so all references stay valid
    state.control.update(loaded.control)
    state.parameters = loaded.parameters
    state.target_media = loaded.target_media
    state.target_faces = loaded.target_faces
    state.input_faces = loaded.input_faces
    state.embeddings = loaded.embeddings
    state.markers = loaded.markers
    state.selected_media_id = loaded.selected_media_id
    state.webcam_transform = loaded.webcam_transform
    state.webrtc_transform = loaded.webrtc_transform
    state.last_target_media_folder = loaded.last_target_media_folder
    state.last_input_media_folder = loaded.last_input_media_folder
    state.loaded_embedding_filename = loaded.loaded_embedding_filename
    state.current_widget_parameters = loaded.current_widget_parameters

    return OkResponse(message=f"Workspace loaded from {path}")


@router.post("/reset", response_model=OkResponse)
def reset_workspace(state: AppState = Depends(get_app_state)):
    """Clear all working-set data (media, faces, embeddings, markers)."""
    state.target_media.clear()
    state.target_faces.clear()
    state.input_faces.clear()
    state.embeddings.clear()
    state.markers.clear()
    state.parameters.clear()
    state.selected_media_id = None
    state.selected_face_id = None
    return OkResponse(message="Workspace reset")
