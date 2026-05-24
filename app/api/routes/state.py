"""
GET  /api/state
PUT  /api/state/control
PUT  /api/state/parameters/{face_id}
POST /api/state/copy/{face_id}
POST /api/state/paste/{face_id}
POST /api/state/reset/{face_id}
"""
from __future__ import annotations

import copy
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state
from app.api.schemas import (
    ControlPatchRequest,
    OkResponse,
    ParameterPatchRequest,
    StateResponse,
)
from app.core.state import AppState
from app.helpers.miscellaneous import ParametersDict

router = APIRouter(prefix="/api/state", tags=["state"])


def _build_state_response(state: AppState) -> StateResponse:
    return StateResponse(
        selected_media_id=state.selected_media_id,
        selected_face_id=state.selected_face_id,
        control=state.control,
        target_media=[m.to_json() for m in state.target_media.values()],
        target_faces={fid: tf.to_json() for fid, tf in state.target_faces.items()},
        input_faces={fid: f.to_json() for fid, f in state.input_faces.items()},
        embeddings={eid: e.to_json() for eid, e in state.embeddings.items()},
        markers={str(pos): m.to_json() for pos, m in state.markers.items()},
        webcam_transform=state.webcam_transform.to_json(),
        webrtc_transform=state.webrtc_transform.to_json(),
        last_target_media_folder=state.last_target_media_folder,
        last_input_media_folder=state.last_input_media_folder,
        output_media_folder=state.output_media_folder,
    )


@router.get("", response_model=StateResponse)
def get_state(state: AppState = Depends(get_app_state)):
    """Return a full snapshot of the current application state."""
    return _build_state_response(state)


@router.put("/control", response_model=StateResponse)
def patch_control(
    body: ControlPatchRequest,
    state: AppState = Depends(get_app_state),
):
    """Patch one or more global control values."""
    for key, value in body.updates.items():
        state.set_control(key, value)
    return _build_state_response(state)


@router.put("/parameters/{face_id}", response_model=StateResponse)
def patch_parameters(
    face_id: str,
    body: ParameterPatchRequest,
    state: AppState = Depends(get_app_state),
):
    """Patch one or more per-face parameter values."""
    if face_id not in state.target_faces:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    for key, value in body.updates.items():
        state.set_parameter(face_id, key, value)
    return _build_state_response(state)


@router.post("/copy/{face_id}", response_model=OkResponse)
def copy_parameters(
    face_id: str,
    state: AppState = Depends(get_app_state),
):
    """Copy the parameters of a target face into the clipboard."""
    if face_id not in state.target_faces:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    params = state.get_parameters(face_id)
    state._clipboard_parameters = copy.deepcopy(
        params.data if isinstance(params, ParametersDict) else params
    )
    return OkResponse(message=f"Parameters copied from face {face_id}")


@router.post("/paste/{face_id}", response_model=StateResponse)
def paste_parameters(
    face_id: str,
    state: AppState = Depends(get_app_state),
):
    """Paste clipboard parameters onto a target face."""
    if face_id not in state.target_faces:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    clipboard: Dict[str, Any] = getattr(state, "_clipboard_parameters", None)
    if clipboard is None:
        raise HTTPException(status_code=400, detail="No parameters in clipboard. Call /copy first.")
    state.parameters[face_id] = ParametersDict(
        copy.deepcopy(clipboard), state.default_parameters
    )
    return _build_state_response(state)


@router.post("/reset/{face_id}", response_model=StateResponse)
def reset_parameters(
    face_id: str,
    state: AppState = Depends(get_app_state),
):
    """Reset a target face's parameters to defaults."""
    if face_id not in state.target_faces:
        raise HTTPException(status_code=404, detail=f"Target face '{face_id}' not found")
    state.parameters[face_id] = ParametersDict(
        copy.deepcopy(state.default_parameters), state.default_parameters
    )
    return _build_state_response(state)
