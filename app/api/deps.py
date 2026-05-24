"""
app/api/deps.py
───────────────
FastAPI dependency injection helpers.

The server module stores the live AppState, ModelsProcessor, and
VideoProcessor on the FastAPI `app.state` object.  Every route that
needs them calls `Depends(get_app_state)` etc. rather than importing
globals, which keeps the code testable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from app.core.state import AppState
    from app.processors.models_processor import ModelsProcessor
    from app.processors.video_processor import VideoProcessor


def get_app_state(request: Request) -> "AppState":
    return request.app.state.app_state


def get_models_processor(request: Request) -> "ModelsProcessor":
    return request.app.state.models_processor


def get_video_processor(request: Request) -> "VideoProcessor":
    return request.app.state.video_processor
