"""
GET  /api/playback
POST /api/playback/play
POST /api/playback/stop
POST /api/playback/seek
POST /api/playback/step
POST /api/playback/swap/enable
POST /api/playback/swap/disable
POST /api/playback/edit/enable
POST /api/playback/edit/disable
POST /api/playback/record/start
POST /api/playback/record/stop
POST /api/playback/save-frame
GET  /api/playback/markers
POST /api/playback/markers
DELETE /api/playback/markers/{frame_number}
GET  /api/preview/snapshot
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.api.deps import get_app_state, get_video_processor
from app.api.schemas import (
    MarkerInfo,
    MarkerListResponse,
    OkResponse,
    PlaybackState,
    PreviewSnapshotResponse,
    RecordStartRequest,
    RecordStopResponse,
    SeekRequest,
    StepRequest,
)
from app.core.state import AppState, Marker
from app.helpers.miscellaneous import get_output_file_path, is_ffmpeg_in_path, get_ffmpeg_path

router = APIRouter(tags=["playback"])


# ── Playback state ────────────────────────────────────────────────────────────

@router.get("/api/playback", response_model=PlaybackState)
def get_playback(
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    return PlaybackState(
        file_type=vp.file_type,
        fps=vp.fps,
        current_frame=vp.current_frame_number,
        max_frame=vp.max_frame_number,
        is_playing=vp.processing,
        is_recording=vp.recording,
        swap_enabled=state.control.get("_swap_enabled", False),
        edit_enabled=state.control.get("_edit_enabled", False),
        loop_enabled=state.loop_enabled,
    )


# ── Play / stop ───────────────────────────────────────────────────────────────

@router.post("/api/playback/play", response_model=OkResponse)
def play(vp=Depends(get_video_processor)):
    """Start playback (non-recording)."""
    if vp.processing:
        return OkResponse(message="Already playing")
    if vp.file_type is None:
        raise HTTPException(status_code=400, detail="No media selected")
    vp.process_video()
    return OkResponse(message="Playback started")


@router.post("/api/playback/stop", response_model=OkResponse)
def stop(vp=Depends(get_video_processor)):
    """Stop playback or recording."""
    vp.stop_processing()
    return OkResponse(message="Stopped")


# ── Seek / step ───────────────────────────────────────────────────────────────

@router.post("/api/playback/seek", response_model=OkResponse)
def seek(
    body: SeekRequest,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Seek to a specific frame and render a preview. Resumes playback if it was playing."""
    if vp.file_type not in ("video", "image"):
        raise HTTPException(status_code=400, detail="Seek only supported for video/image sources")
    frame_number = max(0, min(body.frame, vp.max_frame_number))
    was_playing = vp.processing
    vp.stop_processing()
    vp.current_frame_number = frame_number
    if vp.media_capture:
        vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # Apply marker overrides if present
    if frame_number in state.markers:
        m = state.markers[frame_number]
        state.parameters.update(m.parameters)
        state.control.update(m.control)
    if was_playing and vp.file_type == "video":
        vp.process_video()
    else:
        vp.process_current_frame()
    return OkResponse(message=f"Seeked to frame {frame_number}")


@router.post("/api/playback/step", response_model=OkResponse)
def step(
    body: StepRequest,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Advance or rewind by N frames."""
    if vp.file_type not in ("video",):
        raise HTTPException(status_code=400, detail="Step only supported for video sources")
    new_frame = max(0, min(vp.current_frame_number + body.n, vp.max_frame_number))
    vp.stop_processing()
    vp.current_frame_number = new_frame
    if vp.media_capture:
        vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
    vp.process_current_frame()
    return OkResponse(message=f"Stepped to frame {new_frame}")


# ── Swap / edit toggles ───────────────────────────────────────────────────────

@router.post("/api/playback/swap/enable", response_model=OkResponse)
def enable_swap(state: AppState = Depends(get_app_state), vp=Depends(get_video_processor)):
    state.control["_swap_enabled"] = True
    state.control["_edit_enabled"] = False
    vp.process_current_frame()
    return OkResponse(message="Swap enabled")


@router.post("/api/playback/swap/disable", response_model=OkResponse)
def disable_swap(state: AppState = Depends(get_app_state), vp=Depends(get_video_processor)):
    state.control["_swap_enabled"] = False
    vp.process_current_frame()
    return OkResponse(message="Swap disabled")


@router.post("/api/playback/edit/enable", response_model=OkResponse)
def enable_edit(state: AppState = Depends(get_app_state), vp=Depends(get_video_processor)):
    state.control["_edit_enabled"] = True
    state.control["_swap_enabled"] = False
    vp.process_current_frame()
    return OkResponse(message="Edit enabled")


@router.post("/api/playback/edit/disable", response_model=OkResponse)
def disable_edit(state: AppState = Depends(get_app_state), vp=Depends(get_video_processor)):
    state.control["_edit_enabled"] = False
    vp.process_current_frame()
    return OkResponse(message="Edit disabled")


# ── Loop ──────────────────────────────────────────────────────────────────────

@router.post("/api/playback/loop/enable", response_model=OkResponse)
def enable_loop(state: AppState = Depends(get_app_state)):
    state.loop_enabled = True
    return OkResponse(message="Loop enabled")


@router.post("/api/playback/loop/disable", response_model=OkResponse)
def disable_loop(state: AppState = Depends(get_app_state)):
    state.loop_enabled = False
    return OkResponse(message="Loop disabled")


# ── Recording ─────────────────────────────────────────────────────────────────

@router.post("/api/playback/record/start", response_model=OkResponse)
def record_start(
    body: RecordStartRequest,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Start recording the processed video to disk."""
    if vp.file_type != "video":
        detail = (
            "No video loaded. Please select a video file before recording."
            if vp.file_type is None
            else f"Recording is only supported for video files (current source: {vp.file_type})."
        )
        raise HTTPException(status_code=400, detail=detail)
    if not is_ffmpeg_in_path():
        raise HTTPException(status_code=500, detail=
            f"FFmpeg not found. Expected at '{get_ffmpeg_path()}' or on system PATH.")

    output_folder = body.output_folder or state.control.get("OutputMediaFolder", "")
    if not output_folder:
        raise HTTPException(status_code=400, detail="No output folder configured")
    if not Path(output_folder).is_dir():
        raise HTTPException(status_code=400, detail=f"Output folder does not exist: {output_folder}")

    # Auto-stop playback if running (mirrors Qt UI behaviour)
    if vp.processing:
        vp.stop_processing()

    state.control["OutputMediaFolder"] = output_folder
    vp.recording = True
    vp.process_video()
    return OkResponse(message="Recording started")


@router.post("/api/playback/record/stop", response_model=RecordStopResponse)
def record_stop(
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Stop recording and return the output file path."""
    if not vp.recording:
        raise HTTPException(status_code=400, detail="Not currently recording")
    output_path = get_output_file_path(
        vp.media_path, state.control.get("OutputMediaFolder", ".")
    )
    vp.stop_processing()
    return RecordStopResponse(output_path=output_path)


# ── Save current frame ────────────────────────────────────────────────────────

@router.post("/api/playback/save-frame", response_model=RecordStopResponse)
def save_frame(
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Save the current processed frame as a PNG to the output folder."""
    output_folder = state.control.get("OutputMediaFolder", "")
    if not output_folder:
        raise HTTPException(status_code=400, detail="No output folder configured")
    frame = vp.current_frame
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise HTTPException(status_code=400, detail="No frame available")
    output_path = get_output_file_path(
        vp.media_path or "snapshot.png", output_folder, media_type="image"
    )
    cv2.imwrite(output_path, frame)
    return RecordStopResponse(output_path=output_path)


# ── Markers ───────────────────────────────────────────────────────────────────

@router.get("/api/playback/markers", response_model=MarkerListResponse)
def list_markers(state: AppState = Depends(get_app_state)):
    return MarkerListResponse(markers=sorted(state.markers.keys()))


@router.post("/api/playback/markers", response_model=OkResponse)
def add_marker(
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Add a marker at the current frame position."""
    if vp.file_type != "video":
        raise HTTPException(status_code=400, detail="Markers only supported for video sources")
    if not state.target_faces:
        raise HTTPException(status_code=400, detail="No target faces — add at least one before creating a marker")
    pos = vp.current_frame_number
    if pos in state.markers:
        raise HTTPException(status_code=409, detail=f"Marker already exists at frame {pos}")
    state.markers[pos] = Marker(
        frame_number=pos,
        parameters=copy.deepcopy({
            fid: (p.data if hasattr(p, "data") else p)
            for fid, p in state.parameters.items()
        }),
        control=state.control.copy(),
    )
    return OkResponse(message=f"Marker added at frame {pos}")


@router.delete("/api/playback/markers/{frame_number}", response_model=OkResponse)
def delete_marker(frame_number: int, state: AppState = Depends(get_app_state)):
    if frame_number not in state.markers:
        raise HTTPException(status_code=404, detail=f"No marker at frame {frame_number}")
    del state.markers[frame_number]
    return OkResponse(message=f"Marker removed from frame {frame_number}")


# ── Open / reveal saved files ─────────────────────────────────────────────────

@router.post("/api/system/open-file", response_model=OkResponse)
def open_file(body: dict, state: AppState = Depends(get_app_state)):
    """Open a file with the default OS application (server-side)."""
    import sys
    import subprocess
    path = body.get("path", "")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=400, detail=f"File not found: {path}")
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return OkResponse(message="Opened")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/system/reveal-in-folder", response_model=OkResponse)
def reveal_in_folder(body: dict, state: AppState = Depends(get_app_state)):
    """Reveal a file in the OS file explorer (server-side)."""
    import sys
    import subprocess
    path = body.get("path", "")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=400, detail=f"File not found: {path}")
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", str(Path(path).parent)])
        return OkResponse(message="Revealed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/preview/snapshot")
def preview_snapshot(vp=Depends(get_video_processor)):
    """
    Return the latest processed frame as a JPEG binary response.
    The React UI can poll this or use the WebSocket stream instead.
    """
    frame = vp.current_frame
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise HTTPException(status_code=404, detail="No frame available yet")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")
