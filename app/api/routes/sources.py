"""
GET  /api/sources/webcams
POST /api/sources/webcams/{index}/select
POST /api/sources/webrtc/start
POST /api/sources/webrtc/stop
GET  /api/sources/webrtc/status
PUT  /api/sources/transform
"""
from __future__ import annotations

import socket
from typing import List, Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state, get_video_processor
from app.api.schemas import (
    OkResponse,
    StreamTransformRequest,
    WebcamInfo,
    WebcamListResponse,
    WebRTCStartResponse,
)
from app.core.state import AppState, StreamTransform

router = APIRouter(prefix="/api/sources", tags=["sources"])

# Shared reference to the WebRTC subprocess (lives on the FastAPI app.state)
_webrtc_process = None


def _local_ip() -> str:
    """Best-effort local LAN IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@router.get("/webcams", response_model=WebcamListResponse)
def list_webcams(state: AppState = Depends(get_app_state)):
    """Enumerate available webcam devices."""
    from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
    backend_name = state.control.get("WebcamBackendSelection", "Default")
    backend = CAMERA_BACKENDS.get(backend_name, cv2.CAP_ANY)
    max_cams = int(state.control.get("WebcamMaxNoSelection", 3))

    webcams: List[WebcamInfo] = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            webcams.append(WebcamInfo(index=i, label=f"Webcam {i}"))
            cap.release()
    return WebcamListResponse(webcams=webcams)


@router.post("/webcams/{index}/select", response_model=OkResponse)
def select_webcam(
    index: int,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Open a webcam and make it the active source."""
    from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
    backend_name = state.control.get("WebcamBackendSelection", "Default")
    backend = CAMERA_BACKENDS.get(backend_name, cv2.CAP_ANY)
    res_str = state.control.get("WebcamMaxResSelection", "1280x720")
    try:
        w_str, h_str = res_str.split("x")
        res_w, res_h = int(w_str), int(h_str)
    except ValueError:
        res_w, res_h = 1280, 720

    vp.stop_processing()
    if vp.media_capture:
        vp.media_capture.release()

    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Cannot open webcam {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

    vp.media_capture = cap
    vp.file_type = "webcam"
    vp.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vp.max_frame_number = 999999
    vp.current_frame_number = 0
    vp.media_path = f"Webcam {index}"

    return OkResponse(message=f"Webcam {index} selected")


@router.post("/webrtc/start", response_model=WebRTCStartResponse)
def start_webrtc(
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Spawn the StreamRelay WebRTC server subprocess."""
    global _webrtc_process
    import multiprocessing
    from pathlib import Path
    from streamrelay.server import run_server

    SHM_NAME = "visomaster_webrtc_frame"
    http_port = int(state.control.get("WebRTCHttpPortText", 9091))
    https_port = int(state.control.get("WebRTCHttpsPortText", 9090))
    host = state.control.get("WebRTCBindAddressText", "0.0.0.0").strip() or "0.0.0.0"

    cert_dir = Path(__file__).parent.parent.parent / "ui" / "external" / "certificates"
    cert_file = str(cert_dir / "cert.pem")
    key_file = str(cert_dir / "key.pem")

    if _webrtc_process and _webrtc_process.is_alive():
        pass  # Already running — just return the URLs
    else:
        p = multiprocessing.Process(
            target=run_server,
            kwargs={
                "http_port": http_port,
                "https_port": https_port,
                "cert_file": cert_file,
                "key_file": key_file,
                "host": host,
                "shm_name": SHM_NAME,
            },
            daemon=True,
        )
        p.start()
        _webrtc_process = p

    # Set up VideoProcessor for WebRTC mode
    vp.stop_processing()
    vp.file_type = "webrtc"
    vp.media_capture = None
    vp.fps = 30.0
    vp.max_frame_number = 999999
    vp.current_frame_number = 0
    vp.media_path = "WebRTC"

    ip = _local_ip()
    return WebRTCStartResponse(
        http_url=f"http://{ip}:{http_port}/",
        https_url=f"https://{ip}:{https_port}/",
        whip_url=f"http://{ip}:{http_port}/whip",
        whip_https_url=f"https://{ip}:{https_port}/whip",
    )


@router.post("/webrtc/stop", response_model=OkResponse)
def stop_webrtc(vp=Depends(get_video_processor)):
    """Terminate the StreamRelay subprocess."""
    global _webrtc_process
    vp.stop_processing()
    if _webrtc_process and _webrtc_process.is_alive():
        _webrtc_process.terminate()
        _webrtc_process.join(timeout=3)
    _webrtc_process = None
    return OkResponse(message="WebRTC server stopped")


@router.get("/webrtc/status")
def webrtc_status(vp=Depends(get_video_processor)):
    """Return WebRTC server and frame-counter status."""
    global _webrtc_process
    running = bool(_webrtc_process and _webrtc_process.is_alive())
    frames = 0
    if vp.webrtc_shm is not None:
        import struct
        try:
            frames = struct.unpack_from("<I", vp.webrtc_shm.buf, 0)[0]
        except Exception:
            pass
    return {"running": running, "frames_received": frames}


@router.put("/transform", response_model=OkResponse)
def set_transform(
    body: StreamTransformRequest,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Set rotation/flip for the active streaming source (webcam or webrtc)."""
    transform = StreamTransform(
        rotation=body.rotation,
        flip_h=body.flip_h,
        flip_v=body.flip_v,
    )
    if vp.file_type == "webcam":
        state.webcam_transform = transform
    elif vp.file_type == "webrtc":
        state.webrtc_transform = transform
    else:
        raise HTTPException(status_code=400, detail="No active streaming source")
    return OkResponse(message="Transform updated")
