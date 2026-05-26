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
from fastapi import APIRouter, Depends, HTTPException, Request

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


def stop_webrtc_process(app_state_obj, vp) -> None:
    """Shared helper — tears down the StreamRelay subprocess and SHM.

    Called from both the REST route and the WS source_tab_changed handler
    so teardown logic is never duplicated.
    """
    vp.stop_processing()

    if vp.webrtc_shm is not None:
        try:
            vp.webrtc_shm.close()
        except Exception:
            pass
        vp.webrtc_shm = None

    proc = getattr(app_state_obj, "webrtc_process", None)
    if proc and proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)
    app_state_obj.webrtc_process = None

    vp.file_type = None


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
    """Open a webcam and make it the active source.

    Tries the configured backend first, then falls back through DSHOW → MSMF
    → CAP_ANY so that a device locked by another app (e.g. Teams, OBS) can
    still be opened via a sharing-capable backend.
    """
    from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
    backend_name = state.control.get("WebcamBackendSelection", "Default")
    preferred_backend = CAMERA_BACKENDS.get(backend_name, cv2.CAP_ANY)
    res_str = state.control.get("WebcamMaxResSelection", "1280x720")
    try:
        w_str, h_str = res_str.split("x")
        res_w, res_h = int(w_str), int(h_str)
    except ValueError:
        res_w, res_h = 1280, 720

    vp.stop_processing()
    if vp.media_capture:
        vp.media_capture.release()
        vp.media_capture = None
        # MSMF (Windows Media Foundation) needs a brief moment to fully release
        # the device before it can be reopened. Without this, the next
        # VideoCapture() call races the async teardown and raises -1072873821
        # (MF_E_INVALIDREQUEST). 200 ms is enough on all tested hardware.
        import time as _time
        _time.sleep(0.2)

    # Try backends in order: preferred → DSHOW (shares on Windows) → MSMF → ANY
    backends_to_try = [preferred_backend]
    for fallback in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        if fallback not in backends_to_try:
            backends_to_try.append(fallback)

    cap = None
    for backend in backends_to_try:
        try:
            c = cv2.VideoCapture(index, backend)
            if c.isOpened():
                cap = c
                break
            c.release()
        except Exception:
            pass

    if cap is None or not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot open webcam {index}. "
                "It may be exclusively locked by another application. "
                "Try closing Teams, OBS, or any other app using the camera."
            ),
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

    # Warm up the capture — MSMF/DSHOW sometimes returns black frames for the
    # first few reads after open. Drain up to 5 frames silently.
    for _ in range(5):
        ret, _ = cap.read()
        if ret:
            break

    vp.media_capture = cap
    vp.file_type = "webcam"
    vp.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vp.max_frame_number = 999999
    vp.current_frame_number = 0
    vp.media_path = f"Webcam {index}"

    return OkResponse(message=f"Webcam {index} selected")


@router.post("/webrtc/start", response_model=WebRTCStartResponse)
def start_webrtc(
    request: Request,
    state: AppState = Depends(get_app_state),
    vp=Depends(get_video_processor),
):
    """Spawn the StreamRelay WebRTC server and wire its frames into the VP via SHM."""
    import multiprocessing
    from pathlib import Path

    SHM_NAME = "visomaster_webrtc_frame"
    http_port  = int(state.control.get("WebRTCHttpPortText",  9091))
    https_port = int(state.control.get("WebRTCHttpsPortText", 9090))
    host = state.control.get("WebRTCBindAddressText", "0.0.0.0").strip() or "0.0.0.0"

    cert_dir  = Path(__file__).parent.parent.parent / "ui" / "external" / "certificates"
    cert_file = str(cert_dir / "cert.pem")
    key_file  = str(cert_dir / "key.pem")

    vp.stop_processing()

    vp.file_type             = "webrtc"
    vp.media_capture         = None
    vp.fps                   = 30.0
    vp.max_frame_number      = 999999
    vp.current_frame_number  = 0
    vp.media_path            = "WebRTC"
    vp._last_webrtc_counter  = 0

    existing = getattr(request.app.state, "webrtc_process", None)
    if not (existing and existing.is_alive()):
        from streamrelay.server import run_server
        p = multiprocessing.Process(
            target=run_server,
            kwargs={
                "http_port":  http_port,
                "https_port": https_port,
                "cert_file":  cert_file,
                "key_file":   key_file,
                "host":       host,
                "shm_name":   SHM_NAME,
            },
            daemon=True,
        )
        p.start()
        request.app.state.webrtc_process = p

    import time as _time
    from multiprocessing.shared_memory import SharedMemory

    if vp.webrtc_shm is not None:
        try:
            vp.webrtc_shm.close()
        except Exception:
            pass
        vp.webrtc_shm = None

    deadline = _time.monotonic() + 5.0
    while _time.monotonic() < deadline:
        try:
            vp.webrtc_shm = SharedMemory(name=SHM_NAME, create=False)
            print(f"[WebRTC] Attached shared memory '{SHM_NAME}'")
            break
        except FileNotFoundError:
            _time.sleep(0.1)

    if vp.webrtc_shm is None:
        print("[WebRTC] Warning: shared memory not ready yet; will retry in play loop.")

    vp.process_video()

    ip = _local_ip()
    return WebRTCStartResponse(
        http_url=f"http://{ip}:{http_port}/",
        https_url=f"https://{ip}:{https_port}/",
        ws_url=f"ws://{ip}:{http_port}/ws",
        wss_url=f"wss://{ip}:{https_port}/ws",
    )


@router.post("/webrtc/stop", response_model=OkResponse)
def stop_webrtc(
    request: Request,
    vp=Depends(get_video_processor),
):
    """Terminate the StreamRelay subprocess and release shared memory."""
    stop_webrtc_process(request.app.state, vp)
    return OkResponse(message="WebRTC server stopped")


@router.get("/webrtc/status")
def webrtc_status(request: Request, vp=Depends(get_video_processor)):
    """Return WebRTC server and frame-counter status."""
    proc = getattr(request.app.state, "webrtc_process", None)
    running = bool(proc and proc.is_alive())
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
    """Set rotation/flip for the active source (webcam, webrtc, video, or image)."""
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
        # video, image, or no source — store in media_transform
        state.media_transform = transform
    vp.process_current_frame()
    return OkResponse(message="Transform updated")
