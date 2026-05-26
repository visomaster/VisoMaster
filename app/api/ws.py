"""
app/api/ws.py
─────────────
WebSocket endpoints:

  /ws/events   — bidirectional JSON control channel
  /ws/preview  — server-push binary JPEG frame stream

────────────────────────────────────────────────────────────────────────────
/ws/events — server → client JSON events:
  { "type": "frame_processed",  "payload": { "frame_number": N, "width": W, "height": H } }
  { "type": "fps_update",       "payload": { "fps": 29.7 } }
  { "type": "state_updated",    "payload": { "section": "...", ... } }
  { "type": "playback_state",   "payload": { "is_playing": bool, "is_recording": bool,
                                              "current_frame": N, "max_frame": N, "fps": F } }
  { "type": "error",            "payload": { "message": "..." } }
  { "type": "pong" }

/ws/events — client → server JSON commands:
  { "type": "play" }
  { "type": "stop" }
  { "type": "seek",          "payload": { "frame": N } }
  { "type": "step",          "payload": { "n": N } }          # negative = rewind
  { "type": "set_control",   "payload": { "name": "...", "value": ... } }
  { "type": "set_parameter", "payload": { "face_id": "...", "name": "...", "value": ... } }
  { "type": "swap_enable" }
  { "type": "swap_disable" }
  { "type": "edit_enable" }
  { "type": "edit_disable" }
  { "type": "preview_quality", "payload": { "quality": 75 } }  # JPEG quality 1-100
  { "type": "source_tab_changed", "payload": { "source": "media"|"webcam"|"streaming" } }
  { "type": "ping" }

────────────────────────────────────────────────────────────────────────────
/ws/preview — server pushes raw JPEG bytes as binary WebSocket messages.
  No client → server messages expected on this channel.
  The client renders each message as an image:

    const ws = new WebSocket('ws://localhost:8000/ws/preview');
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (e) => {
      const blob = new Blob([e.data], { type: 'image/jpeg' });
      img.src = URL.createObjectURL(blob);
    };
"""
from __future__ import annotations

import asyncio
import json

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.events import bus

router = APIRouter()


# ── /ws/events ────────────────────────────────────────────────────────────────

@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    client_q = bus.subscribe()

    async def _sender():
        """Forward JSON events from the bus to this client."""
        while True:
            msg = await client_q.get()
            try:
                await websocket.send_text(msg)
            except Exception:
                break

    sender_task = asyncio.create_task(_sender())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                continue

            cmd_type = cmd.get("type", "")
            payload  = cmd.get("payload", {})

            app   = websocket.app
            state = app.state.app_state
            vp    = app.state.video_processor

            match cmd_type:

                # ── Playback ──────────────────────────────────────────────
                case "play":
                    if not vp.processing and vp.file_type:
                        vp.process_video()
                        _emit_playback_state(vp, state)

                case "stop":
                    vp.stop_processing()
                    _emit_playback_state(vp, state)

                case "seek":
                    frame = int(payload.get("frame", 0))
                    frame = max(0, min(frame, vp.max_frame_number))
                    was_playing = vp.processing
                    vp.stop_processing()
                    vp.current_frame_number = frame
                    if vp.media_capture:
                        vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    # Apply marker overrides if present
                    if frame in state.markers:
                        m = state.markers[frame]
                        state.parameters.update(m.parameters)
                        state.control.update(m.control)
                    if was_playing and vp.file_type == "video":
                        vp.process_video()
                    else:
                        vp.process_current_frame()
                    _emit_playback_state(vp, state)

                case "step":
                    n = int(payload.get("n", 1))
                    new_frame = max(0, min(vp.current_frame_number + n, vp.max_frame_number))
                    vp.stop_processing()
                    vp.current_frame_number = new_frame
                    if vp.media_capture:
                        vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    vp.process_current_frame()
                    _emit_playback_state(vp, state)

                # ── Swap / edit toggles ───────────────────────────────────
                case "swap_enable":
                    state.control["_swap_enabled"] = True
                    state.control["_edit_enabled"] = False
                    vp.process_current_frame()
                    bus.emit_sync("state_updated", {"section": "control",
                                                    "name": "_swap_enabled", "value": True})

                case "swap_disable":
                    state.control["_swap_enabled"] = False
                    vp.process_current_frame()
                    bus.emit_sync("state_updated", {"section": "control",
                                                    "name": "_swap_enabled", "value": False})

                case "edit_enable":
                    state.control["_edit_enabled"] = True
                    state.control["_swap_enabled"] = False
                    vp.process_current_frame()
                    bus.emit_sync("state_updated", {"section": "control",
                                                    "name": "_edit_enabled", "value": True})

                case "edit_disable":
                    state.control["_edit_enabled"] = False
                    vp.process_current_frame()
                    bus.emit_sync("state_updated", {"section": "control",
                                                    "name": "_edit_enabled", "value": False})

                # ── State mutations ───────────────────────────────────────
                case "set_control":
                    name  = payload.get("name")
                    value = payload.get("value")
                    if name is not None:
                        state.set_control(name, value)
                        # Virtual camera toggle
                        if name == "SendVirtCamFramesEnableToggle":
                            if value:
                                vp.enable_virtualcam()
                            else:
                                vp.disable_virtualcam()
                            # Emit actual state back so the UI reflects reality
                            # (e.g. if enable_virtualcam failed, virtcam is None)
                            actual = vp.virtcam is not None
                            bus.emit_sync("virtcam_state", {"enabled": actual})
                            if value and not actual:
                                bus.emit_sync("error", {
                                    "message": "Virtual camera failed to start. "
                                               "Check that OBS Virtual Camera (or Unity Capture) is installed."
                                })
                        # Backend change while cam is active — restart with new backend
                        elif name == "VirtCamBackendSelection":
                            if state.control.get("SendVirtCamFramesEnableToggle", False):
                                vp.enable_virtualcam(backend=value)
                                actual = vp.virtcam is not None
                                bus.emit_sync("virtcam_state", {"enabled": actual})
                        vp.process_current_frame()
                        bus.emit_sync("state_updated", {
                            "section": "control", "name": name, "value": value
                        })

                case "set_parameter":
                    face_id = payload.get("face_id")
                    name    = payload.get("name")
                    value   = payload.get("value")
                    if face_id and name is not None:
                        state.set_parameter(face_id, name, value)
                        vp.process_current_frame()
                        bus.emit_sync("state_updated", {
                            "section": "parameters",
                            "face_id": face_id, "name": name, "value": value,
                        })

                # ── Preview quality ───────────────────────────────────────
                case "preview_quality":
                    q = int(payload.get("quality", 75))
                    bus._preview_quality = max(1, min(100, q))

                # ── Native preview window ─────────────────────────────────
                case "open_preview_window":
                    try:
                        from PySide6.QtWidgets import QApplication
                        qt_app = QApplication.instance()

                        if qt_app is not None:
                            # ── Qt desktop mode: find the MainWindow ──────
                            main_windows = [
                                w for w in qt_app.topLevelWidgets()
                                if w.__class__.__name__ == "MainWindow"
                            ]
                            if main_windows:
                                mw = main_windows[0]
                                from app.ui.widgets.preview_window import PreviewWindow
                                if mw._preview_window is not None and mw._preview_window.isVisible():
                                    # Window is open — close it
                                    mw._preview_window.close()
                                    bus.emit_sync("preview_window_closed", {})
                                else:
                                    # Window is closed or crashed — (re)open it
                                    mw._preview_window = PreviewWindow(mw)
                                    mw._preview_window.show()
                                    bus.emit_sync("preview_window_opened", {})
                        else:
                            # ── Headless API mode: spin up Qt on a bg thread ──
                            from app.ui.widgets.headless_preview import headless_preview
                            if headless_preview.is_open:
                                # Window is open — close it
                                headless_preview.close()
                                bus.emit_sync("preview_window_closed", {})
                            else:
                                # Window is closed or crashed — (re)open it
                                headless_preview.open()
                                bus.emit_sync("preview_window_opened", {})
                    except Exception as _pw_err:
                        print(f"[WS] open_preview_window error: {_pw_err}")

                # ── Utility ───────────────────────────────────────────────
                case "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                case "source_tab_changed":
                    # Full source switch:
                    # 1. Tear down whatever is currently running
                    # 2. Auto-start the new source if it's already configured
                    source = payload.get("source", "")
                    prev_type = vp.file_type

                    # ── Teardown previous source ──────────────────────────
                    if prev_type == "webrtc":
                        # Stop the relay server and free SHM
                        from app.api.routes.sources import stop_webrtc_process
                        stop_webrtc_process(websocket.app.state, vp)
                        bus.emit_sync("webrtc_stopped", {})
                    elif prev_type == "webcam":
                        vp.stop_processing()
                        if vp.media_capture:
                            vp.media_capture.release()
                            vp.media_capture = None
                        vp.file_type = None
                    else:
                        vp.stop_processing()

                    # ── Start new source ──────────────────────────────────
                    if source == "media" and vp.file_type == "video":
                        vp.process_video()
                    elif source == "webcam" and vp.file_type == "webcam":
                        vp.process_video()
                    elif source == "streaming" and vp.file_type == "webrtc":
                        vp.process_video()
                    # else: no active source for this tab — stay idle

                    _emit_playback_state(vp, state)

                case _:
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        bus.unsubscribe(client_q)


# ── /ws/preview ───────────────────────────────────────────────────────────────

@router.websocket("/ws/preview")
async def ws_preview(websocket: WebSocket):
    """
    Push-only binary JPEG stream.

    Each message is a raw JPEG byte payload — no framing, no JSON.
    The client renders it via createImageBitmap + canvas.drawImage.

    Uses a latest-frame-wins asyncio.Event so the write buffer stays at
    most one frame deep — prevents the websockets keepalive ping
    AssertionError that occurs when the buffer backs up under load.

    Optional text message from client:
        { "quality": 75 }   — set JPEG quality for this connection (1-100)
    """
    await websocket.accept()
    frame_event = bus.subscribe_preview()

    async def _receiver():
        """Accept optional quality-change messages from the client."""
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                    if "quality" in msg:
                        bus._preview_quality = max(1, min(100, int(msg["quality"])))
                except Exception:
                    pass
        except Exception:
            pass

    receiver_task = asyncio.create_task(_receiver())

    try:
        while True:
            # Wait for the next frame signal
            await frame_event.wait()
            frame_event.clear()

            # Grab the latest encoded frame (may have been replaced since signal)
            frame_bytes = bus._latest_frame
            if frame_bytes is None:
                continue

            try:
                await websocket.send_bytes(frame_bytes)
            except Exception:
                # Connection closed or write buffer error — stop cleanly
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        receiver_task.cancel()
        bus.unsubscribe_preview(frame_event)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _emit_playback_state(vp, state=None) -> None:
    """Push a playback_state event so the client can sync its UI."""
    bus.emit_sync("playback_state", {
        "is_playing":   vp.processing,
        "is_recording": vp.recording,
        "current_frame": vp.current_frame_number,
        "max_frame":    vp.max_frame_number,
        "fps":          vp.fps,
        "file_type":    vp.file_type,
        "loop_enabled": state.loop_enabled if state is not None else False,
    })
