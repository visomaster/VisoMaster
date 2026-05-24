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
                        _emit_playback_state(vp)

                case "stop":
                    vp.stop_processing()
                    _emit_playback_state(vp)

                case "seek":
                    frame = int(payload.get("frame", 0))
                    frame = max(0, min(frame, vp.max_frame_number))
                    vp.stop_processing()
                    vp.current_frame_number = frame
                    if vp.media_capture:
                        vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    # Apply marker overrides if present
                    if frame in state.markers:
                        m = state.markers[frame]
                        state.parameters.update(m.parameters)
                        state.control.update(m.control)
                    vp.process_current_frame()
                    _emit_playback_state(vp)

                case "step":
                    n = int(payload.get("n", 1))
                    new_frame = max(0, min(vp.current_frame_number + n, vp.max_frame_number))
                    vp.stop_processing()
                    vp.current_frame_number = new_frame
                    if vp.media_capture:
                        vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    vp.process_current_frame()
                    _emit_playback_state(vp)

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
                        bus.emit_sync("state_updated", {
                            "section": "control", "name": name, "value": value
                        })

                case "set_parameter":
                    face_id = payload.get("face_id")
                    name    = payload.get("name")
                    value   = payload.get("value")
                    if face_id and name is not None:
                        state.set_parameter(face_id, name, value)
                        bus.emit_sync("state_updated", {
                            "section": "parameters",
                            "face_id": face_id, "name": name, "value": value,
                        })

                # ── Preview quality ───────────────────────────────────────
                case "preview_quality":
                    q = int(payload.get("quality", 75))
                    bus._preview_quality = max(1, min(100, q))

                # ── Utility ───────────────────────────────────────────────
                case "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

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
    The client renders it directly:

        ws.binaryType = 'arraybuffer';
        ws.onmessage = (e) => {
            const blob = new Blob([e.data], { type: 'image/jpeg' });
            img.src = URL.createObjectURL(blob);
        };

    Optional text message from client:
        { "quality": 75 }   — set JPEG quality for this connection (1-100)
    """
    await websocket.accept()
    frame_q = bus.subscribe_preview()

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
            frame_bytes = await frame_q.get()
            await websocket.send_bytes(frame_bytes)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        receiver_task.cancel()
        bus.unsubscribe_preview(frame_q)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _emit_playback_state(vp) -> None:
    """Push a playback_state event so the client can sync its UI."""
    bus.emit_sync("playback_state", {
        "is_playing":   vp.processing,
        "is_recording": vp.recording,
        "current_frame": vp.current_frame_number,
        "max_frame":    vp.max_frame_number,
        "fps":          vp.fps,
        "file_type":    vp.file_type,
    })
