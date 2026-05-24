"""
app/api/ws.py
─────────────
WebSocket endpoint: /ws/events

Server → client: JSON event objects  { type, payload }
Client → server: JSON command objects { type, payload }

Supported client commands:
  { "type": "play" }
  { "type": "stop" }
  { "type": "seek",          "payload": { "frame": N } }
  { "type": "step",          "payload": { "n": N } }
  { "type": "set_control",   "payload": { "name": "...", "value": ... } }
  { "type": "set_parameter", "payload": { "face_id": "...", "name": "...", "value": ... } }
  { "type": "ping" }
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.events import bus

router = APIRouter()


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    client_q = bus.subscribe()

    async def _sender():
        """Forward events from the bus to this client."""
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
            payload = cmd.get("payload", {})

            # Lazy import to avoid circular deps
            from fastapi import Request
            app = websocket.app
            state = app.state.app_state
            vp = app.state.video_processor

            match cmd_type:
                case "play":
                    if not vp.processing and vp.file_type:
                        vp.process_video()
                case "stop":
                    vp.stop_processing()
                case "seek":
                    frame = int(payload.get("frame", 0))
                    frame = max(0, min(frame, vp.max_frame_number))
                    vp.stop_processing()
                    vp.current_frame_number = frame
                    import cv2
                    if vp.media_capture:
                        vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    vp.process_current_frame()
                case "step":
                    n = int(payload.get("n", 1))
                    new_frame = max(0, min(vp.current_frame_number + n, vp.max_frame_number))
                    vp.stop_processing()
                    vp.current_frame_number = new_frame
                    import cv2
                    if vp.media_capture:
                        vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    vp.process_current_frame()
                case "set_control":
                    name = payload.get("name")
                    value = payload.get("value")
                    if name is not None:
                        state.set_control(name, value)
                        bus.emit_sync("state_updated", {"section": "control", "name": name, "value": value})
                case "set_parameter":
                    face_id = payload.get("face_id")
                    name = payload.get("name")
                    value = payload.get("value")
                    if face_id and name is not None:
                        state.set_parameter(face_id, name, value)
                        bus.emit_sync("state_updated", {"section": "parameters", "face_id": face_id, "name": name, "value": value})
                case "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                case _:
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        bus.unsubscribe(client_q)
