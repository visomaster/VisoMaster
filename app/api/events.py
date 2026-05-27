"""
app/api/events.py
─────────────────
Thread-safe event bus bridging sync worker threads → async WebSocket clients.

Two channels:
  1. JSON events  — control/state messages  → /ws/events
  2. Binary frames — JPEG-encoded BGR frames → /ws/preview

Frame delivery strategy
-----------------------
Frames arrive from GPU worker threads at up to 30+ fps. WebSocket clients
may be slower. To prevent write-buffer buildup (which causes the websockets
library's keepalive ping AssertionError), we use a "latest frame wins" slot:

  - A single asyncio.Event signals that a new frame is ready.
  - The latest encoded JPEG is stored in _latest_frame (bytes | None).
  - Each /ws/preview sender loop waits on the event, grabs the latest frame,
    clears the event, and sends. If multiple frames arrive while a send is
    in progress, only the most recent one is sent next — stale frames are
    silently discarded.

This keeps the WebSocket write buffer at most one frame deep, eliminating
the drain() assertion and producing smooth, always-current playback.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class EventBus:
    """Thread-safe event bus bridging sync worker threads → async WS clients."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # ── JSON event channel ────────────────────────────────────────────
        self._queue: asyncio.Queue = asyncio.Queue()
        self._clients: List[asyncio.Queue] = []

        # ── Binary frame channel — latest-frame-wins slot ─────────────────
        self._latest_frame: Optional[bytes] = None
        self._frame_event: Optional[asyncio.Event] = None
        self._preview_clients: List[asyncio.Event] = []
        self._preview_quality: int = 75

        # ── Position channel — latest-wins slot (never queued) ────────────
        # Stores the most recent (frame_number, max_frame) pair.
        # A dedicated broadcast task wakes on _position_event and fans out
        # a single "frame_position" JSON message to all /ws/events clients.
        # Because it's latest-wins, 30 position updates/sec never pile up.
        self._latest_position: Optional[tuple[int, int]] = None
        self._position_event: Optional[asyncio.Event] = None  # created after loop is set

        # ── Dedicated playback channel — latest-wins binary JSON ──────────
        # Carries only { current_frame, max_frame, is_playing, fps } at up
        # to 30 fps without ever blocking the main /ws/events queue.
        self._latest_playback_msg: Optional[bytes] = None
        self._playback_clients: List[asyncio.Event] = []

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._position_event = asyncio.Event()

    # ── JSON event producers (called from sync threads) ───────────────────

    def emit_sync(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        """
        Thread-safe JSON event emit.
        Safe to call from Qt worker threads or the main thread.
        Silently dropped if the event loop isn't running yet.
        """
        if self._loop is None or self._loop.is_closed():
            return
        msg = json.dumps({"type": event_type, "payload": payload or {}})
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)
        except Exception:
            pass

    # ── Binary frame producers (called from on_frame_done callback) ───────

    def emit_frame_sync(self, frame_bgr: np.ndarray, quality: int | None = None) -> None:
        """
        Thread-safe JPEG frame emit.

        Encodes frame_bgr to JPEG, stores it in the latest-frame slot, and
        wakes all /ws/preview sender coroutines. If a sender is still busy
        with the previous frame, the old frame is silently replaced — the
        sender will always pick up the newest available frame.
        """
        if self._loop is None or self._loop.is_closed():
            return
        if not self._preview_clients:
            return  # No subscribers — skip encoding entirely

        q = quality if quality is not None else self._preview_quality
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return
        data = buf.tobytes()

        def _store_and_notify():
            self._latest_frame = data
            for ev in self._preview_clients:
                ev.set()

        try:
            self._loop.call_soon_threadsafe(_store_and_notify)
        except Exception:
            pass

    def emit_position_sync(self, frame_number: int, max_frame: int) -> None:
        """
        Thread-safe position update — latest-wins, never queued.

        Stores the most recent (frame_number, max_frame) and wakes the
        position broadcast task. If multiple frames complete before the
        task wakes, only the latest position is delivered — no pile-up.
        """
        if self._loop is None or self._loop.is_closed():
            return
        if not self._clients:
            return

        def _store_and_notify():
            self._latest_position = (frame_number, max_frame)
            if self._position_event is not None:
                self._position_event.set()

        try:
            self._loop.call_soon_threadsafe(_store_and_notify)
        except Exception:
            pass

    def emit_playback_sync(
        self,
        current_frame: int,
        max_frame: int,
        is_playing: bool,
        fps: float,
        is_recording: bool = False,
    ) -> None:
        """
        Thread-safe playback-state update for the dedicated /ws/playback channel.

        Latest-wins: if multiple frames complete before the sender wakes, only
        the most recent state is delivered — no pile-up at 30 fps.
        """
        if self._loop is None or self._loop.is_closed():
            return
        if not self._playback_clients:
            return

        data = json.dumps({
            "current_frame": current_frame,
            "max_frame": max_frame,
            "is_playing": is_playing,
            "fps": fps,
            "is_recording": is_recording,
        }).encode()

        def _store_and_notify():
            self._latest_playback_msg = data
            for ev in self._playback_clients:
                ev.set()

        try:
            self._loop.call_soon_threadsafe(_store_and_notify)
        except Exception:
            pass

    # ── Dedicated playback channel consumers ──────────────────────────────

    def subscribe_playback(self) -> asyncio.Event:
        """Register a new /ws/playback client; returns a per-client asyncio.Event."""
        ev = asyncio.Event()
        self._playback_clients.append(ev)
        return ev

    def unsubscribe_playback(self, ev: asyncio.Event) -> None:
        try:
            self._playback_clients.remove(ev)
        except ValueError:
            pass

    # ── JSON event consumers ──────────────────────────────────────────────
    def subscribe(self) -> asyncio.Queue:
        """Register a new /ws/events client; returns its personal queue."""
        q: asyncio.Queue = asyncio.Queue()
        self._clients.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._clients.remove(q)
        except ValueError:
            pass

    # ── Binary frame consumers ────────────────────────────────────────────

    def subscribe_preview(self) -> asyncio.Event:
        """
        Register a new /ws/preview client.

        Returns a per-client asyncio.Event. The sender loop should:
          1. await event.wait()
          2. event.clear()
          3. read bus._latest_frame
          4. send the bytes
        """
        ev = asyncio.Event()
        self._preview_clients.append(ev)
        return ev

    def unsubscribe_preview(self, ev: asyncio.Event) -> None:
        try:
            self._preview_clients.remove(ev)
        except ValueError:
            pass

    # ── Broadcast loops (run as asyncio tasks) ────────────────────────────

    async def _broadcast_loop(self) -> None:
        """Drain the JSON queue and fan-out to all /ws/events clients."""
        try:
            while True:
                msg = await self._queue.get()
                dead = []
                for client_q in self._clients:
                    try:
                        client_q.put_nowait(msg)
                    except asyncio.QueueFull:
                        dead.append(client_q)
                for q in dead:
                    self.unsubscribe(q)
        except asyncio.CancelledError:
            pass

    async def _position_broadcast_loop(self) -> None:
        """
        Deliver latest-wins frame position to all /ws/events clients.

        Wakes whenever emit_position_sync() fires, grabs the latest
        (frame, max_frame) pair, and fans out a single 'frame_position'
        JSON message. Because it's latest-wins, bursts of 30+ updates/sec
        collapse into one delivery per event-loop iteration.
        """
        try:
            while True:
                if self._position_event is None:
                    await asyncio.sleep(0.05)
                    continue
                await self._position_event.wait()
                self._position_event.clear()

                pos = self._latest_position
                if pos is None:
                    continue

                frame_number, max_frame = pos
                msg = json.dumps({
                    "type": "frame_position",
                    "payload": {"current_frame": frame_number, "max_frame": max_frame},
                })
                dead = []
                for client_q in self._clients:
                    try:
                        client_q.put_nowait(msg)
                    except asyncio.QueueFull:
                        dead.append(client_q)
                for q in dead:
                    self.unsubscribe(q)
        except asyncio.CancelledError:
            pass
        except asyncio.CancelledError:
            pass


# Singleton — imported by server.py and injected into app.state
bus = EventBus()
