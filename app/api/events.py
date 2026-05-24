"""
app/api/events.py
─────────────────
Thread-safe event bus bridging sync worker threads → async WebSocket clients.

Two channels:
  1. JSON events  — control/state messages  → /ws/events
  2. Binary frames — JPEG-encoded BGR frames → /ws/preview
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

        # ── Binary frame channel ──────────────────────────────────────────
        # Each preview subscriber gets its own queue.
        # We keep at most 2 frames per subscriber so slow clients don't
        # accumulate a backlog — old frames are dropped.
        self._preview_clients: List[asyncio.Queue] = []
        self._preview_quality: int = 75   # JPEG quality 1-100

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

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
        Encodes frame_bgr to JPEG and pushes to all preview subscribers.
        Old frames are dropped when a subscriber's queue is full (maxsize=2).
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

        def _push():
            dead = []
            for client_q in self._preview_clients:
                # Drop oldest frame if queue is full (non-blocking)
                if client_q.full():
                    try:
                        client_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    client_q.put_nowait(data)
                except Exception:
                    dead.append(client_q)
            for q in dead:
                self.unsubscribe_preview(q)

        try:
            self._loop.call_soon_threadsafe(_push)
        except Exception:
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

    def subscribe_preview(self) -> asyncio.Queue:
        """Register a new /ws/preview client; returns its personal queue (maxsize=2)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._preview_clients.append(q)
        return q

    def unsubscribe_preview(self, q: asyncio.Queue) -> None:
        try:
            self._preview_clients.remove(q)
        except ValueError:
            pass

    # ── Broadcast loops (run as asyncio tasks) ────────────────────────────

    async def _broadcast_loop(self) -> None:
        """Drain the JSON queue and fan-out to all /ws/events clients."""
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


# Singleton — imported by server.py and injected into app.state
bus = EventBus()
