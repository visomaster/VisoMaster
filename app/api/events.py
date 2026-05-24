"""
app/api/events.py
─────────────────
A lightweight asyncio-based event bus.

The VideoProcessor (running in Qt threads) pushes events via
`EventBus.emit_sync(...)`.  The WebSocket handler drains them
and forwards to all connected clients.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional


class EventBus:
    """Thread-safe event bus bridging sync worker threads → async WS clients."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._clients: List[Any] = []   # list of asyncio.Queue per WS connection

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ── Producers (called from sync threads) ─────────────────────────────

    def emit_sync(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        """
        Thread-safe emit.  Safe to call from Qt worker threads or the main thread.
        If the event loop isn't running yet, the event is silently dropped.
        """
        if self._loop is None or self._loop.is_closed():
            return
        msg = json.dumps({"type": event_type, "payload": payload or {}})
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)
        except Exception:
            pass

    # ── Consumers (called from async WS handler) ──────────────────────────

    def subscribe(self) -> asyncio.Queue:
        """Register a new WebSocket client; returns its personal queue."""
        q: asyncio.Queue = asyncio.Queue()
        self._clients.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._clients.remove(q)
        except ValueError:
            pass

    async def _broadcast_loop(self) -> None:
        """Drain the shared queue and fan-out to all client queues."""
        while True:
            msg = await self._queue.get()
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self.unsubscribe(q)


# Singleton — imported by server.py and injected into app.state
bus = EventBus()
