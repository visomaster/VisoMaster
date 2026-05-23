"""Consumer-side helper: read frames out of the shared-memory block.

Designed to be the *only* file an integrating project needs to import.
No web server, no aiohttp, no codec libraries — just numpy and stdlib.

Typical usage in a polling loop::

    from streamrelay import FrameReader

    reader = FrameReader()                    # waits for the server
    while running:
        frame = reader.read_latest()          # HxWx3 uint8 BGR or None
        if frame is None:
            time.sleep(0.005)
            continue
        process(frame)

If you want to know whether a returned frame is *new* (vs a re-read of the
same buffer), use ``read_latest_with_info``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple

import numpy as np

from . import protocol


@dataclass(frozen=True)
class FrameInfo:
    """Metadata for a single frame read from shared memory."""
    counter: int
    width:   int
    height:  int


class FrameReader:
    """Polls the shared-memory frame buffer written by ``StreamServer``.

    Parameters
    ----------
    shm_name:
        Name of the shared-memory block. Must match the server's ``shm_name``.
    attach_timeout:
        How long ``__init__`` waits for the producer to create the block.
        Set to 0 to fail immediately if the server isn't running.
    """

    def __init__(
        self,
        shm_name: str = protocol.DEFAULT_SHM_NAME,
        attach_timeout: float = 0.0,
    ):
        self.shm_name = shm_name
        self._shm: Optional[SharedMemory] = None
        self._last_counter = 0
        self._attach(attach_timeout)

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def _attach(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                self._shm = SharedMemory(name=self.shm_name, create=False)
                return
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise FileNotFoundError(
                        f"Shared memory '{self.shm_name}' not found. Is the "
                        "StreamServer running and using the same shm_name?"
                    )
                time.sleep(0.05)

    def try_attach(self) -> bool:
        """Attempt a single non-blocking attach; safe to retry."""
        if self._shm is not None:
            return True
        try:
            self._shm = SharedMemory(name=self.shm_name, create=False)
            return True
        except FileNotFoundError:
            return False

    @property
    def attached(self) -> bool:
        return self._shm is not None

    def close(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            finally:
                self._shm = None

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Frame reads ──────────────────────────────────────────────────────────
    def read_latest(self) -> Optional[np.ndarray]:
        """Return the latest frame as a HxWx3 uint8 BGR ``numpy`` array.

        Returns ``None`` if no frame is available yet. Always returns a
        copy — safe to mutate or pass to threads.
        """
        result = self.read_latest_with_info()
        return None if result is None else result[0]

    def read_latest_with_info(self) -> Optional[Tuple[np.ndarray, FrameInfo]]:
        """Return ``(frame, FrameInfo)`` or ``None``."""
        if self._shm is None:
            if not self.try_attach():
                return None
        counter, w, h = protocol.unpack_header(self._shm.buf)
        if counter == 0 or w == 0 or h == 0:
            return None
        nbytes = w * h * 3
        raw = bytes(
            self._shm.buf[protocol.SHM_HEADER_BYTES:
                          protocol.SHM_HEADER_BYTES + nbytes]
        )
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        self._last_counter = counter
        return frame, FrameInfo(counter=counter, width=w, height=h)

    def read_new(self) -> Optional[np.ndarray]:
        """Like ``read_latest`` but returns ``None`` if the frame counter
        hasn't advanced since the previous successful read.
        """
        result = self.read_new_with_info()
        return None if result is None else result[0]

    def read_new_with_info(self) -> Optional[Tuple[np.ndarray, FrameInfo]]:
        if self._shm is None:
            if not self.try_attach():
                return None
        counter, w, h = protocol.unpack_header(self._shm.buf)
        if counter == 0 or counter == self._last_counter or w == 0 or h == 0:
            return None
        nbytes = w * h * 3
        raw = bytes(
            self._shm.buf[protocol.SHM_HEADER_BYTES:
                          protocol.SHM_HEADER_BYTES + nbytes]
        )
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        self._last_counter = counter
        return frame, FrameInfo(counter=counter, width=w, height=h)

    # ── Misc ────────────────────────────────────────────────────────────────
    @property
    def last_counter(self) -> int:
        return self._last_counter
