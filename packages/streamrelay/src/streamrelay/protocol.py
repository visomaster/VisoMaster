"""Shared-memory frame protocol shared by producer and consumer.

Layout (fixed-size buffer, default 1920×1080×3 = 6 220 800 bytes + header):

    Bytes  0-3 : frame counter (uint32 LE) — incremented on every write
    Bytes  4-7 : frame width   (uint32 LE)
    Bytes  8-11: frame height  (uint32 LE)
    Bytes 12-N : raw BGR frame data (width * height * 3 bytes)

Why fixed-size? The block is allocated once at server start and never
grows or shrinks. Producers write any frame up to MAX_WIDTH x MAX_HEIGHT;
consumers read only the W*H*3 bytes the header advertises. The counter
lets the consumer detect new frames without locks.
"""

from __future__ import annotations

import struct

# ── Defaults that callers can override per StreamServer instance ─────────────
DEFAULT_SHM_NAME: str = "streamrelay_frame"

SHM_MAX_WIDTH:    int = 1920
SHM_MAX_HEIGHT:   int = 1080
SHM_FRAME_BYTES:  int = SHM_MAX_WIDTH * SHM_MAX_HEIGHT * 3  # BGR
SHM_HEADER_BYTES: int = 12                                  # counter + W + H
SHM_TOTAL_BYTES:  int = SHM_HEADER_BYTES + SHM_FRAME_BYTES


def pack_header(buf, counter: int, width: int, height: int) -> None:
    """Write the 12-byte header at offset 0."""
    struct.pack_into("<III", buf, 0, counter & 0xFFFFFFFF, width, height)


def unpack_header(buf) -> tuple[int, int, int]:
    """Return (counter, width, height) from the first 12 bytes."""
    return struct.unpack_from("<III", buf, 0)


def header_size() -> int:
    """Convenience for consumers that don't want to import the constant."""
    return SHM_HEADER_BYTES
