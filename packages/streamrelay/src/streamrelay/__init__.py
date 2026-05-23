"""streamrelay — phone-to-process video streaming over WebSocket.

A drop-in module for AI face-swap and video-processing apps that need a
low-latency way to receive camera frames from a browser or mobile device.

Frames arrive over a single WebSocket (JPEG or H.264), get decoded to BGR,
and land in a named shared-memory block that any consumer process can poll
with O(1) latency. No multiprocessing queues, no pickling, no sockets in
the consumer.

Quick start
-----------

Producer side (run as a subprocess so it doesn't block your UI loop):

    from streamrelay import StreamServer
    StreamServer(shm_name="my_app_frames", http_port=9091).run()

Consumer side (your existing AI pipeline):

    from streamrelay import FrameReader

    reader = FrameReader(shm_name="my_app_frames")
    while True:
        frame = reader.read_latest()         # numpy HxWx3 BGR, or None
        if frame is None:
            time.sleep(0.005)
            continue
        # ... feed `frame` to your face-swap model ...

See the README for ready-made adapters for VisoMaster, DeepFaceLive,
Rope-Live and similar projects.
"""

from .reader import FrameReader, FrameInfo
from .server import StreamServer, run_server
from .protocol import (
    DEFAULT_SHM_NAME,
    SHM_HEADER_BYTES,
    SHM_MAX_WIDTH,
    SHM_MAX_HEIGHT,
    SHM_FRAME_BYTES,
    SHM_TOTAL_BYTES,
)

__all__ = [
    "FrameReader",
    "FrameInfo",
    "StreamServer",
    "run_server",
    "DEFAULT_SHM_NAME",
    "SHM_HEADER_BYTES",
    "SHM_MAX_WIDTH",
    "SHM_MAX_HEIGHT",
    "SHM_FRAME_BYTES",
    "SHM_TOTAL_BYTES",
]

__version__ = "0.1.0"
