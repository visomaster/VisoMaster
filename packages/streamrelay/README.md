# streamrelay

A small, dependency-light Python package that gets camera frames from a phone or browser into your AI process with minimum latency.

It was extracted from [VisoMaster](https://github.com/visomaster/VisoMaster) so other face-swap and real-time vision projects (DeepFaceLive, Rope-Live, etc.) can drop it in without inheriting the rest of the app.

```
┌──────────────┐   WebSocket   ┌────────────────┐  shared  ┌───────────────────┐
│ phone/browser│ ─────────────▶│ streamrelay    │ memory   │ your AI process   │
│  camera      │  JPEG / H.264 │ server (subproc)│ ───────▶│ (face swap, etc.) │
└──────────────┘               └────────────────┘          └───────────────────┘
```

## Features

- **One WebSocket** — no signaling servers, no STUN/TURN, no SDP. Works through any HTTP proxy.
- **JPEG and H.264** transports. Optional PyAV-based H.264 decoder, falls back to JPEG transparently.
- **Shared-memory delivery** — zero-copy hand-off to the consumer process; the consumer never blocks the server.
- **Bundled web UI** so a phone just opens `https://<your-pc>:9090/` and starts streaming.
- **Self-signed TLS** generated automatically on first run (required for `getUserMedia` on mobile).
- **Hot reload** of the bundled HTML/CSS/JS during development (SSE on `/livereload`).

## Install

```bash
pip install streamrelay              # core
pip install "streamrelay[h264]"      # with PyAV for H.264 decoding
pip install "streamrelay[h264,psutil]"  # also reclaim ports automatically on restart
```

## Producer side (run as subprocess)

```python
import multiprocessing as mp
from streamrelay import StreamServer

def _serve():
    StreamServer(shm_name="myapp_frames", http_port=9091, https_port=9090).run()

if __name__ == "__main__":
    proc = mp.Process(target=_serve, daemon=True)
    proc.start()
```

Open `https://<your-host>:9090/` on the phone, allow camera, hit **Start**.

## Consumer side (your AI loop)

```python
import time
from streamrelay import FrameReader

reader = FrameReader(shm_name="myapp_frames", attach_timeout=10.0)

while True:
    frame = reader.read_new()       # HxWx3 BGR, or None if no new frame
    if frame is None:
        time.sleep(0.005)
        continue

    # ── feed `frame` to your model ──
    # swapped = face_swap(frame)
    # display(swapped)
```

`read_new()` returns `None` when the producer hasn't written a new frame since your last read — perfect for tight polling loops. Use `read_latest()` if you want the most recent buffer regardless.

## Integration recipes

### VisoMaster

Already wired in. VisoMaster imports `StreamServer`, `run_server`, and the protocol constants directly from `streamrelay`. The path bootstrap in `main.py` ensures the package resolves without a pip install.

### DeepFaceLive

Add a custom backend at `apps/DeepFaceLive/backend/StreamRelaySource.py`:

```python
import time, numpy as np
from streamrelay import FrameReader
from .BackendBase import BackendHost, BackendWeakHeap

class StreamRelaySource(BackendHost):
    def __init__(self, weak_heap: BackendWeakHeap, bc_out, shm_name="dfl_frames"):
        super().__init__(...)
        self._reader = FrameReader(shm_name=shm_name)

    def on_tick(self):
        frame = self._reader.read_new()
        if frame is None:
            return
        self._send_frame_bgr(frame)
```

Then start the StreamServer subprocess from your launcher with `shm_name="dfl_frames"`.

### Rope / Rope-Live

Rope reads frames from `cv2.VideoCapture` via a queue. Wrap a reader as a fake capture:

```python
import cv2, numpy as np
from streamrelay import FrameReader

class StreamRelayCapture:
    def __init__(self, shm_name="rope_frames"):
        self._reader = FrameReader(shm_name=shm_name, attach_timeout=10.0)
        self._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def read(self):
        f = self._reader.read_latest()
        if f is not None:
            self._last_frame = f
        return True, self._last_frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS: return 30.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:  return self._last_frame.shape[1]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return self._last_frame.shape[0]
        return 0

    def isOpened(self): return self._reader.attached
    def release(self):  self._reader.close()
```

Pass an instance of `StreamRelayCapture` anywhere Rope expects a `cv2.VideoCapture`.

### Generic OpenCV pipeline

```python
import cv2
from streamrelay import FrameReader

reader = FrameReader()
while True:
    frame = reader.read_new()
    if frame is None: continue
    cv2.imshow("stream", frame)
    if cv2.waitKey(1) == 27: break
```

## Shared-memory protocol

If you don't want to use `FrameReader` (different language, custom polling, etc.), the layout is documented in [`protocol.py`](src/streamrelay/protocol.py):

| Bytes  | Field   | Type      |
|--------|---------|-----------|
| 0..3   | counter | uint32 LE |
| 4..7   | width   | uint32 LE |
| 8..11  | height  | uint32 LE |
| 12..N  | pixels  | BGR uint8 |

A non-zero `counter` means a frame is available. Increment-detection is enough for change tracking; no locks needed for single-producer-single-consumer.

## Development

```bash
git clone <this-repo>
cd packages/streamrelay
pip install -e ".[h264,psutil]"
streamrelay-server --shm-name dev_frames
```

Then open the printed URL on your phone.

## License

MIT.
