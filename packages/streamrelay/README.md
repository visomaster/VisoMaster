# streamrelay

A lightweight Python package that streams camera frames from any phone or browser into your Python process over a local network — with near-zero latency and no external infrastructure.

```
┌──────────────────┐   WebSocket    ┌─────────────────┐  shared memory  ┌──────────────────┐
│  phone / browser │ ─────────────▶ │  streamrelay    │ ──────────────▶ │  your Python     │
│  (WebStreamer UI)│  JPEG / H.264  │  server         │                 │  process         │
└──────────────────┘                └─────────────────┘                 └──────────────────┘
```

## How it works

1. **Start the server** in a subprocess from your Python app. It binds an HTTP and HTTPS port on your machine.
2. **Open the URL** on any phone or browser on the same network. The bundled **WebStreamer** web UI loads automatically.
3. **Hit Start** in the UI. The browser captures the camera and streams encoded frames over a WebSocket to the server.
4. **Read frames** in your Python process using `FrameReader`. Each call returns a fresh BGR NumPy array ready for any OpenCV or ML pipeline.

No signaling servers. No STUN/TURN. No accounts. Works entirely on your local network.

---

## Features

- **Simple WebSocket transport** — no WebRTC negotiation, no SDP, works through any HTTP proxy
- **JPEG and H.264** — browser auto-selects the best codec; H.264 uses hardware acceleration where available and falls back to JPEG transparently
- **Shared-memory delivery** — the server writes decoded BGR frames into a named shared-memory block; your consumer reads without copying or blocking the server
- **Bundled WebStreamer UI** — phone opens a URL, grants camera permission, and streams immediately; no app install required
- **Automatic self-signed TLS** — generated on first run so mobile browsers can access `getUserMedia` (required for non-localhost origins)
- **Persistent settings** — codec, resolution, and quality preferences saved in the browser across sessions
- **Responsive UI** — works on phone portrait, phone landscape, and desktop
- **Live reload** during development — edit the client files and the browser reloads automatically

---

## Install

```bash
# Core package
pip install streamrelay

# With H.264 decoding support (recommended)
pip install "streamrelay[h264]"

# Also auto-release ports on restart
pip install "streamrelay[h264,psutil]"
```

---

## Quick start

### Step 1 — Start the server

Run this from your application, typically at startup:

```python
import multiprocessing as mp
from streamrelay import StreamServer

def _serve():
    StreamServer(
        shm_name="myapp_frames",
        http_port=9091,
        https_port=9090,
    ).run()

if __name__ == "__main__":
    proc = mp.Process(target=_serve, daemon=True)
    proc.start()
```

The server prints the URLs it is listening on:

```
[streamrelay] HTTP  on 0.0.0.0:9091
[streamrelay] HTTPS on 0.0.0.0:9090
```

### Step 2 — Open the WebStreamer UI

On any device on the same network, open:

```
https://<your-machine-ip>:9090/
```

> **Note:** The browser will show a certificate warning on first visit because the cert is self-signed. Accept it once — this is expected and safe on a local network.

Grant camera permission when prompted, select your preferred resolution and codec, then tap **Start Streaming**.

### Step 3 — Read frames in your process

```python
import time
from streamrelay import FrameReader

reader = FrameReader(shm_name="myapp_frames", attach_timeout=10.0)

while True:
    frame = reader.read_new()   # returns HxWx3 BGR ndarray, or None if no new frame
    if frame is None:
        time.sleep(0.005)
        continue

    # frame is a standard OpenCV BGR image — pass it to any pipeline
    process(frame)
```

`read_new()` returns `None` when no new frame has arrived since your last read, making it safe to use in a tight polling loop without busy-waiting. Use `read_latest()` if you always want the most recent frame regardless of whether it is new.

---

## StreamServer options

```python
StreamServer(
    shm_name="myapp_frames",   # shared-memory block name — must match FrameReader
    http_port=9091,            # plain HTTP port (redirects to HTTPS on mobile)
    https_port=9090,           # HTTPS port (required for camera access on mobile)
    host="0.0.0.0",            # bind address
    cert_file="",              # path to existing TLS cert (auto-generated if empty)
    key_file="",               # path to existing TLS key  (auto-generated if empty)
    on_frame=None,             # optional callback fn(frame_bgr) called on every frame
)
```

### Using the `on_frame` callback

If you prefer a callback over polling shared memory:

```python
def handle_frame(frame):
    # called in the server's event loop — keep it fast
    result_queue.put(frame.copy())

StreamServer(shm_name="myapp_frames", on_frame=handle_frame).run()
```

---

## FrameReader options

```python
FrameReader(
    shm_name="myapp_frames",   # must match the server's shm_name
    attach_timeout=10.0,       # seconds to wait for the server to create the block
)
```

| Method | Returns | Description |
|---|---|---|
| `read_new()` | `ndarray \| None` | New frame since last read, or `None` |
| `read_latest()` | `ndarray \| None` | Most recent frame regardless of whether it is new |
| `close()` | — | Detach from shared memory |

---

## Shared-memory protocol

If you want to consume frames from a language other than Python, the shared-memory layout is:

| Bytes | Field | Type |
|---|---|---|
| 0–3 | counter | `uint32` little-endian |
| 4–7 | width | `uint32` little-endian |
| 8–11 | height | `uint32` little-endian |
| 12–N | pixels | BGR `uint8`, row-major |

A counter value of `0` means no frame has been written yet. Track the counter value between reads — when it changes, a new frame is available. No locks are needed for a single-producer / single-consumer setup.

---

## Integrating with an existing OpenCV pipeline

Drop `FrameReader` in anywhere you currently use `cv2.VideoCapture`:

```python
import cv2
from streamrelay import FrameReader

reader = FrameReader(shm_name="myapp_frames")

while True:
    frame = reader.read_new()
    if frame is None:
        continue

    # use exactly like a VideoCapture frame
    cv2.imshow("preview", frame)
    if cv2.waitKey(1) == 27:
        break

reader.close()
```

Or wrap it as a drop-in `VideoCapture` replacement:

```python
from streamrelay import FrameReader
import cv2, numpy as np

class StreamRelayCapture:
    def __init__(self, shm_name="myapp_frames"):
        self._reader = FrameReader(shm_name=shm_name, attach_timeout=10.0)
        self._last = np.zeros((480, 640, 3), dtype=np.uint8)

    def read(self):
        f = self._reader.read_latest()
        if f is not None:
            self._last = f
        return True, self._last

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:          return 30.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._last.shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._last.shape[0])
        return 0.0

    def isOpened(self): return True
    def release(self):  self._reader.close()
```

Pass an instance of `StreamRelayCapture` anywhere your code expects a `cv2.VideoCapture`.

---

## Development

```bash
git clone <this-repo>
cd packages/streamrelay
pip install -e ".[h264,psutil]"
streamrelay-server --shm-name dev_frames
```

Open the printed URL in your browser. Edits to `client/app.js`, `client/index.html`, or `client/style.css` trigger an automatic browser reload via the `/livereload` SSE endpoint.

---

## License

MIT.
