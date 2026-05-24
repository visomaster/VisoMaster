# 08 · StreamRelay (WebRTC ingress)

`packages/streamrelay/` is a git submodule (`https://github.com/crazidev/streamrelay.git`) bundled at `packages/streamrelay/src/streamrelay/`. It's an aiohttp + aiortc server that receives video from a phone or browser, decodes it to BGR `numpy.ndarray` frames, and writes them into a shared-memory block that VisoMaster's main process reads.

## Why a separate process?

- aiortc has its own asyncio loop. Mixing it into Qt's event loop is fragile.
- Crashing the relay (e.g. malformed RTP, bad SDP) shouldn't take down the GUI.
- The ML pipeline already runs on the GPU; the relay is mostly CPU/network-bound and benefits from being on its own.
- Shared memory is **zero-copy** — no IPC serialization tax for 1080p frames at 30 fps.

This **is the design template for splitting the rest of VisoMaster.** A future Python service hosting the inference pipeline can use the same idea: a long-running process exposing an HTTP/WebSocket API.

## Endpoints exposed by `StreamServer`

Defined in `packages/streamrelay/src/streamrelay/server.py::StreamServer`:

| Path | Methods | Purpose |
|---|---|---|
| `/` | GET | Serves `client/index.html` — the browser camera client. |
| `/app.js` | GET | Bundled vanilla-JS client logic. |
| `/style.css` | GET | Client styles. |
| `/whip` | POST | WHIP — accepts an SDP offer, returns an answer. Used by Larix Broadcaster, OBS, etc. |
| `/ws/stream` | WebSocket | Browser client signaling channel. |
| `/livereload` | GET (SSE) | Dev-only live-reload of bundled assets. |

The same server binds **two ports** by default:

- HTTP on `9091`
- HTTPS on `9090` (using a self-signed cert; auto-generated on first run)

`generate_self_signed_cert(...)` creates `cert.pem` + `key.pem` if they don't exist. Cert path defaults to `app/ui/external/certificates/`.

## Frame protocol (`packages/streamrelay/src/streamrelay/protocol.py`)

```
Bytes  0..3 : counter (uint32 LE)  — incremented on every write
Bytes  4..7 : width   (uint32 LE)
Bytes  8..11: height  (uint32 LE)
Bytes 12..N : raw BGR frame data (W*H*3 bytes)
```

The block is sized for the worst case (1920×1080×3 + 12 = 6 220 812 bytes) and allocated once. The header lets the consumer detect new frames without locks.

VisoMaster constants (in `app/processors/video_processor.py`):

```python
VISOMASTER_SHM_NAME = "visomaster_webrtc_frame"
```

Producers and consumers must agree on this name. The UI worker passes it to `streamrelay.run_server(shm_name=...)` and the `VideoProcessor` reads it with `SharedMemory(name=...)`.

## How VisoMaster spawns the server

`app/ui/widgets/ui_workers.py::TargetMediaLoaderWorker.load_webrtc`:

```python
import multiprocessing
from streamrelay.server import run_server

http_port  = int(main_window.control.get('WebRTCHttpPortText', 9091))
https_port = int(main_window.control.get('WebRTCHttpsPortText', 9090))
host       = main_window.control.get('WebRTCBindAddressText', '0.0.0.0').strip() or '0.0.0.0'

cert_file = "app/ui/external/certificates/cert.pem"
key_file  = "app/ui/external/certificates/key.pem"

p = multiprocessing.Process(
    target=run_server,
    kwargs={
        'http_port':  http_port,
        'https_port': https_port,
        'cert_file':  cert_file,
        'key_file':   key_file,
        'host':       host,
        'shm_name':   "visomaster_webrtc_frame",
    },
    daemon=True,
)
p.start()
main_window.webrtc_server_process = p
```

## How VisoMaster reads from the relay

`VideoProcessor.process_next_webrtc_frame`:

```python
counter = struct.unpack_from("<I", self.webrtc_shm.buf, 0)[0]
if counter == self._last_webrtc_counter:
    return                              # no new frame
self._last_webrtc_counter = counter
w = struct.unpack_from("<I", self.webrtc_shm.buf, 4)[0]
h = struct.unpack_from("<I", self.webrtc_shm.buf, 8)[0]
raw = bytes(self.webrtc_shm.buf[SHM_HEADER_BYTES : SHM_HEADER_BYTES + w*h*3])
frame_bgr = numpy.frombuffer(raw, dtype=numpy.uint8).reshape((h, w, 3)).copy()
frame_rgb = frame_bgr[..., ::-1]
frame_rgb = self._apply_streaming_transforms(frame_rgb)
self.start_frame_worker(self.current_frame_number, frame_rgb)
```

If shared memory isn't yet allocated (server still starting up), `_try_attach_webrtc_shm` polls every 500 ms until it succeeds, then swaps the timer over.

## Connection methods (from README)

| Method | URL | Use case |
|---|---|---|
| Web client | `http://<host>:9091/` | Browser on phone/tablet |
| WHIP | `http://<host>:9091/whip` | Larix Broadcaster, OBS, GoCoder |
| HTTPS web client | `https://<host>:9090/` | Secure browser connection |
| HTTPS WHIP | `https://<host>:9090/whip` | Secure WHIP |

Larix configuration:

1. Settings → Connections → New Connection
2. URL: `http://<pc-ip>:9091/whip`
3. Codec: H.264 or VP8

## What this looks like in a React/Electron rewrite

**Keep StreamRelay as-is.** It already has a clean public API (`run_server(...)` + the SHM protocol). The new React UI can:

1. Show the current bind URL + a QR code so the user can scan to connect their phone.
2. Hit a future REST endpoint like `POST /api/sources/webrtc/start` that wraps the `multiprocessing.Process` spawn.
3. Stream the **processed** output back to the browser via either:
   - `<video>` + a server-side WebRTC track (re-using aiortc), or
   - WebSocket binary frames (simpler, OK for previews), or
   - HLS/DASH (best for offline or remote use).

The most natural answer: have the same StreamRelay process **also publish** a "swapped" track in addition to receiving the input track. That keeps the WebRTC plumbing in one place.
