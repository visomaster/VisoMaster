# WebSocket Channels

Two WebSocket endpoints provide real-time communication between the server and the React UI.

Base URL: `ws://localhost:8000`

---

## /ws/events — Bidirectional JSON control channel

Used for all control commands and server-push state events. Both sides send JSON text messages.

```js
const ws = new WebSocket('ws://localhost:8000/ws/events');
ws.onmessage = (e) => {
  const event = JSON.parse(e.data);
  console.log(event.type, event.payload);
};
```

---

### Server → Client events

All events have the shape `{ "type": string, "payload": object }`.

#### `frame_processed`

Emitted after every frame is processed. Use this to know a new frame is available on `/api/preview/snapshot`, or to update a frame counter.

```json
{
  "type": "frame_processed",
  "payload": { "frame_number": 142, "width": 1920, "height": 1080 }
}
```

#### `playback_state`

Emitted after play, stop, seek, step, and source-tab-changed commands. Sync your UI controls to this.

```json
{
  "type": "playback_state",
  "payload": {
    "is_playing": true,
    "is_recording": false,
    "current_frame": 142,
    "max_frame": 3600,
    "fps": 29.97,
    "file_type": "video",
    "loop_enabled": false
  }
}
```

#### `frame_position`

High-frequency position update emitted after every processed frame via the latest-wins position channel. Delivered separately from `playback_state` to avoid flooding the main event queue at 30 fps.

```json
{
  "type": "frame_position",
  "payload": { "current_frame": 143, "max_frame": 3600 }
}
```

#### `fps_update`

Emitted once per second during webcam/webrtc playback.

```json
{
  "type": "fps_update",
  "payload": { "fps": 28.4 }
}
```

#### `state_updated`

Emitted when a control or parameter value changes via the WebSocket (not via REST).

```json
{ "type": "state_updated", "payload": { "section": "control", "name": "_swap_enabled", "value": true } }
{ "type": "state_updated", "payload": { "section": "parameters", "face_id": "...", "name": "FaceRestorerBlendSlider", "value": 80 } }
{ "type": "state_updated", "payload": { "section": "playback", "event": "stopped" } }
```

#### `recording_finished`

Emitted when a recording is finalised and the output file is ready.

```json
{
  "type": "recording_finished",
  "payload": { "output_path": "C:/Videos/output/sample_2026_05_24_14_30_00.mp4" }
}
```

#### `virtcam_state`

Emitted after a virtual camera enable/disable attempt to report the actual state (may differ from the requested state if the camera failed to start).

```json
{ "type": "virtcam_state", "payload": { "enabled": true } }
```

#### `error`

Emitted when a processing error occurs (e.g. cannot read frame).

```json
{
  "type": "error",
  "payload": { "message": "Error Reading Frame 142" }
}
```

#### `pong`

Response to a `ping` command.

```json
{ "type": "pong" }
```

---

### Client → Server commands

All commands have the shape `{ "type": string, "payload"?: object }`.

#### Playback

| Command | Payload | Description |
|---|---|---|
| `play` | — | Start the processing loop. |
| `stop` | — | Stop the processing loop. |
| `seek` | `{ "frame": N }` | Seek to frame N and process it. |
| `step` | `{ "n": N }` | Step N frames forward (negative = rewind). |

#### Swap / edit mode

| Command | Payload | Description |
|---|---|---|
| `swap_enable` | — | Enable face swap, disable edit. |
| `swap_disable` | — | Disable face swap. |
| `edit_enable` | — | Enable face editor (LivePortrait), disable swap. |
| `edit_disable` | — | Disable face editor. |

#### State mutations

| Command | Payload | Description |
|---|---|---|
| `set_control` | `{ "name": "...", "value": ... }` | Set one global control value. |
| `set_parameter` | `{ "face_id": "...", "name": "...", "value": ... }` | Set one per-face parameter. |

Use `set_control` and `set_parameter` for low-latency slider updates (e.g. while dragging). For bulk updates use the REST endpoints instead.

#### Preview

| Command | Payload | Description |
|---|---|---|
| `preview_quality` | `{ "quality": 75 }` | Set JPEG quality for `/ws/preview` (1–100). |

#### Utility

| Command | Payload | Description |
|---|---|---|
| `ping` | — | Server responds with `pong`. |
| `open_preview_window` | — | Toggle the native Qt preview window (Qt modes only). |
| `source_tab_changed` | `{ "source": "media"\|"webcam"\|"streaming" }` | Tear down the current source and switch to the new one. |

---

## /ws/playback — Dedicated playback-state stream

Push-only high-frequency channel for playback position and state. Delivers the same data as `playback_state` events on `/ws/events` but via a dedicated latest-wins slot so 30 fps position updates never flood the main event queue.

Each message is a UTF-8 JSON text frame:

```json
{ "current_frame": 143, "max_frame": 3600, "is_playing": true, "fps": 29.97, "is_recording": false }
```

**Optional client → server:** send the text `"sync"` to request an immediate snapshot of the current state.

```js
const ws = new WebSocket('ws://localhost:8000/ws/playback');
ws.onmessage = (e) => {
  const state = JSON.parse(e.data);
  updateSeekBar(state.current_frame, state.max_frame);
  updatePlayButton(state.is_playing);
};

// Request immediate sync on connect
ws.onopen = () => ws.send('sync');
```

**Backpressure:** uses the same latest-frame-wins `asyncio.Event` pattern as `/ws/preview` — if the client is slower than the frame rate, intermediate positions are silently dropped and only the most recent state is delivered.

---

## /ws/preview — Binary JPEG frame stream

Push-only. The server sends a raw JPEG byte payload as a binary WebSocket message after every processed frame. No JSON framing — each message is directly renderable as an image.

```js
const ws = new WebSocket('ws://localhost:8000/ws/preview');
ws.binaryType = 'arraybuffer';

const img = document.getElementById('preview');
ws.onmessage = (e) => {
  const blob = new Blob([e.data], { type: 'image/jpeg' });
  const url = URL.createObjectURL(blob);
  // Revoke the previous URL to avoid memory leaks
  if (img.src) URL.revokeObjectURL(img.src);
  img.src = url;
};
```

**React hook example:**

```tsx
function usePreviewStream(url = 'ws://localhost:8000/ws/preview') {
  const [src, setSrc] = useState<string>('');

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    let prevSrc = '';

    ws.onmessage = (e) => {
      const blob = new Blob([e.data], { type: 'image/jpeg' });
      const next = URL.createObjectURL(blob);
      setSrc(next);
      if (prevSrc) URL.revokeObjectURL(prevSrc);
      prevSrc = next;
    };

    return () => {
      ws.close();
      if (prevSrc) URL.revokeObjectURL(prevSrc);
    };
  }, [url]);

  return src;
}

// Usage
function PreviewCanvas() {
  const src = usePreviewStream();
  return <img src={src} style={{ width: '100%' }} />;
}
```

**Optional quality control** — send a text message to reduce bandwidth:

```js
ws.send(JSON.stringify({ quality: 50 }));  // 1-100, default 75
```

### Backpressure

The server keeps at most 2 frames queued per subscriber. If your client is slower than the processing rate, old frames are dropped silently — you always get the latest frame, never a stale backlog.

### Latency

On localhost: ~15–30 ms end-to-end (encode + WS send + decode + render). For remote connections over LAN: ~30–80 ms depending on frame size and quality setting.
