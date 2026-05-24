# 10 · Proposed API for a React/Electron Frontend

This is one concrete design that maps the existing functionality onto an HTTP/WebSocket service. It's intentionally not the only option — the goal is to give you a mental model concrete enough to start building.

## High-level layout

```
┌────────────────────────────┐
│  React app                 │   Vite + TypeScript + a state lib
│  (or Electron renderer)    │
└─────┬───────────────┬──────┘
      │ REST/WS       │ WebRTC <video> tag
      ▼               │
┌────────────────────────────┐
│  Python service            │   FastAPI / Starlette + uvicorn
│   - thin controllers       │
│   - owns AppState          │
│   - owns ModelsProcessor   │
│   - owns frame pipeline    │
└─────┬───────────────┬──────┘
      │ shared mem    │ aiortc (re)publishing
      ▼               │
┌────────────────────────────┐
│  StreamRelay subprocess    │   already exists, no change
└────────────────────────────┘
```

Recommended stack:

- **API:** FastAPI (built-in Pydantic + WebSocket + auto OpenAPI for the React types).
- **Realtime:** WebSocket for control + status messages; WebRTC for the live preview track.
- **Frontend:** React + TanStack Query + Zustand (or Redux Toolkit) + Tailwind + shadcn/ui.
- **Build:** Vite. For Electron, use Electron Forge with the Vite template.

## REST endpoints

### System

| Method | Path | Purpose | Backed by |
|---|---|---|---|
| GET | `/api/system/info` | OS, GPU, CUDA, ORT/TRT versions, FFmpeg presence | `models_processor.device`, `torch.cuda.get_device_properties()`, `misc_helpers.is_ffmpeg_in_path()` |
| GET | `/api/system/gpu-memory` | Used / total GPU memory MB | `models_processor.get_gpu_memory()` |
| POST | `/api/system/clear-memory` | `torch.cuda.empty_cache()` + `gc.collect()` | `models_processor.clear_gpu_memory()` |
| GET | `/api/system/providers` | Available ORT providers + currently active | `onnxruntime.get_available_providers()` |
| POST | `/api/system/providers` | Switch active provider | `models_processor.switch_providers_priority(name)` |

### Schemas (drives the React parameter UI)

| Method | Path | Returns |
|---|---|---|
| GET | `/api/schema/control` | `SETTINGS_LAYOUT_DATA` as JSON (drop function refs, replace with event names) |
| GET | `/api/schema/parameters/swap` | `SWAPPER_LAYOUT_DATA` |
| GET | `/api/schema/parameters/common` | `COMMON_LAYOUT_DATA` |
| GET | `/api/schema/parameters/face-editor` | `FACE_EDITOR_LAYOUT_DATA` |
| GET | `/api/schema/dfm-models` | `misc_helpers.get_dfm_models_data()` (filename → path) |

### State

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state` | Full snapshot — same shape as `last_workspace.json` |
| PUT | `/api/state/control` | Patch global control (provider, threads, …) |
| PUT | `/api/state/parameters/{face_id}` | Patch one face's parameters |
| POST | `/api/state/copy/{face_id}` | Capture parameters into clipboard |
| POST | `/api/state/paste/{face_id}` | Apply clipboard parameters |
| POST | `/api/state/reset/{face_id}` | Reset to defaults |

### Workspaces

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/workspace` | Returns current workspace JSON |
| POST | `/api/workspace/save` | Body `{filename: string}` — saves the current state to a file |
| POST | `/api/workspace/load` | Body `{filename: string}` — loads from a file |
| POST | `/api/workspace/reset` | Clears all working set |

### Target media

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/target-media` | List target media cards |
| POST | `/api/target-media/scan-folder` | `{path, recursive: bool}` — scans + returns `[{media_id, media_path, file_type, thumbnail_url}]` |
| POST | `/api/target-media/upload` | Multipart upload (Electron only — over the local network upload would be expensive) |
| POST | `/api/target-media/{media_id}/select` | Make this the active capture |
| DELETE | `/api/target-media/{media_id}` | Remove |
| GET | `/api/target-media/{media_id}/thumbnail` | Cached jpg/png from `.thumbnails/` |

### Source faces & embeddings

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/input-faces` | List source face cards |
| POST | `/api/input-faces/scan-folder` | `{path, recursive}` |
| POST | `/api/input-faces/upload` | Multipart |
| DELETE | `/api/input-faces/{face_id}` | |
| GET | `/api/embeddings` | List merged embeddings |
| POST | `/api/embeddings/merge` | `{name, input_face_ids: [...]}` — merge selected sources |
| GET | `/api/embeddings/export` | Download embeddings JSON |
| POST | `/api/embeddings/import` | Upload embeddings JSON |
| DELETE | `/api/embeddings/{embedding_id}` | |

### Target faces (detected in current media)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/target-faces/find` | Run detector on current frame; returns new target face cards |
| GET | `/api/target-faces` | List current target faces |
| POST | `/api/target-faces/{face_id}/select` | Make this the parameter-edit target |
| POST | `/api/target-faces/{face_id}/assign-input/{input_face_id}` | Toggle assignment |
| POST | `/api/target-faces/{face_id}/assign-embedding/{embedding_id}` | Toggle assignment |
| DELETE | `/api/target-faces/{face_id}` | Remove |
| POST | `/api/target-faces/clear` | Remove all |

### Playback

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/playback` | `{file_type, fps, current_frame, max_frame, processing, recording}` |
| POST | `/api/playback/seek` | `{frame: int}` |
| POST | `/api/playback/play` | Start playing |
| POST | `/api/playback/stop` | Stop |
| POST | `/api/playback/step` | `{n: int}` (negative = rewind) |
| POST | `/api/playback/record/start` | Body `{output_folder?: string}` (defaults to control.OutputMediaFolder) |
| POST | `/api/playback/record/stop` | Returns `{output_path: string}` |
| POST | `/api/playback/save-frame` | Saves current frame to disk |
| GET | `/api/playback/markers` | List markers |
| POST | `/api/playback/markers` | Add at current position |
| DELETE | `/api/playback/markers/{frame_number}` | |

### Streaming sources

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/sources/webcams` | Enumerate webcams (returns thumbnails) |
| POST | `/api/sources/webcams/{index}/select` | Open webcam at this index |
| POST | `/api/sources/webrtc/start` | Spawn StreamRelay subprocess; returns `{http_url, https_url, whip_url}` |
| POST | `/api/sources/webrtc/stop` | Kill StreamRelay |
| GET | `/api/sources/webrtc/status` | `{running, frames_received, last_frame_at}` |
| PUT | `/api/sources/transform` | `{rotation: 0|90|180|270, flip_h, flip_v}` for the active source |

### Output / display

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/output/virtualcam/enable` | `{backend: 'obs'|'unitycapture'}` |
| POST | `/api/output/virtualcam/disable` | |
| POST | `/api/output/window/show` | (Electron-only — show borderless OBS window) |
| POST | `/api/output/window/hide` | |

## WebSocket: `/ws/events`

Server pushes JSON events:

```ts
type Event =
  | { type: 'frame_processed', frame_number: number, timestamp: number }
  | { type: 'fps_update', fps: number }
  | { type: 'gpu_memory', used_mb: number, total_mb: number }
  | { type: 'model_loading', model_name: string }
  | { type: 'model_loaded', model_name: string }
  | { type: 'state_updated', section: 'control'|'parameters'|'target_faces'|... }
  | { type: 'recording_progress', frames: number, elapsed_s: number }
  | { type: 'recording_finished', output_path: string, avg_fps: number }
  | { type: 'webrtc_status', running: boolean, frames_received: number }
  | { type: 'error', code: string, message: string };
```

Client → server messages keep state changes interactive without HTTP latency:

```ts
type Command =
  | { type: 'play' }
  | { type: 'stop' }
  | { type: 'seek', frame: number }
  | { type: 'set_parameter', face_id: string, name: string, value: any }
  | { type: 'set_control', name: string, value: any };
```

## Live frame transport

Three viable options:

### A. WebRTC (recommended)

The Python service publishes a single `RTCPeerConnection` track containing the **processed** frames. The React app consumes it with a `<video>` element. Latency on localhost: ~50 ms.

Implementation: reuse aiortc (already a dep). Add a track that pulls frames from the same place `display_next_frame` does today. Signaling endpoint at `POST /api/preview/offer` answers an SDP offer.

### B. WebSocket binary frames

Send JPEG/WebP-encoded frames as binary WebSocket messages. Simpler, works without WebRTC. Latency: ~80–150 ms depending on encoding.

```python
# server side (per-frame in display loop)
ok, jpg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
await ws.send_bytes(jpg.tobytes())
```

```tsx
// client side
ws.onmessage = (ev) => {
  const blob = new Blob([ev.data], { type: 'image/jpeg' });
  setPreviewSrc(URL.createObjectURL(blob));
};
```

### C. MJPEG over HTTP

`GET /api/preview/stream` returns `multipart/x-mixed-replace; boundary=frame`. Simplest of all (works in `<img>` directly), but no fine-grained control.

## React state shape

```ts
// Mirrors Python AppState; keep it as the single source of truth on the client too.
interface AppState {
  control: Record<string, ControlValue>;
  parameters: Record<FaceId, ParametersDict>;
  defaultParameters: ParametersDict;
  control_schema: LayoutData;        // from /api/schema/control
  swap_schema:    LayoutData;
  common_schema:  LayoutData;
  editor_schema:  LayoutData;

  targetMedia:       Record<MediaId, MediaCard>;
  selectedMediaId:   MediaId | null;

  targetFaces:       Record<FaceId, TargetFaceCard>;
  selectedFaceId:    FaceId | null;
  inputFaces:        Record<FaceId, InputFaceCard>;
  embeddings:        Record<EmbeddingId, EmbeddingCard>;

  markers:           Record<number, MarkerData>;
  playback: { fps, currentFrame, maxFrame, isPlaying, isRecording, fileType };

  webrtc: { running: boolean, urls: WebRTCUrls | null, fps: number };
  gpu:    { usedMB: number, totalMB: number };
  modelLoading: string | null;
}
```

State sync strategy:

- On boot: `GET /api/state` → hydrate Zustand store.
- Use TanStack Query for "what's on disk" (target media folders, schemas).
- WebSocket events apply patches to Zustand directly.
- Writes go through HTTP and the success response includes the updated section so the optimistic update is verified.

## Why FastAPI

- Pydantic models can be generated to TypeScript via `pydantic-to-typescript` or by exporting OpenAPI and running `openapi-typescript-codegen`. Free, type-safe client.
- Native WebSocket support in the same app.
- Easy to mount aiortc routes alongside REST.

## Electron vs browser

- **Electron** wins if you need: file system access (drag a folder of source faces), virtual camera output, OBS borderless window, native install. The Python service runs as a sidecar process spawned by Electron's main process.
- **Browser-only** is fine if you can accept: file uploads instead of folder scans, no virtual cam (use a WebRTC pipe instead), no borderless OBS window. The same Python service runs as `python -m visomaster.api`.

The recommended path is **both**: build the React app first, then wrap it in Electron later. The API doesn't change.
