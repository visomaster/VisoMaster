# 11 · React/Electron Migration Roadmap

A staged plan for taking the existing PySide6 codebase and ending up with a React (or Electron) frontend on top of an unchanged inference pipeline. Each phase produces a runnable artifact.

## Guiding principles

1. **Don't fork the inference code.** The pipeline (`app/processors/*`) is the asset. Refactor at the seams, never rewrite. The Qt UI can keep running on the same backend until the new UI is at parity.
2. **Make state explicit before making it networked.** Lifting `AppState` out of `MainWindow` is the unblocker for everything else.
3. **WebRTC ingestion already shows the way.** It runs in a subprocess with a defined protocol. The same shape works for the rest.
4. **Ship something each phase.** Even Phase 1 produces a working CLI replay tool — that's the proof the backend stands alone.

## Phase 0 — Map and pin (1 week)

- ✅ Read these docs.
- Pin all current behaviors with a checklist (load video, find faces, swap, record, webcam, WebRTC, virtual cam, OBS window, marker editing, embedding merge). This is the parity bar.
- Tag the current commit `pre-react`.

## Phase 1 — Extract `AppState` (1–2 weeks)

**Goal.** A pure-Python `AppState` dataclass tree replaces direct `MainWindow.*` access in actions. Qt UI keeps working.

Tasks:

- Create `app/core/state.py`:
  ```python
  @dataclass
  class AppState:
      control: dict[str, Any]
      default_parameters: dict[str, Any]
      parameters: dict[str, ParametersDict]   # face_id → params
      target_faces: dict[str, TargetFace]
      input_faces: dict[str, InputFace]
      embeddings: dict[str, MergedEmbedding]
      markers: dict[int, Marker]
      target_media: list[MediaRef]
      selected_media_id: str | None
      selected_face_id: str | None
      ...
  ```
- Move `convert_parameters_to_supported_type`, `save_current_workspace`, `load_saved_workspace` to operate on `AppState`.
- In `MainWindow.initialize_variables`, instantiate `AppState` and **assign by reference**:
  ```python
  self.state = AppState(...)
  self.parameters = self.state.parameters     # alias for backwards-compat
  self.control    = self.state.control
  self.target_faces = self.state.target_faces
  ```
- Verify `last_workspace.json` round-trips byte-for-byte.

**Deliverable.** Identical Qt app, but every action function is one mechanical refactor away from being UI-free.

## Phase 2 — Decouple `FrameWorker` from Qt (1–2 weeks)

**Goal.** `FrameWorker` and `VideoProcessor` no longer import from `app/ui/*`.

Tasks:

- Replace direct Qt signal calls with a callback interface:
  ```python
  class FrameSink(Protocol):
      def on_frame(self, frame_number: int, frame_bgr: np.ndarray) -> None: ...
      def on_progress(self, frame_number: int, total: int) -> None: ...
      def on_error(self, msg: str) -> None: ...
  ```
- `FrameWorker.run` calls `sink.on_frame(...)` instead of emitting `frame_processed_signal`.
- Move pixmap conversion to a Qt-side `QtFrameSink` adapter.
- Move `update_graphics_view` / `set_play_button_icon_to_play` calls out of `VideoProcessor`. Replace with state events:
  ```python
  class StateBus:
      on_processing_changed: Callable[[bool], None]
      on_recording_changed:  Callable[[bool], None]
      on_fps_update:         Callable[[float], None]
  ```
- Keep `FrameWorker` reading parameters from `AppState` (parameter snapshots happen in `run()` like today).

**Deliverable.** `python -m app.cli.replay <input_video> <swap_config.json> <output.mp4>` — runs the full pipeline headless. Proves the backend is UI-free.

## Phase 3 — FastAPI service skeleton (1 week)

**Goal.** A FastAPI app that hosts `AppState`, `ModelsProcessor`, `VideoProcessor`, and exposes the schema endpoints.

Tasks:

- New module `app/api/server.py`. `uvicorn app.api.server:app --reload`.
- Endpoints (initially): `/api/system/info`, `/api/schema/*`, `/api/state`, `/api/state/control`, `/api/state/parameters/{face_id}`.
- `lifespan` context creates the `AppState`, `ModelsProcessor`, `VideoProcessor` once.
- WebSocket `/ws/events`. Backed by an `asyncio.Queue` that the StateBus pushes into.
- Build the OpenAPI schema → generate `frontend/src/api/types.ts` via `openapi-typescript`.

**Deliverable.** Hit `GET /api/state` from curl, get JSON identical to `last_workspace.json`. Hit `PUT /api/state/control` to switch detector model and watch the WebSocket emit `state_updated`.

## Phase 4 — React shell (1–2 weeks)

**Goal.** A React app that renders the parameter panels, target/source/embedding lists, and a static preview from `/api/preview/snapshot`.

Tasks:

- Vite + React + TypeScript + Tailwind + shadcn/ui scaffold under `frontend/`.
- `useAppState` Zustand store, hydrated from `/api/state` on mount.
- `useEvents` hook subscribed to `/ws/events`, applies patches.
- Generic `<ParameterPanel layoutData={...} values={...} onChange={...} />` that maps the existing `LayoutDictTypes` to widgets:
  - `ToggleButton` → shadcn `Switch`
  - `SelectionBox` → shadcn `Select`
  - `ParameterSlider` → shadcn `Slider`
  - `ParameterDecimalSlider` → shadcn `Slider` with float coercion
  - `ParameterText` → `Input`
- Three panels: Common, Swap, Face Editor (per-face) and Settings (global).
- Card lists for target media / target faces / input faces / embeddings.

**Deliverable.** A static React app that mirrors the Qt UI's parameter editor. No live preview yet.

## Phase 5 — Live preview (1–2 weeks)

**Goal.** The React app shows the processed video in real time.

Path A (simpler): WebSocket binary frames.

- Add `/ws/preview` that the `VideoProcessor`'s display loop pushes JPEG-encoded frames into.
- React: a `<canvas>` or `<img>` consuming the stream.

Path B (lower latency): WebRTC.

- Add `POST /api/preview/offer` answering an SDP offer with a track that pulls from the existing display loop.
- React: `RTCPeerConnection` + `<video>` element.

Either way, also stream the playback controls: `POST /api/playback/play`, `seek`, `step`, `markers`, etc.

**Deliverable.** Pick a target video in the browser, click Play, see the swapped output live.

## Phase 6 — File handling (Electron split) (1–2 weeks)

**Goal.** Decide how the user picks files. Two paths:

- **Electron path (recommended):** wrap the React build in Electron. The Python service runs as a sidecar. Use Electron's `dialog.showOpenDialog` for folder/file pickers; pass the absolute path back to the API. Same UX as today.
- **Browser path:** add `POST /api/upload` for multipart uploads. Add `POST /api/scan` that takes server-relative paths. Less convenient on the user's local machine.

The API doesn't change between these — only the file-picker UX.

**Deliverable.** Working app on Windows + Linux for the developer's local machine. No remote-deploy concerns yet.

## Phase 7 — Streaming sources (1 week)

**Goal.** Webcam + WebRTC sources work end-to-end in the new UI.

Tasks:

- `GET /api/sources/webcams` — enumerate.
- `POST /api/sources/webcams/{index}/select` — open + show preview.
- `POST /api/sources/webrtc/start` — spawn StreamRelay (no change to the relay itself).
- React: a "source picker" UI with three modes (file, webcam, WebRTC); a QR code for the WebRTC URL.
- Streaming transforms (rotation/flip) become `PUT /api/sources/transform`.

**Deliverable.** Phone-as-camera works in the React UI exactly like in the Qt UI.

## Phase 8 — Recording + virtual cam + OBS window (1 week)

**Goal.** All output paths work.

Tasks:

- `POST /api/playback/record/start` / `stop` (already designed).
- Virtual cam endpoints (Electron only — `pyvirtualcam` requires native).
- The borderless output window is **Electron-only**. Implement it as a frameless `BrowserWindow` that loads `preview.html` (same WebRTC track).

**Deliverable.** Stream face-swapped output to OBS via virtual cam or window capture.

## Phase 9 — Polish (1–2 weeks)

- Workspace save/load UI.
- Markers timeline scrub.
- Embedding merge/export/import.
- Theme switcher (Dark / Dark-Blue / Light).
- Keyboard shortcuts (port the ones from `keyPressEvent`).
- GPU memory bar.
- Toast notifications (replace `pyqt-toast-notification` with `sonner`).
- Loading dialog when models load (driven by `model_loading` events).

## Phase 10 — Decommission Qt (optional)

Once parity is reached, the Qt code can stay (great for power users / debugging) or be deleted. Either way, the inference code is unchanged. The success metric is **the React app passes the parity checklist from Phase 0**.

## Risks and how to retire them

| Risk | Mitigation |
|---|---|
| Hidden Qt deps in inference helpers | Run the Phase 2 CLI in CI to guarantee headless. |
| Performance regression from JPEG/WebSocket transport | Have WebRTC ready as the fallback. Localhost should easily hit 30 fps at 1080p. |
| Parameter schema drift between layout-data and React UI | Single source of truth: serve `LAYOUT_DATA` as JSON; React consumes it directly. Don't duplicate. |
| Workspace JSON migration | The `ParametersDict` fallback already handles unknown keys. Use the same pattern in the API: ignore unknown keys; fill defaults on read. |
| GPU memory leaks across reloads | Already present in the Qt app. The new service runs as a long-lived process — no worse than today, and easier to monitor. |
| WebRTC network surface | Bind to `127.0.0.1` by default; require explicit opt-in for `0.0.0.0`. The current app already does this. |

## Recommended skeleton

```
frontend/
├── src/
│   ├── api/                 # generated types + axios/ky client
│   ├── components/
│   │   ├── ParameterPanel.tsx
│   │   ├── CardList.tsx
│   │   ├── PreviewCanvas.tsx
│   │   └── ...
│   ├── store/
│   │   ├── appState.ts      # Zustand
│   │   └── events.ts        # WS subscription
│   ├── pages/
│   │   ├── Editor.tsx
│   │   ├── Settings.tsx
│   │   └── Workspace.tsx
│   └── main.tsx
├── public/
└── vite.config.ts

app/
├── api/
│   ├── server.py            # FastAPI app
│   ├── routes/
│   │   ├── system.py
│   │   ├── state.py
│   │   ├── playback.py
│   │   ├── target_media.py
│   │   ├── input_faces.py
│   │   ├── target_faces.py
│   │   ├── sources.py
│   │   └── workspace.py
│   ├── schemas.py           # Pydantic mirrors of AppState
│   └── ws.py
├── core/
│   └── state.py             # AppState dataclass
├── processors/              # unchanged inference code
└── ui/                      # legacy Qt UI; runs alongside
```
