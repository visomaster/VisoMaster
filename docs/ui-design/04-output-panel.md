# Output Panel

The output panel lives at the bottom of the left column. It controls where the processed frames go after the swap pipeline. Only one output mode can be active at a time, so a tab strip is used.

```
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT                                                 [−] │
│                                                             │
│  [Record] [Virtual Cam] [Window] [Stream]                   │
│  ─────────────────────────────────────────────────────────  │
│  (tab content)                                              │
│                                                             │
│  Output FPS: 28.4  ████████████░░░░ 4.4 / 24 GB            │
└─────────────────────────────────────────────────────────────┘
```

---

## Tab: Record (to file)

```
┌─────────────────────────────────────┐
│  Output folder:                     │
│  [C:/Videos/output          ] [📁]  │
│                                     │
│  [⏺ Start Recording]                │
│  ── or ──                           │
│  [💾 Save Current Frame]            │
│                                     │
│  Status: ● Recording  00:01:23      │  ← timer while recording
└─────────────────────────────────────┘
```

- Folder picker → `PUT /api/state/control { OutputMediaFolder }`.
- **Start Recording** → `POST /api/playback/record/start`. Button becomes "Stop Recording" (red).
- **Stop Recording** → `POST /api/playback/record/stop` → shows output path as a toast.
- **Save Current Frame** → `POST /api/playback/save-frame`.
- Timer counts up from 0 while `is_recording` is true (from `playback_state` WS event).
- On `recording_finished` WS event → show a toast with the output path and a "Open folder" link.

---

## Tab: Virtual Camera

```
┌─────────────────────────────────────┐
│  Backend: [OBS ▾]                   │
│                                     │
│  [Enable Virtual Camera]            │
│  Status: ● Active                   │
└─────────────────────────────────────┘
```

- Backend dropdown → `PUT /api/state/control { VirtCamBackendSelection }`.
- Toggle → `PUT /api/state/control { SendVirtCamFramesEnableToggle: true/false }`.
- Status badge from `control.SendVirtCamFramesEnableToggle`.

---

## Tab: Output Window

```
┌─────────────────────────────────────┐
│  A borderless window for OBS        │
│  "Window Capture" source.           │
│                                     │
│  [Show Output Window]               │
│  Status: ○ Hidden                   │
└─────────────────────────────────────┘
```

- Toggle → `PUT /api/state/control { OutputWindowEnableToggle: true/false }`.
- Note: only meaningful in Electron (native window). In browser mode, show a note explaining this is desktop-only.

---

## Tab: Stream (WebSocket output)

```
┌─────────────────────────────────────┐
│  Stream processed frames via        │
│  WebSocket to external consumers.   │
│                                     │
│  Preview WS: ws://localhost:8000/   │
│              ws/preview             │
│                                     │
│  Quality: [75 ──●── ]               │
│  [Copy URL]                         │
│                                     │
│  Connected clients: 2               │
└─────────────────────────────────────┘
```

- Quality slider → sends `{ "quality": N }` text message on `/ws/preview`.
- "Copy URL" copies the WebSocket URL to clipboard.
- Connected clients count — not yet tracked by the API (future enhancement).

---

## Output FPS bar

Always visible at the bottom of the output panel regardless of active tab:

```
Output FPS: 28.4   VRAM: ████████░░░░ 4.4 / 24 GB
```

- FPS from `fps_update` WS event (for webcam/webrtc) or calculated from `frame_processed` events (for video).
- VRAM from `GET /api/system/gpu-memory` polled every 5s.
- VRAM bar color: green < 70%, amber 70–85%, red > 85%.
