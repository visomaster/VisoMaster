# Left Panel

The left panel has four stacked sections, each collapsible. Order is fixed (not draggable — these are structural, not parameter blocks).

```
┌─────────────────────────────────────┐
│  SOURCE                         [−] │  ← collapsible header
│  ┌─────────────────────────────┐    │
│  │ [Media] [Webcam] [Streaming]│    │  ← tab strip
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │  (tab content — see below)  │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │  Source preview (thumbnail) │    │
│  └─────────────────────────────┘    │
├─────────────────────────────────────┤
│  FACES                          [−] │
│  (see 03-faces-panel.md)            │
├─────────────────────────────────────┤
│  OUTPUT                         [−] │
│  (see 04-output-panel.md)           │
└─────────────────────────────────────┘
```

---

## Source section

### Tab: Media

```
┌─────────────────────────────────────┐
│  [📁 Browse folder]  [🔍 Search...] │
│  [☑ Images] [☑ Videos]             │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐           │
│  │▶  │ │▶  │ │▶  │ │▶  │  ← cards  │
│  │   │ │   │ │   │ │   │           │
│  └───┘ └───┘ └───┘ └───┘           │
│  clip1  clip2  img1  img2           │
└─────────────────────────────────────┘
```

- **Browse folder** button → `POST /api/target-media/scan-folder`
- **Search box** → filters the card grid client-side
- **Image / Video filter toggles** → hide/show by file_type
- **Cards** — thumbnail + filename label. Click to select → `POST /api/target-media/{id}/select`. Right-click context menu: Remove.
- **Drag-and-drop** — drag files/folders from OS onto the list area.
- Selected card gets a sky-500 ring border.

### Tab: Webcam

```
┌─────────────────────────────────────┐
│  Backend: [Default ▾]               │
│  Resolution: [1280×720 ▾]           │
│  FPS: [30 ▾]                        │
│  ┌───┐ ┌───┐                        │
│  │📷 │ │📷 │  ← one card per cam    │
│  │ 0 │ │ 1 │                        │
│  └───┘ └───┘                        │
│                                     │
│  Transform:                         │
│  [↺ CCW] [↻ CW] [↔ Flip H] [↕ V]  │
└─────────────────────────────────────┘
```

- On tab activate → `GET /api/sources/webcams` to enumerate.
- Click a card → `POST /api/sources/webcams/{index}/select`.
- Backend / Resolution / FPS dropdowns → `PUT /api/state/control` with `WebcamBackendSelection`, `WebcamMaxResSelection`, `WebCamMaxFPSSelection`.
- Transform buttons → `PUT /api/sources/transform`.

### Tab: Streaming (WebRTC)

```
┌─────────────────────────────────────┐
│  ● LIVE  FPS: 28.4                  │  ← status badge + fps
│                                     │
│  [▶ Start]  [■ Stop]  [⚙ Settings] │
│                                     │
│  HTTP:  http://192.168.1.10:9091/   │
│  HTTPS: https://192.168.1.10:9090/  │
│  WHIP:  http://192.168.1.10:9091/whip│
│                                     │
│  ┌─────────────────────────────┐    │
│  │  QR code (http_url)         │    │
│  └─────────────────────────────┘    │
│                                     │
│  Transform:                         │
│  [↺ CCW] [↻ CW] [↔ Flip H] [↕ V]  │
└─────────────────────────────────────┘
```

- **Start** → `POST /api/sources/webrtc/start` → shows URLs + QR code.
- **Stop** → `POST /api/sources/webrtc/stop`.
- **Status badge** — green "● LIVE" when `frames_received > 0` (poll `GET /api/sources/webrtc/status` every 2s). Grey "○ Waiting" otherwise.
- **FPS** — from `fps_update` WebSocket event.
- **Settings popup** (⚙ button) → `Dialog` with:
  - HTTP Port (text input, default 9091)
  - HTTPS Port (text input, default 9090)
  - Bind Address (text input, default 0.0.0.0)
  - [Apply] → `PUT /api/state/control` with `WebRTCHttpPortText`, `WebRTCHttpsPortText`, `WebRTCBindAddressText`, then restart server.
- **QR code** — rendered client-side from `http_url` using a QR library (e.g. `qrcode.react`).
- Transform buttons → `PUT /api/sources/transform`.

### Source preview

A small thumbnail below the tab content showing the current source frame. Updated from `/ws/preview` at reduced quality (quality: 30) so it doesn't compete with the main canvas.

```
┌─────────────────────────────────────┐
│  Source preview                     │
│  ┌─────────────────────────────┐    │
│  │  [live thumbnail ~160×90]   │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

This is always the raw (unprocessed) frame — it shows what the source looks like before swap. The center canvas shows the processed output.

> Implementation note: the source preview needs a separate frame path. For now it can be a lower-quality snapshot from `GET /api/preview/snapshot` polled at 5fps, or a second `/ws/preview` connection at quality 20. The processed output goes to the main canvas at quality 75.
