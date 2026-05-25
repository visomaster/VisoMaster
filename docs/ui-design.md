# VisoMaster React UI — Layout Preview

---

## Top Bar (always visible)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  [VM] VisoMaster                                                                                 │
│                                                                                                  │
│  CPU  ████████░░  34%    GPU  ████████████░░  67%    VRAM  ████████░░  4.4/24GB                 │
│                                                                                                  │
│                              [CUDA]  [TensorRT]  [TRT-Engine]          [🗑 Clear VRAM]          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- CPU % polled from system (browser: `navigator` API or server-sent).
- GPU % + VRAM from `GET /api/system/gpu-memory` every 3s.
- **Provider selector** — segmented control. Active segment highlighted sky-500. Clicking calls `POST /api/system/providers { provider }`. While reloading: active segment shows spinner, others disabled. Toast on completion.
- **Clear VRAM** → confirmation popover → `POST /api/system/clear-memory`.
- Bars color: green < 70%, amber 70–85%, red > 85%.

---

## Main Layout — 4 columns

```
┌──────────────────┬──────────────────────┬──────────────────────┬──────────────────┐
│  INPUT SOURCE    │  FACE SWAPPING       │  FACE OPTIONS        │  OUTPUT          │
│  (col 1)         │  (col 2)             │  (col 3)             │  (col 4)         │
└──────────────────┴──────────────────────┴──────────────────────┴──────────────────┘
```

---

## Column 1 — Input Source

```
┌──────────────────────────────────────┐
│  INPUT SOURCE                        │
│                                      │
│  ┌──────────────────────────────┐    │
│  │                              │    │
│  │   Source preview             │    │
│  │   (raw / unprocessed)        │    │
│  │   /ws/preview  quality: 20   │    │
│  │                              │    │
│  │   FPS: 28.4                  │    │
│  └──────────────────────────────┘    │
│                                      │
│  Transform:  [↺][↻]  [↔ H][↕ V]     │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  Source type:                        │
│  [● Media]  [○ Webcam]  [○ Stream]   │
│  (radio / segmented control)         │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── when Media is selected ────────  │
│                                      │
│  [📁 Browse folder]  [🔍 Search...]  │
│  [☑ Images]  [☑ Videos]             │
│                                      │
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐           │
│  │▶ │ │▶ │ │🖼│ │▶ │ │🖼│           │
│  └──┘ └──┘ └──┘ └──┘ └──┘           │
│  clip1 clip2 img1 clip3 img2         │
│  (scrollable grid, click to select) │
│                                      │
│  Selected: clip1.mp4                 │
│                                      │
│  ── Seek bar (media only) ─────────  │
│  ████████░░░░░░░░░░░░░░░░░░░░░░░░   │
│  ▲        ▲              ▲          │
│  [0]                        [3600]  │
│                                      │
│  ◀◀  [⏺ Rec]  [▶ Play]  ▶▶         │
│  ⊞ Marker  ⊟ Remove  ⊣ Prev  ⊢ Next │
│  frame: [  142  ]                    │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── when Webcam is selected ───────  │
│                                      │
│  Backend:    [Default ▾]             │
│  Resolution: [1280×720 ▾]            │
│  FPS:        [30 ▾]                  │
│                                      │
│  ┌──┐ ┌──┐                           │
│  │📷│ │📷│  ← click to activate      │
│  │ 0│ │ 1│                           │
│  └──┘ └──┘                           │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── when Streaming is selected ────  │
│                                      │
│  ┌──────────────────────────────┐    │
│  │  ● LIVE   FPS: 28.4          │    │
│  │  ○ Waiting for connection    │    │
│  └──────────────────────────────┘    │
│                                      │
│  [▶ Start Server]  [■ Stop]          │
│                                      │
│  HTTP:  http://192.168.1.10:9091/    │
│  WHIP:  http://192.168.1.10:9091/whip│
│                                      │
│  ┌──────────────────────────────┐    │
│  │  QR code (scan on phone)     │    │
│  └──────────────────────────────┘    │
│                                      │
│  [⚙ Port Settings]                   │
│  ┌── Popover ──────────────────┐     │
│  │ HTTP Port:  [9091      ]    │     │
│  │ HTTPS Port: [9090      ]    │     │
│  │ Bind Addr:  [0.0.0.0   ]    │     │
│  │             [Apply]         │     │
│  └─────────────────────────────┘     │
│                                      │
└──────────────────────────────────────┘
```

---

## Column 2 — Face Swapping

```
┌──────────────────────────────────────┐
│  FACE SWAPPING                       │
│                                      │
│  [⚡ Activate Swap]                  │
│  (loads models, turns green when on) │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── Model Settings ────────────────  │
│                                      │
│  Detector:   [RetinaFace ▾]          │
│  Swapper:    [Inswapper128 ▾]        │
│  Resolution: [128 ▾]                 │
│  ArcFace:    [Inswapper128ArcFace ▾] │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── Face Pairs ────────────────────  │
│                                      │
│  Each row = one swap pair            │
│  Left = source face (who to swap IN) │
│  Right = target face (who to replace)│
│                                      │
│  ┌──────────────────────────────┐    │
│  │ Row 1                        │    │
│  │  ┌──────────┐  ┌──────────┐  │    │
│  │  │          │  │          │  │    │
│  │  │  Source  │  │  Target  │  │    │
│  │  │  face    │  │  face    │  │    │
│  │  │  (photo) │  │  (from   │  │    │
│  │  │          │  │  video)  │  │    │
│  │  └──────────┘  └──────────┘  │    │
│  │  Alice.jpg     [click to     │    │
│  │                 choose]      │    │
│  │                              │    │
│  │  [🗑 Remove pair]            │    │
│  └──────────────────────────────┘    │
│                                      │
│  ┌──────────────────────────────┐    │
│  │ Row 2                        │    │
│  │  ┌──────────┐  ┌──────────┐  │    │
│  │  │          │  │    👤    │  │    │
│  │  │  Bob.jpg │  │  (empty) │  │    │
│  │  │          │  │  click   │  │    │
│  │  │          │  │  to pick │  │    │
│  │  └──────────┘  └──────────┘  │    │
│  │                              │    │
│  │  [🗑 Remove pair]            │    │
│  └──────────────────────────────┘    │
│                                      │
│  [+ Add Face Pair]                   │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── Embeddings ────────────────────  │
│                                      │
│  [+ Merge]  [📂 Import]  [💾 Export] │
│  ┌──────────────┐ ┌──────────────┐   │
│  │ Alice merged │ │ Bob merged   │   │
│  └──────────────┘ └──────────────┘   │
│                                      │
└──────────────────────────────────────┘
```

### Target face picker dialog

When user clicks the empty target face slot (👤):

```
┌─────────────────────────────────────────────────────┐
│  Choose Target Face                             [×]  │
│                                                      │
│  [📁 Browse folder]  [🔍 Search faces...]            │
│                                                      │
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐           │
│  │😊│ │😊│ │😊│ │😊│ │😊│ │😊│ │😊│ │😊│           │
│  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘           │
│  face1 face2 face3 face4 face5 face6 face7 face8     │
│                                                      │
│  (click a face to assign it to this slot)            │
│                                                      │
│  [Find Faces in Current Frame]                       │
│                                                      │
│                                    [Cancel]          │
└─────────────────────────────────────────────────────┘
```

- Browse folder → `POST /api/target-faces/find` after scanning.
- Search → client-side filter on face thumbnails.
- Click a face → assigns it to the target slot → closes dialog.
- "Find Faces in Current Frame" → `POST /api/target-faces/find`.

---

## Column 3 — Face Options

### Face selection state

Column 3 is **context-sensitive** — it shows parameters for whichever face is currently selected in column 2. When no face is selected it shows an empty state.

**Empty state (no face selected):**

```
┌──────────────────────────────────────┐
│  FACE OPTIONS                        │
│                                      │
│                                      │
│                                      │
│              👤                      │
│                                      │
│       Click on a face to tune        │
│                                      │
│   Select a face pair in the swap     │
│   panel to edit its parameters.      │
│                                      │
│                                      │
└──────────────────────────────────────┘
```

**Selected state** — clicking any face pair row in column 2 loads that face's saved parameters into column 3. The selected face is highlighted with a sky-500 ring in column 2, and the column 3 header shows which face is active:

```
┌──────────────────────────────────────┐
│  FACE OPTIONS  ·  Face 1             │  ← face label / index
│                                      │
│  [📋 Copy]  [📌 Paste]  [↺ Reset]   │  ← per-face actions
│  ─────────────────────────────────   │
│  (blocks shown below)                │
└──────────────────────────────────────┘
```

### Per-face actions

| Button | Behaviour |
|---|---|
| **📋 Copy** | Snapshots this face's full parameter set into a clipboard (client-side). |
| **📌 Paste** | Applies the clipboard parameters to the currently selected face. Calls `PUT /api/state/parameters/{face_id}` with the copied values. |
| **↺ Reset** | Resets all parameters for this face to defaults. Calls `POST /api/state/reset/{face_id}`. |

Paste is greyed out until a Copy has been performed. The clipboard survives face switches within the session but is cleared on page reload.

### Parameter persistence

Each face stores its own independent copy of every parameter. Switching between faces in column 2 instantly swaps the values shown in column 3 — sliders, toggles, and dropdowns all update to reflect the selected face's saved state. Changes are written back to that face's parameter set in real time via the `set_parameter` WebSocket command.

---

### Full column 3 layout (when a face is selected)

```
┌──────────────────────────────────────┐
│  FACE OPTIONS                        │
│                                      │
│  ── Pinned (always visible) ───────  │
│                                      │
│  📌 FACE RESTORER              [−]  │
│    Enable 1:   [○]                   │
│      Type:     [GFPGAN-v1.4 ▾]       │
│      Alignment:[Original ▾]          │
│      Fidelity: [0.9 ──●── ]          │
│      Blend:    [100 ──●── ]          │
│    Enable 2:   [○]                   │
│      (same controls)                 │
│                                      │
│  ─────────────────────────────────   │
│  ── Active blocks ─────────────────  │
│  (drag ⠿ to reorder, [×] to remove) │
│                                      │
│  ⠿ FACE SIMILARITY          [−] [×] │
│    Threshold:  [60 ──●── ]           │
│    Strength:   [○]                   │
│      Amount:   [100 ──●── ]          │
│    Face Likeness: [○]                │
│      Amount:   [0.00 ──●── ]         │
│    Differencing: [○]                 │
│      Amount:   [4 ──●── ]            │
│      Blend:    [5 ──●── ]            │
│                                      │
│  ⠿ FACE MASK                [−] [×] │
│    Border T/B/L/R + Blur             │
│    Occlusion Mask:  [○]              │
│    DFL XSeg Mask:   [○]              │
│    Text Masking:    [○]              │
│    Face Parser Mask:[○]              │
│    Restore Eyes:    [○]              │
│    Restore Mouth:   [○]              │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  [+ Add Block ▾]                     │
│  ┌── dropdown ─────────────────┐     │
│  │  ○ Landmarks Correction     │     │
│  │  ○ Detection                │     │
│  │  ○ Swapper                  │     │
│  │  ○ Frame Enhancer           │     │
│  │  ○ Color Correction         │     │
│  │  ○ Expression Restorer      │     │
│  │  ○ Face Editor              │     │
│  └─────────────────────────────┘     │
│  (already-active blocks are greyed   │
│   out in the list)                   │
│                                      │
└──────────────────────────────────────┘
```

### Block catalogue (available to add)

Each block below is **off by default** — user adds it when needed.

```
LANDMARKS CORRECTION
  Face Adjustments: [○]
    X: [0]  Y: [0]  Scale: [0]  Face Scale: [0]
  5-Keypoints Adj: [○]
    Left Eye X/Y  Right Eye X/Y  Nose X/Y
    Left Mouth X/Y  Right Mouth X/Y

DETECTION
  Model:    [RetinaFace ▾]
  Score:    [50 ──●── ]
  Max Faces:[20 ──●── ]
  Auto Rotation: [○]
  Manual Rotation: [○]  Angle: [0 ▾]
  Landmark Detect: [○]  Model: [203 ▾]  Score: [50 ──●── ]
  From Points: [○]
  Show Landmarks: [○]   Show Bboxes: [○]

SWAPPER
  Model:      [Inswapper128 ▾]
  Resolution: [128 ▾]
  DFM Model:  [my_model.dfm ▾]
  AMP Morph:  [50 ──●── ]
  RCT Color:  [○]

FRAME ENHANCER
  Enable: [○]
    Type:  [RealEsrgan-x2-Plus ▾]
    Blend: [100 ──●── ]

COLOR CORRECTION
  Enable: [○]
    Gamma / Brightness / Contrast / Saturation
    Sharpness / Hue / Noise / R / G / B
  Auto Color: [○]  Type: [Test ▾]  Blend: [0 ──●── ]
  JPEG Compression: [○]  Amount: [0 ──●── ]
  Final Blend Adj: [○]   Amount: [0 ──●── ]
  Overall Mask Blur: [0 ──●── ]

EXPRESSION RESTORER
  Enable: [○]
    Crop Scale:  [2.30 ──●── ]
    VY Ratio:    [-0.125 ──●── ]
    Friendly Factor: [1.0 ──●── ]
    Animation Region: [all ▾]
    Normalize Lips: [●]  Threshold: [0.03 ──●── ]
    Retargeting Eyes: [○]  Multiplier: [1.00 ──●── ]
    Retargeting Lips: [○]  Multiplier: [1.00 ──●── ]

FACE EDITOR (LivePortrait)
  Enable: [○]
    Type:     [Human-Face ▾]
    Eyes Open:   [0.00 ──●── ]
    Lips Open:   [0.00 ──●── ]
    Head Pitch / Yaw / Roll
    X / Y / Z Movement
    Mouth: Pouting / Pursing / Grin / Smile
    Eye Wink / Eyebrows Direction
    Eye Gaze Horizontal / Vertical
  Face Makeup:    [○]  R/G/B + Blend
  Hair Makeup:    [○]  R/G/B + Blend
  Eyebrows Makeup:[○]  R/G/B + Blend
  Lips Makeup:    [○]  R/G/B + Blend
```

---

## Column 4 — Output

```
┌──────────────────────────────────────┐
│  OUTPUT                              │
│                                      │
│  ── Preview ───────────────────────  │
│                                      │
│  [👁 Compare] [🎭 Mask]              │
│  [⬜ BBoxes]  [· Landmarks] [⛶ Win] │
│  (toggle buttons — sky ring = active)│
│                                      │
│  ┌──────────────────────────────┐    │
│  │                              │    │
│  │   Processed output           │    │
│  │   /ws/preview                │    │
│  │   (zoom + pan)               │    │
│  │                              │    │
│  └──────────────────────────────┘    │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── Record to File ────────────────  │
│                                      │
│  Folder: [C:/Videos/output  ] [📁]   │
│                                      │
│  [⏺ Start Recording]                 │
│  [💾 Save Current Frame]             │
│                                      │
│  Status: ● Recording  00:01:23       │
│  (timer counts up while recording)   │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── Virtual Camera ────────────────  │
│                                      │
│  Backend: [OBS ▾]                    │
│  [Enable Virtual Camera ○]           │
│  Status: ● Active                    │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── Stream Output (WebSocket) ─────  │
│                                      │
│  ws://localhost:8000/ws/preview      │
│  Quality: [75 ──●── ]                │
│  [📋 Copy URL]                       │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── Stats ─────────────────────────  │
│                                      │
│  Output FPS:  28.4                   │
│  VRAM:  ████████░░  4.4 / 24 GB      │
│                                      │
│  ─────────────────────────────────   │
│                                      │
│  ── Settings ──────────────────────  │
│                                      │
│  ▸ General                           │
│    Provider:  [CUDA ▾]               │
│    Threads:   [2 ──●── 30]           │
│    Theme:     [Dark ▾]               │
│    Auto Swap: [○]                    │
│                                      │
│  ▸ Recognition                       │
│    Model:     [Inswapper128ArcFace▾] │
│    Similarity:[Opal ▾]               │
│    Emb Merge: [Mean ▾]               │
│                                      │
│  ▸ Webcam                            │
│    Max Cams:  [1 ▾]                  │
│    Backend:   [Default ▾]            │
│    Resolution:[1280×720 ▾]           │
│    FPS:       [30 ▾]                 │
│                                      │
│  ▸ DFM                               │
│    Max Models:[1 ──●── 5]            │
│                                      │
│  ▸ Video Playback                    │
│    Custom FPS:[○]                    │
│      FPS:     [30 ──●── 120]         │
│                                      │
└──────────────────────────────────────┘
```

---

## Interaction notes

### Source type switching
Only one source is active at a time. Switching hides the previous source's controls and shows the new ones. The seek bar and playback buttons are **only visible when Media is selected** — they don't apply to webcam or streaming.

### Face pair rows
- Each row has a **source face** (left) and a **target face** (right).
- Source face: click to open a file picker or browse loaded input faces.
- Target face: shows a 👤 icon when empty. Click → opens the target face picker dialog.
- The dialog has a folder browser + search + grid of detected faces.
- Multiple pairs = multiple simultaneous swaps in one frame.

### Activate Swap button
- Off state: grey, label "Activate Swap".
- Loading state: spinner, label "Loading models...".
- On state: green, label "Swap Active".
- Calls `POST /api/playback/swap/enable` which triggers model loading on first use.

### Draggable blocks (col 3)
- `⠿` grip handle on the left of each block header.
- Drag to reorder. Order saved to `localStorage`.
- Each block collapses with `[−]` / `[+]`.

### Output preview vs source preview
- **Col 1 preview** = raw input (before swap). Small, quality 20.
- **Col 4 preview** = processed output (after swap). Full quality 75.
- Both consume `/ws/preview` — the server always pushes the latest processed frame. The "raw" preview in col 1 is actually the same stream; it just appears smaller and serves as a "what am I looking at" reference.

---

## Tech stack (unchanged)
- Vite + React 18 + TypeScript
- shadcn/ui + Tailwind CSS
- Zustand (state) + TanStack Query (REST)
- @dnd-kit/sortable (draggable blocks)
- react-use-websocket (/ws/events + /ws/preview)
- qrcode.react (WebRTC QR code)
- lucide-react (icons)

---

## API mapping — every UI action to its endpoint

### Top bar

| UI action | API call |
|---|---|
| CPU % display | Browser `performance` API (no server call needed) |
| GPU % + VRAM bar | `GET /api/system/gpu-memory` every 3s → `{ used_mb, total_mb }` |
| 🗑 Clear VRAM | `POST /api/system/clear-memory` |

### Column 1 — Input Source

| UI action | API call |
|---|---|
| Browse folder (Media) | `POST /api/target-media/scan-folder { path, recursive }` |
| Click media card | `POST /api/target-media/{media_id}/select` |
| Remove media card | `DELETE /api/target-media/{media_id}` |
| Load thumbnail | `GET /api/target-media/{media_id}/thumbnail` |
| Filter images/videos | Client-side filter on `file_type` field |
| Search media | Client-side filter on `media_path` |
| Play | `POST /api/playback/play` (or WS `play`) |
| Stop | `POST /api/playback/stop` (or WS `stop`) |
| Seek (drag slider) | WS `seek { frame: N }` on mouseup |
| Step ±30 frames | WS `step { n: ±30 }` |
| Add marker | `POST /api/playback/markers` |
| Remove marker | `DELETE /api/playback/markers/{frame_number}` |
| Prev/next marker | Client-side: find nearest marker in `GET /api/playback/markers`, then WS `seek` |
| Record start | `POST /api/playback/record/start { output_folder }` |
| Record stop | `POST /api/playback/record/stop` |
| Enumerate webcams | `GET /api/sources/webcams` |
| Select webcam | `POST /api/sources/webcams/{index}/select` |
| Webcam backend/res/fps | `PUT /api/state/control { WebcamBackendSelection, WebcamMaxResSelection, WebCamMaxFPSSelection }` |
| Start WebRTC server | `POST /api/sources/webrtc/start` → returns URLs |
| Stop WebRTC server | `POST /api/sources/webrtc/stop` |
| WebRTC status / FPS | `GET /api/sources/webrtc/status` every 2s + WS `fps_update` event |
| WebRTC port settings | `PUT /api/state/control { WebRTCHttpPortText, WebRTCHttpsPortText, WebRTCBindAddressText }` |
| Transform (rotate/flip) | `PUT /api/sources/transform { rotation, flip_h, flip_v }` |
| Source preview | `/ws/preview` at quality 20 |

### Column 2 — Face Swapping

| UI action | API call |
|---|---|
| Activate Swap | `POST /api/playback/swap/enable` |
| Deactivate Swap | `POST /api/playback/swap/disable` |
| Activate Edit | `POST /api/playback/edit/enable` |
| Deactivate Edit | `POST /api/playback/edit/disable` |
| Change detector model | WS `set_control { name: "DetectorModelSelection", value }` |
| Change swapper model | WS `set_control { name: "SwapModelSelection", value }` |
| Change resolution | WS `set_control { name: "SwapperResSelection", value }` |
| Change ArcFace model | WS `set_control { name: "RecognitionModelSelection", value }` |
| Browse source faces | `POST /api/input-faces/scan-folder { path, recursive }` |
| Load source face thumbnail | `GET /api/input-faces/{face_id}/thumbnail` |
| Remove source face | `DELETE /api/input-faces/{face_id}` |
| Assign source → target | `POST /api/target-faces/{face_id}/assign-input/{input_face_id}` |
| Unassign source | `DELETE /api/target-faces/{face_id}/assign-input/{input_face_id}` |
| Open target face picker | `GET /api/target-faces` (list existing) + `POST /api/target-faces/find` |
| Find faces in frame | `POST /api/target-faces/find` |
| Select target face | `POST /api/target-faces/{face_id}/select` |
| Remove target face | `DELETE /api/target-faces/{face_id}` |
| Clear all target faces | `POST /api/target-faces/clear` |
| Load target face thumbnail | `GET /api/target-faces/{face_id}/thumbnail` |
| Merge embeddings | `POST /api/embeddings/merge { name, input_face_ids }` |
| Import embeddings | `POST /api/embeddings/import` (multipart) |
| Export embeddings | `GET /api/embeddings/export` |
| Assign embedding → target | `POST /api/target-faces/{face_id}/assign-embedding/{embedding_id}` |
| Unassign embedding | `DELETE /api/target-faces/{face_id}/assign-embedding/{embedding_id}` |
| Delete embedding | `DELETE /api/embeddings/{embedding_id}` |

### Column 3 — Face Options

| UI action | API call |
|---|---|
| Load face parameters (on select) | Already in `GET /api/state` → `target_faces[face_id].parameters` |
| Change any slider/toggle | WS `set_parameter { face_id, name, value }` |
| Copy parameters | `POST /api/state/copy/{face_id}` |
| Paste parameters | `POST /api/state/paste/{face_id}` |
| Reset parameters | `POST /api/state/reset/{face_id}` |
| Load block schema | `GET /api/schema/parameters/swap` + `/common` + `/face-editor` |
| Block order (drag) | `localStorage` only — no API call |
| Add/remove block | `localStorage` only — no API call |

### Column 4 — Output

| UI action | API call |
|---|---|
| Output preview | `/ws/preview` at quality 75 |
| Open in window | `PUT /api/state/control { OutputWindowEnableToggle: true }` |
| Set output folder | `PUT /api/state/control { OutputMediaFolder: "..." }` |
| Start recording | `POST /api/playback/record/start { output_folder }` |
| Stop recording | `POST /api/playback/record/stop` |
| Save current frame | `POST /api/playback/save-frame` |
| Enable virtual cam | `PUT /api/state/control { SendVirtCamFramesEnableToggle: true }` |
| Virtual cam backend | `PUT /api/state/control { VirtCamBackendSelection: "obs" }` |
| WS stream quality | WS `preview_quality { quality: 75 }` |
| VRAM bar | `GET /api/system/gpu-memory` every 5s |
| Provider dropdown | `POST /api/system/providers { provider }` |
| Settings (all fields) | `PUT /api/state/control { ... }` |
| Save workspace | `POST /api/workspace/save { filename }` |
| Load workspace | `POST /api/workspace/load { filename }` |
| Reset workspace | `POST /api/workspace/reset` |

### WebSocket events the UI listens to

| Event | What the UI does |
|---|---|
| `frame_processed` | Triggers a frame counter update |
| `playback_state` | Syncs play/stop button state, seek bar position, recording indicator |
| `fps_update` | Updates FPS label in col 1 (streaming) and col 4 (output) |
| `state_updated` | Refreshes the affected widget (slider, toggle, etc.) |
| `recording_finished` | Shows a toast with the output file path |
| `error` | Shows an error toast |

---

## Features left out — choose what to add

These exist in the original Qt app but are **not yet in the UI design**. Grouped by priority.

### 🟢 Add now — small effort, high value

| Feature | Where in Qt | API support | Decision |
|---|---|---|---|
| **Provider selector** (CUDA/TRT/TRT-Engine) | `ProvidersPrioritySelection` | `POST /api/system/providers` | ✅ In top bar |
| **View Face Compare** overlay | `faceCompareCheckBox` | WS `set_control { _view_face_compare }` | ✅ Button above output preview |
| **View Face Mask** overlay | `faceMaskCheckBox` | WS `set_control { _view_face_mask }` | ✅ Button above output preview |
| **Show Bounding Boxes** | `ShowAllDetectedFacesBBoxToggle` | WS `set_control` | ✅ Button above output preview + Detection block |
| **Show Landmarks** | `ShowLandmarksEnableToggle` | WS `set_control` | ✅ Button above output preview + Detection block |
| **Recursive folder scan** | `TargetMediaFolderRecursiveToggle` | `scan-folder { recursive: true }` | ✅ Checkbox next to Browse |
| **Recursive input faces scan** | `InputFacesFolderRecursiveToggle` | `scan-folder { recursive: true }` | ✅ Checkbox next to source face Browse |
| **Embedding merge method** | `EmbMergeMethodSelection` | WS `set_control` | ✅ Dropdown in embeddings section |
| **Similarity type** | `SimilarityTypeSelection` | WS `set_control` | ✅ Dropdown in col 2 model settings |
| **Auto Swap on load** | `AutoSwapToggle` | `set_control` | ❌ Skip for now |
| **Custom video playback FPS** | `VideoPlaybackCustomFpsToggle` | `set_control` | ❌ Skip for now |
| **Max DFM models** | `MaxDFMModelsSlider` | `set_control` | ❌ Skip for now |

### Overlay button behaviour (col 4 preview header)

The four overlay buttons above the output preview are **mutually exclusive in pairs**:
- Compare and Mask cannot both be on — activating one turns off the other.
- BBoxes and Landmarks are independent — both can be on simultaneously.
- All four can be off (normal output).

Each sends a WS `set_control` command and the server calls `process_current_frame()` immediately.

### 🟡 Add later — medium effort

| Feature | Where in Qt | API support | Notes |
|---|---|---|---|
| **Video markers** (full UI) | `videoSeekSlider` with painted markers | `GET/POST/DELETE /api/playback/markers` | Seek bar needs custom rendering for marker triangles |
| **Workspace save/load dialog** | Menu bar | `POST /api/workspace/save` / `load` | File picker in Electron; path input in browser |
| **Per-face parameters JSON export** | Right-click context menu on face card | `POST /api/state/copy/{face_id}` + download | Useful for sharing presets |
| **Face card context menu** | Right-click on `TargetFaceCardButton` | copy/paste/reset endpoints | Copy, Paste, Reset, Remove |
| **Fullscreen canvas** | `viewFullScreenButton` + F11 | No API needed | Browser fullscreen API |
| **Keyboard shortcuts** | `keyPressEvent` in MainWindow | WS commands | Space=play, R=record, S=swap, F=marker, A/D=step, Q/W=marker nav |
| **Output window** (Electron only) | `OutputWindowEnableToggle` | `set_control` | Only meaningful in Electron; show note in browser |
| **TensorRT provider** | `ProvidersPrioritySelection` | `POST /api/system/providers` | Already in settings, just needs the TRT option exposed |
| **Manual rotation** (detector) | `ManualRotationEnableToggle` + angle | `set_control` | In Detection block, col 3 |

### 🔴 Leave for later — complex or niche

| Feature | Where in Qt | Reason to defer |
|---|---|---|
| **Multi-GPU device routing** | Not in Qt yet (planned in doc 13) | Requires Phase A–H backend work first |
| **Batch processing** | Mentioned in code comments but not implemented | Not in Qt either |
| **Drag-and-drop files onto media list** | `ListWidgetEventFilter` | Electron-only; browser has security restrictions |
| **Webcam virtual camera output** | `pyvirtualcam` | Requires native driver; Electron-only |
| **Unity Capture backend** | `VirtCamBackendSelection: unitycapture` | Windows-only, niche |
| **TensorRT engine build progress** | `model_loading_signal` | Long-running; needs a progress bar + cancel |
| **DeOldify / DDColor colorization** | `FrameEnhancerTypeSelection` | Works via existing Frame Enhancer block; no extra UI needed |
| **HTTPS WebRTC** | `WebRTCHttpsPortText` | Already in port settings popup; cert generation is automatic |
| **RunPod / remote deployment UI** | `RUNPOD_SETUP.md` | Server-side concern, not a UI feature |
