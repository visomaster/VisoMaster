# VisoMaster React UI — Layout Preview

---

## Top Bar (always visible)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [VM] VisoMaster                                                                    │
│                                                                                     │
│  CPU  ████████░░  34%     GPU  ████████████░░  67%     VRAM  ████████░░  4.4/24GB  │
│                                                                                     │
│                                                              [🗑 Clear VRAM]        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

- CPU % polled from system (browser: `navigator` API or server-sent).
- GPU % + VRAM from `GET /api/system/gpu-memory` every 3s.
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
│  ┌──────────────────────────────┐    │
│  │                              │    │
│  │   Processed output           │    │
│  │   /ws/preview                │    │
│  │   (zoom + pan)               │    │
│  │                              │    │
│  └──────────────────────────────┘    │
│                                      │
│  [⛶ Open in Window]                  │
│  (opens a detached preview window)   │
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
