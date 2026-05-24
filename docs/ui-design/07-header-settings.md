# Header Bar & Global Settings

The header bar is always visible at the top. It shows the app identity, VRAM usage, active provider, and access to global settings.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [VM] VisoMaster          VRAM: ████████░░░░ 4.4 / 24 GB   CUDA ▾   [⚙]  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## VRAM bar

- Polled from `GET /api/system/gpu-memory` every 5 seconds.
- Color: green < 70%, amber 70–85%, red > 85%.
- Tooltip shows exact values.
- Click → `POST /api/system/clear-memory` with a confirmation popover.

---

## Provider dropdown

Shows the active provider (`CUDA`, `TensorRT`, `TensorRT-Engine`, `CPU`). Changing it calls `POST /api/system/providers` and shows a loading toast while models reload.

---

## Settings button (⚙)

Opens a full-screen `Dialog` with tabs:

### Tab: General

```
Provider:        [CUDA ▾]
Threads:         [2 ──●── 30]
Theme:           [Dark ▾]
Auto Swap:       [○]
Output Folder:   [C:/Videos/output    ] [📁]
```

### Tab: Detection

```
Detector Model:  [RetinaFace ▾]
Detect Score:    [50 ──●── ]
Max Faces:       [20 ──●── ]
Auto Rotation:   [○]
Manual Rotation: [○]
  Angle:         [0 ▾]  (0/90/180/270)
Landmark Detect: [○]
  Model:         [203 ▾]
  Score:         [50 ──●── ]
  From Points:   [○]
Show Landmarks:  [○]
Show Bboxes:     [○]
```

### Tab: Recognition

```
Recognition Model:  [Inswapper128ArcFace ▾]
Similarity Type:    [Opal ▾]
Embedding Merge:    [Mean ▾]
```

### Tab: Webcam

```
Max Webcams:     [1 ▾]
Backend:         [Default ▾]
Resolution:      [1280×720 ▾]
FPS:             [30 ▾]
```

### Tab: Virtual Camera

```
Send to VirtCam: [○]
Backend:         [obs ▾]
```

### Tab: WebRTC

```
HTTP Port:       [9091]
HTTPS Port:      [9090]
Bind Address:    [0.0.0.0]
```

### Tab: DFM

```
Max DFM Models:  [1 ──●── 5]
```

### Tab: Video Playback

```
Custom FPS:      [○]
  FPS:           [30 ──●── 120]
```

All settings map to `control` keys and are sent via `PUT /api/state/control`.

---

## Workspace menu

A `[☰]` menu button in the header (or a `File` menu in Electron):

```
New Workspace
Open Workspace...
Save Workspace
Save Workspace As...
─────────────────
Recent workspaces
  my_project.json
  test_run.json
```

- **New** → `POST /api/workspace/reset`
- **Open** → file picker → `POST /api/workspace/load`
- **Save** → `POST /api/workspace/save { filename: last_path }`
- **Save As** → file picker → `POST /api/workspace/save`
- Recent workspaces stored in `localStorage`.
