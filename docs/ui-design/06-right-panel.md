# Right Panel — Parameter Blocks

The right panel contains all per-face and global processing parameters. Each section is a **draggable block** — users can reorder them by dragging the grip handle. Blocks can be collapsed individually.

```
┌──────────────────────────────────────┐
│  ⠿ SWAPPER                      [−] │  ← drag handle + collapse
│  Model: [Inswapper128 ▾]            │
│  Resolution: [128 ▾]                │
├──────────────────────────────────────┤
│  ⠿ FACE SIMILARITY              [−] │
│  Threshold: [60 ──●── ]             │
│  Strength: [○]                      │
├──────────────────────────────────────┤
│  ⠿ FACE MASK                    [−] │
│  Border: T[10] B[10] L[10] R[10]    │
│  Blur: [10 ──●── ]                  │
│  Occlusion: [○]                     │
│  DFL XSeg: [○]                      │
│  Text Mask: [○]                     │
│  Face Parser: [○]                   │
│  Restore Eyes: [○]                  │
│  Restore Mouth: [○]                 │
├──────────────────────────────────────┤
│  ⠿ LANDMARKS CORRECTION         [−] │
│  Face Adjustments: [○]              │
│  5-Keypoints Adj: [○]               │
├──────────────────────────────────────┤
│  ⠿ FACE RESTORER                [−] │
│  Enable: [○]                        │
│  Enable 2: [○]                      │
│  Expression Restorer: [○]           │
├──────────────────────────────────────┤
│  ⠿ FRAME ENHANCER               [−] │
│  Enable: [○]                        │
├──────────────────────────────────────┤
│  ⠿ COLOR CORRECTION             [−] │
│  Enable: [○]                        │
├──────────────────────────────────────┤
│  ⠿ FACE EDITOR (LivePortrait)   [−] │
│  Enable: [○]                        │
│  Makeup: Face / Hair / Brows / Lips │
└──────────────────────────────────────┘
```

---

## Block: Swapper

Source: `SWAPPER_LAYOUT_DATA['Swapper']`

```
Model:       [Inswapper128 ▾]
Resolution:  [128 ▾]              ← visible only when Inswapper128
DFM Model:   [my_model.dfm ▾]    ← visible only when DFM
AMP Morph:   [50 ──●── ]         ← visible only when DFM
RCT Color:   [○]                  ← visible only when DFM
```

---

## Block: Face Similarity

Source: `SWAPPER_LAYOUT_DATA['Face Similarity']`

```
Similarity Threshold:  [60 ──●── ]
Strength:              [○]
  Amount:              [100 ──●── ]   ← when Strength on
Face Likeness:         [○]
  Amount:              [0.00 ──●── ]  ← when Likeness on
Differencing:          [○]
  Amount:              [4 ──●── ]
  Blend Amount:        [5 ──●── ]
```

---

## Block: Face Mask

Source: `SWAPPER_LAYOUT_DATA['Face Mask']`

Four border sliders displayed as a compact 2×2 grid:

```
  Top:    [10 ──●── ]   Bottom: [10 ──●── ]
  Left:   [10 ──●── ]   Right:  [10 ──●── ]
  Border Blur: [10 ──●── ]

  Occlusion Mask:  [○]  Size: [0 ──●── ]
  DFL XSeg Mask:   [○]  Size: [0 ──●── ]
  Occluder/XSeg Blur: [0 ──●── ]

  Text Masking:    [○]
    Text: [glasses, hat...]  Amount: [50 ──●── ]

  Face Parser Mask: [○]
    Background / Face / Eyebrows / Eyes / Glasses /
    Nose / Mouth / Lips / Neck / Hair (each a slider)
    Background Blur / Face Blur
    Hair Makeup: [○]  R/G/B + Blend
    Lips Makeup: [○]  R/G/B + Blend

  Restore Eyes:  [○]
    Blend / Size Factor / Feather / X Radius / Y Radius /
    X Offset / Y Offset / Eye Spacing Offset

  Restore Mouth: [○]
    Blend / Feather / Size Factor / X Radius / Y Radius /
    X Offset / Y Offset

  Eyes+Mouth Blur: [0 ──●── ]
```

---

## Block: Landmarks Correction

Source: `SWAPPER_LAYOUT_DATA['Face Landmarks Correction']`

```
Face Adjustments: [○]
  Keypoints X:    [-100 ──●── 100]
  Keypoints Y:    [-100 ──●── 100]
  Keypoints Scale:[-100 ──●── 100]
  Face Scale:     [-20 ──●── 20]

5-Keypoints Adj:  [○]
  Left Eye X/Y, Right Eye X/Y, Nose X/Y,
  Left Mouth X/Y, Right Mouth X/Y
  (10 sliders in a 2-column grid)
```

---

## Block: Face Restorer

Source: `COMMON_LAYOUT_DATA['Face Restorer']`

```
Enable Restorer 1: [○]
  Type:          [GFPGAN-v1.4 ▾]
  Alignment:     [Original ▾]
  Fidelity:      [0.9 ──●── ]
  Blend:         [100 ──●── ]

Enable Restorer 2: [○]
  (same controls)

Expression Restorer: [○]
  Crop Scale:    [2.30 ──●── ]
  VY Ratio:      [-0.125 ──●── ]
  Friendly Factor: [1.0 ──●── ]
  Animation Region: [all ▾]
  Normalize Lips: [●]
    Threshold:   [0.03 ──●── ]
  Retargeting Eyes: [○]
    Multiplier:  [1.00 ──●── ]
  Retargeting Lips: [○]
    Multiplier:  [1.00 ──●── ]
```

---

## Block: Frame Enhancer

Source: `SETTINGS_LAYOUT_DATA['Frame Enhancer']`

```
Enable: [○]
  Type:  [RealEsrgan-x2-Plus ▾]
  Blend: [100 ──●── ]
```

---

## Block: Color Correction

Source: `SWAPPER_LAYOUT_DATA` (color section — `ColorEnableToggle` etc.)

```
Enable: [○]
  Gamma:      [1.0 ──●── ]
  Brightness: [1.0 ──●── ]
  Contrast:   [1.0 ──●── ]
  Saturation: [1.0 ──●── ]
  Sharpness:  [1.0 ──●── ]
  Hue:        [0.0 ──●── ]
  Noise:      [0.0 ──●── ]
  R/G/B:      three sliders
Auto Color:   [○]
  Type:       [Test ▾]
  Blend:      [0 ──●── ]
JPEG Compression: [○]
  Amount:     [0 ──●── ]
Final Blend Adj: [○]
  Amount:     [0 ──●── ]
Overall Mask Blur: [0 ──●── ]
```

---

## Block: Face Editor (LivePortrait)

Source: `FACE_EDITOR_LAYOUT_DATA`

```
Crop Scale:  [2.50 ──●── ]
VY Ratio:    [-0.125 ──●── ]
Blur Amount: [5 ──●── ]

Enable Pose/Expression Editor: [○]
  Type:          [Human-Face ▾]
  Eyes Open:     [-0.80 ──●── 0.80]
  Lips Open:     [-0.80 ──●── 0.80]
  Head Pitch:    [-15 ──●── 15]
  Head Yaw:      [-15 ──●── 15]
  Head Roll:     [-15 ──●── 15]
  X/Y/Z Movement (3 decimal sliders)
  Mouth Pouting / Pursing / Grin / Smile
  Lips Close/Open
  Eye Wink / Eyebrows / Gaze H / Gaze V

Face Makeup:   [○]  R/G/B + Blend
Hair Makeup:   [○]  R/G/B + Blend
Eyebrows Makeup: [○]  R/G/B + Blend
Lips Makeup:   [○]  R/G/B + Blend
```

---

## Drag-and-drop reordering

Each block has a `⠿` grip handle on the left of its header. Dragging reorders blocks within the right panel. Order is persisted to `localStorage` (not the server — it's a UI preference, not a processing parameter).

Implementation: `@dnd-kit/sortable` wrapping each `ParameterBlock` component.

---

## Per-face vs global parameters

The right panel always shows parameters for the **currently selected target face**. When no face is selected, sliders show default values and are dimmed (but still editable — changes apply to `current_widget_parameters` and will be used for the next detected face).

The "Face Editor" and "Frame Enhancer" blocks are global (they apply to all faces), so they're always active regardless of face selection.

---

## Parameter changes

All slider/toggle/select changes go through the WebSocket `set_parameter` command for low latency:

```ts
ws.send(JSON.stringify({
  type: 'set_parameter',
  payload: { face_id: selectedFaceId, name: 'FaceRestorerBlendSlider', value: 80 }
}));
```

The server calls `process_current_frame()` after each change, and the result appears on `/ws/preview` within ~50ms.

For bulk changes (e.g. loading a preset), use `PUT /api/state/parameters/{face_id}` instead.
