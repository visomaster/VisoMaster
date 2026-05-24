# Faces Panel

The faces panel lives in the left column below the source section. It handles target face detection, source face loading, embedding management, and the swap/edit mode toggles.

```
┌─────────────────────────────────────────────────────────────┐
│  FACES                                                  [−] │
│                                                             │
│  [Find Faces]  [Clear]  [Swap ●]  [Edit ○]                 │
│                                                             │
│  ── Target Faces ──────────────────────────────────────     │
│  ┌────┐ ┌────┐ ┌────┐                                       │
│  │ 😊 │ │ 😊 │ │ 😊 │  ← detected faces, click to select   │
│  └────┘ └────┘ └────┘                                       │
│  [selected face gets sky ring]                              │
│                                                             │
│  ── Source Faces ──────────────────────────────────────     │
│  [📁 Browse]  [🔍 Search...]                                │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐                               │
│  │ 👤 │ │ 👤 │ │ 👤 │ │ 👤 │  ← source images              │
│  └────┘ └────┘ └────┘ └────┘                               │
│  [checked = assigned to selected target face]               │
│                                                             │
│  ── Embeddings ────────────────────────────────────────     │
│  [🔍 Search...]  [📂 Import]  [💾 Export]  [+ Merge]       │
│  ┌────────────────┐ ┌────────────────┐                      │
│  │ Alice (10 ph.) │ │ Bob merged     │                      │
│  └────────────────┘ └────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Action bar

| Button | Action | API |
|---|---|---|
| **Find Faces** | Detect faces in current frame | `POST /api/target-faces/find` |
| **Clear** | Remove all target faces | `POST /api/target-faces/clear` |
| **Swap** | Toggle swap mode (checkable) | `POST /api/playback/swap/enable` or `swap_disable` WS |
| **Edit** | Toggle edit mode (checkable, mutually exclusive with Swap) | `POST /api/playback/edit/enable` or `edit_disable` WS |

Swap and Edit are mutually exclusive toggle buttons. When Swap is active it gets a sky-500 background. When Edit is active it gets an amber-500 background.

---

## Target faces

- Horizontal scrollable row of face cards (82×82px thumbnails).
- Click a card → `POST /api/target-faces/{face_id}/select` → that face's parameters load into the right panel.
- Selected card: sky-500 ring.
- Right-click context menu:
  - Copy Parameters
  - Paste Parameters
  - Reset to Defaults
  - Remove Face

---

## Source faces

- **Browse** → `POST /api/input-faces/scan-folder`.
- **Search** → client-side filter.
- Cards are checkable. Checking a card while a target face is selected → `POST /api/target-faces/{target_id}/assign-input/{input_id}`. Unchecking → `DELETE /api/target-faces/{target_id}/assign-input/{input_id}`.
- Multiple source faces can be checked for the same target (they get averaged into the assigned embedding).
- Right-click: Remove.

---

## Embeddings

- **Import** → `POST /api/embeddings/import` (file picker).
- **Export** → `GET /api/embeddings/export` (triggers download).
- **+ Merge** → opens a `Dialog`:
  - Name input
  - Checkboxes for each loaded source face
  - [Create] → `POST /api/embeddings/merge`
- Embedding cards are checkable. Checking assigns to selected target face → `POST /api/target-faces/{target_id}/assign-embedding/{embedding_id}`.
- Right-click: Rename, Delete.

---

## Detector & recognition settings

A small collapsible "Detection Settings" row below the Find Faces button (collapsed by default):

```
▸ Detection Settings
  Detector:    [RetinaFace ▾]   Score: [50 ──●── ]
  Max faces:   [20 ──●── ]      Auto-rotate: [○]
  Landmark:    [○]  Model: [203 ▾]  Score: [50 ──●── ]
  Recognition: [Inswapper128ArcFace ▾]
  Similarity:  [Opal ▾]
```

These map directly to `control` keys. Changes go via `PUT /api/state/control` or the `set_control` WS command.
