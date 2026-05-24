# VisoMaster React UI — Design Docs

Tech stack: React 18 + TypeScript + Vite + shadcn/ui + Tailwind + Zustand + @dnd-kit

## Documents

| File | What it covers |
|---|---|
| [01-overview.md](./01-overview.md) | Three-column layout, color tokens, tech stack, file structure |
| [02-left-panel.md](./02-left-panel.md) | Source section: Media / Webcam / Streaming tabs, source preview |
| [03-faces-panel.md](./03-faces-panel.md) | Target faces, source faces, embeddings, detection settings |
| [04-output-panel.md](./04-output-panel.md) | Record / Virtual Cam / Output Window / Stream tabs, VRAM bar |
| [05-center-canvas.md](./05-center-canvas.md) | Preview canvas, seek bar with markers, playback controls |
| [06-right-panel.md](./06-right-panel.md) | All draggable parameter blocks with full control inventory |
| [07-header-settings.md](./07-header-settings.md) | Header bar, VRAM, provider dropdown, global settings dialog |

## Quick summary

```
┌──────────────┬──────────────────────────────────┬──────────────────────────┐
│ LEFT (320px) │ CENTER (flex)                    │ RIGHT (340px)            │
│              │                                  │                          │
│ ▣ Source     │  ┌────────────────────────────┐  │ ▣ Swapper                │
│   Media tab  │  │  /ws/preview frame         │  │ ▣ Face Similarity        │
│   Webcam tab │  │  zoom + pan                │  │ ▣ Face Mask              │
│   Stream tab │  └────────────────────────────┘  │ ▣ Landmarks Correction   │
│   (preview)  │  seek bar + markers              │ ▣ Face Restorer          │
│              │  ◀◀  ⏺  ▶  ▶▶  ⊞  ⊟  ⊣  ⊢  │  │ ▣ Frame Enhancer         │
│ ▣ Faces      │                                  │ ▣ Color Correction       │
│   Find/Clear │                                  │ ▣ Face Editor            │
│   Swap/Edit  │                                  │                          │
│   Target ↔   │                                  │ (blocks are draggable)   │
│   Source     │                                  │                          │
│   Embeddings │                                  │                          │
│              │                                  │                          │
│ ▣ Output     │                                  │                          │
│   Record     │                                  │                          │
│   VirtCam    │                                  │                          │
│   Window     │                                  │                          │
│   Stream     │                                  │                          │
└──────────────┴──────────────────────────────────┴──────────────────────────┘
```

## Key design decisions

**Parameter blocks are draggable** — users reorder them to match their workflow. Order saved to `localStorage`. Implemented with `@dnd-kit/sortable`.

**WebSocket-first for parameters** — slider changes go through `set_parameter` WS command, not REST. This gives ~50ms preview latency while dragging.

**Schema-driven widgets** — the right panel reads `GET /api/schema/parameters/*` and renders the correct widget type for each descriptor. No hardcoded parameter lists in the React code.

**Source preview is separate from output preview** — the center canvas shows the processed output. The small thumbnail in the source section shows the raw input. Both use `/ws/preview` at different quality settings.

**Streaming settings in a popup** — the WebRTC settings (ports, bind address) live in a `Popover` triggered by a ⚙ button, not in the main settings dialog. This keeps the streaming tab self-contained.

**Mutual exclusion: Swap vs Edit** — these are toggle buttons that can't both be active. Activating one deactivates the other. Implemented as a single `mode: 'swap' | 'edit' | 'none'` state.

**Markers on the seek bar** — rendered as small colored triangles at their frame positions. Clicking a marker seeks to it. Right-clicking removes it.
