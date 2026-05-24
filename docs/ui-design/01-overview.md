# VisoMaster React UI — Design Overview

## Layout philosophy

Three-column layout at 1440px+. Collapsible panels. Dark theme (zinc-900 base, matching the existing Qt dark theme). All interactive blocks are draggable within their column so users can reorder them to match their workflow.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Header bar                                                                 │
│  [VisoMaster logo]  [VRAM bar ████░░ 4.4/24 GB]  [Provider: CUDA ▾]  [⚙]  │
├──────────────┬──────────────────────────────────┬──────────────────────────┤
│              │                                  │                          │
│  LEFT PANEL  │       CENTER CANVAS              │   RIGHT PANEL            │
│  (source)    │       (preview)                  │   (parameters)           │
│              │                                  │                          │
│  ┌──────────┐│  ┌────────────────────────────┐  │  ┌────────────────────┐  │
│  │ Source   ││  │                            │  │  │ ▣ Swapper          │  │
│  │ selector ││  │   processed frame          │  │  │ ▣ Face Similarity  │  │
│  └──────────┘│  │   (WebSocket stream)       │  │  │ ▣ Face Mask        │  │
│              │  │                            │  │  │ ▣ Landmarks Corr.  │  │
│  ┌──────────┐│  └────────────────────────────┘  │  │ ▣ Face Restorer    │  │
│  │ Source   ││                                  │  │ ▣ Expression Rest. │  │
│  │ preview  ││  ┌────────────────────────────┐  │  │ ▣ Frame Enhancer   │  │
│  └──────────┘│  │  Seek bar + markers        │  │  │ ▣ Color Correction │  │
│              │  │  ◀◀  ●  ▶  ⏺  ▶▶  ⊞  ⊟  │  │  │ ▣ Face Editor      │  │
│  ┌──────────┐│  └────────────────────────────┘  │  └────────────────────┘  │
│  │ Faces    ││                                  │                          │
│  │ panel    ││                                  │                          │
│  └──────────┘│                                  │                          │
│              │                                  │                          │
│  ┌──────────┐│                                  │                          │
│  │ Output   ││                                  │                          │
│  │ panel    ││                                  │                          │
│  └──────────┘│                                  │                          │
│              │                                  │                          │
└──────────────┴──────────────────────────────────┴──────────────────────────┘
```

## Column widths

| Column | Default width | Min | Resizable |
|---|---|---|---|
| Left (source + faces + output) | 320px | 280px | Yes (drag divider) |
| Center (canvas) | flex-1 | 400px | Yes |
| Right (parameters) | 340px | 280px | Yes |

All three columns scroll independently. The center canvas is sticky — the seek bar stays visible even when the parameter list is long.

## Color tokens (shadcn/ui zinc dark)

```
background:       zinc-950   (#09090b)
surface:          zinc-900   (#18181b)
surface-elevated: zinc-800   (#27272a)
border:           zinc-700   (#3f3f46)
text-primary:     zinc-50    (#fafafa)
text-muted:       zinc-400   (#a1a1aa)
accent:           sky-500    (#0ea5e9)   ← matches Qt #4facc9
accent-hover:     sky-400    (#38bdf8)
destructive:      red-500    (#ef4444)
success:          green-500  (#22c55e)
warning:          amber-500  (#f59e0b)
```

## Tech stack

- **Vite + React 18 + TypeScript**
- **shadcn/ui** — Button, Slider, Switch, Select, Tabs, Dialog, Popover, Badge, ScrollArea, Tooltip, Separator, Input, Label, Card
- **Tailwind CSS v3**
- **Zustand** — global state store
- **TanStack Query** — server state (REST calls)
- **@dnd-kit/core** — drag-and-drop for parameter blocks
- **react-use-websocket** — `/ws/events` and `/ws/preview`
- **lucide-react** — icons

## File structure

```
frontend/
├── src/
│   ├── api/                    ← generated from OpenAPI
│   │   └── client.ts
│   ├── store/
│   │   ├── appState.ts         ← Zustand: mirrors AppState
│   │   └── events.ts           ← WS event subscriptions
│   ├── hooks/
│   │   ├── usePreviewStream.ts ← /ws/preview binary consumer
│   │   ├── useEvents.ts        ← /ws/events JSON consumer
│   │   └── useSchema.ts        ← GET /api/schema/* with caching
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── LeftPanel.tsx
│   │   │   ├── CenterCanvas.tsx
│   │   │   └── RightPanel.tsx
│   │   ├── source/
│   │   │   ├── SourcePanel.tsx
│   │   │   ├── MediaSource.tsx
│   │   │   ├── WebcamSource.tsx
│   │   │   └── WebRTCSource.tsx
│   │   ├── faces/
│   │   │   ├── FacesPanel.tsx
│   │   │   ├── TargetFaceCard.tsx
│   │   │   ├── InputFaceCard.tsx
│   │   │   └── EmbeddingCard.tsx
│   │   ├── output/
│   │   │   └── OutputPanel.tsx
│   │   ├── canvas/
│   │   │   ├── PreviewCanvas.tsx
│   │   │   ├── SeekBar.tsx
│   │   │   └── PlaybackControls.tsx
│   │   ├── parameters/
│   │   │   ├── ParameterBlock.tsx  ← draggable container
│   │   │   ├── ParameterPanel.tsx  ← renders schema widgets
│   │   │   └── widgets/
│   │   │       ├── ToggleWidget.tsx
│   │   │       ├── SliderWidget.tsx
│   │   │       ├── SelectWidget.tsx
│   │   │       └── TextWidget.tsx
│   │   └── shared/
│   │       ├── VramBar.tsx
│   │       ├── FpsLabel.tsx
│   │       └── SettingsDialog.tsx
│   └── App.tsx
└── vite.config.ts
```
