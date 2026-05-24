# Center Canvas

The center column is the main viewing area. It shows the processed output frame in real time, the seek bar with markers, and playback controls.

```
┌──────────────────────────────────────────────────────────┐
│  [View Face Compare ○]  [View Face Mask ○]  [⛶ Fullscreen]│
├──────────────────────────────────────────────────────────┤
│                                                          │
│                                                          │
│              processed frame                             │
│              (WebSocket /ws/preview)                     │
│                                                          │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ▲         ▲                    ▲                        │
│  marker    marker               marker                   │
│  [0]                                              [3600] │
├──────────────────────────────────────────────────────────┤
│  [◀◀]  [⏺ Rec]  [▶ Play]  [▶▶]  [⊞ Add Marker]  [⊟]  [⊣⊢]│
│         frame: [  142  ]                                 │
└──────────────────────────────────────────────────────────┘
```

---

## Preview canvas

- `<img>` element fed by the `/ws/preview` WebSocket hook.
- Aspect ratio preserved, letterboxed in the available space.
- **Zoom** — mouse wheel zooms in/out (CSS transform scale).
- **Pan** — right-click drag pans the zoomed view.
- **Double-click** — reset zoom to fit.
- When no frame is available: dark placeholder with "Select a source to begin" text.

### Overlay toggles

| Toggle | Control key | Effect |
|---|---|---|
| View Face Compare | `_view_face_compare` | Shows original / swapped / mask side-by-side |
| View Face Mask | `_view_face_mask` | Shows the alpha mask overlay |

Both send `set_control` WS commands and trigger `process_current_frame`.

---

## Seek bar

- Custom `<input type="range">` styled with Tailwind.
- Markers rendered as small colored triangles above the track at their frame positions.
- Dragging → `seek` WS command on `mouseup`.
- Frame number input on the right — type a number and press Enter to seek.
- Keyboard shortcuts (when canvas is focused):
  - `Space` — play/pause
  - `←` / `→` — step ±1 frame
  - `A` / `D` — step ±30 frames
  - `Z` — seek to start
  - `F` — add marker
  - `Alt+F` — remove marker
  - `Q` / `W` — previous/next marker
  - `R` — toggle record
  - `S` — toggle swap
  - `F11` — fullscreen

---

## Playback controls

```
[◀◀ -30]  [⏺ Record]  [▶ Play / ■ Stop]  [▶▶ +30]
[⊞ Add Marker]  [⊟ Remove Marker]  [⊣ Prev Marker]  [⊢ Next Marker]
[⛶ Fullscreen]
```

| Button | Action |
|---|---|
| ◀◀ | `step { n: -30 }` WS |
| ▶▶ | `step { n: 30 }` WS |
| ▶ Play | `play` WS → becomes ■ Stop |
| ■ Stop | `stop` WS |
| ⏺ Record | `POST /api/playback/record/start` → becomes ⏹ Stop Recording |
| ⊞ Add Marker | `POST /api/playback/markers` |
| ⊟ Remove Marker | `DELETE /api/playback/markers/{current_frame}` |
| ⊣ Prev Marker | seek to previous marker frame |
| ⊢ Next Marker | seek to next marker frame |
| ⛶ Fullscreen | browser fullscreen API |

Play button state syncs from `playback_state` WS events.

---

## Fullscreen mode

When fullscreen:
- Left and right panels slide off-screen (CSS transition).
- Canvas fills the viewport.
- A minimal floating toolbar appears at the bottom (play/stop/seek only).
- Press `Escape` or `F11` to exit.
