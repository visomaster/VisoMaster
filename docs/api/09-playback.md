# Playback & Recording

Playback endpoints control the processing loop — play, stop, seek, step, record, and save frames. The preview snapshot endpoint gives a single-frame JPEG; for a live stream use `/ws/preview` instead.

---

## GET /api/playback

Returns the current playback state.

**Response**

```json
{
  "file_type": "video",
  "fps": 29.97,
  "current_frame": 142,
  "max_frame": 3600,
  "is_playing": true,
  "is_recording": false,
  "swap_enabled": true,
  "edit_enabled": false
}
```

`file_type` is `null` when no media is selected.

---

## POST /api/playback/play

Starts the processing loop. For video files this reads frames at the source FPS (×0.8 to stay ahead of processing). For webcam/webrtc it polls the capture continuously.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Playback started" }
```

**Errors**
- `400` — no media selected.

Returns `{ "ok": true, "message": "Already playing" }` if already running (not an error).

---

## POST /api/playback/stop

Stops the processing loop. For recordings, finalises the ffmpeg subprocess and muxes audio from the original file.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Stopped" }
```

---

## POST /api/playback/seek

Seeks to a specific frame, processes it, and pushes the result to `/ws/preview`. Stops any active playback first.

**Request body**

```json
{ "frame": 300 }
```

If a marker exists at that frame, its parameters and control overrides are applied.

**Response**

```json
{ "ok": true, "message": "Seeked to frame 300" }
```

**Errors**
- `400` — source is not a video or image (seek not supported for webcam/webrtc).

---

## POST /api/playback/step

Advances or rewinds by N frames, then processes the new frame.

**Request body**

```json
{ "n": 30 }
```

Use negative `n` to rewind: `{ "n": -1 }` steps back one frame.

**Response**

```json
{ "ok": true, "message": "Stepped to frame 172" }
```

**Errors**
- `400` — source is not a video.

---

## POST /api/playback/swap/enable

Enables face swap processing. Disables face editor if it was active. Processes the current frame immediately.

**No request body.**

**Response** `{ "ok": true, "message": "Swap enabled" }`

## POST /api/playback/swap/disable

Disables face swap. Frames are passed through unmodified (unless frame enhancer is on).

**Response** `{ "ok": true, "message": "Swap disabled" }`

## POST /api/playback/edit/enable

Enables the LivePortrait face editor. Disables swap if it was active.

**Response** `{ "ok": true, "message": "Edit enabled" }`

## POST /api/playback/edit/disable

**Response** `{ "ok": true, "message": "Edit disabled" }`

---

## POST /api/playback/record/start

Starts recording the processed video to disk. Spawns an ffmpeg subprocess that receives raw BGR frames via stdin. Audio is muxed from the original file when recording stops.

**Request body**

```json
{ "output_folder": "C:/Users/Miles/Videos/output" }
```

`output_folder` is optional — falls back to `control.OutputMediaFolder` if omitted.

**Response**

```json
{ "ok": true, "message": "Recording started" }
```

**Errors**
- `400` — source is not a video, playback is already running, or output folder doesn't exist.
- `500` — ffmpeg not found in PATH.

---

## POST /api/playback/record/stop

Stops recording. The ffmpeg subprocess is closed, audio is muxed, and the final file is written to the output folder.

**No request body.**

**Response**

```json
{ "output_path": "C:/Users/Miles/Videos/output/sample_2026_05_24_14_30_00.mp4" }
```

**Errors**
- `400` — not currently recording.

A `recording_finished` event is also pushed to `/ws/events`:
```json
{ "type": "recording_finished", "payload": { "output_path": "..." } }
```

---

## POST /api/playback/save-frame

Saves the current processed frame as a PNG to the output folder.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Frame saved to C:/Users/Miles/Videos/output/sample_2026_05_24_14_30_00.png" }
```

**Errors**
- `400` — no output folder configured, or no frame available.

---

## GET /api/playback/markers

Returns a sorted list of frame numbers that have markers.

**Response**

```json
{ "markers": [120, 300, 750, 1200] }
```

---

## POST /api/playback/markers

Adds a marker at the current frame position. Snapshots the current `parameters` and `control` so they can be restored when playback reaches that frame.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Marker added at frame 300" }
```

**Errors**
- `400` — source is not a video, or no target faces exist.
- `409` — a marker already exists at this frame.

---

## DELETE /api/playback/markers/{frame_number}

Removes the marker at the given frame number.

**Response**

```json
{ "ok": true, "message": "Marker removed from frame 300" }
```

**Errors**
- `404` — no marker at that frame.

---

## GET /api/preview/snapshot

Returns the latest processed frame as a JPEG binary response. Use this for a static preview or when WebSocket is not available. For live streaming use `/ws/preview` instead.

**Response** — binary `image/jpeg` (quality 80).

**Errors**
- `404` — no frame has been processed yet.
- `500` — JPEG encoding failed.

```tsx
// Polling example (React)
useEffect(() => {
  const interval = setInterval(async () => {
    const res = await fetch('http://localhost:8000/api/preview/snapshot');
    if (res.ok) {
      const blob = await res.blob();
      setPreviewSrc(URL.createObjectURL(blob));
    }
  }, 100);
  return () => clearInterval(interval);
}, []);
```
