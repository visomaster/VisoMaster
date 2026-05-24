# Typical Workflows

End-to-end call sequences for the most common tasks. All examples use `fetch` / `WebSocket` from a browser or Node.js client.

---

## 1. Swap a face in a video file

```
1. POST /api/target-media/scan-folder        → register video files
2. POST /api/target-media/{id}/select        → open the video
3. POST /api/input-faces/scan-folder         → register source face images
4. POST /api/target-faces/find               → detect faces in frame 0
5. POST /api/target-faces/{id}/select        → pick the face to swap
6. POST /api/target-faces/{id}/assign-input/{input_id}  → assign source
7. POST /api/playback/swap/enable            → turn on swap
8. Connect /ws/preview                       → start receiving frames
9. POST /api/playback/play                   → start the loop
   ... watch frames arrive on /ws/preview ...
10. POST /api/playback/stop                  → stop
```

---

## 2. Record the processed video

```
1–7. Same as above
8. PUT  /api/state/control  { "updates": { "OutputMediaFolder": "C:/output" } }
9. POST /api/playback/record/start
   ... recording runs, frames arrive on /ws/preview ...
10. POST /api/playback/record/stop  → returns { "output_path": "..." }
```

Or listen on `/ws/events` for `recording_finished` instead of polling.

---

## 3. Preview a single frame with swap

```
1. POST /api/target-media/scan-folder
2. POST /api/target-media/{id}/select
3. POST /api/input-faces/scan-folder
4. POST /api/target-faces/find
5. POST /api/target-faces/{id}/assign-input/{input_id}
6. POST /api/playback/swap/enable
7. POST /api/playback/seek  { "frame": 300 }
8. GET  /api/preview/snapshot   → JPEG of frame 300 with swap applied
```

---

## 4. Adjust parameters and see the result live

```
// Via REST (one-shot)
PUT /api/state/parameters/{face_id}
  { "updates": { "FaceRestorerEnableToggle": true, "FaceRestorerBlendSlider": 80 } }

// Via WebSocket (low-latency, good for sliders)
ws.send(JSON.stringify({
  type: "set_parameter",
  payload: { face_id: "...", name: "FaceRestorerBlendSlider", value: 80 }
}));
```

After either call, `process_current_frame()` runs automatically and the result appears on `/ws/preview`.

---

## 5. Webcam live swap

```
1. POST /api/input-faces/scan-folder         → register source faces
2. GET  /api/sources/webcams                 → list available cameras
3. POST /api/sources/webcams/0/select        → open webcam 0
4. POST /api/target-faces/find               → detect face in first frame
5. POST /api/target-faces/{id}/assign-input/{input_id}
6. POST /api/playback/swap/enable
7. Connect /ws/preview
8. POST /api/playback/play
```

---

## 6. WebRTC phone-as-camera

```
1. POST /api/sources/webrtc/start
   → returns { http_url, https_url, whip_url, whip_https_url }
   → show http_url as QR code in the UI

2. User opens http_url on phone browser (or enters whip_url in Larix)
   → StreamRelay receives the video stream

3. GET  /api/sources/webrtc/status  → poll until frames_received > 0

4. POST /api/input-faces/scan-folder
5. POST /api/target-faces/find
6. POST /api/target-faces/{id}/assign-input/{input_id}
7. POST /api/playback/swap/enable
8. Connect /ws/preview
9. POST /api/playback/play
```

---

## 7. Build the parameter panel from schema

```ts
// On app startup
const [swapSchema, commonSchema, controlSchema] = await Promise.all([
  fetch('/api/schema/parameters/swap').then(r => r.json()),
  fetch('/api/schema/parameters/common').then(r => r.json()),
  fetch('/api/schema/control').then(r => r.json()),
]);

// Render each widget
swapSchema.widgets.forEach(w => {
  const value = state.target_faces[selectedFaceId]?.parameters[w.widget_name]
             ?? w.default;
  renderWidget(w, value, (newVal) => {
    ws.send(JSON.stringify({
      type: 'set_parameter',
      payload: { face_id: selectedFaceId, name: w.widget_name, value: newVal }
    }));
  });
});
```

---

## 8. Save and restore a workspace

```
// Save
POST /api/workspace/save  { "filename": "C:/workspaces/project.json" }

// Later — restore
POST /api/workspace/load  { "filename": "C:/workspaces/project.json" }
// Media is registered but not opened yet
POST /api/target-media/{selected_media_id}/select
```

---

## 9. Use merged embeddings for better accuracy

```
// Load 10 photos of the same person
POST /api/input-faces/scan-folder  { "path": "C:/Photos/alice" }

// Merge them
POST /api/embeddings/merge
  { "name": "Alice", "input_face_ids": ["id1", "id2", ..., "id10"] }
  → returns { "embedding_id": "..." }

// Assign the merged embedding to a target face
POST /api/target-faces/{face_id}/assign-embedding/{embedding_id}

// Export for reuse
GET /api/embeddings/export  → download embeddings.json

// Next session — import instead of re-scanning
POST /api/embeddings/import  (multipart, file=embeddings.json)
POST /api/target-faces/{face_id}/assign-embedding/{embedding_id}
```

---

## 10. Add per-frame parameter markers

```
// Seek to the frame where you want different settings
POST /api/playback/seek  { "frame": 300 }

// Change parameters for this frame
ws.send({ type: "set_parameter", payload: { face_id: "...", name: "FaceRestorerEnableToggle", value: true } })

// Save as a marker
POST /api/playback/markers

// Seek to another frame and add another marker
POST /api/playback/seek  { "frame": 600 }
ws.send({ type: "set_parameter", payload: { ..., name: "FaceRestorerBlendSlider", value: 50 } })
POST /api/playback/markers

// Now play — parameters change automatically at frames 300 and 600
POST /api/playback/play
```
