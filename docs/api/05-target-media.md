# Target Media

Target media are the video files, images, or live sources you want to process. You register them first, then select one to make it active.

---

## GET /api/target-media

Returns all registered target media cards.

**Response**

```json
[
  {
    "media_id": "247013112649665177041422535211796140790",
    "media_path": "C:/Videos/sample.mp4",
    "file_type": "video",
    "thumbnail_url": "/api/target-media/247013112649665177041422535211796140790/thumbnail"
  },
  {
    "media_id": "246025604671586647369822850939522388726",
    "media_path": "C:/Photos/portrait.jpg",
    "file_type": "image",
    "thumbnail_url": "/api/target-media/246025604671586647369822850939522388726/thumbnail"
  }
]
```

`file_type` is one of `"video"`, `"image"`, `"webcam"`, `"webrtc"`.

---

## POST /api/target-media/scan-folder

Scans a folder for video and image files, registers each one, generates thumbnails, and returns the new cards.

**Request body**

```json
{
  "path": "C:/Videos/project",
  "recursive": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | string | required | Absolute path to the folder. |
| `recursive` | bool | `false` | Include files in subfolders. |

**Response**

```json
{
  "items": [
    {
      "media_id": "...",
      "media_path": "C:/Videos/project/clip1.mp4",
      "file_type": "video",
      "thumbnail_url": "/api/target-media/.../thumbnail"
    }
  ]
}
```

**Errors**
- `400` — path is not a directory.

---

## POST /api/target-media/{media_id}/select

Makes a registered media item the active source. Opens the `cv2.VideoCapture` for videos, or sets up the WebRTC shared-memory handle. Stops any current processing first.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Selected media 247013... (video)" }
```

**Errors**
- `404` — media_id not found.
- `400` — video file cannot be opened by OpenCV.

After selecting, call `POST /api/playback/play` to start processing, or `POST /api/target-faces/find` to detect faces on the first frame.

---

## DELETE /api/target-media/{media_id}

Removes a media card from the list. If it was the active source, processing is stopped.

**Response**

```json
{ "ok": true, "message": "Removed media 247013..." }
```

**Errors**
- `404` — media_id not found.

---

## GET /api/target-media/{media_id}/thumbnail

Returns the cached thumbnail image (PNG or JPEG) for a media card. Generates it on demand if not cached.

**Response** — binary image (`image/png` or `image/jpeg`).

**Errors**
- `404` — media_id not found, or thumbnail could not be generated.

Use directly in an `<img>` tag:

```tsx
<img src={`http://localhost:8000${card.thumbnail_url}`} />
```
