# Input Faces

Input faces are the source face images you supply to swap onto target faces. Each image is scanned for a face, run through all four ArcFace recognition models, and stored with its embeddings.

---

## GET /api/input-faces

Returns all registered input face cards.

**Response**

```json
[
  {
    "face_id": "246025604671586647369822850939522388726",
    "media_path": "C:/Photos/source_alice.jpg",
    "thumbnail_url": "/api/input-faces/246025.../thumbnail"
  }
]
```

---

## POST /api/input-faces/scan-folder

Scans a folder for image files, runs face detection and recognition on each, and registers them as input faces. Only the first detected face per image is used.

**Request body**

```json
{
  "path": "C:/Photos/source_faces",
  "recursive": false
}
```

Detection uses the current `control` settings (`DetectorModelSelection`, `DetectorScoreSlider`, `RecognitionModelSelection`, `SimilarityTypeSelection`).

**Response**

```json
{
  "items": [
    {
      "face_id": "246025604671586647369822850939522388726",
      "media_path": "C:/Photos/source_faces/alice.jpg",
      "thumbnail_url": "/api/input-faces/246025.../thumbnail"
    }
  ]
}
```

Images where no face is detected are silently skipped.

**Errors**
- `400` — path is not a directory.

---

## DELETE /api/input-faces/{face_id}

Removes an input face. Also removes it from any target face assignments and recalculates their merged embeddings.

**Response**

```json
{ "ok": true, "message": "Removed input face 246025..." }
```

**Errors**
- `404` — face_id not found.

---

## POST /api/input-faces/clear

Removes all input faces and clears all target face assignments.

**No request body.**

**Response**

```json
{ "ok": true, "message": "All input faces cleared" }
```

---

## GET /api/input-faces/{face_id}/thumbnail

Returns the cropped face thumbnail (PNG) for an input face card.

**Response** — binary `image/png`.

**Errors**
- `404` — face_id not found or thumbnail unavailable.
