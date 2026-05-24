# Target Faces

Target faces are faces detected in the active media. Each one has its own parameter set and can be assigned one or more source faces or embeddings to swap from.

---

## POST /api/target-faces/find

Runs face detection on the current frame of the active media source and registers any new faces found. Faces already registered (matched by cosine similarity above `SimilarityThresholdSlider`) are skipped.

**No request body.**

Detection uses the current `control` settings:
- `DetectorModelSelection` — which detector to use (RetinaFace, Yolov8, SCRFD, Yunet)
- `DetectorScoreSlider` — confidence threshold (0–100)
- `MaxFacesToDetectSlider` — max faces per frame
- `AutoRotationToggle` — try 0°/90°/180°/270° rotations
- `LandmarkDetectToggle` + `LandmarkDetectModelSelection` — optional landmark refinement

**Response**

```json
{
  "found": 2,
  "faces": [
    {
      "face_id": "246025604671586647369822850939522388726",
      "thumbnail_url": "/api/target-faces/246025.../thumbnail",
      "assigned_input_face_ids": [],
      "assigned_embedding_ids": []
    },
    {
      "face_id": "246089298251364089844116488877329879798",
      "thumbnail_url": "/api/target-faces/246089.../thumbnail",
      "assigned_input_face_ids": [],
      "assigned_embedding_ids": []
    }
  ]
}
```

`found` is the count of **newly** registered faces (not total faces in frame).

**Errors**
- `400` — no media selected, or no frame available yet.

---

## GET /api/target-faces

Returns all registered target face cards.

**Response**

```json
[
  {
    "face_id": "246025604671586647369822850939522388726",
    "thumbnail_url": "/api/target-faces/246025.../thumbnail",
    "assigned_input_face_ids": ["input_face_id_1"],
    "assigned_embedding_ids": []
  }
]
```

---

## POST /api/target-faces/{face_id}/select

Sets the given face as the currently selected target face. Parameter panel edits apply to this face.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Selected face 246025..." }
```

---

## POST /api/target-faces/{face_id}/assign-input/{input_face_id}

Assigns an input face as a swap source for this target face. Recalculates the merged embedding immediately.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Assigned input face <input_id> to target <face_id>" }
```

**Errors**
- `404` — either face_id or input_face_id not found.

---

## DELETE /api/target-faces/{face_id}/assign-input/{input_face_id}

Removes an input face assignment from this target face. Recalculates the merged embedding.

**Response**

```json
{ "ok": true, "message": "Unassigned input face <input_id> from target <face_id>" }
```

---

## POST /api/target-faces/{face_id}/assign-embedding/{embedding_id}

Assigns a merged embedding as a swap source for this target face.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Assigned embedding <embedding_id> to target <face_id>" }
```

---

## DELETE /api/target-faces/{face_id}/assign-embedding/{embedding_id}

Removes an embedding assignment from this target face.

**Response**

```json
{ "ok": true, "message": "Unassigned embedding <embedding_id> from target <face_id>" }
```

---

## DELETE /api/target-faces/{face_id}

Removes a target face and its parameters. If it was selected, the next face in the list becomes selected.

**Response**

```json
{ "ok": true, "message": "Removed target face 246025..." }
```

---

## POST /api/target-faces/clear

Removes all target faces and clears all per-face parameters.

**No request body.**

**Response**

```json
{ "ok": true, "message": "All target faces cleared" }
```

---

## GET /api/target-faces/{face_id}/thumbnail

Returns the cropped face thumbnail (PNG) for a target face card.

**Response** — binary `image/png`.

**Errors**
- `404` — face_id not found or thumbnail unavailable.
