# Embeddings

Merged embeddings are averaged ArcFace vectors created from multiple input faces. They give better identity consistency than a single source image, especially when you have many photos of the same person.

---

## GET /api/embeddings

Returns all registered merged embeddings.

**Response**

```json
{
  "embeddings": [
    {
      "embedding_id": "246025604671586647369822850939522388726",
      "name": "Alice (10 photos)"
    }
  ]
}
```

---

## POST /api/embeddings/merge

Averages (or takes the median of) the embeddings of the listed input faces into a single named embedding. The merge method is controlled by `EmbMergeMethodSelection` in `control` (`"Mean"` or `"Median"`).

**Request body**

```json
{
  "name": "Alice (10 photos)",
  "input_face_ids": [
    "246025604671586647369822850939522388726",
    "246089298251364089844116488877329879798"
  ]
}
```

**Response**

```json
{
  "embedding_id": "247013112649665177041422535211796140790",
  "name": "Alice (10 photos)"
}
```

**Errors**
- `400` — `input_face_ids` is empty.
- `404` — one or more face IDs not found.

---

## GET /api/embeddings/export

Downloads all merged embeddings as a JSON file. The file format is compatible with the Qt app's embedding export.

**Response** — `application/json` with `Content-Disposition: attachment; filename="embeddings.json"`.

```json
[
  {
    "name": "Alice (10 photos)",
    "embedding_store": {
      "Inswapper128ArcFace": [0.0123, -0.4567, ...],
      "SimSwapArcFace": [...],
      "GhostArcFace": [...],
      "CSCSArcFace": [...]
    }
  }
]
```

---

## POST /api/embeddings/import

Uploads a previously exported embeddings JSON file and registers all embeddings in it.

**Request** — multipart form upload with field name `file`.

```bash
curl -X POST http://localhost:8000/api/embeddings/import \
  -F "file=@embeddings.json"
```

**Response**

```json
{
  "embeddings": [
    { "embedding_id": "...", "name": "Alice (10 photos)" }
  ]
}
```

**Errors**
- `400` — file is not valid JSON, or not a JSON array.

---

## DELETE /api/embeddings/{embedding_id}

Removes a merged embedding. Also removes it from any target face assignments.

**Response**

```json
{ "ok": true, "message": "Removed embedding 247013..." }
```

**Errors**
- `404` — embedding_id not found.

---

## POST /api/embeddings/clear

Removes all merged embeddings and clears all target face embedding assignments.

**No request body.**

**Response**

```json
{ "ok": true, "message": "All embeddings cleared" }
```
