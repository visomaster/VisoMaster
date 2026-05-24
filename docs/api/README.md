# VisoMaster API Reference

Base URL: `http://localhost:8000`

Start the server:
```bash
conda activate visomaster
python -m app.api.server
# or
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs (auto-generated): `http://localhost:8000/docs`

---

## Contents

| Category | Doc |
|---|---|
| System & Hardware | [01-system.md](./01-system.md) |
| Schema (UI widget definitions) | [02-schema.md](./02-schema.md) |
| Application State | [03-state.md](./03-state.md) |
| Workspace | [04-workspace.md](./04-workspace.md) |
| Target Media | [05-target-media.md](./05-target-media.md) |
| Target Faces | [06-target-faces.md](./06-target-faces.md) |
| Input Faces | [07-input-faces.md](./07-input-faces.md) |
| Embeddings | [08-embeddings.md](./08-embeddings.md) |
| Playback & Recording | [09-playback.md](./09-playback.md) |
| Streaming Sources | [10-sources.md](./10-sources.md) |
| WebSocket Channels | [11-websockets.md](./11-websockets.md) |
| Typical Workflows | [12-workflows.md](./12-workflows.md) |

---

## Common response shapes

Every mutating endpoint returns at minimum:

```json
{ "ok": true, "message": "Human-readable confirmation" }
```

Errors follow FastAPI's default:

```json
{ "detail": "Error description" }
```

HTTP status codes used:

| Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Bad request (wrong state, missing config, invalid input) |
| 404 | Resource not found |
| 409 | Conflict (e.g. marker already exists at that frame) |
| 500 | Server error (ffmpeg missing, encoding failure) |
