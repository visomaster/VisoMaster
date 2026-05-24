# Workspace

Workspace endpoints save and restore the complete session — target media list, detected faces, source faces, embeddings, markers, parameters, and control settings. The format is compatible with the Qt app's `last_workspace.json`.

---

## GET /api/workspace

Returns the current workspace as a raw JSON object. Same format as `last_workspace.json`. Useful for debugging or building a custom save dialog.

**No request body.**

**Response** — the full workspace dict (see `AppState.to_json()` in `app/core/state.py`).

---

## POST /api/workspace/save

Saves the current workspace to a file on the server's filesystem.

**Request body**

```json
{ "filename": "C:/Users/Miles/workspaces/my_project.json" }
```

The parent directory is created if it doesn't exist.

**Response**

```json
{ "ok": true, "message": "Workspace saved to C:/Users/Miles/workspaces/my_project.json" }
```

**Errors**
- `500` — filesystem error (permissions, disk full).

---

## POST /api/workspace/load

Loads a workspace JSON file into the live state. Media files are registered but **not opened** — call `POST /api/target-media/{media_id}/select` afterwards to activate one.

**Request body**

```json
{ "filename": "C:/Users/Miles/workspaces/my_project.json" }
```

**Response**

```json
{ "ok": true, "message": "Workspace loaded from C:/Users/Miles/workspaces/my_project.json" }
```

**Errors**
- `404` — file not found.
- `400` — file is not valid JSON.

> The server auto-saves to `last_workspace.json` on shutdown and auto-loads it on startup.

---

## POST /api/workspace/reset

Clears all working-set data: target media, target faces, input faces, embeddings, markers, and parameters. Control settings (provider, threads, etc.) are preserved.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Workspace reset" }
```
