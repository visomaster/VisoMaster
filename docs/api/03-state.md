# Application State

The state endpoints give direct read/write access to the live `AppState` object — the single source of truth for all session data.

---

## GET /api/state

Returns a complete snapshot of the current application state.

**Response** (abbreviated)

```json
{
  "selected_media_id": "247013112649665177041422535211796140790",
  "selected_face_id": "246025604671586647369822850939522388726",
  "control": {
    "DetectorModelSelection": "RetinaFace",
    "DetectorScoreSlider": 50,
    "SwapModelSelection": "Inswapper128",
    "_swap_enabled": true,
    "_edit_enabled": false,
    "ProvidersPrioritySelection": "CUDA",
    "nThreadsSlider": 2,
    "OutputMediaFolder": "C:/Users/Miles/Videos/output",
    "..."
  },
  "target_media": [
    {
      "media_id": "247013112649665177041422535211796140790",
      "media_path": "C:/Videos/sample.mp4",
      "file_type": "video"
    }
  ],
  "target_faces": {
    "<face_id>": {
      "face_id": "<face_id>",
      "embedding_store": { "Inswapper128ArcFace": [...512 floats...] },
      "assigned_input_face_ids": ["<input_face_id>"],
      "assigned_embedding_ids": [],
      "assigned_input_embedding": { "Inswapper128ArcFace": [...512 floats...] },
      "parameters": {
        "SwapModelSelection": "Inswapper128",
        "FaceRestorerEnableToggle": false,
        "SimilarityThresholdSlider": 65,
        "..."
      }
    }
  },
  "input_faces": {
    "<face_id>": {
      "face_id": "<face_id>",
      "media_path": "C:/Photos/source.jpg",
      "embedding_store": { "Inswapper128ArcFace": [...512 floats...] }
    }
  },
  "embeddings": {},
  "markers": {
    "120": {
      "frame_number": 120,
      "parameters": { "<face_id>": { "FaceRestorerEnableToggle": true, "..." } },
      "control": { "DetectorScoreSlider": 70, "..." }
    }
  },
  "webcam_transform": { "rotation": 0, "flip_h": false, "flip_v": false },
  "webrtc_transform": { "rotation": 0, "flip_h": true, "flip_v": false },
  "last_target_media_folder": "C:/Videos",
  "last_input_media_folder": "C:/Photos",
  "output_media_folder": "C:/Videos/output"
}
```

### Special control keys

These keys live in `control` but are not part of the settings schema — they are set by the API/UI to drive processing mode:

| Key | Type | Description |
|---|---|---|
| `_swap_enabled` | bool | Whether face swap is active during processing. |
| `_edit_enabled` | bool | Whether face editor (LivePortrait) is active. |
| `_view_face_compare` | bool | Show side-by-side face comparison overlay. |
| `_view_face_mask` | bool | Show face mask overlay. |

---

## PUT /api/state/control

Patches one or more global control values. Returns the full updated state.

**Request body**

```json
{
  "updates": {
    "DetectorModelSelection": "Yolov8",
    "DetectorScoreSlider": 60,
    "_swap_enabled": true
  }
}
```

**Response** — full `StateResponse` (same shape as `GET /api/state`).

> Changing `nThreadsSlider` takes effect on the next `play` call. Changing `ProvidersPrioritySelection` here does **not** reload models — use `POST /api/system/providers` for that.

---

## PUT /api/state/parameters/{face_id}

Patches one or more per-face parameter values for the given target face.

**Path parameter** — `face_id`: the UUID string from a target face card.

**Request body**

```json
{
  "updates": {
    "SwapModelSelection": "SimSwap512",
    "FaceRestorerEnableToggle": true,
    "FaceRestorerTypeSelection": "CodeFormer",
    "FaceRestorerBlendSlider": 80
  }
}
```

**Response** — full `StateResponse`.

**Errors**
- `404` — face_id not found.

---

## POST /api/state/copy/{face_id}

Copies the parameters of the given target face into a server-side clipboard.

**No request body.**

**Response**

```json
{ "ok": true, "message": "Parameters copied from face <face_id>" }
```

---

## POST /api/state/paste/{face_id}

Pastes the clipboard parameters onto the given target face. Call `copy` first.

**No request body.**

**Response** — full `StateResponse`.

**Errors**
- `400` — clipboard is empty (no prior `copy` call).
- `404` — face_id not found.

---

## POST /api/state/reset/{face_id}

Resets the given target face's parameters to their default values.

**No request body.**

**Response** — full `StateResponse`.

**Errors**
- `404` — face_id not found.
