# Streaming Sources

Streaming source endpoints manage webcam capture and WebRTC ingestion. After selecting a source, call `POST /api/playback/play` to start the processing loop.

---

## GET /api/sources/webcams

Enumerates available webcam devices by trying to open each index up to `WebcamMaxNoSelection` (from `control`).

**Response**

```json
{
  "webcams": [
    { "index": 0, "label": "Webcam 0", "thumbnail_url": null },
    { "index": 1, "label": "Webcam 1", "thumbnail_url": null }
  ]
}
```

The backend used is controlled by `WebcamBackendSelection` in `control` (`"Default"`, `"DirectShow"`, `"MSMF"`, `"V4L"`, `"V4L2"`, `"GSTREAMER"`).

---

## POST /api/sources/webcams/{index}/select

Opens the webcam at the given index and makes it the active source. Stops any current processing first.

**Path parameter** — `index`: integer webcam index (0, 1, 2, …).

**No request body.**

Resolution is set from `WebcamMaxResSelection` in `control` (e.g. `"1280x720"`).

**Response**

```json
{ "ok": true, "message": "Webcam 0 selected" }
```

**Errors**
- `400` — webcam cannot be opened.

---

## POST /api/sources/webrtc/start

Spawns the StreamRelay WebRTC server subprocess and sets the video processor to WebRTC mode. If the server is already running, just returns the connection URLs.

**No request body.**

Ports and bind address are read from `control`:
- `WebRTCHttpPortText` (default `9091`)
- `WebRTCHttpsPortText` (default `9090`)
- `WebRTCBindAddressText` (default `0.0.0.0`)

**Response**

```json
{
  "http_url":       "http://192.168.1.10:9091/",
  "https_url":      "https://192.168.1.10:9090/",
  "whip_url":       "http://192.168.1.10:9091/whip",
  "whip_https_url": "https://192.168.1.10:9090/whip"
}
```

Show these URLs in the UI (with a QR code for the phone). The user opens `http_url` in their phone browser, or enters `whip_url` in Larix Broadcaster / OBS.

---

## POST /api/sources/webrtc/stop

Terminates the StreamRelay subprocess and stops processing.

**No request body.**

**Response**

```json
{ "ok": true, "message": "WebRTC server stopped" }
```

---

## GET /api/sources/webrtc/status

Returns whether the WebRTC server is running and how many frames have been received.

**Response**

```json
{
  "running": true,
  "frames_received": 1842
}
```

`frames_received` is the shared-memory frame counter — it increments on every frame written by the relay. Use it to detect whether a device is actively streaming.

---

## PUT /api/sources/transform

Sets the rotation and flip transforms for the active streaming source (webcam or webrtc). Applied to every frame before processing.

**Request body**

```json
{
  "rotation": 90,
  "flip_h": true,
  "flip_v": false
}
```

| Field | Type | Values | Description |
|---|---|---|---|
| `rotation` | int | `0`, `90`, `180`, `270` | Clockwise rotation in degrees. |
| `flip_h` | bool | — | Mirror horizontally (selfie mode). |
| `flip_v` | bool | — | Mirror vertically. |

**Response**

```json
{ "ok": true, "message": "Transform updated" }
```

**Errors**
- `400` — no active streaming source (file_type is not `"webcam"` or `"webrtc"`).
