# VisoMaster Architecture Docs

These documents describe the VisoMaster backend pipeline and API. The project runs as a FastAPI service (`app/api/server.py`) with a React frontend (`visomaster-ui/`) connecting over REST and WebSocket. The desktop entry point (`web_main.py`) wraps the same backend in a Qt WebEngine window using a bridge.

## Doc index

| Doc | What it covers |
|---|---|
| [04-backend-pipeline.md](./04-backend-pipeline.md) | `VideoProcessor`, `FrameWorker`, `ModelsProcessor`, threading model. |
| [05-models-and-inference.md](./05-models-and-inference.md) | ONNX model registry, TensorRT engines, DFM, ArcFace/landmark mappings. |
| [06-data-flows.md](./06-data-flows.md) | End-to-end flows: load video, find faces, swap, record, webcam, WebRTC. |
| [07-state-and-persistence.md](./07-state-and-persistence.md) | `parameters` / `control` / `markers` schema, `last_workspace.json`, embeddings JSON. |
| [08-streamrelay-webrtc.md](./08-streamrelay-webrtc.md) | The bundled `streamrelay` package: WHIP, WebSocket, shared-memory frame transport. |
| [12-glossary.md](./12-glossary.md) | Terms used in the codebase (target face, source face, embedding, marker, DFM, ArcFace…). |
| [ui-design.md](./ui-design.md) | UI design reference for the React frontend. |
| [api/README.md](./api/README.md) | API reference index. |

## Architecture summary

- **Backend** — `app/processors/*`: PyTorch / ONNX Runtime inference pipeline. UI-free.
- **API server** — `app/api/*`: FastAPI service exposing REST endpoints and WebSocket streams. Single source of truth is `app/core/state.py` (`AppState`).
- **Web frontend** — `visomaster-ui/`: React + TypeScript + Tailwind. Communicates exclusively through `api.*` client methods and the `/ws/events` WebSocket.
- **Desktop wrapper** — `web_main.py`: Qt WebEngine window that loads the Vite dev server (or built frontend) and bridges native capabilities (file dialogs, virtual cam, etc.) via a Qt/JS bridge.
- **WebRTC ingestion** — `packages/streamrelay/`: separate subprocess writing BGR frames into shared memory (`visomaster_webrtc_frame`).
