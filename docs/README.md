# VisoMaster — Architecture Docs

These documents describe the VisoMaster backend pipeline, API, and UI layers. The project runs as a FastAPI service (`app/api/server.py`) with a React frontend (`visomaster-ui/`) connecting over REST and WebSocket. The desktop entry point (`web_main.py`) wraps the same backend in a Qt WebEngine window using a bridge. A legacy Qt-only desktop mode is also available via `main.py`.

> For installation, launch instructions, and feature overview see the [root README](../README.md).

## Doc index

| Doc | What it covers |
|---|---|
| [01-overview.md](./01-overview.md) | Full repo layout, runtime modes, runtime layer diagrams, key dependencies. |
| [02-startup-and-bootstrap.md](./02-startup-and-bootstrap.md) | Entry sequences for all three modes, model bootstrap, provider detection. |
| [03-ui-layer.md](./03-ui-layer.md) | Qt desktop UI (MainWindow, layout-data widgets, action modules, card buttons) and React web UI (TanStack Start, shadcn/ui, transport layer). |
| [04-backend-pipeline.md](./04-backend-pipeline.md) | `VideoProcessor`, `FrameWorker`, `ModelsProcessor`, threading model. |
| [05-models-and-inference.md](./05-models-and-inference.md) | ONNX model registry, TensorRT engines, DFM, ArcFace/landmark mappings. |
| [06-data-flows.md](./06-data-flows.md) | End-to-end flows: load video, find faces, swap, record, webcam, WebRTC. |
| [07-state-and-persistence.md](./07-state-and-persistence.md) | `parameters` / `control` / `markers` schema, `last_workspace.json`, embeddings JSON. |
| [08-streamrelay-webrtc.md](./08-streamrelay-webrtc.md) | The bundled `streamrelay` package: WHIP, WebSocket, shared-memory frame transport. |
| [api/README.md](./api/README.md) | API reference index (74 endpoints + 3 WebSocket channels). |
| [ui-design.md](./ui-design.md) | UI design reference for the React frontend. |
| [12-glossary.md](./12-glossary.md) | Terms used in the codebase (target face, source face, embedding, marker, DFM, ArcFace…). |

## Architecture summary

- **Backend** — `app/processors/*`: PyTorch / ONNX Runtime inference pipeline. UI-free.
- **API server** — `app/api/*`: FastAPI service exposing REST endpoints and 3 WebSocket channels. Single source of truth is `app/core/state.py` (`AppState`).
- **Web frontend** — `visomaster-ui/`: React + TypeScript + Tailwind + TanStack Start + shadcn/ui. Communicates through the `transport/` layer (WebSocket/REST in headless mode, QWebChannel bridge in Qt WebEngine mode).
- **Qt WebEngine desktop** — `web_main.py` + `app/ui/web_main.py`: `QWebEngineView` loads the React UI; `BackendBridge` (`app/ui/bridge.py`) exposes `@Slot`/`Signal` methods via `QWebChannel`, mirroring the REST API 1-to-1.
- **Legacy Qt desktop** — `main.py`: PySide6 `QApplication` + `MainWindow` with declarative layout-data parameter widgets.
- **WebRTC ingestion** — `packages/streamrelay/`: separate subprocess writing BGR frames into shared memory (`visomaster_webrtc_frame`).
- **Launcher** — `Start.bat`: unified Windows launcher with mode menu; manages Vite + API server background processes.
