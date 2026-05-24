# VisoMaster Architecture Docs

These documents describe the **existing** VisoMaster codebase, then propose a path to **extract the backend** so a new React (or Electron) UI can sit on top of it.

The goal: tell two stories.

1. **What VisoMaster is today** — a PySide6 desktop app where the Qt `MainWindow` is also the runtime "controller", parameters are stored on the window object, and processing happens in Python `threading.Thread` workers driven by Qt timers.
2. **How to split it** — the inference pipeline (ONNX Runtime + TensorRT + DFM models) is mostly UI-free and can be lifted into a long-running Python service. The UI's role boils down to: media browsing, parameter editing, frame display, recording control, and (now) WebRTC ingestion. All of those map cleanly onto a REST + WebSocket API.

## Doc index

| Doc | What it covers |
|---|---|
| [01-overview.md](./01-overview.md) | Repo layout, runtime layers, what each top-level thing does. |
| [02-startup-and-bootstrap.md](./02-startup-and-bootstrap.md) | `main.py` → `QApplication` → `MainWindow`, model paths, last workspace dialog. |
| [03-ui-layer.md](./03-ui-layer.md) | `MainWindow`, widgets, layout-data driven parameter system, signals. |
| [04-backend-pipeline.md](./04-backend-pipeline.md) | `VideoProcessor`, `FrameWorker`, `ModelsProcessor`, threading model. |
| [05-models-and-inference.md](./05-models-and-inference.md) | ONNX model registry, TensorRT engines, DFM, ArcFace/landmark mappings. |
| [06-data-flows.md](./06-data-flows.md) | End-to-end flows: load video, find faces, swap, record, webcam, WebRTC. |
| [07-state-and-persistence.md](./07-state-and-persistence.md) | `parameters` / `control` / `markers` schema, `last_workspace.json`, embeddings JSON. |
| [08-streamrelay-webrtc.md](./08-streamrelay-webrtc.md) | The bundled `streamrelay` package: WHIP, WebSocket, shared-memory frame transport. |
| [09-coupling-and-seams.md](./09-coupling-and-seams.md) | Where UI and backend are tangled, and where the natural seams are. |
| [10-proposed-api.md](./10-proposed-api.md) | A REST + WebSocket API design that mirrors current functionality. |
| [11-react-electron-roadmap.md](./11-react-electron-roadmap.md) | Concrete migration plan: phases, milestones, what to build first. |
| [12-glossary.md](./12-glossary.md) | Terms used in the codebase (target face, source face, embedding, marker, DFM, ArcFace…). |

## TL;DR

- Backend is `app/processors/*` — already standalone-ish PyTorch / ONNX Runtime code, plus a Qt `QObject` (`VideoProcessor`) that owns timers and threads.
- UI is `app/ui/*` — Qt widgets, but **most "logic" lives in `app/ui/widgets/actions/*.py`**, which mutate state on the `MainWindow` instance. That state (`main_window.control`, `main_window.parameters`, `main_window.target_faces`, etc.) is the de-facto API surface.
- WebRTC ingestion is **already a separate process** (`packages/streamrelay`) talking via shared memory — that's the template for how the rest of the app should be split.
- A clean split: keep `app/processors/*` as a Python service, replace `app/ui/*` with a REST/WS server + a React frontend. State that currently lives on `MainWindow` becomes server-side session state.
