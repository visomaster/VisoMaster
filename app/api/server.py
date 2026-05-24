"""
app/api/server.py
─────────────────
FastAPI application factory and lifespan.

Usage:
    uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload

Or via the CLI helper:
    python -m app.api.server
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Bootstrap streamrelay path (same as main.py) ─────────────────────────────
_streamrelay_src = Path(__file__).parent.parent.parent / "packages" / "streamrelay" / "src"
if _streamrelay_src.is_dir() and str(_streamrelay_src) not in sys.path:
    sys.path.insert(0, str(_streamrelay_src))


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: build AppState + ModelsProcessor + VideoProcessor.
    Shutdown: stop processing, save workspace.
    """
    from app.api.events import bus
    from app.core.state import AppState
    from app.processors.models_processor import ModelsProcessor
    from app.processors.video_processor import VideoProcessor

    # ── Build default parameters from layout-data ─────────────────────────
    default_parameters: Dict[str, Any] = {}
    default_control: Dict[str, Any] = {}

    def _collect_defaults(layout_data: dict, target: dict) -> None:
        for section, widgets in layout_data.items():
            for name, cfg in widgets.items():
                default = cfg.get("default")
                if callable(default):
                    default = default()
                target[name] = default

    from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
    from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
    from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
    from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA

    _collect_defaults(COMMON_LAYOUT_DATA, default_parameters)
    _collect_defaults(SWAPPER_LAYOUT_DATA, default_parameters)
    _collect_defaults(FACE_EDITOR_LAYOUT_DATA, default_parameters)
    _collect_defaults(SETTINGS_LAYOUT_DATA, default_control)

    # ── AppState ──────────────────────────────────────────────────────────
    state = AppState(
        control=copy.deepcopy(default_control),
        default_parameters=default_parameters,
    )
    state.output_media_folder = default_control.get("OutputMediaFolder", "")

    # Load last workspace if it exists
    ws_path = Path("last_workspace.json")
    if ws_path.is_file():
        try:
            with open(ws_path, "r", encoding="utf-8") as f:
                ws_data = json.load(f)
            loaded = AppState.from_json(ws_data, default_parameters)
            state.control.update(loaded.control)
            state.target_media = loaded.target_media
            state.target_faces = loaded.target_faces
            state.input_faces = loaded.input_faces
            state.embeddings = loaded.embeddings
            state.markers = loaded.markers
            state.parameters = loaded.parameters
            state.selected_media_id = loaded.selected_media_id
            state.webcam_transform = loaded.webcam_transform
            state.webrtc_transform = loaded.webrtc_transform
            state.last_target_media_folder = loaded.last_target_media_folder
            state.last_input_media_folder = loaded.last_input_media_folder
            state.loaded_embedding_filename = loaded.loaded_embedding_filename
            print("[API] Last workspace loaded.")
        except Exception as exc:
            print(f"[API] Could not load last workspace: {exc}")

    # ── ModelsProcessor (headless stub — no MainWindow) ───────────────────
    # We pass a lightweight proxy so ModelsProcessor's signal calls are no-ops.
    class _HeadlessProxy:
        """Minimal stand-in for MainWindow used by ModelsProcessor."""
        class _Signal:
            def emit(self, *a, **kw): pass
            def connect(self, *a, **kw): pass

        model_loading_signal = _Signal()
        model_loaded_signal = _Signal()
        display_messagebox_signal = _Signal()

        def __init__(self, ctrl: dict, params: dict, dfm_data: dict):
            self.control = ctrl
            self.parameters = params
            self.dfm_models_data = dfm_data

    from app.helpers.miscellaneous import get_dfm_models_data
    proxy = _HeadlessProxy(
        ctrl=state.control,
        params=state.parameters,
        dfm_data=get_dfm_models_data(),
    )

    mp = ModelsProcessor(proxy)  # type: ignore[arg-type]

    # ── VideoProcessor (headless stub) ────────────────────────────────────
    # VideoProcessor is tightly coupled to Qt timers and MainWindow.
    # For the API we use a thin wrapper that exposes only the state fields
    # the routes need, without starting any Qt timers.
    class _HeadlessVideoProcessor:
        """
        Minimal VideoProcessor stand-in for the API layer.

        Full Qt-timer-driven processing is not available here — the API
        routes call process_current_frame() directly for single-frame
        previews.  Playback (play/stop) is deferred to Phase 2 when the
        Qt coupling is removed from VideoProcessor.
        """
        def __init__(self, mp_ref, state_ref: AppState):
            import queue
            self.models_processor = mp_ref
            self._state = state_ref
            self.media_capture = None
            self.file_type = None
            self.fps = 0.0
            self.current_frame_number = 0
            self.max_frame_number = 0
            self.media_path = None
            self.processing = False
            self.recording = False
            self.current_frame = []
            self.webrtc_shm = None
            self._last_webrtc_counter = 0
            self.current_fps = 0.0
            self.fps_start_time = 0.0
            self.fps_frame_count = 0
            self.num_threads = int(state_ref.control.get("nThreadsSlider", 2))
            self.frame_queue = queue.Queue(maxsize=self.num_threads)
            self.next_frame_to_display = 0
            self.frames_to_display = {}

            # Callbacks wired below
            self.on_frame_done   = lambda fn, f, s: None
            self.on_state_change = lambda ev, **kw: None
            self.on_fps_update   = lambda fps: None

        def stop_processing(self) -> bool:
            if not self.processing:
                return False
            self.processing = False
            self.on_state_change('stopped')
            return True

        def process_video(self):
            """Stub — full playback loop requires Qt timers (Phase 2)."""
            self.processing = True
            self.on_state_change('playing')

        def process_current_frame(self):
            """
            Process a single frame synchronously and store it in current_frame.
            Works for image and video sources; webcam/webrtc read the latest frame.
            """
            import numpy as np
            import struct
            from app.helpers.miscellaneous import read_image_file, read_frame
            from app.processors.workers.frame_worker import FrameWorker

            frame = None
            if self.file_type == "image" and self.media_path:
                frame = read_image_file(self.media_path)
                if frame is not None:
                    frame = frame[..., ::-1]  # BGR → RGB
            elif self.file_type == "video" and self.media_capture:
                ret, frame = read_frame(self.media_capture)
                if ret:
                    frame = frame[..., ::-1]
                    self.media_capture.set(
                        cv2.CAP_PROP_POS_FRAMES, self.current_frame_number
                    )
            elif self.file_type == "webcam" and self.media_capture:
                import cv2 as cv2_local
                ret, frame = self.media_capture.read()
                if ret:
                    frame = frame[..., ::-1]
            elif self.file_type == "webrtc" and self.webrtc_shm is not None:
                from streamrelay.protocol import SHM_HEADER_BYTES
                w = struct.unpack_from("<I", self.webrtc_shm.buf, 4)[0]
                h = struct.unpack_from("<I", self.webrtc_shm.buf, 8)[0]
                if w > 0 and h > 0:
                    raw = bytes(self.webrtc_shm.buf[SHM_HEADER_BYTES: SHM_HEADER_BYTES + w * h * 3])
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
                    frame = frame[..., ::-1]

            if frame is None:
                return

            import queue as _queue
            self.frame_queue.put(self.current_frame_number)

            # Build a minimal proxy that FrameWorker can read from
            class _FWProxy:
                def __init__(self, vp, st, mp_ref):
                    self.video_processor = vp
                    self.models_processor = mp_ref
                    self.parameters = st.parameters
                    self.target_faces = {
                        fid: tf for fid, tf in st.target_faces.items()
                    }
                    self.control = st.control
                    self.markers = {
                        pos: {'parameters': m.parameters, 'control': m.control}
                        for pos, m in st.markers.items()
                    }
                    self.default_parameters = st.default_parameters

            fw_proxy = _FWProxy(self, self._state, self.models_processor)
            worker = FrameWorker(
                frame, fw_proxy, self.current_frame_number,  # type: ignore[arg-type]
                self.frame_queue, is_single_frame=True
            )
            worker.run()
            # FrameWorker calls on_frame_done directly now; current_frame is
            # updated there via the callback wired in lifespan.

    import cv2  # noqa: F401 — needed inside _HeadlessVideoProcessor
    vp = _HeadlessVideoProcessor(mp, state)

    # Wire callbacks
    import numpy as _np

    def _headless_on_frame_done(frame_number: int, frame_bgr, is_single_frame: bool):
        if isinstance(frame_bgr, _np.ndarray):
            vp.current_frame = frame_bgr
            bus.emit_sync("frame_processed", {
                "frame_number": frame_number,
                "width": frame_bgr.shape[1],
                "height": frame_bgr.shape[0],
            })

    vp.on_frame_done   = _headless_on_frame_done
    vp.on_state_change = lambda ev, **kw: bus.emit_sync("state_updated", {"section": "playback", "event": ev, **kw})
    vp.on_fps_update   = lambda fps: bus.emit_sync("fps_update", {"fps": fps})

    # ── Store on app.state ────────────────────────────────────────────────
    app.state.app_state = state
    app.state.models_processor = mp
    app.state.video_processor = vp
    app.state.event_bus = bus

    # ── Start event bus broadcast loop ────────────────────────────────────
    loop = asyncio.get_event_loop()
    bus.set_loop(loop)
    broadcast_task = asyncio.create_task(bus._broadcast_loop())

    print("[API] VisoMaster API server ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    broadcast_task.cancel()
    vp.stop_processing()
    try:
        with open("last_workspace.json", "w", encoding="utf-8") as f:
            json.dump(state.to_json(), f, indent=4)
        print("[API] Workspace saved.")
    except Exception as exc:
        print(f"[API] Could not save workspace: {exc}")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="VisoMaster API",
        description="REST + WebSocket backend for the VisoMaster face-swap engine.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Allow the React dev server (Vite default: 5173) and any localhost origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    from app.api.routes.system import router as system_router
    from app.api.routes.schema import router as schema_router
    from app.api.routes.state import router as state_router
    from app.api.routes.workspace import router as workspace_router
    from app.api.routes.target_media import router as target_media_router
    from app.api.routes.faces import router as faces_router
    from app.api.routes.embeddings import router as embeddings_router
    from app.api.routes.playback import router as playback_router
    from app.api.routes.sources import router as sources_router
    from app.api.ws import router as ws_router

    app.include_router(system_router)
    app.include_router(schema_router)
    app.include_router(state_router)
    app.include_router(workspace_router)
    app.include_router(target_media_router)
    app.include_router(faces_router)
    app.include_router(embeddings_router)
    app.include_router(playback_router)
    app.include_router(sources_router)
    app.include_router(ws_router)

    return app


app = create_app()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["*.onnx", "*.trt", "*.dfm"],
    )
