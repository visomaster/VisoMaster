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
import gc
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy
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
                # Layout data stores numeric slider defaults as strings (e.g. '60').
                # Coerce them to int/float so numeric comparisons (e.g. sim >= threshold) work.
                if isinstance(default, str):
                    try:
                        default = int(default)
                    except ValueError:
                        try:
                            default = float(default)
                        except ValueError:
                            pass  # keep as string (e.g. model name selections)
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

    # ── VideoProcessor (headless — Phase 3 + 4) ──────────────────────────
    # Full thread-based play loop; no Qt timers required.
    class _HeadlessVideoProcessor:
        """
        Headless VideoProcessor for the FastAPI server.

        Phase 3: on_frame_done pushes JPEG frames to /ws/preview subscribers.
        Phase 4: process_video() runs a real threading.Thread play loop so
                 full video/webcam/webrtc playback works without a QApplication.
        """

        def __init__(self, mp_ref, state_ref: AppState):
            import queue as _queue
            self.models_processor = mp_ref
            self._state           = state_ref

            # Media state
            self.media_capture        = None
            self.file_type            = None
            self.fps                  = 0.0
            self.current_frame_number = 0
            self.max_frame_number     = 0
            self.media_path           = None
            self.processing           = False
            self.recording            = False
            self.current_frame        = []
            self.webrtc_shm           = None
            self._last_webrtc_counter = 0

            # FPS tracking
            self.current_fps    = 0.0
            self.fps_start_time = 0.0
            self.fps_frame_count = 0

            # Threading
            self.num_threads  = int(state_ref.control.get("nThreadsSlider", 2))
            self.frame_queue  = _queue.Queue(maxsize=self.num_threads)
            self._play_thread: threading.Thread | None = None
            self._play_lock   = threading.Lock()

            # Playback position tracking (mirrors Qt VideoProcessor interface used by FrameWorker)
            self.next_frame_to_display = 0

            # Recording
            self.recording_sp: subprocess.Popen | None = None
            self.temp_file    = ""
            self.start_time   = 0.0
            self.play_start_time = 0.0
            self.play_end_time   = 0.0

            # Virtual camera
            self.virtcam = None  # pyvirtualcam.Camera | None

            # Callbacks — wired by lifespan after construction
            self.on_frame_done   = lambda fn, f, s: None
            self.on_state_change = lambda ev, **kw: None
            self.on_fps_update   = lambda fps: None

        # ── Virtual camera ────────────────────────────────────────────────

        def enable_virtualcam(self, backend: str | bool = False) -> None:
            """Start (or restart) the pyvirtualcam output."""
            import pyvirtualcam as _pvc

            # Determine frame dimensions from the current frame or capture
            if isinstance(self.current_frame, numpy.ndarray) and self.current_frame.size > 0:
                frame_height, frame_width = self.current_frame.shape[:2]
            elif self.media_capture:
                frame_height = int(self.media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_width  = int(self.media_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            else:
                print("[VP] enable_virtualcam: no frame/capture available yet — deferring.")
                return

            self.disable_virtualcam()
            if not backend:
                backend = self._state.control.get("VirtCamBackendSelection", "obs")
            try:
                self.virtcam = _pvc.Camera(
                    width=frame_width,
                    height=frame_height,
                    fps=int(max(self.fps, 1)),
                    backend=backend,
                    fmt=_pvc.PixelFormat.BGR,
                )
                print(f"[VP] Virtual camera started: {frame_width}x{frame_height} @ {int(self.fps)}fps backend={backend}")
            except Exception as exc:
                print(f"[VP] enable_virtualcam failed: {exc}")
                self.virtcam = None

        def disable_virtualcam(self) -> None:
            """Stop and release the pyvirtualcam output."""
            if self.virtcam is not None:
                try:
                    self.virtcam.close()
                except Exception:
                    pass
                self.virtcam = None
                print("[VP] Virtual camera stopped.")

        def send_frame_to_virtualcam(self, frame_bgr: numpy.ndarray) -> None:
            """Send a BGR frame to the virtual camera if enabled."""
            if not self._state.control.get("SendVirtCamFramesEnableToggle", False):
                return
            # Auto-init on first frame if enable_virtualcam was called before
            # a frame was available (e.g. toggled before media started).
            if self.virtcam is None:
                self.enable_virtualcam()
                if self.virtcam is None:
                    return
            try:
                h, w = frame_bgr.shape[:2]
                # Reinitialise if dimensions changed (e.g. new video loaded)
                if self.virtcam.height != h or self.virtcam.width != w:
                    self.enable_virtualcam()
                    if self.virtcam is None:
                        return
                self.virtcam.send(frame_bgr)
                self.virtcam.sleep_until_next_frame()
            except Exception as exc:
                print(f"[VP] send_frame_to_virtualcam error: {exc}")

        # ── Internal helpers ──────────────────────────────────────────────

        def _make_fw_proxy(self):
            """Build the minimal object FrameWorker reads from."""
            st = self._state
            class _FWProxy:
                def __init__(self, vp, state, mp):
                    self.video_processor   = vp
                    self.models_processor  = mp
                    self.parameters        = state.parameters
                    self.target_faces      = dict(state.target_faces)
                    self.control           = state.control
                    self.markers           = {
                        pos: {"parameters": m.parameters, "control": m.control}
                        for pos, m in state.markers.items()
                    }
                    self.default_parameters = state.default_parameters
            return _FWProxy(self, st, self.models_processor)

        def _run_frame_worker(self, frame_rgb: numpy.ndarray, frame_number: int,
                              is_single_frame: bool = False) -> None:
            """Enqueue and run a FrameWorker for one frame."""
            from app.processors.workers.frame_worker import FrameWorker
            self.frame_queue.put(frame_number)
            proxy  = self._make_fw_proxy()
            worker = FrameWorker(frame_rgb, proxy, frame_number,  # type: ignore[arg-type]
                                 self.frame_queue, is_single_frame=is_single_frame)
            if is_single_frame:
                worker.run()
            else:
                worker.start()

        def _read_next_frame(self) -> numpy.ndarray | None:
            """Read the next raw BGR frame from the active source. Returns RGB or None.

            All cv2.VideoCapture reads go through the global lock in
            app.helpers.miscellaneous.read_frame so that concurrent calls from
            the play loop thread and the API request threads never touch the
            capture object simultaneously (which triggers the FFmpeg async_lock
            assertion).
            """
            import struct as _struct
            from app.helpers.miscellaneous import read_frame

            if self.file_type == "video" and self.media_capture:
                ret, frame = read_frame(self.media_capture)
                if ret:
                    return frame[..., ::-1]   # BGR → RGB
                return None

            elif self.file_type == "webcam" and self.media_capture:
                ret, frame = read_frame(self.media_capture)
                if ret:
                    return frame[..., ::-1]
                return None

            elif self.file_type == "webrtc":
                # Lazy-attach shared memory if not yet connected (subprocess may
                # still be starting up when the play loop begins).
                if self.webrtc_shm is None:
                    try:
                        from multiprocessing.shared_memory import SharedMemory as _SHM
                        self.webrtc_shm = _SHM(name="visomaster_webrtc_frame", create=False)
                        print("[VP] Lazily attached WebRTC shared memory.")
                    except FileNotFoundError:
                        return None  # subprocess not ready yet — keep polling

                from streamrelay.protocol import SHM_HEADER_BYTES
                counter = _struct.unpack_from("<I", self.webrtc_shm.buf, 0)[0]
                if counter == self._last_webrtc_counter or counter == 0:
                    return None   # No new frame yet
                self._last_webrtc_counter = counter
                w = _struct.unpack_from("<I", self.webrtc_shm.buf, 4)[0]
                h = _struct.unpack_from("<I", self.webrtc_shm.buf, 8)[0]
                if w == 0 or h == 0:
                    return None
                raw = bytes(self.webrtc_shm.buf[SHM_HEADER_BYTES: SHM_HEADER_BYTES + w * h * 3])
                frame = numpy.frombuffer(raw, dtype=numpy.uint8).reshape((h, w, 3)).copy()
                return frame[..., ::-1]   # BGR → RGB

            return None

        def _update_fps(self) -> None:
            """Update FPS counter and fire on_fps_update once per second."""
            self.fps_frame_count += 1
            now = time.time()
            if self.fps_start_time == 0:
                self.fps_start_time = now
            elapsed = now - self.fps_start_time
            if elapsed >= 1.0:
                self.current_fps     = self.fps_frame_count / elapsed
                self.fps_frame_count = 0
                self.fps_start_time  = now
                self.on_fps_update(self.current_fps)
                bus.emit_sync("fps_update", {"fps": round(self.current_fps, 1)})

        # ── Play loop (Phase 4) ───────────────────────────────────────────

        def _play_loop_video(self) -> None:
            """Thread body for video file playback."""
            interval = (1.0 / self.fps * 0.8) if self.fps > 0 else 0.033

            while self.processing:
                if self.current_frame_number > self.max_frame_number:
                    if self._state.loop_enabled:
                        # Seek back to the beginning and keep playing
                        self.current_frame_number = 0
                        if self.media_capture:
                            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        bus.emit_sync("playback_state", {
                            "current_frame": 0,
                            "max_frame": self.max_frame_number,
                            "is_playing": True,
                            "is_recording": self.recording,
                            "loop_enabled": True,
                        })
                        continue
                    break

                # Back-pressure: wait if workers are saturated
                if self.frame_queue.qsize() >= self.num_threads:
                    time.sleep(0.005)
                    continue

                frame_rgb = self._read_next_frame()
                if frame_rgb is None:
                    print(f"[VP] Cannot read frame {self.current_frame_number}, stopping.")
                    break

                fn = self.current_frame_number
                self._run_frame_worker(frame_rgb, fn, is_single_frame=False)
                self.current_frame_number += 1
                time.sleep(interval)

            self.stop_processing()

        def _play_loop_live(self) -> None:
            """Thread body for webcam / webrtc live sources."""
            interval = (1.0 / self.fps * 0.8) if self.fps > 0 else 0.033

            while self.processing:
                if self.frame_queue.qsize() >= self.num_threads:
                    time.sleep(0.005)
                    continue

                frame_rgb = self._read_next_frame()
                if frame_rgb is None:
                    time.sleep(0.01)   # No new frame yet — poll
                    continue

                self._update_fps()
                self._run_frame_worker(frame_rgb, self.current_frame_number,
                                       is_single_frame=False)
                self.current_frame_number += 1
                time.sleep(interval)

        # ── Public API ────────────────────────────────────────────────────

        def process_video(self) -> None:
            """Start the play loop in a background thread."""
            with self._play_lock:
                if self.processing:
                    print("[VP] Already processing — ignoring start request.")
                    return
                if self.file_type is None:
                    print("[VP] No media selected.")
                    return

                self.processing = True
                self.fps_start_time  = 0.0
                self.fps_frame_count = 0
                self.current_fps     = 0.0
                self.start_time      = time.perf_counter()

                # Clear queues
                with self.frame_queue.mutex:
                    self.frame_queue.queue.clear()

                if self.file_type == "video":
                    if not self.media_capture or not self.media_capture.isOpened():
                        print("[VP] Video capture not open.")
                        self.processing = False
                        self.on_state_change("error", message="Unable to open video")
                        return
                    self.play_start_time = float(
                        self.media_capture.get(cv2.CAP_PROP_POS_FRAMES) / max(self.fps, 1)
                    )
                    if self.recording:
                        self._start_ffmpeg()
                        self.on_state_change("recording_started")
                    target = self._play_loop_video
                else:
                    target = self._play_loop_live

                self._play_thread = threading.Thread(target=target, daemon=True,
                                                     name="vp-play-loop")
                self._play_thread.start()
                self.on_state_change("playing")
                print(f"[VP] Play loop started for file_type={self.file_type}")

        def stop_processing(self) -> bool:
            with self._play_lock:
                if not self.processing:
                    self.on_state_change("stopped")
                    return False

                print("[VP] Stopping processing.")
                self.processing = False

            # Join the play thread outside the lock to avoid deadlock.
            # Skip the join when stop_processing() is called from within the play
            # thread itself (e.g. at end-of-video) — joining the current thread
            # raises RuntimeError("cannot join current thread").
            caller_is_play_thread = (
                self._play_thread is not None
                and self._play_thread == threading.current_thread()
            )
            if self._play_thread and self._play_thread.is_alive() and not caller_is_play_thread:
                self._play_thread.join(timeout=3.0)
            self._play_thread = None

            # Drain queues
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()

            # Finalise recording
            if self.recording and self.file_type == "video" and self.recording_sp:
                self.recording_sp.stdin.close()
                self.recording_sp.wait()
                self.play_end_time = float(
                    self.media_capture.get(cv2.CAP_PROP_POS_FRAMES) / max(self.fps, 1)
                ) if self.media_capture else 0.0
                final_path = self._mux_audio()
                self.on_state_change("recording_stopped", output_path=final_path)

            self.recording    = False
            self.recording_sp = None

            torch.cuda.empty_cache()
            gc.collect()
            self.on_state_change("stopped")
            print("[VP] Stopped.")
            return True

        def process_current_frame(self) -> None:
            """Process a single frame synchronously (preview / seek).

            When the play loop is already running, skip the capture read entirely —
            the loop is already pushing frames and cv2.VideoCapture is not
            thread-safe (concurrent reads trigger the FFmpeg async_lock assertion).
            """
            import struct as _struct
            from app.helpers.miscellaneous import read_image_file, read_frame

            # Don't touch the capture while the play loop thread is running.
            if self.processing and self.file_type == "video":
                return

            frame_rgb = None

            if self.file_type == "image" and self.media_path:
                bgr = read_image_file(self.media_path)
                if bgr is not None:
                    frame_rgb = bgr[..., ::-1]

            elif self.file_type == "video" and self.media_capture:
                ret, bgr = read_frame(self.media_capture)
                if ret:
                    frame_rgb = bgr[..., ::-1]
                    self.media_capture.set(cv2.CAP_PROP_POS_FRAMES,
                                           self.current_frame_number)

            elif self.file_type == "webcam" and self.media_capture:
                ret, bgr = read_frame(self.media_capture)
                if ret:
                    frame_rgb = bgr[..., ::-1]

            elif self.file_type == "webrtc":
                if self.webrtc_shm is None:
                    try:
                        from multiprocessing.shared_memory import SharedMemory as _SHM
                        self.webrtc_shm = _SHM(name="visomaster_webrtc_frame", create=False)
                    except FileNotFoundError:
                        pass
                if self.webrtc_shm is not None:
                    from streamrelay.protocol import SHM_HEADER_BYTES
                    w = _struct.unpack_from("<I", self.webrtc_shm.buf, 4)[0]
                    h = _struct.unpack_from("<I", self.webrtc_shm.buf, 8)[0]
                    if w > 0 and h > 0:
                        raw = bytes(self.webrtc_shm.buf[SHM_HEADER_BYTES:
                                                        SHM_HEADER_BYTES + w * h * 3])
                        frame_rgb = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(
                            (h, w, 3)).copy()[..., ::-1]

            if frame_rgb is None:
                return

            self._run_frame_worker(frame_rgb, self.current_frame_number,
                                   is_single_frame=True)

        # ── Recording helpers ─────────────────────────────────────────────

        def _start_ffmpeg(self) -> None:
            """Spawn the ffmpeg stdin-pipe encoder."""
            if not isinstance(self.current_frame, numpy.ndarray) or self.current_frame.size == 0:
                print("[VP] No reference frame for ffmpeg dimensions — using 1280x720.")
                h, w = 720, 1280
            else:
                h, w = self.current_frame.shape[:2]

            self.temp_file = "temp_output.mp4"
            if Path(self.temp_file).is_file():
                os.remove(self.temp_file)

            from app.helpers.miscellaneous import get_ffmpeg_path as _get_ffmpeg
            args = [
                _get_ffmpeg(), "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(max(self.fps, 1)),
                "-i", "pipe:",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p",
                "-c:v", "libx264", "-crf", "18",
                self.temp_file,
            ]
            self.recording_sp = subprocess.Popen(args, stdin=subprocess.PIPE)

        def _mux_audio(self) -> str:
            """Mux audio from the original file into the recorded video.
            Returns the final output path."""
            from app.helpers.miscellaneous import get_output_file_path, get_ffmpeg_path as _get_ffmpeg
            output_folder = self._state.control.get("OutputMediaFolder", ".")
            final_path = get_output_file_path(self.media_path or "output.mp4", output_folder)
            if Path(final_path).is_file():
                os.remove(final_path)
            args = [
                _get_ffmpeg(), "-hide_banner", "-loglevel", "error",
                "-i", self.temp_file,
                "-ss", str(self.play_start_time),
                "-to", str(self.play_end_time),
                "-i", self.media_path,
                "-c", "copy",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest", final_path,
            ]
            subprocess.run(args, check=False)
            if Path(self.temp_file).is_file():
                os.remove(self.temp_file)
            elapsed = time.perf_counter() - self.start_time
            print(f"[VP] Recording saved to {final_path} ({elapsed:.1f}s)")
            bus.emit_sync("recording_finished", {"output_path": final_path})
            return final_path

    import cv2  # noqa: already imported at module level
    vp = _HeadlessVideoProcessor(mp, state)

    # ── Wire callbacks ────────────────────────────────────────────────────
    def _headless_on_frame_done(frame_number: int, frame_bgr, is_single_frame: bool):
        if isinstance(frame_bgr, numpy.ndarray):
            vp.current_frame = frame_bgr
            vp.next_frame_to_display = frame_number + 1
            # Push JPEG to all /ws/preview subscribers (latest-wins, no queue)
            bus.emit_frame_sync(frame_bgr)
            # Push position update — use latest-wins slot so the JSON channel
            # is never flooded; only the most recent frame number is delivered
            bus.emit_position_sync(frame_number, vp.max_frame_number)
            # Feed the native Qt preview window if it is open
            from app.ui.widgets.headless_preview import headless_preview
            if headless_preview.is_open:
                headless_preview.push_frame(frame_bgr)
                headless_preview.sync_state(
                    frame_number,
                    vp.max_frame_number,
                    vp.processing,
                    state.loop_enabled,
                    set(state.markers.keys()),
                )
            # Send to virtual camera if enabled
            vp.send_frame_to_virtualcam(frame_bgr)
    vp.on_frame_done   = _headless_on_frame_done
    vp.on_state_change = lambda ev, **kw: bus.emit_sync(
        "state_updated", {"section": "playback", "event": ev, **kw}
    )
    vp.on_fps_update   = lambda fps: bus.emit_sync("fps_update", {"fps": round(fps, 1)})

    # ── Store on app.state ────────────────────────────────────────────────
    app.state.app_state = state
    app.state.models_processor = mp
    app.state.video_processor = vp
    app.state.event_bus = bus
    app.state.webrtc_process = None   # StreamRelay subprocess — shared by sources.py and ws.py

    # ── Start event bus broadcast loops ──────────────────────────────────
    loop = asyncio.get_event_loop()
    bus.set_loop(loop)
    broadcast_task = asyncio.create_task(bus._broadcast_loop())
    position_task  = asyncio.create_task(bus._position_broadcast_loop())

    async def _gpu_memory_broadcast_loop() -> None:
        """Emit gpu_memory events to all /ws/events clients every ~2 s."""
        try:
            # First emit fires almost immediately so newly-connected clients
            # see real values instead of zeros.
            await asyncio.sleep(0.5)
            while True:
                if bus._clients:
                    try:
                        used, total = mp.get_gpu_memory()
                        bus.emit_sync(
                            "gpu_memory",
                            {"used_mb": int(used), "total_mb": int(total)},
                        )
                    except Exception as exc:
                        print(f"[api] gpu_memory broadcast failed: {exc}", flush=True)
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass

    gpu_memory_task = asyncio.create_task(_gpu_memory_broadcast_loop())

    print("[API] VisoMaster API server ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    broadcast_task.cancel()
    position_task.cancel()
    gpu_memory_task.cancel()
    for t in (broadcast_task, position_task, gpu_memory_task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    # stop_processing() may join a background thread — run it in an executor
    # so it doesn't block the event loop and stall uvicorn's reload.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, vp.stop_processing)
    vp.disable_virtualcam()

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
    from app.api.routes.client_log import router as client_log_router

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
    app.include_router(client_log_router)
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
        # Disable WebSocket keepalive pings — the /ws/preview endpoint pushes
        # high-frequency binary frames and the ping waiter assertion in the
        # websockets library fires when the write buffer is under load.
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )
