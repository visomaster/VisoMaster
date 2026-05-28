"""
app/ui/bridge.py
────────────────
QWebChannel backend bridge — exposes Python slots and signals to the
React frontend running inside QWebEngineView.

Replaces the FastAPI + WebSocket layer for the Qt desktop mode.
All slots mirror the existing API routes 1-to-1 so the frontend
transport adapter can call them identically.
"""
from __future__ import annotations

import copy
import json
import os
import traceback as _tb
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot
from PySide6 import QtWidgets

if TYPE_CHECKING:
    from app.ui.web_main import WebMainWindow


# ── Tiny logger — prefix every line so it's easy to grep ─────────────────────
def _log(tag: str, msg: str) -> None:
    print(f"[bridge:{tag}] {msg}", flush=True)

def _log_err(tag: str, msg: str, exc: Exception | None = None) -> None:
    print(f"[bridge:{tag}] ERROR — {msg}", flush=True)
    if exc is not None:
        print(_tb.format_exc(), flush=True)


# ── Swapper model name → ONNX model key(s) in ModelsProcessor.models ─────────
# Used to unload the old swapper when the user switches models so the new one
# is loaded fresh with the correct emap on the next frame.
_SWAPPER_MODEL_KEYS: dict[str, list[str]] = {
    'Inswapper128':                ['Inswapper128'],
    'InStyleSwapper256 Version A': ['InStyleSwapper256 Version A'],
    'InStyleSwapper256 Version B': ['InStyleSwapper256 Version B'],
    'InStyleSwapper256 Version C': ['InStyleSwapper256 Version C'],
    'SimSwap512':                  ['SimSwap512'],
    'GhostFace-v1':                ['GhostFacev1'],
    'GhostFace-v2':                ['GhostFacev2'],
    'GhostFace-v3':                ['GhostFacev3'],
    'CSCS':                        ['CSCS'],
    # DFM models are managed separately via dfm_models dict — no unload needed here
}


def _unload_swapper_model(mp, model_selection: str) -> None:
    """Unload the ONNX session(s) for the given SwapModelSelection value.

    This forces the new model to be loaded fresh on the next frame, which
    also ensures load_inswapper_iss_emap() runs and sets the correct emap.
    """
    keys = _SWAPPER_MODEL_KEYS.get(model_selection, [])
    for key in keys:
        if mp.models.get(key) is not None:
            print(f"[bridge] Unloading swapper model '{key}' for model switch", flush=True)
            mp.unload_model(key)


class BackendBridge(QObject):
    # ── Signals → WebUI ───────────────────────────────────────────────────
    # All carry JSON strings; JS side does JSON.parse()

    # Full playback state snapshot
    playbackStateChanged = Signal(str)
    # Lightweight per-frame position (high-frequency, from FrameWorker thread)
    framePositionChanged = Signal(str)
    # GPU memory every 3 s
    gpuMemoryChanged = Signal(str)
    # State mutation echo (control / parameters)
    stateUpdated = Signal(str)
    # FPS for streaming sources
    fpsUpdated = Signal(str)
    # Recording mux finished
    recordingFinished = Signal(str)
    # Model load spinner
    modelLoading = Signal()
    modelLoaded = Signal()
    # Face detection complete
    facesFound = Signal(str)
    # Workspace loaded — frontend should re-pull state
    workspaceLoaded = Signal(str)
    # Virtual camera actual state (after enable/disable attempt)
    virtcamStateChanged = Signal(str)
    # General error notification
    errorOccurred = Signal(str)
    # Preview window opened/closed
    previewWindowOpened = Signal()
    previewWindowClosed = Signal()

    def __init__(self, main_window: "WebMainWindow") -> None:
        super().__init__()
        self._mw = main_window
        _log("init", "BackendBridge created — wiring VP callbacks")
        self._setup_vp_callbacks()
        self._setup_gpu_timer()
        # Pending frame-refresh worker — kept so we can cancel a stale one
        # when a new parameter change arrives before the previous render finishes.
        self._pending_frame_worker: "QtCore.QThread | None" = None
        _log("init", "BackendBridge ready")

    # ── Async frame refresh ───────────────────────────────────────────────

    def _process_frame_async(self) -> None:
        """Run process_current_frame() on a background QThread so the main
        thread (and therefore the UI) is never blocked by GPU inference.

        If a previous render is still in flight we let it finish — the
        VideoProcessor's frame_queue already serialises work, so queuing
        another one on top would just waste GPU time.  We simply skip the
        new request in that case; the debounce on the frontend means the
        final resting value will always be sent once the user stops dragging.
        """
        from PySide6.QtCore import QThread

        # If the previous worker is still running, skip — the VP is busy.
        if self._pending_frame_worker is not None and self._pending_frame_worker.isRunning():
            return

        vp = self._mw.video_processor

        class _FrameThread(QThread):
            def run(self_t):
                try:
                    vp.process_current_frame()
                except Exception as exc:
                    print(f"[bridge:_process_frame_async] error: {exc}", flush=True)

        worker = _FrameThread(self)
        self._pending_frame_worker = worker
        worker.start()

    # ── Native preview window helpers ─────────────────────────────────────

    def _open_preview_window(self) -> None:
        """Open (or raise) the native Qt PreviewWindow.

        Called automatically when media / webcam / webrtc is selected so the
        user always has a visible preview surface — frame delivery flows from
        VideoProcessor._send_frame_to_output_window → preview_frame_signal →
        PreviewWindow.update_frame.
        """
        from app.ui.widgets.preview_window import PreviewWindow
        mw = self._mw
        if getattr(mw, "_preview_window", None) is not None:
            try:
                if mw._preview_window.isVisible():
                    mw._preview_window.raise_()
                    mw._preview_window.activateWindow()
                    _log("preview", "window already open — raised")
                    return
            except RuntimeError:
                # Underlying C++ object was deleted — fall through and recreate
                mw._preview_window = None

        _log("preview", "creating new PreviewWindow")
        mw._preview_window = PreviewWindow(mw)
        mw._preview_window.show()
        mw._preview_window.raise_()
        mw._preview_window.activateWindow()
        self.previewWindowOpened.emit()

        # Sync playback controls immediately so the slider/buttons reflect
        # the current state rather than showing defaults.
        vp = mw.video_processor
        loop = bool(mw.control.get("LoopVideoToggle", False) or mw.control.get("loop_enabled", False))
        mw._preview_window.sync_playback_state(
            vp.current_frame_number,
            vp.max_frame_number,
            bool(vp.processing),
            loop,
        )
        mw._preview_window.set_markers(set(mw.markers.keys()))

        # Push the last processed frame so the window isn't black on open.
        # For paused video or static images: re-run the pipeline to get a fresh frame.
        # For live streams already playing: deliver the last known frame directly —
        # the next frame from the play loop will arrive within milliseconds anyway.
        import numpy as _np
        if vp.file_type in ("video", "image") and not vp.processing:
            _log("preview", "pushing last frame via process_current_frame() async")
            self._process_frame_async()
        elif isinstance(getattr(vp, "current_frame", None), _np.ndarray) and vp.current_frame.size > 0:
            _log("preview", "delivering last live frame directly to preview window")
            mw._preview_window.update_frame(vp.current_frame)

    # ── Internal wiring ───────────────────────────────────────────────────

    def _setup_vp_callbacks(self) -> None:
        vp = self._mw.video_processor
        _log("setup", f"Wrapping VP callbacks (vp id={id(vp)})")

        _orig_frame_done = vp.on_frame_done

        def _on_frame_done(frame_number: int, frame_bgr, is_single_frame: bool):
            _orig_frame_done(frame_number, frame_bgr, is_single_frame)
            # Schedule the position-update signal on the main thread —
            # FrameWorker runs on a worker thread, and QWebChannel signal
            # delivery to JavaScript only works reliably from the main thread.
            pos_json = json.dumps({
                "current_frame": frame_number,
                "max_frame":     vp.max_frame_number,
                "is_playing":    True,
            })
            QTimer.singleShot(0, lambda j=pos_json: self.framePositionChanged.emit(j))
        vp.on_frame_done = _on_frame_done
        _log("setup", "on_frame_done wrapped → emits framePositionChanged")

        _orig_state_change = vp.on_state_change
        def _on_state_change(event: str, **kwargs):
            _log("vp_state", f"event='{event}' kwargs={kwargs}")
            _orig_state_change(event, **kwargs)
            if event in ("stopped", "playing", "recording_started", "recording_stopped"):
                _log("vp_state", f"emitting playbackStateChanged for event='{event}'")
                self._emit_playback()
            if event == "recording_stopped":
                # Use the path passed by stop_processing(); fall back to
                # computing it only if not provided (shouldn't happen).
                out_path = kwargs.get("output_path", "")
                if not out_path:
                    try:
                        from app.helpers.miscellaneous import get_output_file_path as _gofp
                        out_path = _gofp(
                            vp.media_path or "",
                            self._mw.control.get("OutputMediaFolder", ".")
                        )
                    except Exception:
                        out_path = ""
                self.recordingFinished.emit(json.dumps({
                    "output_path": out_path
                }))
        vp.on_state_change = _on_state_change
        _log("setup", "on_state_change wrapped → emits playbackStateChanged")

        _orig_fps = vp.on_fps_update
        def _on_fps_update(fps: float):
            _orig_fps(fps)
            self.fpsUpdated.emit(json.dumps({"fps": round(fps, 1)}))
        vp.on_fps_update = _on_fps_update
        _log("setup", "on_fps_update wrapped → emits fpsUpdated")

        vp.fps_update_signal.connect(
            lambda fps: self.fpsUpdated.emit(json.dumps({"fps": round(fps, 1)})),
            Qt.ConnectionType.QueuedConnection,
        )
        _log("setup", "fps_update_signal connected (queued)")

    def _setup_gpu_timer(self) -> None:
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(2000)  # 2 s — feels live without thrashing
        self._gpu_timer.timeout.connect(self._emit_gpu_memory)
        self._gpu_timer.start()
        self._gpu_last_error_time: float = 0.0
        # Fire once right away so the TopBar shows a value within the first
        # render frame instead of "—" for 2-3 s.
        QTimer.singleShot(0, self._emit_gpu_memory)
        # And again once models start being loaded — first call may report
        # zero before CUDA is initialised.
        QTimer.singleShot(1500, self._emit_gpu_memory)

    def _emit_gpu_memory(self) -> None:
        import time
        try:
            used, total = self._mw.models_processor.get_gpu_memory()
        except Exception as e:
            # Log at most once every 60 s so the console isn't spammed,
            # but keep retrying — CUDA may not be ready on the first few calls.
            now = time.monotonic()
            if now - self._gpu_last_error_time > 60.0:
                _log_err("gpu_memory", "get_gpu_memory() raised", e)
                self._gpu_last_error_time = now
            return

        # Always emit, even when total is 0 — frontend should show "—" but
        # at least we know the bridge is alive.
        payload = json.dumps({"used_mb": int(used), "total_mb": int(total)})
        self.gpuMemoryChanged.emit(payload)

    def _emit_playback(self) -> None:
        vp = self._mw.video_processor
        payload = {
            "is_playing":    vp.processing,
            "is_recording":  vp.recording,
            "current_frame": vp.current_frame_number,
            "max_frame":     vp.max_frame_number,
            "fps":           vp.fps,
            "file_type":     vp.file_type,
            "loop_enabled":  self._mw.control.get("loop_enabled", False),
            "swap_enabled":  self._mw.control.get("_swap_enabled", False),
            "edit_enabled":  self._mw.control.get("_edit_enabled", False),
        }
        _log("emit_playback", f"is_playing={payload['is_playing']} file_type={payload['file_type']} "
             f"frame={payload['current_frame']}/{payload['max_frame']} fps={payload['fps']}")
        self.playbackStateChanged.emit(json.dumps(payload))

    # ── Playback slots ────────────────────────────────────────────────────

    @Slot()
    def play(self) -> None:
        vp = self._mw.video_processor
        _log("play", f"called — vp.processing={vp.processing} vp.file_type={vp.file_type}")
        if not vp.processing and vp.file_type:
            _log("play", "calling vp.process_video()")
            vp.process_video()
            _log("play", f"process_video() returned — vp.processing={vp.processing}")
        else:
            _log("play", "skipped process_video() (already processing or no file_type)")
        QTimer.singleShot(100, self._emit_playback)

    @Slot()
    def stop(self) -> None:
        _log("stop", "called")
        self._mw.video_processor.stop_processing()
        self._emit_playback()

    @Slot(int)
    def seek(self, frame: int) -> None:
        import cv2
        vp = self._mw.video_processor
        clamped = max(0, min(frame, vp.max_frame_number))
        _log("seek", f"frame={frame} → clamped={clamped} max={vp.max_frame_number} was_playing={vp.processing}")
        was_playing = vp.processing
        vp.stop_processing()
        vp.current_frame_number = clamped
        vp.next_frame_to_display = clamped
        if vp.media_capture:
            vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, clamped)
        if self._mw.markers.get(clamped):
            _log("seek", f"marker found at frame {clamped} — applying parameters")
            self._mw.parameters = copy.deepcopy(self._mw.markers[clamped]["parameters"])
            self._mw.control.update(self._mw.markers[clamped]["control"].copy())
            self.stateUpdated.emit(json.dumps({"section": "marker_applied", "frame": clamped}))
        if was_playing and vp.file_type == "video":
            _log("seek", "resuming playback after seek")
            vp.process_video()
        else:
            vp.process_current_frame()
        self._emit_playback()

    @Slot(int)
    def step(self, n: int) -> None:
        import cv2
        vp = self._mw.video_processor
        new_frame = max(0, min(vp.current_frame_number + n, vp.max_frame_number))
        _log("step", f"n={n} cur={vp.current_frame_number} → new={new_frame}")
        vp.stop_processing()
        vp.current_frame_number = new_frame
        vp.next_frame_to_display = new_frame
        if vp.media_capture:
            vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        vp.process_current_frame()
        self._emit_playback()

    @Slot(result=str)
    def getPlayback(self) -> str:
        vp = self._mw.video_processor
        return json.dumps({
            "is_playing":    vp.processing,
            "is_recording":  vp.recording,
            "current_frame": vp.current_frame_number,
            "max_frame":     vp.max_frame_number,
            "fps":           vp.fps,
            "file_type":     vp.file_type,
            "loop_enabled":  self._mw.control.get("loop_enabled", False),
            "swap_enabled":  self._mw.control.get("_swap_enabled", False),
            "edit_enabled":  self._mw.control.get("_edit_enabled", False),
        })

    # ── State slots ───────────────────────────────────────────────────────

    @Slot(result=str)
    def getState(self) -> str:
        mw = self._mw
        vp = mw.video_processor

        target_faces: dict = {}
        for fid, btn in mw.target_faces.items():
            target_faces[str(fid)] = {
                "face_id": str(fid),
                "assigned_input_face_ids": list(getattr(btn, "assigned_input_face_ids", [])),
                "assigned_embedding_ids":  list(getattr(btn, "assigned_embedding_ids", [])),
            }

        input_faces: dict = {}
        for fid, btn in mw.input_faces.items():
            input_faces[str(fid)] = {
                "face_id":    str(fid),
                "media_path": getattr(btn, "media_path", ""),
            }

        media_list = []
        for mid, btn in mw.target_videos.items():
            media_list.append({
                "media_id":   str(mid),
                "media_path": getattr(btn, "media_path", ""),
                "file_type":  getattr(btn, "file_type", "video"),
            })

        return json.dumps({
            "control":           mw.control,
            "target_faces":      target_faces,
            "input_faces":       input_faces,
            "target_media":      media_list,
            "selected_media_id": str(mw.selected_video_button.media_id)
                                  if mw.selected_video_button else None,
            "markers":           list(mw.markers.keys()),
            "last_target_media_folder_path": mw.last_target_media_folder_path,
            "last_input_media_folder_path":  getattr(mw, 'last_input_media_folder_path', '') or '',
            "playback": {
                "is_playing":    vp.processing,
                "is_recording":  vp.recording,
                "current_frame": vp.current_frame_number,
                "max_frame":     vp.max_frame_number,
                "fps":           vp.fps,
                "file_type":     vp.file_type,
            },
        })

    @Slot(str, str)
    def setControl(self, name: str, value_json: str) -> None:
        value = json.loads(value_json)
        self._mw.control[name] = value
        # Any control change that isn't a pure UI toggle should re-render the
        # current frame so the user sees the effect immediately.
        _no_refresh = {
            # These only affect the React UI state, not the rendered frame
            "_source_tab", "_preview_quality",
            "OutputMediaFolder",
        }
        if name not in _no_refresh:
            self._process_frame_async()

        # Virtual camera toggle — call enable/disable on the VideoProcessor
        # and emit the actual state back so the UI reflects reality.
        vp = self._mw.video_processor
        if name == "SendVirtCamFramesEnableToggle":
            if value:
                vp.enable_virtualcam()
            else:
                vp.disable_virtualcam()
            actual = vp.virtcam is not None
            self.virtcamStateChanged.emit(json.dumps({"enabled": actual}))
            if value and not actual:
                self.errorOccurred.emit(json.dumps({
                    "message": "Virtual camera failed to start. "
                               "Check that OBS Virtual Camera (or Unity Capture) is installed."
                }))
        elif name == "VirtCamBackendSelection":
            # Backend changed while cam is active — restart with new backend
            if self._mw.control.get("SendVirtCamFramesEnableToggle", False):
                vp.enable_virtualcam(backend=value)
                actual = vp.virtcam is not None
                self.virtcamStateChanged.emit(json.dumps({"enabled": actual}))

        self.stateUpdated.emit(json.dumps({"section": "control", "name": name, "value": value}))

    @Slot(str, str, str)
    def setParameter(self, face_id: str, name: str, value_json: str) -> None:
        from app.helpers.miscellaneous import ParametersDict
        value = json.loads(value_json)
        if face_id not in self._mw.parameters:
            self._mw.parameters[face_id] = ParametersDict(
                copy.deepcopy(self._mw.default_parameters),
                self._mw.default_parameters,
            )

        # ── Swapper model switch: unload the old model so the new one is
        # loaded fresh with the correct emap on the next frame. ──────────
        if name == 'SwapModelSelection':
            old_model = self._mw.parameters[face_id].get('SwapModelSelection', 'Inswapper128')
            _log("setParameter", f"SwapModelSelection: '{old_model}' → '{value}' (face {face_id})")
            if old_model != value:
                _unload_swapper_model(self._mw.models_processor, old_model)

        self._mw.parameters[face_id][name] = value
        # Re-render the current frame off the main thread so the UI stays responsive.
        self._process_frame_async()
        self.stateUpdated.emit(json.dumps({
            "section": "parameters", "face_id": face_id, "name": name, "value": value,
        }))

    # ── Media slots ───────────────────────────────────────────────────────

    @Slot(result=str)
    def pickFolder(self) -> str:
        """Open a native Qt folder picker rooted at the last media folder."""
        return self._pick_folder(self._mw.last_target_media_folder_path or "",
                                 "Select Media Folder")

    @Slot(str, result=str)
    def pickFolderAt(self, initial_dir: str) -> str:
        """Open a native Qt folder picker rooted at the supplied directory.
        Lets dialogs (e.g. the source face picker) start in their own
        last-used folder rather than the global media folder."""
        return self._pick_folder(initial_dir or "", "Select Folder")

    def _pick_folder(self, initial_dir: str, title: str) -> str:
        from PySide6.QtWidgets import QFileDialog
        _log("pickFolder", f"opening dialog title='{title}' initial='{initial_dir}'")
        folder = QFileDialog.getExistingDirectory(self._mw, title, initial_dir)
        _log("pickFolder", f"selected='{folder}'")
        return json.dumps(folder or "")

    @Slot(str, bool, result=str)
    def scanFolder(self, path: str, recursive: bool) -> str:
        """
        Scan a folder for media files and return the list as JSON.

        Walks the filesystem directly (no Qt worker thread) so the slot
        can return synchronously without blocking the main event loop.
        Thumbnails are generated on demand via getThumbnail().
        """
        import traceback
        from app.ui.widgets.actions.list_view_actions import clear_stop_loading_target_media
        from app.ui.widgets import widget_components
        from PySide6 import QtGui, QtCore

        trimmed = path.strip()
        _log("scanFolder", f"path='{trimmed}' recursive={recursive}")
        if not trimmed:
            _log("scanFolder", "empty path — returning early")
            return json.dumps({"items": [], "error": "Empty path"})

        folder = Path(trimmed)
        if not folder.is_dir():
            _log("scanFolder", f"not a directory: '{trimmed}'")
            return json.dumps({"items": [], "error": f"Not a directory: {trimmed}"})

        # Use the same extensions as app/helpers/miscellaneous.py
        IMAGE_EXT = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".tif", ".tiff",
                     ".jp2", ".exr", ".hdr", ".ras", ".pnm", ".ppm", ".pgm", ".pbm", ".pfm"}
        VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
                     ".m4v", ".3gp", ".gif", ".ts"}

        try:
            # Stop any active processing and clear existing media state
            _log("scanFolder", "stopping VP and clearing state")
            self._mw.video_processor.stop_processing()
            clear_stop_loading_target_media(self._mw)

            # Clear target faces inline (avoids dependency on targetFacesList stub)
            for _, face_btn in self._mw.target_faces.items():
                try:
                    face_btn.deleteLater()
                except Exception:
                    pass
            self._mw.target_faces = {}
            self._mw.parameters = {}
            self._mw.selected_target_face_id = False

            self._mw.selected_video_button = False
            self._mw.target_videos = {}
            self._mw.targetVideosList.clear()
            self._mw.last_target_media_folder_path = trimmed

            # Walk the folder
            pattern = "**/*" if recursive else "*"
            found_files: list[tuple[str, str]] = []  # (path, file_type)
            for p in sorted(folder.glob(pattern)):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in IMAGE_EXT:
                    found_files.append((str(p), "image"))
                elif ext in VIDEO_EXT:
                    found_files.append((str(p), "video"))

            _log("scanFolder", f"glob found {len(found_files)} media files")

            # Build card buttons, add to targetVideosList, and populate target_videos
            items = []
            for file_path, file_type in found_files:
                # Generate a stable integer ID from the path
                media_id = abs(hash(file_path)) % (10 ** 9)
                # Avoid collisions
                while media_id in self._mw.target_videos:
                    media_id += 1

                btn = widget_components.TargetMediaCardButton(
                    file_path, file_type, media_id,
                    main_window=self._mw,
                )

                # Add to the hidden QListWidget so btn.list_widget is set
                list_item = QtWidgets.QListWidgetItem(self._mw.targetVideosList)
                list_item.setSizeHint(btn.sizeHint())
                self._mw.targetVideosList.addItem(list_item)
                self._mw.targetVideosList.setItemWidget(list_item, btn)
                btn.list_item   = list_item
                btn.list_widget = self._mw.targetVideosList

                self._mw.target_videos[media_id] = btn

                items.append({
                    "media_id":   str(media_id),
                    "media_path": file_path,
                    "file_type":  file_type,
                })

            print(f"[bridge:scanFolder] Found {len(items)} files in '{trimmed}' (recursive={recursive})")
            return json.dumps({"items": items})

        except Exception as e:
            tb = traceback.format_exc()
            _log_err("scanFolder", str(e))
            print(tb, flush=True)
            return json.dumps({"error": str(e), "traceback": tb, "items": []})

    @Slot(str, result=str)
    def selectMedia(self, media_id: str) -> str:
        import cv2
        _log("selectMedia", f"media_id={media_id} target_videos keys={list(self._mw.target_videos.keys())[:10]}")
        try:
            btn = self._mw.target_videos.get(int(media_id))
            if btn is None:
                _log_err("selectMedia", f"media_id {media_id} not found in target_videos")
                return json.dumps({"ok": False, "error": f"Media id {media_id} not found in target_videos"})

            vp = self._mw.video_processor
            file_type  = getattr(btn, 'file_type', None)
            media_path = getattr(btn, 'media_path', '')
            _log("selectMedia", f"file_type={file_type} path='{media_path}'")

            vp.stop_processing()

            if file_type == 'video':
                _log("selectMedia", f"opening VideoCapture for '{media_path}'")
                cap = cv2.VideoCapture(media_path)
                if not cap.isOpened():
                    _log_err("selectMedia", f"cv2.VideoCapture failed to open '{media_path}'")
                    return json.dumps({"ok": False, "error": f"Cannot open video: {media_path}"})
                vp.media_capture = cap
                vp.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                vp.max_frame_number = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                vp.current_frame_number = 0
                vp.next_frame_to_display = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _log("selectMedia", f"video opened — fps={vp.fps:.1f} max_frame={vp.max_frame_number}")

            elif file_type == 'image':
                _log("selectMedia", "image selected — releasing any existing capture")
                if vp.media_capture:
                    vp.media_capture.release()
                vp.media_capture = None
                vp.fps = 0
                vp.max_frame_number = 0
                vp.current_frame_number = 0

            vp.media_path = media_path
            vp.file_type  = file_type
            self._mw.selected_video_button = btn
            self._mw.parameters = {}
            self._mw.selected_target_face_id = False

            if file_type == 'image':
                _log("selectMedia", "calling process_current_frame() for image")
                vp.process_current_frame()
            else:
                _log("selectMedia", "calling process_video()")
                vp.process_video()
                _log("selectMedia", f"process_video() returned — vp.processing={vp.processing}")

            # Auto-open native preview window so the user can see frames
            self._open_preview_window()

            QTimer.singleShot(200, self._emit_playback)
            result = {"ok": True, "max_frame": vp.max_frame_number, "fps": vp.fps, "file_type": file_type}
            _log("selectMedia", f"returning {result}")
            return json.dumps(result)
        except Exception as e:
            tb = _tb.format_exc()
            _log_err("selectMedia", str(e))
            print(tb, flush=True)
            return json.dumps({"ok": False, "error": str(e), "traceback": tb})

    @Slot(str, result=str)
    def deleteMedia(self, media_id: str) -> str:
        try:
            btn = self._mw.target_videos.pop(int(media_id), None)
            if btn:
                btn.deleteLater()
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    # ── Thumbnail slot (returns base64 JPEG) ──────────────────────────────

    @Slot(str, str, result=str)
    def getThumbnail(self, thumb_type: str, item_id: str) -> str:
        """Returns a JSON-encoded data URI string: '"data:image/jpeg;base64,..."'"""
        import base64
        import cv2

        def _safe_int(v):
            try: return int(v)
            except (TypeError, ValueError): return None

        try:
            if thumb_type == "media":
                btn = self._mw.target_videos.get(int(item_id))
                if btn is None:
                    return json.dumps("")
                path = getattr(btn, "media_path", "")
                file_type = getattr(btn, "file_type", "video")
                if file_type == "image":
                    frame = cv2.imread(path)
                else:
                    cap = cv2.VideoCapture(path)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        return json.dumps("")
            elif thumb_type == "face":
                # Buttons are keyed by face_id which is a long uuid string;
                # the main_ui legacy code uses ints. Try both so either keying
                # convention works.
                btn = self._mw.target_faces.get(item_id) or self._mw.target_faces.get(_safe_int(item_id))
                if btn is None:
                    return json.dumps("")
                frame = getattr(btn, "cropped_face", None)
                if frame is None:
                    return json.dumps("")
            elif thumb_type == "input":
                btn = self._mw.input_faces.get(item_id) or self._mw.input_faces.get(_safe_int(item_id))
                if btn is None:
                    return json.dumps("")
                frame = getattr(btn, "cropped_face", None)
                if frame is None:
                    path = getattr(btn, "media_path", "")
                    frame = cv2.imread(path)
                    if frame is None:
                        return json.dumps("")
            else:
                return json.dumps("")

            if frame is None:
                return json.dumps("")

            # Resize to thumbnail size
            h, w = frame.shape[:2]
            max_dim = 160
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf.tobytes()).decode()
            return json.dumps(f"data:image/jpeg;base64,{b64}")
        except Exception:
            return json.dumps("")

    # ── Webcam slots ──────────────────────────────────────────────────────

    @Slot(result=str)
    def getWebcams(self) -> str:
        """Enumerate available webcams by probing indices directly (no worker thread)."""
        import cv2
        _log("getWebcams", "enumerating webcams")
        try:
            max_cams = int(self._mw.control.get('WebcamMaxNoSelection', 5))
            from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
            backend_name = self._mw.control.get('WebcamBackendSelection', 'Default')
            backend = CAMERA_BACKENDS.get(backend_name, cv2.CAP_ANY)
            _log("getWebcams", f"probing up to {max_cams} cameras with backend='{backend_name}'")
            webcams = []
            for i in range(max_cams):
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    _log("getWebcams", f"  camera {i} — OPEN")
                    webcams.append({"index": i, "label": f"Camera {i}"})
                    cap.release()
                else:
                    _log("getWebcams", f"  camera {i} — not available")
            _log("getWebcams", f"found {len(webcams)} webcam(s)")
            return json.dumps({"webcams": webcams})
        except Exception as e:
            tb = _tb.format_exc()
            _log_err("getWebcams", str(e))
            print(tb, flush=True)
            return json.dumps({"webcams": [], "error": str(e), "traceback": tb})

    @Slot(int, result=str)
    def selectWebcam(self, index: int) -> str:
        """Select a webcam by index and start processing."""
        import cv2
        _log("selectWebcam", f"index={index}")
        try:
            from app.ui.widgets.settings_layout_data import CAMERA_BACKENDS
            backend_name = self._mw.control.get('WebcamBackendSelection', 'Default')
            backend = CAMERA_BACKENDS.get(backend_name, cv2.CAP_ANY)
            _log("selectWebcam", f"backend='{backend_name}'")
            vp = self._mw.video_processor
            vp.stop_processing()

            webcam_path = f"Webcam {index}"
            existing = next(
                (btn for btn in self._mw.target_videos.values()
                 if getattr(btn, 'is_webcam', False) and getattr(btn, 'webcam_index', -1) == index),
                None
            )
            if existing is None:
                _log("selectWebcam", f"creating new TargetMediaCardButton for '{webcam_path}'")
                from app.ui.widgets import widget_components
                media_id = abs(hash(webcam_path)) % (10 ** 9)
                while media_id in self._mw.target_videos:
                    media_id += 1
                btn = widget_components.TargetMediaCardButton(
                    webcam_path, 'webcam', media_id,
                    is_webcam=True, webcam_index=index, webcam_backend=backend,
                    main_window=self._mw,
                )
                list_item = QtWidgets.QListWidgetItem(self._mw.webcamList)
                list_item.setSizeHint(btn.sizeHint())
                self._mw.webcamList.addItem(list_item)
                self._mw.webcamList.setItemWidget(list_item, btn)
                btn.list_item   = list_item
                btn.list_widget = self._mw.webcamList
                self._mw.target_videos[media_id] = btn
                existing = btn
            else:
                _log("selectWebcam", f"reusing existing button for '{webcam_path}'")

            _log("selectWebcam", f"opening VideoCapture({index}, {backend_name})")
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                _log_err("selectWebcam", f"cannot open webcam {index}")
                return json.dumps({"ok": False, "error": f"Cannot open webcam {index}"})

            vp.media_capture = cap
            vp.media_path    = webcam_path
            vp.file_type     = 'webcam'
            vp.fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
            vp.max_frame_number   = 0
            vp.current_frame_number = 0
            self._mw.selected_video_button = existing
            _log("selectWebcam", f"capture opened — fps={vp.fps:.1f} — calling process_video()")

            vp.process_video()
            _log("selectWebcam", f"process_video() returned — vp.processing={vp.processing}")

            # Auto-open native preview window
            self._open_preview_window()

            QTimer.singleShot(200, self._emit_playback)
            return json.dumps({"ok": True})
        except Exception as e:
            tb = _tb.format_exc()
            _log_err("selectWebcam", str(e))
            print(tb, flush=True)
            return json.dumps({"ok": False, "error": str(e), "traceback": tb})

    # ── WebRTC slots ──────────────────────────────────────────────────────

    @Slot(result=str)
    def startWebrtc(self) -> str:
        """Start the WebRTC relay server subprocess and return connection URLs."""
        _log("startWebrtc", "called")
        try:
            from app.ui.widgets import ui_workers
            from functools import partial
            from app.ui.widgets.actions.list_view_actions import add_webrtc_thumbnail_to_streaming_list

            mw = self._mw
            vp = mw.video_processor

            vp.stop_processing()
            _log("startWebrtc", "VP stopped")

            if mw.webrtc_server_process and mw.webrtc_server_process.is_alive():
                _log("startWebrtc", "terminating existing WebRTC process")
                mw.webrtc_server_process.terminate()
                mw.webrtc_server_process.join(timeout=3)
                mw.webrtc_server_process = None

            if vp.webrtc_shm is not None:
                _log("startWebrtc", "closing old shared memory")
                try:
                    vp.webrtc_shm.close()
                except Exception:
                    pass
                vp.webrtc_shm = None

            http_port  = int(mw.control.get('WebRTCHttpPortText',  9091))
            https_port = int(mw.control.get('WebRTCHttpsPortText', 9090))
            bind_addr  = str(mw.control.get('WebRTCBindAddressText', '0.0.0.0'))
            _log("startWebrtc", f"ports http={http_port} https={https_port} bind={bind_addr}")

            worker = ui_workers.TargetMediaLoaderWorker(main_window=mw, webrtc_mode=True)
            worker.webrtc_thumbnail_ready.connect(
                partial(add_webrtc_thumbnail_to_streaming_list, mw)
            )
            worker.start()
            mw.video_loader_worker = worker
            _log("startWebrtc", "relay worker started")

            vp.media_path  = 'webrtc'
            vp.file_type   = 'webrtc'
            vp.fps         = 30.0
            vp.max_frame_number   = 0
            vp.current_frame_number = 0
            _log("startWebrtc", "calling process_video() for webrtc")
            vp.process_video()
            _log("startWebrtc", f"process_video() returned — vp.processing={vp.processing}")

            # Auto-open native preview window
            self._open_preview_window()

            import socket
            local_ip = socket.gethostbyname(socket.gethostname())
            http_url   = f"http://{local_ip}:{http_port}"
            https_url  = f"https://{local_ip}:{https_port}"
            whip_url   = f"http://{local_ip}:{http_port}/whip"
            whip_https = f"https://{local_ip}:{https_port}/whip"
            _log("startWebrtc", f"URLs — http={http_url} whip={whip_url}")

            QTimer.singleShot(200, self._emit_playback)
            return json.dumps({
                "http_url":       http_url,
                "https_url":      https_url,
                "whip_url":       whip_url,
                "whip_https_url": whip_https,
            })
        except Exception as e:
            tb = _tb.format_exc()
            _log_err("startWebrtc", str(e))
            print(tb, flush=True)
            return json.dumps({"error": str(e), "traceback": tb,
                               "http_url": "", "https_url": "",
                               "whip_url": "", "whip_https_url": ""})

    @Slot(result=str)
    def stopWebrtc(self) -> str:
        """Stop the WebRTC relay server and processing."""
        _log("stopWebrtc", "called")
        try:
            mw = self._mw
            vp = mw.video_processor
            vp.stop_processing()

            if mw.webrtc_server_process and mw.webrtc_server_process.is_alive():
                _log("stopWebrtc", "terminating relay process")
                mw.webrtc_server_process.terminate()
                mw.webrtc_server_process.join(timeout=3)
                mw.webrtc_server_process = None

            if vp.webrtc_shm is not None:
                _log("stopWebrtc", "closing shared memory")
                try:
                    vp.webrtc_shm.close()
                except Exception:
                    pass
                vp.webrtc_shm = None

            _log("stopWebrtc", "done")
            self._emit_playback()
            return json.dumps({"ok": True})
        except Exception as e:
            _log_err("stopWebrtc", str(e))
            return json.dumps({"ok": False, "error": str(e)})

    # ── System slots ──────────────────────────────────────────────────────

    @Slot(result=str)
    def getGpuMemory(self) -> str:
        """Return current GPU memory usage as JSON — used by the frontend for polling."""
        try:
            used, total = self._mw.models_processor.get_gpu_memory()
            return json.dumps({"used_mb": int(used), "total_mb": int(total)})
        except Exception as e:
            _log_err("getGpuMemory", str(e), e)
            return json.dumps({"used_mb": 0, "total_mb": 0})

    @Slot(str, result=str)
    def setProvider(self, provider: str) -> str:
        _log("setProvider", f"provider='{provider}'")
        try:
            self._mw.control["ProvidersPrioritySelection"] = provider
            return json.dumps({"ok": True})
        except Exception as e:
            _log_err("setProvider", str(e))
            return json.dumps({"ok": False, "error": str(e)})

    @Slot(int, bool, bool, result=str)
    def setTransform(self, rotation: int, flip_h: bool, flip_v: bool) -> str:
        """Apply rotation/flip to the active source and re-render the current frame."""
        _log("setTransform", f"rotation={rotation} flip_h={flip_h} flip_v={flip_v}")
        try:
            mw = self._mw
            vp = mw.video_processor
            if vp.file_type == "webcam":
                mw.webcam_rotation = rotation
                mw.webcam_flip_h   = flip_h
                mw.webcam_flip_v   = flip_v
            elif vp.file_type == "webrtc":
                mw.webrtc_rotation = rotation
                mw.webrtc_flip_h   = flip_h
                mw.webrtc_flip_v   = flip_v
            else:
                mw.media_rotation = rotation
                mw.media_flip_h   = flip_h
                mw.media_flip_v   = flip_v
            self._process_frame_async()
            return json.dumps({"ok": True})
        except Exception as e:
            _log_err("setTransform", str(e))
            return json.dumps({"ok": False, "error": str(e)})

    @Slot(result=str)
    def clearMemory(self) -> str:
        _log("clearMemory", "called")
        from app.ui.widgets.actions import common_actions
        try:
            common_actions.clear_gpu_memory(self._mw)
            _log("clearMemory", "done")
            return json.dumps({"ok": True})
        except Exception as e:
            _log_err("clearMemory", str(e))
            return json.dumps({"ok": False, "error": str(e)})

    @Slot(result=str)
    def getLoadedModels(self) -> str:
        """Return all currently loaded models as JSON — mirrors GET /api/models."""
        _log("getLoadedModels", "called")
        try:
            models = self._mw.models_processor.get_loaded_models()
            return json.dumps({"models": models})
        except Exception as e:
            _log_err("getLoadedModels", str(e))
            return json.dumps({"models": [], "error": str(e)})

    @Slot(str, result=str)
    def unloadModel(self, model_name: str) -> str:
        """Unload a single model by name — mirrors DELETE /api/models/{name}."""
        _log("unloadModel", f"model_name='{model_name}'")
        try:
            unloaded = self._mw.models_processor.unload_model(model_name)
            if not unloaded:
                return json.dumps({"ok": False, "error": f"Model '{model_name}' is not currently loaded."})
            return json.dumps({"ok": True, "message": f"Model '{model_name}' unloaded."})
        except Exception as e:
            _log_err("unloadModel", str(e))
            return json.dumps({"ok": False, "error": str(e)})

    # ── Workspace slots ───────────────────────────────────────────────────

    @Slot(result=str)
    def togglePreviewWindow(self) -> str:
        """Open or close the native preview window."""
        mw = self._mw
        try:
            if getattr(mw, "_preview_window", None) is not None:
                try:
                    if mw._preview_window.isVisible():
                        _log("togglePreviewWindow", "closing")
                        mw._preview_window.close()
                        mw._preview_window = None
                        return json.dumps({"ok": True, "open": False})
                except RuntimeError:
                    mw._preview_window = None
            self._open_preview_window()
            return json.dumps({"ok": True, "open": True})
        except Exception as e:
            _log_err("togglePreviewWindow", str(e))
            return json.dumps({"ok": False, "error": str(e)})

    @Slot(result=str)
    def saveWorkspace(self) -> str:
        from app.ui.widgets.actions import save_load_actions
        try:
            save_load_actions.save_current_workspace(self._mw, "last_workspace.json")
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    # ── Recording / save-frame slots ──────────────────────────────────────

    @Slot(str, result=str)
    def recordStart(self, output_folder: str) -> str:
        """Start recording the processed video to disk."""
        _log("recordStart", f"output_folder='{output_folder}'")
        try:
            from app.helpers.miscellaneous import is_ffmpeg_in_path, get_ffmpeg_path
            from pathlib import Path as _Path

            vp = self._mw.video_processor

            if vp.file_type != "video":
                msg = (
                    "No video loaded. Please select a video file before recording."
                    if vp.file_type is None
                    else f"Recording is only supported for video files (current source: {vp.file_type})."
                )
                return json.dumps({"ok": False, "error": msg})

            if not is_ffmpeg_in_path():
                ffmpeg_path = get_ffmpeg_path()
                return json.dumps({"ok": False, "error":
                    f"FFmpeg not found. Expected at '{ffmpeg_path}' or on system PATH. "
                    "Install FFmpeg or launch via Start.bat so the bundled copy is used."})

            folder = output_folder or self._mw.control.get("OutputMediaFolder", "")
            if not folder:
                return json.dumps({"ok": False, "error": "No output folder configured"})
            if not _Path(folder).is_dir():
                return json.dumps({"ok": False, "error": f"Output folder does not exist: {folder}"})

            # If playback is already running, stop it first (mirrors Qt UI behaviour)
            if vp.processing:
                _log("recordStart", "stopping active playback before starting recording")
                vp.stop_processing()

            self._mw.control["OutputMediaFolder"] = folder
            vp.recording = True
            vp.process_video()
            QTimer.singleShot(200, self._emit_playback)
            return json.dumps({"ok": True})
        except Exception as e:
            _log_err("recordStart", str(e), e)
            return json.dumps({"ok": False, "error": str(e), "traceback": _tb.format_exc()})

    @Slot(result=str)
    def recordStop(self) -> str:
        """Stop recording and return the output file path."""
        _log("recordStop", "called")
        try:
            from app.helpers.miscellaneous import get_output_file_path

            vp = self._mw.video_processor
            if not vp.recording:
                return json.dumps({"ok": False, "error": "Not currently recording"})

            output_path = get_output_file_path(
                vp.media_path, self._mw.control.get("OutputMediaFolder", ".")
            )
            vp.stop_processing()
            QTimer.singleShot(200, self._emit_playback)
            return json.dumps({"ok": True, "output_path": output_path})
        except Exception as e:
            _log_err("recordStop", str(e), e)
            return json.dumps({"ok": False, "error": str(e), "traceback": _tb.format_exc()})

    @Slot(result=str)
    def saveFrame(self) -> str:
        """Save the current processed frame as an image to the output folder."""
        _log("saveFrame", "called")
        try:
            import cv2 as _cv2
            import numpy as _np
            from app.helpers.miscellaneous import get_output_file_path
            from pathlib import Path as _Path

            vp = self._mw.video_processor
            output_folder = self._mw.control.get("OutputMediaFolder", "")
            if not output_folder:
                return json.dumps({"ok": False, "error": "No output folder configured"})

            frame = vp.current_frame
            if not isinstance(frame, _np.ndarray) or frame.size == 0:
                return json.dumps({"ok": False, "error": "No frame available"})

            output_path = get_output_file_path(
                vp.media_path or "snapshot.png", output_folder, media_type="image"
            )
            _cv2.imwrite(output_path, frame)
            _log("saveFrame", f"saved to '{output_path}'")
            return json.dumps({"ok": True, "message": f"Frame saved to {output_path}", "output_path": output_path})
        except Exception as e:
            _log_err("saveFrame", str(e), e)
            return json.dumps({"ok": False, "error": str(e), "traceback": _tb.format_exc()})

    # ── File system helpers ───────────────────────────────────────────────

    @Slot(str, result=str)
    def openFile(self, path: str) -> str:
        """Open a file with the default OS application."""
        _log("openFile", f"path='{path}'")
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            ok = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            return json.dumps({"ok": ok})
        except Exception as e:
            _log_err("openFile", str(e), e)
            return json.dumps({"ok": False, "error": str(e)})

    @Slot(str, result=str)
    def revealInFolder(self, path: str) -> str:
        """Reveal a file in the OS file explorer, selecting it."""
        _log("revealInFolder", f"path='{path}'")
        try:
            import sys as _sys
            import subprocess as _sp
            from pathlib import Path as _Path
            p = _Path(path)
            if _sys.platform == "win32":
                # /select highlights the file in Explorer
                _sp.Popen(["explorer", "/select,", str(p)])
            elif _sys.platform == "darwin":
                _sp.Popen(["open", "-R", str(p)])
            else:
                # Linux: open the parent folder
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))
            return json.dumps({"ok": True})
        except Exception as e:
            _log_err("revealInFolder", str(e), e)
            return json.dumps({"ok": False, "error": str(e)})

    # ── Face slots ────────────────────────────────────────────────────────

    @Slot(result=str)
    def findFaces(self) -> str:
        """Run face detection on the current frame and return the resulting
        target faces as JSON. Mirrors the FastAPI POST /api/target-faces/find
        endpoint so the React frontend transport gets identical behaviour
        whether it's running over QWebChannel or HTTP."""
        _log("findFaces", "called")
        try:
            from app.ui.widgets.actions import card_actions

            mw = self._mw

            # Run the same detection routine the legacy Qt UI uses. It mutates
            # mw.target_faces in place by appending TargetFaceCardButton items
            # via list_view_actions.add_media_thumbnail_to_target_faces_list.
            card_actions.find_target_faces(mw)

            # find_target_faces clears mw.target_faces internally then
            # repopulates it. Whatever ends up in `after` is the fresh
            # detection result; report that count as "found".
            after = list(mw.target_faces.keys())
            faces = []
            for fid in after:
                faces.append({
                    "face_id":      str(fid),
                    "thumbnail_url": "",  # Qt mode uses getThumbnail() instead
                    "assigned_input_face_ids": [],
                    "assigned_embedding_ids":  [],
                })

            payload = {"found": len(after), "faces": faces}
            self.facesFound.emit(json.dumps(payload))
            return json.dumps(payload)
        except Exception as e:
            _log_err("findFaces", str(e), e)
            return json.dumps({"error": str(e), "traceback": _tb.format_exc(),
                               "found": 0, "faces": []})

    @Slot()
    def clearFaces(self) -> None:
        """Remove all target faces from the legacy Qt list and from state."""
        _log("clearFaces", "called")
        try:
            from app.ui.widgets.actions import card_actions
            card_actions.clear_target_faces(self._mw)
        except Exception as e:
            _log_err("clearFaces", str(e), e)

    @Slot(str)
    def selectFace(self, face_id: str) -> None:
        """Select a target face by id."""
        _log("selectFace", f"face_id={face_id}")
        try:
            mw = self._mw
            btn = mw.target_faces.get(face_id)
            if btn is None:
                try:
                    btn = mw.target_faces.get(int(face_id))
                except (TypeError, ValueError):
                    btn = None
            if btn is not None:
                btn.click()
        except Exception as e:
            _log_err("selectFace", str(e), e)

    @Slot(str, bool, result=str)
    def scanInputFolder(self, path: str, recursive: bool) -> str:
        """Scan a folder for source/input face images and load them into
        main_window.input_faces.

        Detection runs in a QThread so the main-thread event loop stays alive
        (UI remains responsive).  A local QEventLoop spins here until the
        worker finishes, then we return the JSON result synchronously to the
        JS caller (required by QWebChannel's callback protocol).

        Qt widget construction (buttons, list items) is marshalled back to the
        main thread via a direct-connected signal so it happens safely while
        the event loop is running.
        """
        _log("scanInputFolder", f"path='{path}' recursive={recursive}")
        try:
            import torch
            import numpy
            from app.helpers.miscellaneous import get_image_files, read_image_file
            from app.ui.widgets.actions import card_actions as _ca
            from app.ui.widgets.actions import common_actions as common_widget_actions
            from app.ui.widgets import widget_components
            from PySide6 import QtWidgets, QtGui, QtCore

            mw = self._mw
            control = mw.control.copy()

            # ── Clear previous input faces ────────────────────────────────
            try:
                _ca.clear_input_faces(mw)
            except Exception as exc:
                _log_err("scanInputFolder", f"clear_input_faces failed: {exc}")
                mw.input_faces = {}

            mw.last_input_media_folder_path = path
            mw.last_input_media_folder = path

            # ── Discover image files ──────────────────────────────────────
            image_files = get_image_files(path, recursive)
            image_files.sort()
            _log("scanInputFolder", f"found {len(image_files)} image file(s) in '{path}' (recursive={recursive})")

            if not image_files:
                return json.dumps({"items": [], "new_face_ids": []})

            # ── Collect model settings ────────────────────────────────────
            mp               = mw.models_processor
            detector_name    = control.get('DetectorModelSelection', 'RetinaFace')
            recognition_name = control.get('RecognitionModelSelection', 'Inswapper128ArcFace')
            similarity_type  = control.get('SimilarityTypeSelection', 'Opal')
            det_score        = float(control.get('DetectorScoreSlider', 50)) / 100.0
            use_lmk          = bool(control.get('LandmarkDetectToggle', False))
            lmk_mode         = control.get('LandmarkDetectModelSelection', '203')
            lmk_score        = float(control.get('LandmarkDetectScoreSlider', 50)) / 100.0
            from_pts         = bool(control.get('DetectFromPointsToggle', False))

            # ── Worker: runs detection on a background thread ─────────────
            # Results are collected into a list; Qt widget construction is
            # done on the main thread via _add_face_to_ui() below.
            _results: list[dict] = []   # filled by worker, read after join

            class _ScanWorker(QtCore.QThread):
                # Emitted for each detected face so the main thread can build
                # the Qt button while the event loop is still running.
                face_ready = QtCore.Signal(str, object, object, str)  # path, cropped_bgr, emb_store, face_id

                def run(self_w):
                    found = skipped = 0
                    for image_path in image_files:
                        frame = read_image_file(image_path)
                        if frame is None:
                            skipped += 1
                            continue

                        frame_rgb = frame[..., ::-1]
                        img = torch.from_numpy(frame_rgb.astype('uint8')).to(mp.device).permute(2, 0, 1)

                        try:
                            _, kpss_5, _ = mp.run_detect(
                                img, detector_name,
                                max_num=1, score=det_score, input_size=(512, 512),
                                use_landmark_detection=use_lmk,
                                landmark_detect_mode=lmk_mode,
                                landmark_score=lmk_score,
                                from_points=from_pts,
                                rotation_angles=[0],
                            )
                        except Exception:
                            skipped += 1
                            continue

                        if not len(kpss_5):
                            skipped += 1
                            continue

                        try:
                            face_emb, cropped_img = mp.run_recognize_direct(
                                img, kpss_5[0], similarity_type, recognition_name
                            )
                        except Exception:
                            skipped += 1
                            continue

                        cropped_bgr = numpy.ascontiguousarray(cropped_img.cpu().numpy()[..., ::-1])
                        face_id     = str(uuid.uuid1().int)
                        emb_store   = {recognition_name: face_emb}

                        self_w.face_ready.emit(image_path, cropped_bgr, emb_store, face_id)
                        found += 1

                    _log("scanInputFolder", f"worker done — {found} found, {skipped} skipped")

            # ── Main-thread slot: build Qt button for each detected face ──
            def _add_face_to_ui(image_path: str, cropped_bgr, emb_store, face_id: str):
                pixmap = common_widget_actions.get_pixmap_from_frame(mw, cropped_bgr)
                if not pixmap:
                    return
                button_size = QtCore.QSize(70, 70)
                btn = widget_components.InputFaceCardButton(
                    image_path, cropped_bgr, emb_store, face_id, main_window=mw,
                )
                btn.setIcon(QtGui.QIcon(pixmap))
                btn.setIconSize(button_size - QtCore.QSize(8, 8))
                btn.setFixedSize(button_size)
                btn.setCheckable(True)

                list_item = QtWidgets.QListWidgetItem(mw.inputFacesList)
                list_item.setSizeHint(button_size)
                btn.list_item   = list_item
                btn.list_widget = mw.inputFacesList
                list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                mw.inputFacesList.setItemWidget(list_item, btn)
                grid_size = button_size + QtCore.QSize(4, 4)
                mw.inputFacesList.setGridSize(grid_size)
                mw.inputFacesList.setWrapping(True)
                mw.inputFacesList.setFlow(QtWidgets.QListView.LeftToRight)
                mw.inputFacesList.setResizeMode(QtWidgets.QListView.Adjust)

                mw.input_faces[face_id] = btn
                _results.append({"face_id": face_id, "media_path": image_path, "thumbnail_url": ""})

            # ── Spin a local event loop until the worker finishes ─────────
            loop   = QtCore.QEventLoop()
            worker = _ScanWorker()
            # face_ready uses a direct connection so _add_face_to_ui runs on
            # the main thread immediately as each signal is delivered through
            # the event loop.
            worker.face_ready.connect(_add_face_to_ui, QtCore.Qt.ConnectionType.QueuedConnection)
            worker.finished.connect(loop.quit)
            worker.start()
            loop.exec()   # blocks this slot but keeps the event loop alive

            _log("scanInputFolder", f"scan complete — {len(_results)} face(s) registered")
            return json.dumps({"items": _results, "new_face_ids": [x["face_id"] for x in _results]})

        except Exception as e:
            _log_err("scanInputFolder", str(e), e)
            return json.dumps({"error": str(e), "traceback": _tb.format_exc(), "items": []})

    @Slot(str, str)
    def assignInput(self, target_face_id: str, input_face_id: str) -> None:
        """Assign an input face to a target face for swapping."""
        _log("assignInput", f"target={target_face_id} input={input_face_id}")
        try:
            mw = self._mw

            def _lookup(d, key):
                hit = d.get(key)
                if hit is not None:
                    return hit
                try:
                    return d.get(int(key))
                except (TypeError, ValueError):
                    return None

            target = _lookup(mw.target_faces, target_face_id)
            input_btn = _lookup(mw.input_faces, input_face_id)
            if target is None or input_btn is None:
                return
            target.assigned_input_faces[input_btn.face_id] = input_btn.embedding_store
            target.calculate_assigned_input_embedding()
            from app.ui.widgets.actions import common_actions as _ca
            _ca.refresh_frame(mw)
        except Exception as e:
            _log_err("assignInput", str(e), e)

    @Slot(str, str)
    def unassignInput(self, target_face_id: str, input_face_id: str) -> None:
        """Remove an input face from a target face's swap source set."""
        _log("unassignInput", f"target={target_face_id} input={input_face_id}")
        try:
            mw = self._mw
            target = mw.target_faces.get(target_face_id)
            if target is None:
                try:
                    target = mw.target_faces.get(int(target_face_id))
                except (TypeError, ValueError):
                    target = None
            if target is None:
                return
            target.remove_assigned_input_face(input_face_id)
            from app.ui.widgets.actions import common_actions as _ca
            _ca.refresh_frame(mw)
        except Exception as e:
            _log_err("unassignInput", str(e), e)

    @Slot(result=str)
    def loadLastWorkspace(self) -> str:
        """Load last_workspace.json and notify the frontend to re-pull state."""
        _log("loadLastWorkspace", "called")
        try:
            from app.ui.widgets.actions import save_load_actions as _sl
            _sl.load_saved_workspace(self._mw, "last_workspace.json")
            self._emit_workspace_loaded()
            return json.dumps({"ok": True})
        except Exception as e:
            _log_err("loadLastWorkspace", str(e), e)
            return json.dumps({"ok": False, "error": str(e), "traceback": _tb.format_exc()})

    def _emit_workspace_loaded(self) -> None:
        """Emit workspaceLoaded with the post-load paths so the React UI can
        update its folder inputs and trigger a fresh getState() call."""
        mw = self._mw
        try:
            payload = {
                "last_target_media_folder_path": getattr(mw, 'last_target_media_folder_path', '') or '',
                "last_input_media_folder_path":  getattr(mw, 'last_input_media_folder_path', '') or '',
            }
            self.workspaceLoaded.emit(json.dumps(payload))
            # Also emit playback state so the loaded media id / file_type
            # propagates to the React UI immediately.
            self._emit_playback()
        except Exception as exc:
            print(f"[bridge:_emit_workspace_loaded] {exc}", flush=True)
