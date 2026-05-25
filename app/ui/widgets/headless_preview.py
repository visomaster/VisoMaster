"""
app/ui/widgets/headless_preview.py
───────────────────────────────────
Singleton that manages a native Qt preview window when the server is running
in headless (API-only) mode — i.e. no QApplication exists yet.

Thread safety
-------------
Qt widgets must only be touched on the thread that owns the QApplication.
Frames arrive from FrameWorker threads.  We bridge the gap with a
QObject-based relay that lives on the Qt thread and exposes Qt Signals —
cross-thread signal delivery is handled safely by Qt's queued connection
mechanism, which is the only correct way to push data to a widget from
another thread.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from PySide6 import QtCore


# ── Cross-thread relay ────────────────────────────────────────────────────────

class _FrameRelay(QtCore.QObject):
    """
    Lives on the Qt thread.  Worker threads call the thread-safe emit_*
    methods; Qt delivers the signals to the connected slots on the Qt thread.
    """
    frame_ready   = QtCore.Signal(object)   # np.ndarray (BGR)
    state_changed = QtCore.Signal(int, int, bool, bool, object)  # cf, mf, playing, loop, markers
    show_requested  = QtCore.Signal()
    close_requested = QtCore.Signal()

    def emit_frame(self, frame_bgr: np.ndarray) -> None:
        """Thread-safe: called from any thread."""
        self.frame_ready.emit(frame_bgr)

    def emit_state(self, cf: int, mf: int, playing: bool,
                   loop: bool, markers: set) -> None:
        self.state_changed.emit(cf, mf, playing, loop, markers)

    def emit_show(self) -> None:
        self.show_requested.emit()

    def emit_close(self) -> None:
        self.close_requested.emit()


# ── Manager singleton ─────────────────────────────────────────────────────────

class _HeadlessPreviewManager:
    """Module-level singleton — use the `headless_preview` instance below."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._app    = None          # QApplication
        self._window = None          # PreviewWindow
        self._relay: Optional[_FrameRelay] = None
        self._ready  = threading.Event()

    # ── Public API (safe to call from any thread) ─────────────────────────

    @property
    def is_open(self) -> bool:
        return self._window is not None and self._window.isVisible()

    def open(self) -> None:
        """Open the preview window, starting the Qt thread if needed."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                # Qt thread already running — just show the window
                if self._relay is not None:
                    self._relay.emit_show()
                return
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._qt_main,
                name="headless-preview-qt",
                daemon=True,
            )
            self._thread.start()
        # Wait outside the lock so the Qt thread can acquire it if needed
        self._ready.wait(timeout=5.0)

    def close(self) -> None:
        """Close the preview window (Qt thread stays alive for re-open)."""
        if self._relay is not None:
            self._relay.emit_close()

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Deliver a BGR frame to the preview window.
        Safe to call from any thread at high frequency.
        """
        if self._relay is None or self._window is None:
            return
        # Copy so the numpy buffer stays valid after the caller moves on
        self._relay.emit_frame(frame_bgr.copy())

    def sync_state(
        self,
        current_frame: int,
        max_frame: int,
        is_playing: bool,
        loop_enabled: bool = False,
        markers: Optional[set] = None,
    ) -> None:
        """Sync playback controls. Safe to call from any thread."""
        if self._relay is None or self._window is None:
            return
        self._relay.emit_state(
            current_frame, max_frame, is_playing, loop_enabled,
            set(markers) if markers else set(),
        )

    # ── Qt thread entry point ─────────────────────────────────────────────

    def _qt_main(self) -> None:
        import sys
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv[:1])
        self._app = app

        # Create relay on the Qt thread so its signals are delivered here
        relay = _FrameRelay()
        self._relay = relay

        # Create the window
        self._create_window()

        # Wire relay signals → window slots (all on Qt thread)
        relay.frame_ready.connect(self._on_frame_ready,
                                  QtCore.Qt.ConnectionType.QueuedConnection)
        relay.state_changed.connect(self._on_state_changed,
                                    QtCore.Qt.ConnectionType.QueuedConnection)
        relay.show_requested.connect(self._on_show,
                                     QtCore.Qt.ConnectionType.QueuedConnection)
        relay.close_requested.connect(self._on_close,
                                      QtCore.Qt.ConnectionType.QueuedConnection)

        self._ready.set()
        app.exec()

    # ── Qt-thread slots ───────────────────────────────────────────────────

    @QtCore.Slot(object)
    def _on_frame_ready(self, frame_bgr: np.ndarray) -> None:
        if self._window is not None and self._window.isVisible():
            self._window.update_frame(frame_bgr)

    @QtCore.Slot(int, int, bool, bool, object)
    def _on_state_changed(self, cf: int, mf: int,
                          playing: bool, loop: bool, markers: set) -> None:
        if self._window is not None and self._window.isVisible():
            self._window.sync_playback_state(cf, mf, playing, loop)
            self._window.set_markers(markers)

    @QtCore.Slot()
    def _on_show(self) -> None:
        if self._window is None:
            self._create_window()
        else:
            self._window.show()
            self._window.raise_()
            self._window.activateWindow()

    @QtCore.Slot()
    def _on_close(self) -> None:
        if self._window is not None:
            self._window.close()
            self._window = None

    def _create_window(self) -> None:
        from app.ui.widgets.preview_window import PreviewWindow

        mgr = self  # capture for closure

        class _StandaloneWindow(PreviewWindow):
            def __init__(self_w):
                super().__init__(_HeadlessProxy())

            def closeEvent(self_w, event):
                mgr._window = None
                from app.api.events import bus
                bus.emit_sync("preview_window_closed", {})
                event.accept()

        self._window = _StandaloneWindow()
        self._window.show()


# ── Headless proxy ────────────────────────────────────────────────────────────

class _HeadlessProxy:
    """
    Minimal MainWindow stand-in for PreviewWindow button handlers.
    Reads the live headless VideoProcessor from FastAPI app state.
    """

    def __init__(self) -> None:
        self.control: dict = {}
        self.markers: dict = {}
        self._preview_window = None

    @property
    def video_processor(self):
        try:
            from app.api.server import app as _fa
            return _fa.state.video_processor
        except Exception:
            return _NullVP()


class _NullVP:
    processing = False
    file_type = None
    current_frame_number = 0
    max_frame_number = 0
    media_capture = None

    def process_video(self): pass
    def stop_processing(self): return False
    def process_current_frame(self): pass


# Module-level singleton
headless_preview = _HeadlessPreviewManager()
