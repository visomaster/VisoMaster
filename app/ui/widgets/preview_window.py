"""
Native Qt preview window with seeker and playback controls.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# ── Palette ───────────────────────────────────────────────────────────────────
_BG       = "#141414"
_SURFACE  = "#1e1e1e"
_BORDER   = "#2a2a2a"
_CTRL_BG  = "#181818"
_TEXT     = "#e2e2e2"
_MUTED    = "#666"
_ACCENT   = "#3b82f6"       # blue-500
_ACCENT_H = "#2563eb"       # blue-600
_ACTIVE   = "#1d4ed8"       # blue-700
_AMBER    = "#f59e0b"
_RADIUS   = "6px"

_QSS = f"""
QWidget#ctrl_bar {{
    background: {_CTRL_BG};
    border-top: 1px solid {_BORDER};
}}
QLabel {{
    background: transparent;
    color: {_TEXT};
}}
/* ── Seek slider ── */
QSlider::groove:horizontal {{
    height: 4px;
    background: #2e2e2e;
    border-radius: 2px;
    margin: 0 2px;
}}
QSlider::sub-page:horizontal {{
    background: {_ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px;
    margin: -5px 0;
    background: #ffffff;
    border-radius: 7px;
    border: 2px solid {_ACCENT};
}}
QSlider::handle:horizontal:hover {{
    background: {_ACCENT};
    border-color: #93c5fd;
}}
/* ── Icon buttons ── */
QPushButton.icon_btn {{
    background: transparent;
    border: none;
    border-radius: {_RADIUS};
    padding: 0;
}}
QPushButton.icon_btn:hover  {{ background: #2a2a2a; }}
QPushButton.icon_btn:pressed {{ background: #1a1a1a; }}
QPushButton.icon_btn:checked {{ background: {_ACTIVE}; }}
QPushButton.icon_btn:disabled {{ opacity: 0.35; }}
/* ── Play button (larger, pill) ── */
QPushButton#play_btn {{
    background: {_ACCENT};
    border: none;
    border-radius: 18px;
    padding: 0;
}}
QPushButton#play_btn:hover  {{ background: {_ACCENT_H}; }}
QPushButton#play_btn:pressed {{ background: {_ACTIVE}; }}
QPushButton#play_btn:checked {{ background: {_ACTIVE}; }}
"""


# ── Icon painter helpers ──────────────────────────────────────────────────────

def _make_icon(draw_fn, size=20, color="#e2e2e2") -> QtGui.QIcon:
    """Create a QIcon by calling draw_fn(painter, rect, color)."""
    px = QtGui.QPixmap(size, size)
    px.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(px)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    draw_fn(p, QtCore.QRectF(0, 0, size, size), color)
    p.end()
    return QtGui.QIcon(px)


def _icon_skip_back(p: QtGui.QPainter, r: QtCore.QRectF, c: str):
    col = QtGui.QColor(c)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(col)
    cx, cy = r.center().x(), r.center().y()
    w = r.width()
    # Left bar
    bar_w = w * 0.12
    p.drawRect(QtCore.QRectF(cx - w * 0.38, cy - w * 0.28, bar_w, w * 0.56))
    # Triangle pointing left
    tri = QtGui.QPolygonF([
        QtCore.QPointF(cx + w * 0.28, cy - w * 0.28),
        QtCore.QPointF(cx + w * 0.28, cy + w * 0.28),
        QtCore.QPointF(cx - w * 0.18, cy),
    ])
    p.drawPolygon(tri)


def _icon_skip_fwd(p: QtGui.QPainter, r: QtCore.QRectF, c: str):
    col = QtGui.QColor(c)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(col)
    cx, cy = r.center().x(), r.center().y()
    w = r.width()
    bar_w = w * 0.12
    p.drawRect(QtCore.QRectF(cx + w * 0.26, cy - w * 0.28, bar_w, w * 0.56))
    tri = QtGui.QPolygonF([
        QtCore.QPointF(cx - w * 0.28, cy - w * 0.28),
        QtCore.QPointF(cx - w * 0.28, cy + w * 0.28),
        QtCore.QPointF(cx + w * 0.18, cy),
    ])
    p.drawPolygon(tri)


def _icon_step_back(p: QtGui.QPainter, r: QtCore.QRectF, c: str):
    col = QtGui.QColor(c)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(col)
    cx, cy = r.center().x(), r.center().y()
    w = r.width()
    # Two triangles pointing left
    for dx in (0.14, -0.10):
        tri = QtGui.QPolygonF([
            QtCore.QPointF(cx + dx * w + w * 0.18, cy - w * 0.24),
            QtCore.QPointF(cx + dx * w + w * 0.18, cy + w * 0.24),
            QtCore.QPointF(cx + dx * w - w * 0.10, cy),
        ])
        p.drawPolygon(tri)


def _icon_step_fwd(p: QtGui.QPainter, r: QtCore.QRectF, c: str):
    col = QtGui.QColor(c)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(col)
    cx, cy = r.center().x(), r.center().y()
    w = r.width()
    for dx in (-0.14, 0.10):
        tri = QtGui.QPolygonF([
            QtCore.QPointF(cx + dx * w - w * 0.18, cy - w * 0.24),
            QtCore.QPointF(cx + dx * w - w * 0.18, cy + w * 0.24),
            QtCore.QPointF(cx + dx * w + w * 0.10, cy),
        ])
        p.drawPolygon(tri)


def _icon_play(p: QtGui.QPainter, r: QtCore.QRectF, c: str):
    col = QtGui.QColor(c)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(col)
    cx, cy = r.center().x(), r.center().y()
    w = r.width()
    tri = QtGui.QPolygonF([
        QtCore.QPointF(cx - w * 0.22, cy - w * 0.30),
        QtCore.QPointF(cx - w * 0.22, cy + w * 0.30),
        QtCore.QPointF(cx + w * 0.28, cy),
    ])
    p.drawPolygon(tri)


def _icon_stop(p: QtGui.QPainter, r: QtCore.QRectF, c: str):
    col = QtGui.QColor(c)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(col)
    cx, cy = r.center().x(), r.center().y()
    w = r.width()
    s = w * 0.30
    p.drawRoundedRect(QtCore.QRectF(cx - s, cy - s, s * 2, s * 2), 3, 3)


def _icon_loop(p: QtGui.QPainter, r: QtCore.QRectF, c: str):
    col = QtGui.QColor(c)
    pen = QtGui.QPen(col, r.width() * 0.10)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    cx, cy = r.center().x(), r.center().y()
    w = r.width()
    # Arc
    arc_rect = QtCore.QRectF(cx - w * 0.30, cy - w * 0.28, w * 0.60, w * 0.56)
    p.drawArc(arc_rect, 30 * 16, 300 * 16)
    # Arrow head at end of arc
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(col)
    tip = QtCore.QPointF(cx + w * 0.30, cy - w * 0.05)
    arr = QtGui.QPolygonF([
        QtCore.QPointF(tip.x() - w * 0.12, tip.y() - w * 0.12),
        QtCore.QPointF(tip.x() + w * 0.06, tip.y()),
        QtCore.QPointF(tip.x() - w * 0.12, tip.y() + w * 0.12),
    ])
    p.drawPolygon(arr)


def _make_icon_btn(draw_fn, size=28, icon_size=16,
                   checkable=False, tooltip="",
                   obj_name="icon_btn") -> QtWidgets.QPushButton:
    btn = QtWidgets.QPushButton()
    btn.setObjectName(obj_name)
    btn.setProperty("class", "icon_btn")
    btn.setFixedSize(size, size)
    btn.setCheckable(checkable)
    btn.setToolTip(tooltip)
    btn.setIcon(_make_icon(draw_fn, icon_size))
    btn.setIconSize(QtCore.QSize(icon_size, icon_size))
    return btn


# ── Main window ───────────────────────────────────────────────────────────────

class PreviewWindow(QtWidgets.QWidget):
    seek_requested = QtCore.Signal(int)

    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._drag_pos: QtCore.QPoint | None = None
        self._slider_dragging = False

        self.setWindowTitle("VisoMaster Preview")
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumSize(400, 280)
        self.resize(840, 540)

        # Base palette
        pal = self.palette()
        pal.setColor(QtGui.QPalette.ColorRole.Window,     QtGui.QColor(_BG))
        pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(_TEXT))
        pal.setColor(QtGui.QPalette.ColorRole.Base,       QtGui.QColor(_SURFACE))
        self.setPalette(pal)
        self.setAutoFillBackground(True)
        self.setStyleSheet(_QSS)

        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Frame area ────────────────────────────────────────────────────
        self._frame_label = QtWidgets.QLabel()
        self._frame_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet("background: #000;")
        self._frame_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self._frame_label, stretch=1)

        # FPS badge (absolute child of frame_label)
        self._fps_badge = QtWidgets.QLabel("", self._frame_label)
        self._fps_badge.setStyleSheet(
            "background: rgba(0,0,0,0.65); color: #fff; "
            "padding: 2px 6px; border-radius: 4px; font-size: 11px;"
        )
        self._fps_badge.hide()

        # ── Controls bar ──────────────────────────────────────────────────
        ctrl = QtWidgets.QWidget()
        ctrl.setObjectName("ctrl_bar")
        ctrl.setFixedHeight(76)
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(12, 8, 12, 8)
        ctrl_layout.setSpacing(6)

        # Seek slider
        self._slider = _MarkerSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.valueChanged.connect(self._on_slider_value_changed)
        ctrl_layout.addWidget(self._slider)

        # Bottom row: frame counter left, buttons center, loop right
        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(0)

        # Frame counter
        self._lbl_cur = QtWidgets.QLabel("0")
        self._lbl_cur.setStyleSheet(f"font-size: 10px; color: {_MUTED}; font-variant-numeric: tabular-nums;")
        self._lbl_max = QtWidgets.QLabel("0")
        self._lbl_max.setStyleSheet(f"font-size: 10px; color: {_MUTED}; font-variant-numeric: tabular-nums;")
        self._lbl_max.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        bottom.addWidget(self._lbl_cur)
        bottom.addStretch()

        # Playback buttons — centered
        self._btn_start = _make_icon_btn(_icon_skip_back,  size=30, icon_size=16, tooltip="Jump to start")
        self._btn_back  = _make_icon_btn(_icon_step_back,  size=30, icon_size=16, tooltip="−30 frames")
        self._btn_play  = _make_icon_btn(_icon_play,       size=36, icon_size=18,
                                         checkable=True, tooltip="Play / Stop",
                                         obj_name="play_btn")
        self._btn_fwd   = _make_icon_btn(_icon_step_fwd,   size=30, icon_size=16, tooltip="+30 frames")
        self._btn_end   = _make_icon_btn(_icon_skip_fwd,   size=30, icon_size=16, tooltip="Jump to end")
        self._btn_loop  = _make_icon_btn(_icon_loop,       size=30, icon_size=16,
                                         checkable=True, tooltip="Toggle loop")

        for btn in (self._btn_start, self._btn_back, self._btn_play,
                    self._btn_fwd, self._btn_end):
            bottom.addWidget(btn)
            if btn is not self._btn_end:
                bottom.addSpacing(4)

        bottom.addStretch()

        # Loop button right-aligned, same level as frame counter
        bottom.addWidget(self._btn_loop)
        bottom.addSpacing(4)
        bottom.addWidget(self._lbl_max)

        ctrl_layout.addLayout(bottom)
        root.addWidget(ctrl)

        # Wire buttons
        self._btn_start.clicked.connect(self._on_jump_start)
        self._btn_back.clicked.connect(self._on_step_back)
        self._btn_play.toggled.connect(self._on_play_toggled)
        self._btn_fwd.clicked.connect(self._on_step_fwd)
        self._btn_end.clicked.connect(self._on_jump_end)
        self._btn_loop.toggled.connect(self._on_loop_toggled)

    # ── Public API ────────────────────────────────────────────────────────

    def update_frame(self, frame: np.ndarray):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return
        h, w = frame.shape[:2]
        ch = frame.shape[2] if frame.ndim == 3 else 1
        if ch == 1:
            q_img = QtGui.QImage(frame.data, w, h, w,
                                 QtGui.QImage.Format.Format_Grayscale8)
        else:
            q_img = QtGui.QImage(frame.data, w, h, 3 * w,
                                 QtGui.QImage.Format.Format_RGB888).rgbSwapped()
        self._set_pixmap(QtGui.QPixmap.fromImage(q_img))

    def update_frame_from_pixmap(self, pixmap: QtGui.QPixmap):
        if pixmap and not pixmap.isNull():
            self._set_pixmap(pixmap)

    def sync_playback_state(self, current_frame: int, max_frame: int,
                             is_playing: bool, loop_enabled: bool):
        self._slider.blockSignals(True)
        self._slider.setRange(0, max_frame)
        if not self._slider_dragging:
            self._slider.setValue(current_frame)
        self._slider.blockSignals(False)

        self._lbl_cur.setText(str(current_frame))
        self._lbl_max.setText(str(max_frame))

        # Update play button icon and checked state without re-firing toggled
        self._btn_play.blockSignals(True)
        self._btn_play.setChecked(is_playing)
        self._btn_play.setIcon(_make_icon(_icon_stop if is_playing else _icon_play, 18))
        self._btn_play.blockSignals(False)

        self._btn_loop.blockSignals(True)
        self._btn_loop.setChecked(loop_enabled)
        self._btn_loop.blockSignals(False)

    def set_markers(self, markers: set):
        self._slider.markers = set(markers)
        self._slider.update()

    # ── Internal ──────────────────────────────────────────────────────────

    def _set_pixmap(self, pixmap: QtGui.QPixmap):
        scaled = pixmap.scaled(
            self._frame_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(scaled)
        # Reposition FPS badge
        self._fps_badge.adjustSize()
        self._fps_badge.move(self._frame_label.width() - self._fps_badge.width() - 8, 8)

    # ── Slider ────────────────────────────────────────────────────────────

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        self._seek_to(self._slider.value())

    def _on_slider_value_changed(self, value: int):
        self._lbl_cur.setText(str(value))

    # ── Buttons ───────────────────────────────────────────────────────────

    def _on_play_toggled(self, checked: bool):
        vp = self.main_window.video_processor
        if checked:
            if not vp.processing and vp.file_type:
                vp.process_video()
        else:
            vp.stop_processing()
        self._btn_play.setIcon(_make_icon(_icon_stop if checked else _icon_play, 18))

    def _on_jump_start(self):
        """Seek to frame 0. If video was playing, resume after seek."""
        vp = self.main_window.video_processor
        was_playing = vp.processing
        self._seek_to(0)
        if was_playing and vp.file_type == "video":
            vp.process_video()

    def _on_jump_end(self):
        vp = self.main_window.video_processor
        self._seek_to(vp.max_frame_number)

    def _on_step_back(self):
        vp = self.main_window.video_processor
        self._seek_to(max(0, vp.current_frame_number - 30))

    def _on_step_fwd(self):
        vp = self.main_window.video_processor
        self._seek_to(min(vp.max_frame_number, vp.current_frame_number + 30))

    def _on_loop_toggled(self, checked: bool):
        """Toggle loop on the VP / AppState — works for both Qt and headless modes."""
        vp = self.main_window.video_processor

        # Headless VP (server.py _HeadlessVideoProcessor) exposes _state
        if hasattr(vp, '_state'):
            vp._state.loop_enabled = checked
            vp._state.control['LoopVideoToggle'] = checked
            return

        # Qt VP — update control dict and widget if present
        mw = self.main_window
        mw.control['LoopVideoToggle'] = checked
        if hasattr(mw, 'parameter_widgets') and 'LoopVideoToggle' in mw.parameter_widgets:
            mw.parameter_widgets['LoopVideoToggle'].blockSignals(True)
            mw.parameter_widgets['LoopVideoToggle'].set_value(checked)
            mw.parameter_widgets['LoopVideoToggle'].blockSignals(False)

    def _seek_to(self, frame: int):
        mw = self.main_window
        vp = mw.video_processor
        vp.stop_processing()
        vp.current_frame_number = frame
        if hasattr(vp, 'next_frame_to_display'):
            vp.next_frame_to_display = frame
        if vp.media_capture:
            import cv2
            vp.media_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        # Sync main window slider if present
        if hasattr(mw, 'videoSeekSlider'):
            mw.videoSeekSlider.blockSignals(True)
            mw.videoSeekSlider.setValue(frame)
            mw.videoSeekSlider.blockSignals(False)
        vp.process_current_frame()
        self.seek_requested.emit(frame)

    # ── Resize ────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        px = self._frame_label.pixmap()
        if px and not px.isNull():
            self._set_pixmap(px)

    # ── Window drag ───────────────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._drag_pos and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    # ── Context menu ──────────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        stay = bool(self.windowFlags() & QtCore.Qt.WindowType.WindowStaysOnTopHint)
        menu.addAction("Always on Top ✓" if stay else "Always on Top",
                       self._toggle_always_on_top)
        menu.addSeparator()
        menu.addAction("Close", self.close)
        menu.exec(event.globalPos())

    def _toggle_always_on_top(self):
        pos, size = self.pos(), self.size()
        flags = self.windowFlags()
        if flags & QtCore.Qt.WindowType.WindowStaysOnTopHint:
            flags &= ~QtCore.Qt.WindowType.WindowStaysOnTopHint
        else:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.move(pos); self.resize(size); self.show()

    # ── Close ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        mw = self.main_window
        if hasattr(mw, '_preview_window') and mw._preview_window is self:
            mw._preview_window = None
        if hasattr(mw, 'control'):
            mw.control['PreviewWindowEnableToggle'] = False
        event.accept()


# ── Marker slider ─────────────────────────────────────────────────────────────

class _MarkerSlider(QtWidgets.QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markers: set[int] = set()

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        if not self.markers or self.maximum() <= 0:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt,
            QtWidgets.QStyle.SubControl.SC_SliderGroove, self,
        )
        gx, gw = groove.left(), groove.width()
        p.setPen(QtGui.QPen(QtGui.QColor(_AMBER), 2))
        for m in self.markers:
            x = gx + int(m / self.maximum() * gw)
            p.drawLine(x, groove.top() - 3, x, groove.bottom() + 3)
        p.end()
