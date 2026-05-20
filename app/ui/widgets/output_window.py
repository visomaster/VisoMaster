"""
Borderless output window for OBS/screen capture.

Inspired by DeepFaceLive's streaming output approach: a minimal, frameless window
that displays only the processed frame. OBS (or any screen capture tool) can capture
this window using "Window Capture" source by selecting "VisoMaster Output".

This is useful for systems that don't support virtual cameras or when the user
prefers a simpler capture workflow.
"""

from typing import TYPE_CHECKING

from PySide6 import QtWidgets, QtGui, QtCore
import numpy as np

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


class OutputWindow(QtWidgets.QWidget):
    """
    A frameless, always-on-top window that mirrors the processed output frame.
    
    Features:
    - Frameless (no title bar, borders) for clean OBS capture
    - Window title set to "VisoMaster Output" for easy identification in OBS
    - Resizable by dragging edges
    - Movable by dragging anywhere on the frame
    - Right-click context menu to close or toggle always-on-top
    - Maintains aspect ratio of the source frame
    - Double-click to toggle borderless/bordered mode
    """

    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._drag_position = None
        self._resizing = False
        self._resize_edge = None
        self._aspect_ratio = 16 / 9  # Default, updated when frames arrive
        self._bordered = False  # Track border mode
        self._last_frame_size = QtCore.QSize(640, 360)

        # Window setup - frameless, always on top, with a recognizable title for OBS
        self.setWindowTitle("VisoMaster Output")
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMinimumSize(160, 90)
        self.resize(640, 360)

        # Black background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(0, 0, 0))
        self.setPalette(palette)

        # QLabel to display the frame
        self._label = QtWidgets.QLabel(self)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setScaledContents(False)
        self._label.setStyleSheet("background-color: black;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        # Edge resize margin in pixels
        self._resize_margin = 6

    def update_frame(self, frame: np.ndarray):
        """
        Update the displayed frame. Expects a BGR numpy array (same format as current_frame).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1

        if channels == 1:
            bytes_per_line = width
            q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)
        else:
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).rgbSwapped()

        pixmap = QtGui.QPixmap.fromImage(q_img)
        self._aspect_ratio = width / height if height > 0 else 16 / 9
        self._last_frame_size = QtCore.QSize(width, height)

        # Scale pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self._label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self._label.setPixmap(scaled_pixmap)

    def update_frame_from_pixmap(self, pixmap: QtGui.QPixmap):
        """
        Update the displayed frame from an existing QPixmap.
        """
        if pixmap is None or pixmap.isNull():
            return

        self._aspect_ratio = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 16 / 9
        self._last_frame_size = pixmap.size()

        scaled_pixmap = pixmap.scaled(
            self._label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self._label.setPixmap(scaled_pixmap)

    # ── Resize handling ──────────────────────────────────────────────────────

    def resizeEvent(self, event):
        """Re-scale the current pixmap when the window is resized."""
        super().resizeEvent(event)
        if self._label.pixmap() and not self._label.pixmap().isNull():
            # Re-render from last frame size to avoid quality loss from repeated scaling
            pass  # The next frame update will handle proper scaling

    def _edge_at(self, pos):
        """Determine which edge (if any) the mouse is near for resizing."""
        rect = self.rect()
        m = self._resize_margin
        edges = []
        if pos.x() <= m:
            edges.append('left')
        elif pos.x() >= rect.width() - m:
            edges.append('right')
        if pos.y() <= m:
            edges.append('top')
        elif pos.y() >= rect.height() - m:
            edges.append('bottom')
        return edges if edges else None

    def _cursor_for_edges(self, edges):
        """Return the appropriate resize cursor for the given edges."""
        if not edges:
            return QtCore.Qt.CursorShape.ArrowCursor
        edge_set = set(edges)
        if edge_set == {'left'} or edge_set == {'right'}:
            return QtCore.Qt.CursorShape.SizeHorCursor
        if edge_set == {'top'} or edge_set == {'bottom'}:
            return QtCore.Qt.CursorShape.SizeVerCursor
        if edge_set == {'top', 'left'} or edge_set == {'bottom', 'right'}:
            return QtCore.Qt.CursorShape.SizeFDiagCursor
        if edge_set == {'top', 'right'} or edge_set == {'bottom', 'left'}:
            return QtCore.Qt.CursorShape.SizeBDiagCursor
        return QtCore.Qt.CursorShape.ArrowCursor

    # ── Mouse events for move and resize ─────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            edges = self._edge_at(event.position().toPoint())
            if edges:
                self._resizing = True
                self._resize_edge = edges
                self._drag_position = event.globalPosition().toPoint()
            else:
                self._resizing = False
                self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._resizing and self._drag_position is not None:
            delta = event.globalPosition().toPoint() - self._drag_position
            self._drag_position = event.globalPosition().toPoint()
            geo = self.geometry()

            if 'right' in self._resize_edge:
                geo.setRight(geo.right() + delta.x())
            if 'bottom' in self._resize_edge:
                geo.setBottom(geo.bottom() + delta.y())
            if 'left' in self._resize_edge:
                geo.setLeft(geo.left() + delta.x())
            if 'top' in self._resize_edge:
                geo.setTop(geo.top() + delta.y())

            # Enforce minimum size
            if geo.width() >= self.minimumWidth() and geo.height() >= self.minimumHeight():
                self.setGeometry(geo)

        elif self._drag_position is not None and not self._resizing:
            # Moving the window
            self.move(event.globalPosition().toPoint() - self._drag_position)
        else:
            # Update cursor based on position
            edges = self._edge_at(event.position().toPoint())
            self.setCursor(self._cursor_for_edges(edges))

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self._drag_position = None
        self._resizing = False
        self._resize_edge = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        """Double-click toggles between frameless and bordered mode."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._toggle_border_mode()
        super().mouseDoubleClickEvent(event)

    def _toggle_border_mode(self):
        """Toggle between frameless and normal window frame."""
        self._bordered = not self._bordered
        pos = self.pos()
        size = self.size()

        if self._bordered:
            self.setWindowFlags(
                QtCore.Qt.WindowType.Window
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
            )
        else:
            self.setWindowFlags(
                QtCore.Qt.WindowType.Window
                | QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
            )

        # Restore position and size after flag change
        self.move(pos)
        self.resize(size)
        self.show()

    # ── Context menu ─────────────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)

        # Toggle always on top
        stay_on_top = self.windowFlags() & QtCore.Qt.WindowType.WindowStaysOnTopHint
        toggle_top_action = menu.addAction("Always on Top ✓" if stay_on_top else "Always on Top")
        toggle_top_action.triggered.connect(self._toggle_always_on_top)

        # Toggle border
        border_action = menu.addAction("Show Border" if not self._bordered else "Hide Border")
        border_action.triggered.connect(self._toggle_border_mode)

        menu.addSeparator()

        # Fit to frame size
        fit_action = menu.addAction("Fit to Frame Size")
        fit_action.triggered.connect(self._fit_to_frame)

        menu.addSeparator()

        # Close
        close_action = menu.addAction("Close")
        close_action.triggered.connect(self.close)

        menu.exec(event.globalPos())

    def _toggle_always_on_top(self):
        """Toggle the always-on-top flag."""
        pos = self.pos()
        size = self.size()
        flags = self.windowFlags()

        if flags & QtCore.Qt.WindowType.WindowStaysOnTopHint:
            flags &= ~QtCore.Qt.WindowType.WindowStaysOnTopHint
        else:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint

        self.setWindowFlags(flags)
        self.move(pos)
        self.resize(size)
        self.show()

    def _fit_to_frame(self):
        """Resize window to match the native frame dimensions."""
        if self._last_frame_size.width() > 0 and self._last_frame_size.height() > 0:
            self.resize(self._last_frame_size)

    def closeEvent(self, event):
        """When closed, update the control toggle if it exists."""
        if hasattr(self.main_window, 'control') and 'OutputWindowEnableToggle' in self.main_window.control:
            # Update the toggle without triggering the exec_function
            if self.main_window.control['OutputWindowEnableToggle']:
                self.main_window.control['OutputWindowEnableToggle'] = False
                if 'OutputWindowEnableToggle' in self.main_window.parameter_widgets:
                    widget = self.main_window.parameter_widgets['OutputWindowEnableToggle']
                    widget.blockSignals(True)
                    widget.set_value(False)
                    widget.blockSignals(False)
        event.accept()
