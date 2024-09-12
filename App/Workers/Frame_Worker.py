from PySide6.QtCore import QRunnable,QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QGraphicsPixmapItem
import cv2
import App.Helpers.UI_Helpers as ui_helpers

class FrameWorker(QRunnable):
    def __init__(self, frame, main_window, current_frame_number):
        super().__init__()
        self.current_frame_number = current_frame_number
        self.frame = frame
        self.main_window = main_window
        # self.graphicsViewFrame = graphicsViewFrame

    def run(self):
        # Convert the frame (which is a NumPy array) to QImage
        scaled_pixmap = ui_helpers.get_pixmap_from_frame(self.main_window, self.frame)
        # q_img = QImage(self.frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        # pixmap = QPixmap.fromImage(q_img)
        # scaled_pixmap = ui_helpers.scale_pixmap_to_view(self.main_window.graphicsViewFrame, pixmap)
        self.main_window.update_frame_signal.emit(self.main_window, scaled_pixmap, self.current_frame_number)
        # self.main_window.update_frame_signal.emit(self.main_window, scaled_pixmap, self.current_frame_number)