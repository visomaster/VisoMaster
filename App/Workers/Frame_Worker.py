from PySide6.QtCore import QRunnable,QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QGraphicsPixmapItem
import cv2
import App.Helpers.UI_Helpers as ui_helpers
import numpy

class FrameWorker(QRunnable):
    def __init__(self, frame, main_window, current_frame_number):
        super().__init__()
        self.current_frame_number = current_frame_number
        self.frame = frame
        self.main_window = main_window
        # self.graphicsViewFrame = graphicsViewFrame

    def run(self):
        self.frame = self.process_swap_on_frame()
        # Convert the frame (which is a NumPy array) to QImage
        scaled_pixmap = ui_helpers.get_pixmap_from_frame(self.main_window, self.frame)
        self.main_window.update_frame_signal.emit(self.main_window, scaled_pixmap, self.current_frame_number)

    def process_swap_on_frame(self):
        frame = self.frame
        return numpy.ascontiguousarray(frame)