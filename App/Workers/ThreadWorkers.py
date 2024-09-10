from PySide6 import QtCore as qtc
from PySide6.QtGui import QPixmap, QImage, Qt
from App.Helpers import MiscHelpers as misc_helpers
import os
import cv2

class VideoLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, QPixmap)  # Signal with video path and QPixmap
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(self, folder_name, parent=None):
        super().__init__(parent)
        self.folder_name = folder_name

    def run(self):
        self.load_video_images_from_folder(self.folder_name)
        self.finished.emit()

    def load_video_images_from_folder(self, folder_name):
        print("Hello", folder_name)
        video_files = [f for f in misc_helpers.absoluteFilePaths(folder_name) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.png'))]

        for video_file in video_files:
            video_path = os.path.join(folder_name, video_file)
            pixmap = self.extract_frame_as_pixmap(video_path)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(video_path, pixmap)

    def extract_frame_as_pixmap(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            # Convert the frame to QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio)  # Adjust size as needed
            cap.release()
            return pixmap
        cap.release()
        return None