
from PySide6 import QtCore as qtc
from PySide6.QtGui import QPixmap, QImage, Qt
from App.Helpers import Misc_Helpers as misc_helpers
import os
import cv2

class TargetMediaLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, QPixmap, str)  # Signal with media path and QPixmap and file_type
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(self, folder_name=False, parent=None):
        super().__init__(parent)
        self.folder_name = folder_name

    def run(self):
        if self.folder_name:
            self.load_videos_and_images_from_folder(self.folder_name)
        self.finished.emit()

    def load_videos_and_images_from_folder(self, folder_name):
        video_files = misc_helpers.get_video_files(folder_name)
        image_files = misc_helpers.get_image_files(folder_name)

        media_files = video_files + image_files
        for media_file in media_files:
            media_file_path = os.path.join(folder_name, media_file)
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = self.extract_frame_as_pixmap(media_file_path, file_type)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type)

    def extract_frame_as_pixmap(self, media_file_path, file_type):
        frame = False
        if file_type=='image':
            frame = cv2.imread(media_file_path)
        elif file_type=='video':    
            cap = cv2.VideoCapture(media_file_path)
            ret, frame = cap.read()
            cap.release()

        if not isinstance(frame, bool):
            # Convert the frame to QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(70, 70, Qt.AspectRatioMode.KeepAspectRatio)  # Adjust size as needed
            return pixmap
        return None