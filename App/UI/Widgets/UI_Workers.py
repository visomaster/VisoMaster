
from PySide6 import QtCore as qtc
from PySide6.QtGui import QPixmap, QImage, Qt
from App.Helpers import Misc_Helpers as misc_helpers
from App.UI.Widgets import WidgetActions as widget_actions
import os
import cv2
import torch
import numpy

class TargetMediaLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, QPixmap, str)  # Signal with media path and QPixmap and file_type
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(self, folder_name=False, files_list=[],  parent=None):
        super().__init__(parent)
        self.folder_name = folder_name
        self.files_list = files_list

    def run(self):
        if self.folder_name:
            self.load_videos_and_images_from_folder(self.folder_name)
        if self.files_list:
            self.load_videos_and_images_from_files_list(self.files_list)
        self.finished.emit()

    def load_videos_and_images_from_folder(self, folder_name):
        video_files = misc_helpers.get_video_files(folder_name)
        image_files = misc_helpers.get_image_files(folder_name)

        media_files = video_files + image_files
        for media_file in media_files:
            media_file_path = os.path.join(folder_name, media_file)
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = widget_actions.extract_frame_as_pixmap(media_file_path, file_type)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type)

    def load_videos_and_images_from_files_list(self, files_list):
        media_files = files_list
        for media_file_path in media_files:
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = widget_actions.extract_frame_as_pixmap(media_file_path, file_type)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type)


class InputFacesLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, numpy.ndarray, numpy.ndarray, QPixmap)
    finished = qtc.Signal()  # Signal to indicate completion
    def __init__(self, main_window, media_path=False, folder_name=False, files_list=[],  parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list

    def run(self):
        if self.folder_name or self.files_list:
            self.load_faces(self.folder_name, self.files_list)

    def load_faces(self, folder_name=False, files_list=[]):
        if folder_name:
            image_files = misc_helpers.get_image_files(self.folder_name)
        elif files_list:
            image_files = files_list

        for image_file_path in image_files:
            if folder_name:
                image_file_path = os.path.join(folder_name, image_file_path)
            frame = cv2.imread(image_file_path)
            img = torch.from_numpy(frame.astype('uint8')).to('cuda')
            img = img.permute(2,0,1)
            bboxes, kpss_5, _ = self.main_window.models_processor.run_detect(img,max_num=1,)

            # If atleast one face is found
            found_face = []
            face_kps = False
            try:
                face_kps = kpss_5[0]
            except:
                return
            if face_kps.any():
                face_emb, cropped_img = self.main_window.models_processor.run_recognize(img, face_kps)
                face_img = numpy.ascontiguousarray(cropped_img.cpu().numpy())
                # crop = cv2.resize(face[2].cpu().numpy(), (82, 82))
                pixmap = widget_actions.get_pixmap_from_frame(self.main_window, face_img)
                self.thumbnail_ready.emit(image_file_path, face_img, face_emb, pixmap)