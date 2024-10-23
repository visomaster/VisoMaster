
from PySide6 import QtCore as qtc
from PySide6.QtGui import QPixmap, QImage, Qt
from App.Helpers import Misc_Helpers as misc_helpers
from App.UI.Widgets import WidgetActions as widget_actions
from App.UI.Widgets.SettingsLayoutData import SETTINGS_LAYOUT_DATA
import os
import cv2
import torch
import numpy
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

class TargetMediaLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, QPixmap, str)  # Signal with media path and QPixmap and file_type
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(self, folder_name=False, files_list=[],  parent=None):
        super().__init__(parent)
        self.folder_name = folder_name
        self.files_list = files_list
        self._running = True  # Flag to control the running state

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
            if not self._running:  # Check if the thread is still running
                break
            media_file_path = os.path.join(folder_name, media_file)
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = widget_actions.extract_frame_as_pixmap(media_file_path, file_type)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type)

    def load_videos_and_images_from_files_list(self, files_list):
        media_files = files_list
        for media_file_path in media_files:
            if not self._running:  # Check if the thread is still running
                break
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = widget_actions.extract_frame_as_pixmap(media_file_path, file_type)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type)

    def stop(self):
        """Stop the thread by setting the running flag to False."""
        self._running = False
        self.wait()

class InputFacesLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, numpy.ndarray, object, QPixmap)
    finished = qtc.Signal()  # Signal to indicate completion
    def __init__(self, main_window: 'MainWindow', media_path=False, folder_name=False, files_list=[],  parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list
        self._running = True  # Flag to control the running state

    def run(self):
        if self.folder_name or self.files_list:
            self.load_faces(self.folder_name, self.files_list)

    def load_faces(self, folder_name=False, files_list=[]):
        control = self.main_window.control.copy()
        if folder_name:
            image_files = misc_helpers.get_image_files(self.folder_name)
        elif files_list:
            image_files = files_list

        for image_file_path in image_files:
            if not self._running:  # Check if the thread is still running
                break
            if folder_name:
                image_file_path = os.path.join(folder_name, image_file_path)
            frame = cv2.imread(image_file_path)
            # Frame must be in RGB format
            frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

            img = torch.from_numpy(frame.astype('uint8')).to(self.main_window.models_processor.device)
            img = img.permute(2,0,1)
            bboxes, kpss_5, _ = self.main_window.models_processor.run_detect(img, control['DetectorModelSelection'], max_num=1, score=control['DetectorScoreSlider']/100.0, use_landmark_detection=control['LandmarkDetectToggle'], landmark_detect_mode=control['LandmarkDetectModelSelection'], landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=control["DetectFromPointsToggle"], rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270])

            # If atleast one face is found
            found_face = []
            face_kps = False
            try:
                face_kps = kpss_5[0]
            except:
                continue
            if face_kps.any():
                face_emb, cropped_img = self.main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                cropped_img = cropped_img.cpu().numpy()
                cropped_img = cropped_img[..., ::-1]  # Swap the channels from RGB to BGR
                face_img = numpy.ascontiguousarray(cropped_img)
                # crop = cv2.resize(face[2].cpu().numpy(), (82, 82))
                pixmap = widget_actions.get_pixmap_from_frame(self.main_window, face_img)

                embedding_store: Dict[str, numpy.ndarray] = {}
                # Ottenere i valori di 'options'
                options = SETTINGS_LAYOUT_DATA['Face Recognition']['RecognitionModelSelection']['options']
                for option in options:
                    if option != control['RecognitionModelSelection']:
                        target_emb, _ = self.main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], option)
                        embedding_store[option] = target_emb
                    else:
                        embedding_store[control['RecognitionModelSelection']] = face_emb

                self.thumbnail_ready.emit(image_file_path, face_img, embedding_store, pixmap)

        torch.cuda.empty_cache()

    def stop(self):
        """Stop the thread by setting the running flag to False."""
        self._running = False
        self.wait()