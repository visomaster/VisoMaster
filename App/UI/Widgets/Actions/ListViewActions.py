import cv2
from PySide6 import QtWidgets, QtGui, QtCore
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

import torch
import numpy
from torchvision.transforms import v2
from App.UI.Widgets.SettingsLayoutData import SETTINGS_LAYOUT_DATA
import App.UI.Widgets.Actions.CommonActions as common_widget_actions
import App.UI.Widgets.Actions.CardActions as card_actions

from App.UI.Widgets.WidgetComponents import TargetFaceCardButton, InputFaceCardButton, EmbeddingCardButton, TargetMediaCardButton, CardButton

import App.Helpers.Misc_Helpers as misc_helpers
import App.UI.Widgets.UI_Workers as ui_workers
import time
from functools import partial
def find_target_faces(main_window: 'MainWindow'):
    control = main_window.control.copy()
    video_processor = main_window.video_processor
    if video_processor.media_path:
        print(video_processor.media_capture)
        if video_processor.file_type=='image':
            frame = cv2.imread(video_processor.media_path)
        elif video_processor.file_type=='video' and video_processor.media_capture:
            media_capture = cv2.VideoCapture(video_processor.media_path)
            media_capture.set(cv2.CAP_PROP_POS_FRAMES, video_processor.current_frame_number)
            ret,frame = media_capture.read()
            media_capture.release()
        
        # Frame must be in RGB format
        frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

        # print(frame)
        img = torch.from_numpy(frame.astype('uint8')).to(main_window.models_processor.device)
        img = img.permute(2,0,1)
        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        bboxes, kpss_5, _ = main_window.models_processor.run_detect(img, control['DetectorModelSelection'], max_num=control['MaxFacesToDetectSlider'], score=control['DetectorScoreSlider']/100.0, input_size=(512, 512), use_landmark_detection=control['LandmarkDetectToggle'], landmark_detect_mode=control['LandmarkDetectModelSelection'], landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=control["DetectFromPointsToggle"], rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270])

        ret = []
        for face_kps in kpss_5:
            face_emb, cropped_img = main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
            ret.append([face_kps, face_emb, cropped_img, img])

        if ret:
            # Loop through all faces in video frame
            for face in ret:
                found = False
                # Check if this face has already been found
                for target_face in main_window.target_faces:
                    parameters = main_window.parameters[target_face.face_id]
                    threshhold = parameters['SimilarityThresholdSlider']
                    if main_window.models_processor.findCosineDistance(target_face.get_embedding(control['RecognitionModelSelection']), face[1]) >= threshhold:
                        found = True
                        break
                if not found:
                    face_img = face[2].cpu().numpy()
                    face_img = face_img[..., ::-1]  # Swap the channels from RGB to BGR
                    face_img = numpy.ascontiguousarray(face_img)
                    # crop = cv2.resize(face[2].cpu().numpy(), (82, 82))
                    pixmap = common_widget_actions.get_pixmap_from_frame(main_window, face_img)

                    embedding_store: Dict[str, numpy.ndarray] = {}
                    # Ottenere i valori di 'options'
                    options = SETTINGS_LAYOUT_DATA['Face Recognition']['RecognitionModelSelection']['options']
                    for option in options:
                        if option != control['RecognitionModelSelection']:
                            target_emb, _ = main_window.models_processor.run_recognize_direct(face[3], face[0], control['SimilarityTypeSelection'], option)
                            embedding_store[option] = target_emb
                        else:
                            embedding_store[control['RecognitionModelSelection']] = face[1]

                    add_media_thumbnail_to_target_faces_list(main_window, face_img, embedding_store, pixmap)
        # Select the first target face if no target face is already selected
        if main_window.target_faces and not main_window.selected_target_face_id:
            main_window.target_faces[0].click()

    common_widget_actions.update_gpu_memory_progressbar(main_window)

# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QPixmap)
def add_media_thumbnail_to_target_videos_list(main_window: 'MainWindow', media_path, pixmap, file_type):
    add_media_thumbnail_button(main_window, TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, pixmap, media_path=media_path, file_type=file_type)

@QtCore.Slot()
def add_media_thumbnail_to_target_faces_list(main_window: 'MainWindow', cropped_face, embedding_store, pixmap):
    add_media_thumbnail_button(main_window, TargetFaceCardButton, main_window.targetFacesList, main_window.target_faces, pixmap, cropped_face=cropped_face, embedding_store=embedding_store )

@QtCore.Slot()
def add_media_thumbnail_to_source_faces_list(main_window: 'MainWindow', media_path, cropped_face, embedding_store, pixmap):
    add_media_thumbnail_button(main_window, InputFaceCardButton, main_window.inputFacesList, main_window.input_faces, pixmap, media_path=media_path, cropped_face=cropped_face, embedding_store=embedding_store )


def add_media_thumbnail_button(main_window: 'MainWindow', buttonClass: CardButton, listWidget:QtWidgets.QListWidget, buttons_list:list, pixmap, **kwargs):
    if buttonClass==TargetMediaCardButton:
        constructor_args = (kwargs.get('media_path'), kwargs.get('file_type'))
    elif buttonClass in (TargetFaceCardButton, InputFaceCardButton):
        constructor_args = (kwargs.get('media_path',''), kwargs.get('cropped_face'), kwargs.get('embedding_store'))
    button_size = QtCore.QSize(70, 70)  # Set a fixed size for the buttons
    button: CardButton = buttonClass(*constructor_args, main_window=main_window)
    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(button_size - QtCore.QSize(3, 3))  # Slightly smaller than the button size to add some margin
    button.setFixedSize(button_size)
    button.setCheckable(True)
    buttons_list.append(button)
    # Create a QListWidgetItem and set the button as its widget
    list_item = QtWidgets.QListWidgetItem(listWidget)
    list_item.setSizeHint(button_size)
    button.list_item = list_item
    # Align the item to center
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    listWidget.setItemWidget(list_item, button)
    # Adjust the QListWidget properties to handle the grid layout
    grid_size_with_padding = button_size + QtCore.QSize(4, 4)  # Add padding around the buttons
    listWidget.setGridSize(grid_size_with_padding)  # Set grid size with padding
    listWidget.setWrapping(True)  # Enable wrapping to have items in rows
    listWidget.setFlow(QtWidgets.QListView.LeftToRight)  # Set flow direction
    listWidget.setResizeMode(QtWidgets.QListView.Adjust)  # Adjust layout automatically


def clear_stop_loading_target_media(main_window: 'MainWindow'):
    if main_window.video_loader_worker:
        main_window.video_loader_worker.stop()
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        time.sleep(0.5)
        main_window.targetVideosList.clear()

@QtCore.Slot()
def onClickSelectTargetVideos(main_window: 'MainWindow', source_type='folder', folder_name=False, files_list=[]):
    if source_type=='folder':
        folder_name = QtWidgets.QFileDialog.getExistingDirectory()
        if not folder_name:
            return
        main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelTargetVideosPath.setToolTip(folder_name)

    elif source_type=='files':
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        main_window.labelTargetVideosPath.setText('Selected Files') #Just a temp text until i think of something better
        main_window.labelTargetVideosPath.setToolTip('Selected Files')

    clear_stop_loading_target_media(main_window)
    card_actions.clear_target_faces(main_window)
    
    main_window.selected_video_buttons = []
    main_window.target_videos = []

    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(folder_name=folder_name, files_list=files_list)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_target_videos_list, main_window))
    main_window.video_loader_worker.start()

def clear_stop_loading_input_media(main_window: 'MainWindow'):
    if main_window.input_faces_loader_worker:
        main_window.input_faces_loader_worker.stop()
        main_window.input_faces_loader_worker.terminate()
        main_window.input_faces_loader_worker = False
        time.sleep(0.5)
        main_window.inputFacesList.clear()

@QtCore.Slot()
def onClickSelectInputImages(main_window: 'MainWindow', source_type='folder', folder_name=False, files_list=[]):
    if source_type=='folder':
        folder_name = QtWidgets.QFileDialog.getExistingDirectory()
        main_window.labelInputFacesPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelInputFacesPath.setToolTip(folder_name)
        if not folder_name:
            return

    elif source_type=='files':
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        main_window.labelInputFacesPath.setText('Selected Files') #Just a temp text until i think of something better
        main_window.labelInputFacesPath.setToolTip('Selected Files')
        if not files_list:
            return

    clear_stop_loading_input_media(main_window)
    card_actions.clear_input_faces(main_window)
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(main_window=main_window, folder_name=folder_name, files_list=files_list)
    main_window.input_faces_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_source_faces_list, main_window))
    main_window.input_faces_loader_worker.start()