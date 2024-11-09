
import PySide6.QtCore as qtc
from PySide6 import QtWidgets, QtGui
import time
import App.Helpers.Misc_Helpers as misc_helpers 
import App.UI.Widgets.UI_Workers as ui_workers
from App.UI.Widgets.WidgetComponents import TargetMediaCardButton, ProgressDialog, TargetFaceCardButton, InputFaceCardButton, FormGroupBox, ToggleButton, SelectionBox, ParameterSlider, ParameterDecimalSlider, ParameterLineEdit, ParameterText, ParameterLineDecimalEdit, ParameterResetDefaultButton, CardButton, EmbeddingCardButton
from PySide6.QtWidgets import QComboBox
from pyqttoast import Toast, ToastPreset, ToastPosition

import App.UI.Widgets.WidgetActions as widget_actions 
from functools import partial
import cv2
from App.UI.Core import media_rc
import torch
import numpy
from torchvision.transforms import v2
from App.UI.Widgets.SwapperLayoutData import SWAPPER_LAYOUT_DATA
from App.UI.Widgets.SettingsLayoutData import SETTINGS_LAYOUT_DATA
from App.UI.Widgets.FaceEditorLayoutData import FACE_EDITOR_LAYOUT_DATA
from typing import TYPE_CHECKING, Dict
import json

if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
def clear_stop_loading_target_media(main_window: 'MainWindow'):
    if main_window.video_loader_worker:
        main_window.video_loader_worker.stop()
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        time.sleep(0.5)
        main_window.targetVideosList.clear()

@qtc.Slot()
def onClickSelectTargetVideos(main_window: 'MainWindow', source_type='folder', folder_name=False, files_list=[]):
    if source_type=='folder':
        folder_name = QtWidgets.QFileDialog.getExistingDirectory()
        main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelTargetVideosPath.setToolTip(folder_name)

        if not folder_name:
            return
    elif source_type=='files':
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        main_window.labelTargetVideosPath.setText('Selected Files') #Just a temp text until i think of something better
        main_window.labelTargetVideosPath.setToolTip('Selected Files')
        if not files_list:
            return

    clear_stop_loading_target_media(main_window)
    clear_target_faces(main_window)
    
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

@qtc.Slot()
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
    clear_input_faces(main_window)
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(main_window=main_window, folder_name=folder_name, files_list=files_list)
    main_window.input_faces_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_source_faces_list, main_window))
    main_window.input_faces_loader_worker.start()

def set_up_video_seek_line_edit(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    videoSeekLineEdit = main_window.videoSeekLineEdit
    videoSeekLineEdit.setAlignment(qtc.Qt.AlignCenter)
    videoSeekLineEdit.setText('0')
    videoSeekLineEdit.setValidator(QtGui.QIntValidator(0, video_processor.max_frame_number))  # Restrict input to numbers
    
    
@qtc.Slot(int)
def OnChangeSlider(main_window: 'MainWindow', new_position=0):
    print("Called OnChangeSlider()")
    video_processor = main_window.video_processor

    was_processing = video_processor.stop_processing()
    if was_processing:
        print("OnChangeSlider: Processing in progress. Stopping current processing.")

    video_processor.current_frame_number = new_position
    video_processor.next_frame_to_display = new_position
    if video_processor.media_capture:
        video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
        ret, frame = video_processor.media_capture.read()
        if ret:
            pixmap = widget_actions.get_pixmap_from_frame(main_window, frame)
            widget_actions.update_graphics_view(main_window, pixmap, new_position)
            # restore slider position 
            video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)

    # Do not automatically restart the video, let the user press Play to resume
    print("OnChangeSlider: Video stopped after slider movement.")

def on_slider_moved(main_window: 'MainWindow'):
    # print("Called on_slider_moved()")
    position = main_window.videoSeekSlider.value()
    print(f"\nSlider Moved. position: {position}\n")

def on_slider_pressed(main_window: 'MainWindow'):

    position = main_window.videoSeekSlider.value()
    print(f"\nSlider Pressed. position: {position}\n")

def on_slider_released(main_window: 'MainWindow'):
    # print("Called on_slider_released()")

    new_position = main_window.videoSeekSlider.value()
    print(f"\nSlider released. New position: {new_position}\n")

    # Perform the update to the new frame
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        video_processor.process_current_frame()  # Process the current frame

# Functions to add Buttons with thumbnail for selecting videos/images and faces
@qtc.Slot(str, QtGui.QPixmap)
def add_media_thumbnail_to_target_videos_list(main_window: 'MainWindow', media_path, pixmap, file_type):
    add_media_thumbnail_button(main_window, TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, pixmap, media_path=media_path, file_type=file_type)

@qtc.Slot()
def add_media_thumbnail_to_target_faces_list(main_window: 'MainWindow', cropped_face, embedding_store, pixmap):
    add_media_thumbnail_button(main_window, TargetFaceCardButton, main_window.targetFacesList, main_window.target_faces, pixmap, cropped_face=cropped_face, embedding_store=embedding_store )

@qtc.Slot()
def add_media_thumbnail_to_source_faces_list(main_window: 'MainWindow', media_path, cropped_face, embedding_store, pixmap):
    add_media_thumbnail_button(main_window, InputFaceCardButton, main_window.inputFacesList, main_window.input_faces, pixmap, media_path=media_path, cropped_face=cropped_face, embedding_store=embedding_store )


def add_media_thumbnail_button(main_window: 'MainWindow', buttonClass: CardButton, listWidget:QtWidgets.QListWidget, buttons_list:list, pixmap, **kwargs):
    if buttonClass==TargetMediaCardButton:
        constructor_args = (kwargs.get('media_path'), kwargs.get('file_type'))
    elif buttonClass in (TargetFaceCardButton, InputFaceCardButton):
        constructor_args = (kwargs.get('media_path',''), kwargs.get('cropped_face'), kwargs.get('embedding_store'))
    button_size = qtc.QSize(70, 70)  # Set a fixed size for the buttons
    button: CardButton = buttonClass(*constructor_args, main_window=main_window)
    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(button_size - qtc.QSize(3, 3))  # Slightly smaller than the button size to add some margin
    button.setFixedSize(button_size)
    button.setCheckable(True)
    buttons_list.append(button)
    # Create a QListWidgetItem and set the button as its widget
    list_item = QtWidgets.QListWidgetItem(listWidget)
    list_item.setSizeHint(button_size)
    button.list_item = list_item
    # Align the item to center
    list_item.setTextAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
    listWidget.setItemWidget(list_item, button)
    # Adjust the QListWidget properties to handle the grid layout
    grid_size_with_padding = button_size + qtc.QSize(4, 4)  # Add padding around the buttons
    listWidget.setGridSize(grid_size_with_padding)  # Set grid size with padding
    listWidget.setWrapping(True)  # Enable wrapping to have items in rows
    listWidget.setFlow(QtWidgets.QListView.LeftToRight)  # Set flow direction
    listWidget.setResizeMode(QtWidgets.QListView.Adjust)  # Adjust layout automatically

def extract_frame_as_pixmap(media_file_path, file_type):
    frame = False
    if file_type=='image':
        frame = cv2.imread(media_file_path)
    elif file_type=='video':    
        cap = cv2.VideoCapture(media_file_path)
        ret, frame = cap.read()
        cap.release()

    if isinstance(frame, numpy.ndarray):
        # Convert the frame to QPixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(70, 70, qtc.Qt.AspectRatioMode.KeepAspectRatio)  # Adjust size as needed
        return pixmap
    return None

def update_graphics_view(main_window: 'MainWindow', pixmap: QtGui.QPixmap, current_frame_number, reset_fit=False):
    print('(update_graphics_view) current_frame_number', current_frame_number)
    
    # Update the video seek slider and line edit
    main_window.videoSeekSlider.blockSignals(True)
    main_window.videoSeekSlider.setValue(current_frame_number)
    main_window.videoSeekSlider.blockSignals(False)
    main_window.videoSeekLineEdit.setText(str(current_frame_number))

    # Preserve the current transform (zoom and pan state)
    current_transform = main_window.graphicsViewFrame.transform()

    #Check if there is any Previous QGraphicsItem, if yes then resize the current pixmap to size of that item 
    #This is to handle cases where the size of processed frame if larger than the previous one (Eg: Using the Frame Enhancer)
    previous_graphics_item = main_window.scene.items()[0] if len(main_window.scene.items())>0 else False
    if previous_graphics_item:
        bounding_rect = previous_graphics_item.boundingRect()
        pixmap = pixmap.scaled(bounding_rect.width(), bounding_rect.height())

    # Clear the scene and add the new pixmap
    main_window.graphicsViewFrame.scene().clear()
    pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
    main_window.graphicsViewFrame.scene().addItem(pixmap_item)

    # Optionally set the scene rectangle (helps keep boundaries consistent)
    main_window.graphicsViewFrame.setSceneRect(pixmap_item.boundingRect())

    if reset_fit:
        # Reset the view to fit the new frame
        fit_image_to_view(main_window, pixmap_item)
    else:
        # Restore the previous zoom and pan state
        zoom_and_fit_image_to_view(main_window, current_transform)


def zoom_and_fit_image_to_view(main_window: 'MainWindow', new_transform):
    print("Called zoom_and_fit_image_to_view()")
    """Restore the previous transform (zoom and pan state) and update the view."""
    main_window.graphicsViewFrame.setTransform(new_transform)
    main_window.graphicsViewFrame.update()


def fit_image_to_view(main_window: 'MainWindow', pixmap_item: QtWidgets.QGraphicsPixmapItem):
    """Reset the view and fit the image to the view, keeping the aspect ratio."""
    print("Called fit_image_to_view()")
    graphicsViewFrame = main_window.graphicsViewFrame
    # Reset the transform to ensure no previous transformations affect the new fit
    graphicsViewFrame.resetTransform()
    # Set the scene rectangle to the bounding rectangle of the pixmap item
    graphicsViewFrame.setSceneRect(pixmap_item.boundingRect())
    # Fit the image to the view, keeping the aspect ratio
    graphicsViewFrame.fitInView(pixmap_item, qtc.Qt.AspectRatioMode.KeepAspectRatio)
    graphicsViewFrame.update()

def get_pixmap_from_frame(main_window: 'MainWindow', frame,):
    height, width = frame.shape[:2]
    if len(frame.shape) == 2:
        # Frame in grayscale
        bytes_per_line = width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)
    else:
        # Frame in color
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(q_img)
    return pixmap

def OnClickPlayButton(main_window: 'MainWindow', checked):
    video_processor = main_window.video_processor
    if checked:
        if video_processor.processing or video_processor.current_frame_number==video_processor.max_frame_number:
            print("OnClickPlayButton: Video already playing. Stopping the current video before starting a new one.")
            video_processor.stop_processing()
            return
        print("OnClickPlayButton: Starting video processing.")
        setPlayButtonIconToStop(main_window)
        video_processor.process_video()
    else:
        print("OnClickPlayButton: Stopping video processing.")
        setPlayButtonIconToPlay(main_window)
        video_processor.stop_processing()

def setPlayButtonIconToPlay(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_off.png"))
    main_window.buttonMediaPlay.setToolTip("Play")

def setPlayButtonIconToStop(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_on.png"))
    main_window.buttonMediaPlay.setToolTip("Stop")

def resetMediaButtons(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.blockSignals(True)
    main_window.buttonMediaPlay.setChecked(False)
    main_window.buttonMediaPlay.blockSignals(False)
    setPlayButtonIcon(main_window)

def setPlayButtonIcon(main_window: 'MainWindow'):
    if main_window.buttonMediaPlay.isChecked(): 
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_on.png"))
        main_window.buttonMediaPlay.setToolTip("Stop")
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_off.png"))
        main_window.buttonMediaPlay.setToolTip("Play")

def filterTargetVideos(main_window: 'MainWindow', search_text: str = ''):
    main_window.target_videos_filter_worker.stop_thread()
    main_window.target_videos_filter_worker.search_text = search_text
    main_window.target_videos_filter_worker.start()

def filterInputFaces(main_window: 'MainWindow', search_text: str = ''):
    main_window.input_faces_filter_worker.stop_thread()
    main_window.input_faces_filter_worker.search_text = search_text
    main_window.input_faces_filter_worker.start()

def filterMergedEmbeddings(main_window: 'MainWindow', search_text: str = ''):
    main_window.merged_embeddings_filter_worker.stop_thread()
    main_window.merged_embeddings_filter_worker.search_text = search_text
    main_window.merged_embeddings_filter_worker.start()

def updateFilteredList(main_window: 'MainWindow', filter_list_widget: QtWidgets.QListWidget, visible_indices: list):
    for i in range(filter_list_widget.count()):
        filter_list_widget.item(i).setHidden(True)

    # Show only the items in the visible_indices list
    for i in visible_indices:
        filter_list_widget.item(i).setHidden(False)

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
                    pixmap = get_pixmap_from_frame(main_window, face_img)

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

def clear_target_faces(main_window: 'MainWindow', refresh_frame=True):
    main_window.targetFacesList.clear()
    for target_face in main_window.target_faces:
        target_face.deleteLater()
    main_window.target_faces = []
    main_window.parameters = {}

    # Set Parameter widget values to default
    widget_actions.set_widgets_values_using_face_id_parameters(main_window=main_window, face_id=False)
    if refresh_frame:
        widget_actions.refresh_frame(main_window=main_window)
    
def clear_input_faces(main_window: 'MainWindow'):
    main_window.inputFacesList.clear()
    for input_face in main_window.input_faces:
        input_face.deleteLater()
    main_window.input_faces = []

    for target_face in main_window.target_faces:
        target_face.assigned_input_face_buttons = {}
        target_face.calculateAssignedInputEmbedding()
    widget_actions.refresh_frame(main_window=main_window)

def clear_merged_embeddings(main_window: 'MainWindow'):
    main_window.inputEmbeddingsList.clear()
    for embed_button in main_window.merged_embeddings:
        embed_button.deleteLater()
    main_window.merged_embeddings = []

    for target_face in main_window.target_faces:
        target_face.assigned_embed_buttons = {}
        target_face.calculateAssignedInputEmbedding()
    widget_actions.refresh_frame(main_window=main_window)


def uncheck_all_input_faces(main_window: 'MainWindow'):
    # Uncheck All other input faces 
    for input_face_button in main_window.input_faces:
        input_face_button.setChecked(False)

def uncheck_all_merged_embeddings(main_window: 'MainWindow'):
    for embed_button in  main_window.merged_embeddings:
        embed_button.setChecked(False)

def process_swap_faces(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

def process_edit_faces(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

def add_widgets_to_tab_layout(main_window: 'MainWindow', LAYOUT_DATA: dict, layoutWidget: QtWidgets.QVBoxLayout, data_type='parameter'):
    layout = QtWidgets.QVBoxLayout()
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_content = QtWidgets.QWidget()
    scroll_content.setLayout(layout)
    scroll_area.setWidget(scroll_content)

    for category, widgets in LAYOUT_DATA.items():
        group_box = FormGroupBox(main_window, title=category)
        category_layout = QtWidgets.QFormLayout()
        group_box.setLayout(category_layout)

        for widget_name, widget_data in widgets.items():
            spacing_level = widget_data['level']
            label = QtWidgets.QLabel(widget_data['label'])
            label.setToolTip(widget_data['help'])

            # Create a horizontal layout for the toggle button and its label
            if 'Toggle' in widget_name:
                widget = ToggleButton(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, main_window=main_window)
                widget.setChecked(widget_data['default'])
                widget.reset_default_button = ParameterResetDefaultButton(related_widget=widget)

                # Create a horizontal layout
                horizontal_layout = QtWidgets.QHBoxLayout()
                # In case of toggle button, add show widget first, then its label
                horizontal_layout.addWidget(widget)  # Add the toggle button
                horizontal_layout.addWidget(label)  # Add the label
                horizontal_layout.addWidget(widget.reset_default_button)
                
                category_layout.addRow(horizontal_layout)  # Add the horizontal layout to the form layout

                if data_type=='parameter':
                    # Initialize parameter value
                    create_default_parameter(main_window, widget_name, widget_data['default'])
                else:
                    create_control(main_window, widget_name, widget_data['default'])
                # Set onclick function for toggle button
                def onchange(toggle_widget: ToggleButton, toggle_widget_name, widget_data: dict, *args):
                    toggle_state = toggle_widget.isChecked()
                    if data_type=='parameter':
                        update_parameter(main_window, toggle_widget_name, toggle_state, enable_refresh_frame=toggle_widget.enable_refresh_frame)    
                    elif data_type=='control':
                        update_control(main_window, toggle_widget_name, toggle_state, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.toggled.connect(partial(onchange, widget, widget_name, widget_data))

            elif 'Selection' in widget_name:
                widget = SelectionBox(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, main_window=main_window, default_value=widget_data['default'], selection_values=widget_data['options'])
                if callable(widget_data['options']):
                    widget.addItems(widget_data['options']())
                    widget.setCurrentText(widget_data['default']())
                else:
                    widget.addItems(widget_data['options'])
                    widget.setCurrentText(widget_data['default'])

                widget.reset_default_button = ParameterResetDefaultButton(related_widget=widget)

                horizontal_layout = QtWidgets.QHBoxLayout()
                horizontal_layout.addWidget(label)
                horizontal_layout.addWidget(widget)
                horizontal_layout.addWidget(widget.reset_default_button)

                category_layout.addRow(horizontal_layout)

                if data_type=='parameter':
                    # Initialize parameter value
                    create_default_parameter(main_window, widget_name, widget_data['default'] if not callable(widget_data['default']) else widget_data['default']())
                else:
                    create_control(main_window, widget_name, widget_data['default'] if not callable(widget_data['default']) else widget_data['default']())
                # Set onchange function for select box (Selected value is passed by the signal)
                def onchange(selection_widget: SelectionBox, selection_widget_name, widget_data: dict = {}, selected_value=False):
                    # selected_value = selection_widget.currentText()
                    if data_type=='parameter':
                        update_parameter(main_window, selection_widget_name, selected_value, enable_refresh_frame=selection_widget.enable_refresh_frame)
                    elif data_type=='control':
                        update_control(main_window, selection_widget_name, selected_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.currentTextChanged.connect(partial(onchange, widget, widget_name, widget_data))

            elif 'DecimalSlider' in widget_name:
                widget = ParameterDecimalSlider(
                    label=widget_data['label'], 
                    widget_name=widget_name, 
                    group_layout_data=widgets, 
                    label_widget=label, 
                    min_value=float(widget_data['min_value']),  # Ensure min_value is float
                    max_value=float(widget_data['max_value']),  # Ensure max_value is float
                    default_value=float(widget_data['default']),  # Ensure default_value is float
                    decimals=int(widget_data['decimals']),
                    step_size=float(widget_data['step']),
                    main_window=main_window
                )
                # Use the new ParameterLineDecimalEdit class
                widget.line_edit = ParameterLineDecimalEdit(
                    min_value=float(widget_data['min_value']), 
                    max_value=float(widget_data['max_value']), 
                    default_value=str(widget_data['default']),
                    decimals=int(widget_data['decimals']),  # Ensure it uses decimals place for consistency
                    step_size=float(widget_data['step']),
                    fixed_width=48,
                    max_length=7 if int(widget_data['decimals']) > 1 else 5
                )
                widget.reset_default_button = ParameterResetDefaultButton(related_widget=widget)

                # Layout for widgets
                horizontal_layout = QtWidgets.QHBoxLayout()
                horizontal_layout.addWidget(label)
                horizontal_layout.addWidget(widget)
                horizontal_layout.addWidget(widget.line_edit)
                horizontal_layout.addWidget(widget.reset_default_button)

                category_layout.addRow(horizontal_layout)

                if data_type=='parameter':
                    # Initialize parameter value
                    create_default_parameter(main_window, widget_name, float(widget_data['default']))

                else:
                    create_control(main_window, widget_name, float(widget_data['default']))

                # When slider value changes
                def onchange_slider(slider_widget: ParameterDecimalSlider, slider_widget_name, widget_data: dict, new_value=False):
                    # Update the slider text box value too
                    actual_value = slider_widget.value()  # Get float value from the slider
                    if data_type=='parameter':
                        update_parameter(main_window, slider_widget_name, actual_value, enable_refresh_frame=slider_widget.enable_refresh_frame)
                    elif data_type=='control':
                        update_control(main_window, slider_widget_name, actual_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                    slider_widget.line_edit.set_value(actual_value)  # Update the line edit with the actual value

                # Invece di collegare direttamente onchange_slider, fallo dopo il debounce
                widget.debounce_timer.timeout.connect(partial(onchange_slider, widget, widget_name, widget_data))

                # When line edit value changes
                def onchange_line_edit(slider_widget: ParameterDecimalSlider, slider_widget_name, widget_data, new_value=False):
                    """Handle changes in the line edit and update the slider accordingly."""
                    if not new_value:
                        new_value = 0.0

                    try:
                        # Convert the text box input to float
                        new_value = float(new_value)
                    except ValueError:
                        # If the conversion fails, reset the line edit to the current slider value
                        new_value = slider_widget.value()

                    # Prevent text box value from exceeding the slider range or going below minimum
                    if new_value > (slider_widget.max_value / slider_widget.scale_factor):
                        new_value = slider_widget.max_value / slider_widget.scale_factor
                    elif new_value < (slider_widget.min_value / slider_widget.scale_factor):
                        new_value = slider_widget.min_value / slider_widget.scale_factor

                    # Update the slider's internal value
                    slider_widget.setValue(new_value)
                    # Update the line_edit text to reflect the current value
                    slider_widget.line_edit.set_value(new_value)

                    if data_type=='parameter':
                        update_parameter(main_window, slider_widget_name, new_value, enable_refresh_frame=slider_widget.enable_refresh_frame)
                    elif data_type=='control':
                        update_control(main_window, slider_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))

                widget.line_edit.textChanged.connect(partial(onchange_line_edit, widget, widget_name, widget_data))
 
            elif 'Slider' in widget_name:
                widget = ParameterSlider(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, min_value=widget_data['min_value'], max_value=widget_data['max_value'], default_value=widget_data['default'], step_size=widget_data['step'], main_window=main_window)
                widget.line_edit = ParameterLineEdit(min_value=int(widget_data['min_value']), max_value=int(widget_data['max_value']), default_value=widget_data['default'])
                widget.reset_default_button = ParameterResetDefaultButton(related_widget=widget)
                horizontal_layout = QtWidgets.QHBoxLayout()
                horizontal_layout.addWidget(label)
                horizontal_layout.addWidget(widget)
                horizontal_layout.addWidget(widget.line_edit)
                horizontal_layout.addWidget(widget.reset_default_button)

                category_layout.addRow(horizontal_layout)

                if data_type=='parameter':
                    # Initialize parameter value
                    create_default_parameter(main_window, widget_name, int(widget_data['default']))
                else:
                    create_control(main_window, widget_name, int(widget_data['default']))

                # When slider value is change
                def onchange_slider(slider_widget: ParameterSlider, slider_widget_name, widget_data: dict, new_value=False):
                    if data_type=='parameter':
                        update_parameter(main_window, slider_widget_name, new_value, enable_refresh_frame=slider_widget.enable_refresh_frame)
                    elif data_type=='control':
                        update_control(main_window, slider_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                    # Update the slider text box value too
                    slider_widget.line_edit.setText(str(new_value))

                # Invece di collegare direttamente onchange_slider, fallo dopo il debounce
                widget.debounce_timer.timeout.connect(partial(onchange_slider, widget, widget_name, widget_data))

                # When slider textbox value is changed
                def onchange_line_edit(slider_widget: ParameterSlider, slider_widget_name, widget_data, new_value=False):
                    """Handle changes in the line edit and update the slider accordingly."""
                    if not new_value:
                        new_value = 0
                    try:
                        # Prova a convertire il valore in intero, se fallisce usa il valore corrente dello slider
                        new_value = int(new_value)
                    except ValueError:
                        # Se non è possibile convertire, mantieni il valore corrente dello slider
                        new_value = slider_widget.value()
                    # Prevent the text box value from going above the maximum value of the slider
                    if new_value > slider_widget.max_value:
                        new_value = slider_widget.max_value
                    elif new_value < slider_widget.min_value:
                        new_value = slider_widget.min_value

                    slider_widget.line_edit.set_value(new_value)
                    slider_widget.setValue(int(new_value)) #Update the value of slider too
                    if data_type=='parameter':
                        update_parameter(main_window, slider_widget_name, new_value, enable_refresh_frame=slider_widget.enable_refresh_frame)
                    elif data_type=='control':
                        update_control(main_window, slider_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))

                widget.line_edit.textChanged.connect(partial(onchange_line_edit, widget, widget_name, widget_data))

            elif 'Text' in widget_name:
                widget = ParameterText(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, default_value=widget_data['default'], fixed_width=widget_data['width'], main_window=main_window, data_type=data_type, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                widget.reset_default_button = ParameterResetDefaultButton(related_widget=widget)
                horizontal_layout = QtWidgets.QHBoxLayout()
                horizontal_layout.addWidget(label)
                horizontal_layout.addWidget(widget)
                horizontal_layout.addWidget(widget.reset_default_button)

                category_layout.addRow(horizontal_layout)

                if data_type=='parameter':
                    # Initialize parameter value
                    create_default_parameter(main_window, widget_name, widget_data['default'])

                else:
                    create_control(main_window, widget_name, widget_data['default'])

                 # Handle 'Enter' key press to confirm input
                def on_enter_pressed(text_widget: ParameterText, text_widget_name):
                    # Logic to confirm input or trigger any action when Enter is pressed
                    new_value = text_widget.text()  # Get the current text value
                    if data_type == 'parameter':
                        update_parameter(main_window, text_widget_name, new_value, enable_refresh_frame=text_widget.enable_refresh_frame)
                    else:
                        update_control(main_window, text_widget_name, new_value, exec_function=widget_data.get('exec_function'), exec_function_args=widget_data.get('exec_function_args', []))
                
                # Connect the 'returnPressed' signal to the on_enter_pressed function
                widget.returnPressed.connect(partial(on_enter_pressed, widget, widget_name))

            horizontal_layout.setContentsMargins(spacing_level * 10, 0, 0, 0)

            main_window.parameter_widgets[widget_name] = widget

        layout.addWidget(group_box)

    layoutWidget.addWidget(scroll_area)

    # Default show/hide widgets
    for category, widgets in LAYOUT_DATA.items():
        for widget_name, widget_data in widgets.items():
            widget = main_window.parameter_widgets[widget_name]
            show_hide_related_widgets(main_window, widget, widget_name)

# Function to Hide Elements conditionally from values in LayoutData (Currently supports using Selection box and Toggle button to hide other widgets)
def show_hide_related_widgets(main_window: 'MainWindow', parent_widget: ToggleButton | SelectionBox, parent_widget_name: str, value1=False, value2=False):
    if main_window.parameter_widgets:
        group_layout_data = parent_widget.group_layout_data #Dictionary contaning layout data of all elements in the group of the parent_widget
        if 'Selection' in parent_widget_name:
            # Loop through all widgets data in the parent widget's group layout data
            for widget_name in group_layout_data.keys():
                # Store the widget object (instance) from the parameters_widgets Dictionary
                current_widget = main_window.parameter_widgets.get(widget_name, False)
                # Check if the current_widget depends on the Parent Widget's (selection) value 
                if group_layout_data[widget_name].get('parentSelection', '') == parent_widget_name and current_widget:
                    # Check if the current_widget has the required value of Parent Widget's (selection) current value to hide/show the current_widget
                    if group_layout_data[widget_name].get('requiredSelectionValue') != parent_widget.currentText():
                        current_widget.hide()
                        current_widget.label_widget.hide()
                        current_widget.reset_default_button.hide()
                        if current_widget.line_edit:
                            current_widget.line_edit.hide()
                    else:
                        current_widget.show()
                        current_widget.label_widget.show()
                        current_widget.reset_default_button.show()
                        if current_widget.line_edit:
                            current_widget.line_edit.show()

        elif 'Toggle' in parent_widget_name:
            # Loop through all widgets data in the parent widget's group layout data
            for widget_name in group_layout_data.keys():
                # Store the widget object (instance) from the parameters_widgets Dictionary
                if not widget_name in main_window.parameter_widgets:
                    continue
                current_widget = main_window.parameter_widgets[widget_name]
                # Check if the current_widget depends on the Parent Widget's (toggle) value 
                parentToggles = group_layout_data[widget_name].get('parentToggle', '')
                if parent_widget_name in parentToggles:
                    if ',' in parentToggles:
                        result = [item.strip() for item in parentToggles.split(',')]
                        parentToggle_ischecked = False
                        for index, required_widget_name in enumerate(result):
                            parentToggle_ischecked = main_window.parameter_widgets[required_widget_name].isChecked()
                        # Check if the current_widget has the required toggle value of Parent Widget's (toggle) checked state to hide/show the current_widget
                        if group_layout_data[widget_name].get('requiredToggleValue') != parentToggle_ischecked:
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()
                    elif '|' in parentToggles:
                        result = [item.strip() for item in parentToggles.split('|')]
                        parentToggle_ischecked = True
                        for index, required_widget_name in enumerate(result):
                            ischecked = main_window.parameter_widgets[required_widget_name].isChecked()
                            if not ischecked:
                                parentToggle_ischecked = False
                        # Check if the current_widget has the required toggle value of Parent Widget's (toggle) checked state to hide/show the current_widget
                        if group_layout_data[widget_name].get('requiredToggleValue') != parentToggle_ischecked:
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()
                    else:
                        parentToggle_ischecked = main_window.parameter_widgets[parentToggles].isChecked()
                        if group_layout_data[widget_name].get('requiredToggleValue') != parentToggle_ischecked:
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()

            parent_widget.start_animation()
                    

def create_control(main_window: 'MainWindow', control_name, control_value):
    main_window.control[control_name] = control_value

def update_control(main_window: 'MainWindow', control_name, control_value, exec_function=None, exec_function_args=[]):
    if exec_function:
        # Only execute the function if the value is different from current
        if main_window.control[control_name] != control_value:
            # By default an exec function definition should have atleast one parameter : MainWindow
            exec_function_args = [main_window, control_value] + exec_function_args
            exec_function(*exec_function_args)
    main_window.control[control_name] = control_value
    refresh_frame(main_window)


def create_default_parameter(main_window: 'MainWindow', parameter_name, parameter_value):
    main_window.default_parameters[parameter_name] = parameter_value

def create_parameter_dict_for_face_id(main_window: 'MainWindow', face_id=0):
    if not main_window.parameters.get(face_id):
        parameters =  main_window.parameters.get(main_window.selected_target_face_id) or main_window.default_parameters
        main_window.parameters[face_id] = parameters.copy()
    print("Created parameter_dict_for_face_id", face_id)

# def create_parameter(main_window: 'MainWindow', parameter_name, parameter_value, face_id=False):
#     if not main_window.parameters.get(face_id):
#         create_parameter_dict_using_face_id(main_window, face_id)
#     main_window.parameters[parameter_name] = parameter_value

def update_parameter(main_window: 'MainWindow', parameter_name, parameter_value, enable_refresh_frame=True):
    if main_window.target_faces:
        face_id = main_window.selected_target_face_id
        main_window.parameters[face_id][parameter_name] = parameter_value

        if enable_refresh_frame:
            refresh_frame(main_window)

def refresh_frame(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

def create_and_show_messagebox(main_window: 'MainWindow', window_title: str, message: str, parent_widget: QtWidgets.QWidget):
    messagebox = QtWidgets.QMessageBox(parent_widget)
    messagebox.setWindowTitle(window_title)
    messagebox.setText(message)
    messagebox.exec_()

def create_and_add_embed_button_to_list(main_window: 'MainWindow', embedding_name, embedding_store):
    inputEmbeddingsList = main_window.inputEmbeddingsList
    # Passa l'intero embedding_store
    embed_button = EmbeddingCardButton(main_window=main_window, embedding_name=embedding_name, embedding_store=embedding_store)

    button_size = qtc.QSize(120, 30)  # Imposta una dimensione fissa per i pulsanti
    embed_button.setFixedSize(button_size)
    
    list_item = QtWidgets.QListWidgetItem(inputEmbeddingsList)
    list_item.setSizeHint(button_size)
    embed_button.list_item = list_item
    list_item.setTextAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
    
    inputEmbeddingsList.setItemWidget(list_item, embed_button)
    
    # Aggiungi padding attorno ai pulsanti
    grid_size_with_padding = button_size + qtc.QSize(4, 4)
    inputEmbeddingsList.setGridSize(grid_size_with_padding)  # Add padding around the buttons
    inputEmbeddingsList.setWrapping(True)  # Set grid size with padding
    inputEmbeddingsList.setFlow(QtWidgets.QListView.LeftToRight)  # Set flow direction
    inputEmbeddingsList.setResizeMode(QtWidgets.QListView.Adjust)  # Adjust layout automatically

    main_window.merged_embeddings.append(embed_button)

def open_embeddings_from_file(main_window: 'MainWindow'):
    embedding_filename, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, filter='JSON (*.json)')
    if embedding_filename:
        with open(embedding_filename, 'r') as embed_file:
            embeddings_list = json.load(embed_file)
            clear_merged_embeddings(main_window)

            # Reset per ogni target face
            for target_face in main_window.target_faces:
                target_face.assigned_embed_buttons = {}
                target_face.assigned_input_embedding = {}

            # Carica gli embedding dal file e crea il dizionario embedding_store
            for embed_data in embeddings_list:
                embedding_store = embed_data.get('embedding_store', {})
                # Converte ogni embedding in numpy array
                for recogn_model, embed in embedding_store.items():
                    embedding_store[recogn_model] = numpy.array(embed)

                # Passa l'intero embedding_store alla funzione
                widget_actions.create_and_add_embed_button_to_list(
                    main_window, 
                    embed_data['name'], 
                    embedding_store  # Passa l'intero embedding_store
                )

    main_window.loaded_embedding_filename = embedding_filename or main_window.loaded_embedding_filename

def save_embeddings_to_file(main_window: 'MainWindow', save_as=False):
    if not main_window.merged_embeddings:
        create_and_show_messagebox(main_window, 'Embeddings List Empty!', 'No Embeddings available to save', parent_widget=main_window)
        return

    # Definisce il nome del file di salvataggio
    embedding_filename = main_window.loaded_embedding_filename
    if not embedding_filename or save_as:
        embedding_filename, _ = QtWidgets.QFileDialog.getSaveFileName(main_window, filter='JSON (*.json)')

    # Crea una lista di dizionari, ciascuno con il nome dell'embedding e il relativo embedding_store
    embeddings_list = [
        {
            'name': embed_button.embedding_name,
            'embedding_store': {k: v.tolist() for k, v in embed_button.embedding_store.items()}  # Converti gli embedding in liste
        }
        for embed_button in main_window.merged_embeddings
    ]

    # Salva su file
    if embedding_filename:
        with open(embedding_filename, 'w') as embed_file:
            embeddings_as_json = json.dumps(embeddings_list, indent=4)  # Salva con indentazione per leggibilità
            embed_file.write(embeddings_as_json)

            # Mostra un messaggio di conferma
            create_and_show_toast_message(main_window, 'Embeddings Saved', f'Saved Embeddings to file: {embedding_filename}')

        main_window.loaded_embedding_filename = embedding_filename

def create_and_show_toast_message(main_window: 'MainWindow', title: str, message: str, style_type='information'):
    style_preset_map = {
        'success': ToastPreset.SUCCESS,
        'warning': ToastPreset.WARNING,
        'error': ToastPreset.ERROR,
        'information': ToastPreset.INFORMATION,
        'success_dark': ToastPreset.SUCCESS_DARK,
        'warning_dark': ToastPreset.WARNING_DARK,
        'error_dark': ToastPreset.ERROR_DARK,
        'information_dark': ToastPreset.INFORMATION_DARK,
    }
    toast = Toast(main_window)
    toast.setTitle(title)
    toast.setText(message)
    toast.setDuration(1400)
    toast.setPosition(ToastPosition.TOP_RIGHT)  # Default: ToastPosition.BOTTOM_RIGHT
    toast.applyPreset(style_preset_map[style_type])  # Apply style preset
    toast.show()


def set_widgets_values_using_face_id_parameters(main_window: 'MainWindow', face_id=False):
    if face_id is False:
        print("Set widgets values using default parameters")
        parameters = main_window.default_parameters
    else:
        print(f"Set widgets values using face_id {face_id}")
        parameters = main_window.parameters[face_id].copy()
    parameter_widgets = main_window.parameter_widgets
    for parameter_name, parameter_value in parameters.items():
        # temporarily disable refreshing the frame to prevent slowing due to unnecessary processing
        parameter_widgets[parameter_name].enable_refresh_frame = False
        parameter_widgets[parameter_name].set_value(parameter_value)
        parameter_widgets[parameter_name].enable_refresh_frame = True

def view_fullscreen(main_window: 'MainWindow'):
    if main_window.is_full_screen:
        main_window.showNormal()  # Exit full-screen mode
    else:
        main_window.showFullScreen()  # Enter full-screen mode
    main_window.is_full_screen = not main_window.is_full_screen



def enable_zoom_and_pan(view: QtWidgets.QGraphicsView):
    SCALE_FACTOR = 1.5
    view._zoom = 0  # Track zoom level
    view._last_scale_factor = 1.0  # Track the last scale factor (1.0 = no scaling)

    def zoom(self, step=False):
        """Zoom in or out by a step."""
        if not step:
            factor = self._last_scale_factor
        else:
            self._zoom += step
            factor = SCALE_FACTOR ** step
            self._last_scale_factor *= factor  # Update the last scale factor
        if factor > 0:
            self.scale(factor, factor)

    def wheelEvent(self, event):
        """Handle mouse wheel event for zooming."""
        delta = event.angleDelta().y()
        if delta != 0:
            zoom(self, delta // abs(delta))
    
    def resetZoom(self):
        print("Called resetZoom()")
        """Reset zoom level to fit the view."""
        self._zoom = 0
        if not self.scene():
            return
        items = self.scene().items()
        if not items:
            return
        rect = self.scene().itemsBoundingRect()
        self.setSceneRect(rect)
        unity = self.transform().mapRect(qtc.QRectF(0, 0, 1, 1))
        self.scale(1 / unity.width(), 1 / unity.height())
        view_rect = self.viewport().rect()
        scene_rect = self.transform().mapRect(rect)
        factor = min(view_rect.width() / scene_rect.width(),
                    view_rect.height() / scene_rect.height())
        self.scale(factor, factor)

    # Attach methods to the view
    view.zoom = zoom.__get__(view)
    view.resetZoom = resetZoom.__get__(view)
    view.wheelEvent = wheelEvent.__get__(view)

    # Set anchors for better interaction
    view.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    view.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)