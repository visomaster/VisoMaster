
import PySide6.QtCore as qtc
from PySide6 import QtWidgets, QtGui
import time
import App.Helpers.Misc_Helpers as misc_helpers 
import App.UI.Widgets.UI_Workers as ui_workers
from App.UI.Widgets.WidgetComponents import TargetMediaCardButton, ProgressDialog, TargetFaceCardButton, InputFaceCardButton, FormGroupBox, ToggleButton, SelectionBox, ParameterSlider, ParameterLineEdit, ParameterResetDefaultButton
from PySide6.QtWidgets import QComboBox

import App.UI.Widgets.WidgetActions as widget_actions 
from functools import partial
import cv2
from App.UI.Core import media_rc
import torch
import numpy
from App.UI.Widgets.LayoutData import SWAPPER_LAYOUT_DATA
from typing import TYPE_CHECKING

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

    # Stop the current video if it's playing
    '''
    video_processor = main_window.video_processor
    if video_processor.processing:
        print("Stopping the current video before loading a new video or image.")
        video_processor.stop_processing()
    '''

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

    # Stop the current video if it's playing
    '''
    video_processor = main_window.video_processor
    if video_processor.processing:
        print("Stopping the current video before loading a new video or image.")
        video_processor.stop_processing()
    '''

    clear_stop_loading_input_media(main_window)
    clear_input_faces(main_window)
    main_window.selected_input_face_buttons = []
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(main_window=main_window, folder_name=folder_name, files_list=files_list)
    main_window.input_faces_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_source_faces_list, main_window))
    main_window.input_faces_loader_worker.start()

    
@qtc.Slot(int)
def OnChangeSlider(main_window: 'MainWindow', new_position=0):
    video_processor = main_window.video_processor
    was_processing = video_processor.processing

    if was_processing:
        print("OnChangeSlider: Processing in progress. Stopping current processing.")
        video_processor.stop_processing()

    video_processor.current_frame_number = new_position
    if video_processor.media_capture:
        video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
        ret, frame = video_processor.media_capture.read()
        if ret:
            pixmap = widget_actions.get_pixmap_from_frame(main_window, frame)
            widget_actions.update_graphics_view(main_window, pixmap, new_position)

    # Do not automatically restart the video, let the user press Play to resume
    print("OnChangeSlider: Video stopped after slider movement.")

# Functions to add Buttons with thumbnail for selecting videos/images and faces
@qtc.Slot(str, QtGui.QPixmap)
def add_media_thumbnail_to_target_videos_list(main_window: 'MainWindow', media_path, pixmap, file_type):
    add_media_thumbnail_button(TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, pixmap, media_path=media_path, file_type=file_type)

@qtc.Slot()
def add_media_thumbnail_to_target_faces_list(main_window: 'MainWindow', cropped_face, embedding, pixmap):
    add_media_thumbnail_button(TargetFaceCardButton, main_window.targetFacesList, main_window.target_faces, pixmap, cropped_face=cropped_face, embedding=embedding )

@qtc.Slot()
def add_media_thumbnail_to_source_faces_list(main_window: 'MainWindow', media_path, cropped_face, embedding, pixmap):
    add_media_thumbnail_button(InputFaceCardButton, main_window.inputFacesList, main_window.input_faces, pixmap, media_path=media_path, cropped_face=cropped_face, embedding=embedding )


def add_media_thumbnail_button(buttonClass:QtWidgets.QPushButton, listWidget:QtWidgets.QListWidget, buttons_list:list, pixmap, **kwargs):
    if buttonClass==TargetMediaCardButton:
        constructor_args = (kwargs.get('media_path'), kwargs.get('file_type'))
    elif buttonClass in (TargetFaceCardButton, InputFaceCardButton):
        constructor_args = (kwargs.get('media_path',''), kwargs.get('cropped_face'), kwargs.get('embedding'))
    button_size = qtc.QSize(70, 70)  # Set a fixed size for the buttons
    button = buttonClass(*constructor_args)
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

    if not isinstance(frame, bool):
        # Convert the frame to QPixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(70, 70, qtc.Qt.AspectRatioMode.KeepAspectRatio)  # Adjust size as needed
        return pixmap
    return None

# from App.UI.MainUI import Ui_MainWindow
def update_graphics_view(main_window: 'MainWindow' , pixmap, current_frame_number):
    print(current_frame_number)
    main_window.videoSeekSlider.blockSignals(True)
    main_window.videoSeekSlider.setValue(current_frame_number)
    main_window.videoSeekSlider.blockSignals(False)
    # print(main_window.graphicsViewFrame.scene, pixmap)
    main_window.graphicsViewFrame.scene().clear()
    pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
    main_window.graphicsViewFrame.scene().addItem(pixmap_item)
    # Optionally fit the image to the view
    fit_image_to_view(main_window, pixmap_item)

def fit_image_to_view(main_window: 'MainWindow', pixmap_item: QtWidgets.QGraphicsPixmapItem):
    graphicsViewFrame = main_window.graphicsViewFrame
    # Reset the transform to ensure no previous transformations affect the new fit
    graphicsViewFrame.resetTransform()
    # Set the scene rectangle to the bounding rectangle of the pixmap item
    graphicsViewFrame.setSceneRect(pixmap_item.boundingRect())
    # Fit the image to the view, keeping the aspect ratio
    graphicsViewFrame.fitInView(pixmap_item, qtc.Qt.AspectRatioMode.KeepAspectRatio)
    graphicsViewFrame.update()

def get_pixmap_from_frame(main_window: 'MainWindow', frame, scale=True):
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

    if scale:
        pixmap = scale_pixmap_to_view(main_window.graphicsViewFrame, pixmap)
    return pixmap

def scale_pixmap_to_view(view: QtWidgets.QGraphicsView, pixmap: QtGui.QPixmap):
    # Get the size of the view
    view_size = view.viewport().size()
    pixmap_size = pixmap.size()
    # Calculate the scale factor
    scale_factor = min(view_size.width() / pixmap_size.width(), view_size.height() / pixmap_size.height())
    # Scale the pixmap
    scaled_pixmap = pixmap.scaled(
        pixmap_size.width() * scale_factor,
        pixmap_size.height() * scale_factor,
        qtc.Qt.AspectRatioMode.KeepAspectRatio
    )
    return scaled_pixmap

def OnClickPlayButton(main_window: 'MainWindow', checked):
    video_processor = main_window.video_processor
    if checked:
        if video_processor.processing:
            print("OnClickPlayButton: Video already playing. Stopping the current video before starting a new one.")
            video_processor.stop_processing()
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
    main_window.buttonMediaPlay.setChecked(False)
    setPlayButtonIcon(main_window)

def setPlayButtonIcon(main_window: 'MainWindow'):
    if main_window.buttonMediaPlay.isChecked(): 
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_on.png"))
        main_window.buttonMediaPlay.setToolTip("Stop")
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_off.png"))
        main_window.buttonMediaPlay.setToolTip("Play")

def filterTargetVideos(main_window: 'MainWindow', search_text: str):
    search_text = search_text.lower()
    if search_text:
        for i in range(main_window.targetVideosList.count()):
            list_item = main_window.targetVideosList.item(i)
            if search_text not in main_window.target_videos[i].media_path.lower():
                list_item.setHidden(True)

            else:
                list_item.setHidden(False)


    else:
        for i in range(main_window.targetVideosList.count()):
            main_window.targetVideosList.item(i).setHidden(False)

def filterInputFaces(main_window: 'MainWindow', search_text: str):
    search_text = search_text.lower()
    if search_text:
        for i in range(main_window.inputFacesList.count()):
            list_item = main_window.inputFacesList.item(i)
            if search_text not in main_window.input_faces[i].media_path.lower():
                list_item.setHidden(True)

            else:
                list_item.setHidden(False)


    else:
        for i in range(main_window.inputFacesList.count()):
            main_window.inputFacesList.item(i).setHidden(False)

def initializeModelLoadDialog(main_window: 'MainWindow'):
    main_window.model_load_dialog = ProgressDialog("Loading Models...This is gonna take a while.", "Cancel", 0, 100, main_window)
    main_window.model_load_dialog.setWindowModality(qtc.Qt.ApplicationModal)
    main_window.model_load_dialog.setMinimumDuration(2000)
    main_window.model_load_dialog.setWindowTitle("Loading Models")
    main_window.model_load_dialog.setAutoClose(True)  # Close the dialog when finished
    main_window.model_load_dialog.setCancelButton(None)
    main_window.model_load_dialog.setWindowFlag(qtc.Qt.WindowCloseButtonHint, False)
    main_window.model_load_dialog.setValue(0)
    main_window.model_load_dialog.close()

def find_target_faces(main_window: 'MainWindow'):
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
        
        # print(frame)
        img = torch.from_numpy(frame.astype('uint8')).to('cuda')
        img = img.permute(2,0,1)
        bboxes, kpss_5, _ = main_window.models_processor.run_detect(img,max_num=50)

        ret = []
        for face_kps in kpss_5:
            face_emb, cropped_img = main_window.models_processor.run_recognize(img, face_kps)
            ret.append([face_kps, face_emb, cropped_img])

        if ret:
            # Apply threshold tolerence
            threshhold = 50
            # if self.parameters["ThresholdState"]:
            if 1:
                threshhold = 60

            # Loop through all faces in video frame
            for face in ret:
                found = False
                # Check if this face has already been found
                for target_face in main_window.target_faces:
                    if main_window.models_processor.findCosineDistance(target_face.embedding, face[1]) >= threshhold:
                        found = True
                        break
                if not found:
                    face_img = numpy.ascontiguousarray(face[2].cpu().numpy())
                    # crop = cv2.resize(face[2].cpu().numpy(), (82, 82))
                    pixmap = get_pixmap_from_frame(main_window, face_img)
                    add_media_thumbnail_to_target_faces_list(main_window, face_img, face[1], pixmap)

def clear_target_faces(main_window: 'MainWindow'):
    main_window.targetFacesList.clear()
    for target_face in main_window.target_faces:
        del target_face
    main_window.target_faces = []
    main_window.selected_target_face_buttons = []

def clear_input_faces(main_window: 'MainWindow'):
    main_window.inputFacesList.clear()
    for target_face in main_window.input_faces:
        del target_face
    main_window.input_faces = []
    main_window.selected_input_face_buttons = []


def add_parameter_widgets(main_window: 'MainWindow', LAYOUT_DATA: dict, layoutWidget: QtWidgets.QVBoxLayout):
    layout = QtWidgets.QVBoxLayout()
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_content = QtWidgets.QWidget()
    scroll_content.setLayout(layout)
    scroll_area.setWidget(scroll_content)

    for category, widgets in LAYOUT_DATA.items():
        group_box = FormGroupBox(main_window, title=category)
        group_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        category_layout = QtWidgets.QFormLayout()
        group_box.setLayout(category_layout)

        for widget_name, widget_data in widgets.items():
            spacing_level = widget_data['level']
            label = QtWidgets.QLabel(widget_data['label'])

            # Create a horizontal layout for the toggle button and its label
            if 'Toggle' in widget_name:
                widget = ToggleButton(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=None, main_window=main_window)
                widget.setChecked(widget_data['default'])
                # Create a horizontal layout
                horizontal_layout = QtWidgets.QHBoxLayout()
                # In case of toggle button, add show widget first, then its label
                horizontal_layout.addWidget(widget)  # Add the toggle button
                horizontal_layout.addWidget(label)  # Add the label
                
                category_layout.addRow(horizontal_layout)  # Add the horizontal layout to the form layout

                # Initialize parameter value
                create_parameter(main_window, widget_name, widget_data['default'])
                # Set onclick function for toggle button
                def onchange(toggle_widget: ToggleButton, toggle_widget_name):
                    toggle_state = toggle_widget.isChecked()
                    update_parameter(main_window, toggle_widget_name, toggle_state)    
                widget.clicked.connect(partial(onchange, widget, widget_name))

            elif 'Selection' in widget_name:
                widget = SelectionBox(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, main_window=main_window, default_value=widget_data['default'])
                widget.addItems(widget_data['options'])
                widget.setCurrentText(widget_data['default'])

                widget.reset_default_button = ParameterResetDefaultButton(related_widget=widget)

                horizontal_layout = QtWidgets.QHBoxLayout()
                horizontal_layout.addWidget(label)
                horizontal_layout.addWidget(widget)
                horizontal_layout.addWidget(widget.reset_default_button)

                category_layout.addRow(horizontal_layout)

                # Initialize parameter value
                create_parameter(main_window, widget_name, widget_data['default'])
                # Set onchange function for select box (Selected value is passed by the signal)
                def onchange(selection_widget: SelectionBox, selection_widget_name, selected_value=False):
                    # selected_value = selection_widget.currentText()
                    update_parameter(main_window, selection_widget_name, selected_value)
                widget.currentTextChanged.connect(partial(onchange, widget, widget_name))

            elif 'Slider' in widget_name:
                widget = ParameterSlider(label=widget_data['label'], widget_name=widget_name, group_layout_data=widgets, label_widget=label, min_value=widget_data['min_value'], max_value=widget_data['max_value'], default_value=widget_data['default'], main_window=main_window)
                widget.line_edit = ParameterLineEdit(min_value=int(widget_data['min_value']), max_value=int(widget_data['max_value']), default_value=widget_data['default'])
                widget.reset_default_button = ParameterResetDefaultButton(related_widget=widget)
                horizontal_layout = QtWidgets.QHBoxLayout()
                horizontal_layout.addWidget(label)
                horizontal_layout.addWidget(widget)
                horizontal_layout.addWidget(widget.line_edit)
                horizontal_layout.addWidget(widget.reset_default_button)

                category_layout.addRow(horizontal_layout)

                # Initialize parameter value
                create_parameter(main_window, widget_name, int(widget_data['default']))

                # When slider value is change
                def onchange_slider(slider_widget: ParameterSlider, slider_widget_name, new_value=False):
                    update_parameter(main_window, slider_widget_name, new_value)
                    # Update the slider text box value too
                    slider_widget.line_edit.setText(str(new_value))
                widget.valueChanged.connect(partial(onchange_slider, widget, widget_name))

                # When slider textbox value is changed
                def onchange_line_edit(slider_widget: ParameterSlider, slider_widget_name, new_value=False):
                    if not new_value:
                        new_value = 0
                    new_value = int(new_value) #Text box value is sent as str by default
                    update_parameter(main_window, slider_widget_name, new_value)
                    # Prevent the text box value from going above the maximum value of the slider
                    if new_value>slider_widget.max_value and new_value>0:
                        new_value = new_value % (slider_widget.max_value+1)
                        slider_widget.line_edit.setText(str(new_value))
                    slider_widget.setValue(int(new_value)) #Update the value of slider too
                widget.line_edit.textChanged.connect(partial(onchange_line_edit, widget, widget_name))

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
def show_hide_related_widgets(main_window: 'MainWindow', parent_widget: ToggleButton | QComboBox, parent_widget_name: str, value1=False, value2=False):
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
                    else:
                        current_widget.show()
                        current_widget.label_widget.show()
                        current_widget.reset_default_button.show()

        elif 'Toggle' in parent_widget_name:
            # Loop through all widgets data in the parent widget's group layout data
            for widget_name in group_layout_data.keys():
                # Store the widget object (instance) from the parameters_widgets Dictionary
                current_widget = main_window.parameter_widgets[widget_name]
                # Check if the current_widget depends on the Parent Widget's (toggle) value 
                if group_layout_data[widget_name].get('parentToggle', '') == parent_widget_name:
                    # Check if the current_widget has the required toggle value of Parent Widget's (toggle) checked state to hide/show the current_widget
                    if group_layout_data[widget_name].get('requiredToggleValue') != parent_widget.isChecked():
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

def create_parameter(main_window: 'MainWindow', parameter_name, parameter_value):
    main_window.parameters[parameter_name] = parameter_value

def update_parameter(main_window: 'MainWindow', parameter_name, parameter_value):
    main_window.parameters[parameter_name] = parameter_value
    refresh_frame(main_window)

def refresh_frame(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()