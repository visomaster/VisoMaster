
import PySide6.QtCore as qtc
from PySide6 import QtWidgets, QtGui
import time
import App.Helpers.Misc_Helpers as misc_helpers 
import App.UI.Widgets.UI_Workers as ui_workers
from App.UI.Widgets.WidgetComponents import TargetMediaCardButton, ProgressDialog, TargetFaceCardButton, InputFaceCardButton
import App.UI.Widgets.WidgetActions as widget_actions 
from functools import partial
import cv2
from App.UI.Core import media_rc
import torch
import numpy
def scale_pixmap_to_view(view, pixmap):
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

def fit_image_to_view(main_window, pixmap_item):
    graphicsViewFrame = main_window.graphicsViewFrame
    # Reset the transform to ensure no previous transformations affect the new fit
    graphicsViewFrame.resetTransform()
    # Set the scene rectangle to the bounding rectangle of the pixmap item
    graphicsViewFrame.setSceneRect(pixmap_item.boundingRect())
    # Fit the image to the view, keeping the aspect ratio
    graphicsViewFrame.fitInView(pixmap_item, qtc.Qt.AspectRatioMode.KeepAspectRatio)
    graphicsViewFrame.update()

def clear_stop_loading_target_media(main_window):
    if main_window.video_loader_worker:
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        time.sleep(0.5)
        main_window.targetVideosList.clear()

@qtc.Slot()
def onClickSelectTargetVideosFolder(main_window):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory()
    main_window.selected_video_buttons = []
    main_window.target_videos = []
    main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
    main_window.labelTargetVideosPath.setToolTip(folder_name)
    clear_stop_loading_target_media(main_window)
    clear_target_faces(main_window)
    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(folder_name=folder_name)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_target_videos_list, main_window))
    main_window.video_loader_worker.start()

@qtc.Slot()
def onClickSelectTargetVideosFiles(main_window):
    files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
    main_window.selected_video_buttons = []
    main_window.target_videos = []
    main_window.labelTargetVideosPath.setText('Selected Files')
    main_window.labelTargetVideosPath.setToolTip('Selected Files')
    clear_stop_loading_target_media(main_window)
    clear_target_faces(main_window)
    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(files_list=files_list)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_target_videos_list, main_window))
    main_window.video_loader_worker.start()


def clear_stop_loading_input_media(main_window):
    if main_window.input_faces_loader_worker:
        main_window.input_faces_loader_worker.terminate()
        main_window.input_faces_loader_worker = False
        time.sleep(0.5)
        main_window.inputFacesList.clear()

@qtc.Slot()
def onClickSelectInputImagesFolder(main_window):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory()
    main_window.selected_input_face_buttons = []
    main_window.labelInputFacesPath.setText(misc_helpers.truncate_text(folder_name))
    main_window.labelInputFacesPath.setToolTip(folder_name)
    clear_stop_loading_input_media(main_window)
    clear_input_faces(main_window)
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(main_window=main_window, folder_name=folder_name)
    main_window.input_faces_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_source_faces_list, main_window))
    main_window.input_faces_loader_worker.start()

    
@qtc.Slot()
def OnChangeSlider(main_window, new_position=0):
    video_processor = main_window.video_processor
    video_processor.stop_processing()
    video_processor.current_frame_number = new_position
    if video_processor.media_capture:
        video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
    video_processor.processing=True
    video_processor.create_threads(threads_count=1)
    video_processor.process_next_frame()
    widget_actions.resetMediaButtons(main_window)


@qtc.Slot(str, QtGui.QPixmap)
def add_media_thumbnail_to_target_videos_list(main_window, media_path, pixmap, file_type):
    add_media_thumbnail_button(TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, pixmap, media_path=media_path, file_type=file_type)


@qtc.Slot()
def add_media_thumbnail_to_target_faces_list(main_window, cropped_face, embedding, pixmap):
    add_media_thumbnail_button(TargetFaceCardButton, main_window.targetFacesList, main_window.target_faces, pixmap, cropped_face=cropped_face, embedding=embedding )

@qtc.Slot()
def add_media_thumbnail_to_source_faces_list(main_window, cropped_face, embedding, pixmap):
    add_media_thumbnail_button(InputFaceCardButton, main_window.inputFacesList, main_window.input_faces, pixmap, cropped_face=cropped_face, embedding=embedding )


def add_media_thumbnail_button(buttonClass:QtWidgets.QPushButton, listWidget, buttons_list, pixmap, **kwargs):
    if buttonClass==TargetMediaCardButton:
        constructor_args = (kwargs.get('media_path'), kwargs.get('file_type'))
    elif buttonClass in (TargetFaceCardButton, InputFaceCardButton):
        constructor_args = (kwargs.get('cropped_face'), kwargs.get('embedding'))
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
def update_graphics_view(main_window , pixmap, current_frame_number):
    print(current_frame_number)
    main_window.videoSeekSlider.blockSignals(True)
    main_window.videoSeekSlider.setValue(current_frame_number)
    main_window.videoSeekSlider.blockSignals(False)
    # print(main_window.graphicsViewFrame.scene, pixmap)
    main_window.graphicsViewFrame.scene().clear()
    pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
    main_window.graphicsViewFrame.scene().addItem(pixmap_item)
    # Optionally fit the image to the view
    widget_actions.fit_image_to_view(main_window, pixmap_item)


def get_pixmap_from_frame(main_window, frame, scale=True):
    height, width, channel = frame.shape
    bytes_per_line = 3 * width
    q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(q_img)

    if scale:
        pixmap = widget_actions.scale_pixmap_to_view(main_window.graphicsViewFrame, pixmap)
    return pixmap

def resetMediaButtons(main_window):
    main_window.buttonMediaPlay.setChecked(False)
    setPlayButtonIcon(main_window)

def setPlayButtonIcon(main_window):
    if main_window.buttonMediaPlay.isChecked(): 
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_on.png"))
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_off.png"))

def OnClickPlayButton(main_window):
    setPlayButtonIcon(main_window)
    main_window.video_processor.process_video()

def filterTargetVideos(main_window, search_text):
    # QtWidgets.QListWidgetItem.hide
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

def initializeModelLoadDialog(main_window):
    main_window.model_load_dialog = ProgressDialog("Loading Models...This is gonna take a while.", "Cancel", 0, 100, main_window)
    main_window.model_load_dialog.setWindowModality(qtc.Qt.ApplicationModal)
    main_window.model_load_dialog.setMinimumDuration(2000)
    main_window.model_load_dialog.setWindowTitle("Loading Models")
    main_window.model_load_dialog.setAutoClose(True)  # Close the dialog when finished
    main_window.model_load_dialog.setCancelButton(None)
    main_window.model_load_dialog.setWindowFlag(qtc.Qt.WindowCloseButtonHint, False)
    main_window.model_load_dialog.setValue(0)
    main_window.model_load_dialog.close()

def find_target_faces(main_window):
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

def clear_target_faces(main_window):
    main_window.targetFacesList.clear()
    for target_face in main_window.target_faces:
        del target_face
    main_window.target_faces = []
    main_window.selected_target_face_buttons = []

def clear_input_faces(main_window):
    main_window.inputFacesList.clear()
    for target_face in main_window.input_faces:
        del target_face
    main_window.input_faces = []
    main_window.selected_input_face_buttons = []