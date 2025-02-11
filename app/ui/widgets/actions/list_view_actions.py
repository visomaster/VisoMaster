import time
from functools import partial
from typing import TYPE_CHECKING

from PySide6 import QtWidgets, QtGui, QtCore

from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets import widget_components
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets import ui_workers
if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QPixmap)
def add_media_thumbnail_to_target_videos_list(main_window: 'MainWindow', media_path, pixmap, file_type, media_id):
    add_media_thumbnail_button(main_window, widget_components.TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, pixmap, media_path=media_path, file_type=file_type, media_id=media_id)

# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QPixmap, str, int, int)
def add_webcam_thumbnail_to_target_videos_list(main_window: 'MainWindow', media_path, pixmap, file_type, media_id, webcam_index, webcam_backend):
    add_media_thumbnail_button(main_window, widget_components.TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, pixmap, media_path=media_path, file_type=file_type, media_id=media_id, is_webcam=True, webcam_index=webcam_index, webcam_backend=webcam_backend)

@QtCore.Slot()
def add_media_thumbnail_to_target_faces_list(main_window: 'MainWindow', cropped_face, embedding_store, pixmap, face_id):
    add_media_thumbnail_button(main_window, widget_components.TargetFaceCardButton, main_window.targetFacesList, main_window.target_faces, pixmap, cropped_face=cropped_face, embedding_store=embedding_store, face_id=face_id )

@QtCore.Slot()
def add_media_thumbnail_to_source_faces_list(main_window: 'MainWindow', media_path, cropped_face, embedding_store, pixmap, face_id):
    add_media_thumbnail_button(main_window, widget_components.InputFaceCardButton, main_window.inputFacesList, main_window.input_faces, pixmap, media_path=media_path, cropped_face=cropped_face, embedding_store=embedding_store, face_id=face_id )


def add_media_thumbnail_button(main_window: 'MainWindow', buttonClass: 'widget_components.CardButton', listWidget:QtWidgets.QListWidget, buttons_list:list, pixmap, **kwargs):
    if buttonClass==widget_components.TargetMediaCardButton:
        constructor_args = (kwargs.get('media_path'), kwargs.get('file_type'), kwargs.get('media_id'))
        if kwargs.get('is_webcam'):
            constructor_args+=(kwargs.get('is_webcam'), kwargs.get('webcam_index'), kwargs.get('webcam_backend'))
    elif buttonClass in (widget_components.TargetFaceCardButton, widget_components.InputFaceCardButton):
        constructor_args = (kwargs.get('media_path',''), kwargs.get('cropped_face'), kwargs.get('embedding_store'), kwargs.get('face_id'))
    if buttonClass==widget_components.TargetMediaCardButton:
        button_size = QtCore.QSize(90, 90)  # Set a fixed size for the buttons
    else:
        button_size = QtCore.QSize(70, 70)  # Set a fixed size for the buttons

    button: widget_components.CardButton = buttonClass(*constructor_args, main_window=main_window)
    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(button_size - QtCore.QSize(3, 3))  # Slightly smaller than the button size to add some margin
    button.setFixedSize(button_size)
    button.setCheckable(True)
    if buttonClass in [widget_components.TargetFaceCardButton, widget_components.InputFaceCardButton]:
        buttons_list[button.face_id] = button
    elif buttonClass == widget_components.TargetMediaCardButton:
        buttons_list[button.media_id] = button
    elif buttonClass == widget_components.EmbeddingCardButton:
        buttons_list[button.embedding_id] = button
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


def create_and_add_embed_button_to_list(main_window: 'MainWindow', embedding_name, embedding_store, embedding_id):
    inputEmbeddingsList = main_window.inputEmbeddingsList
    # Passa l'intero embedding_store
    embed_button = widget_components.EmbeddingCardButton(main_window=main_window, embedding_name=embedding_name, embedding_store=embedding_store, embedding_id=embedding_id)

    button_size = QtCore.QSize(120, 30)  # Imposta una dimensione fissa per i pulsanti
    embed_button.setFixedSize(button_size)
    
    list_item = QtWidgets.QListWidgetItem(inputEmbeddingsList)
    list_item.setSizeHint(button_size)
    embed_button.list_item = list_item
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    
    inputEmbeddingsList.setItemWidget(list_item, embed_button)
    
    # Aggiungi padding attorno ai pulsanti
    grid_size_with_padding = button_size + QtCore.QSize(4, 4)
    inputEmbeddingsList.setGridSize(grid_size_with_padding)  # Add padding around the buttons
    inputEmbeddingsList.setWrapping(True)  # Set grid size with padding
    inputEmbeddingsList.setFlow(QtWidgets.QListView.LeftToRight)  # Set flow direction
    inputEmbeddingsList.setResizeMode(QtWidgets.QListView.Adjust)  # Adjust layout automatically

    main_window.merged_embeddings[embed_button.embedding_id] = embed_button

def clear_stop_loading_target_media(main_window: 'MainWindow'):
    if main_window.video_loader_worker:
        main_window.video_loader_worker.stop()
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        time.sleep(0.5)
        main_window.targetVideosList.clear()

@QtCore.Slot()
def select_target_medias(main_window: 'MainWindow', source_type='folder', folder_name=False, files_list=None):
    files_list = files_list or []
    if source_type=='folder':
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(dir=main_window.last_target_media_folder_path)
        if not folder_name:
            return
        main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelTargetVideosPath.setToolTip(folder_name)
        main_window.last_target_media_folder_path = folder_name

    elif source_type=='files':
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        main_window.labelTargetVideosPath.setText('Selected Files') #Just a temp text until i think of something better
        main_window.labelTargetVideosPath.setToolTip('Selected Files')

    clear_stop_loading_target_media(main_window)
    card_actions.clear_target_faces(main_window)
    
    main_window.selected_video_button = False
    main_window.target_videos = {}

    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(main_window=main_window, folder_name=folder_name, files_list=files_list)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_target_videos_list, main_window))
    main_window.video_loader_worker.start()

@QtCore.Slot()
def load_target_webcams(main_window: 'MainWindow',):
    if main_window.filterWebcamsCheckBox.isChecked():
        main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(main_window=main_window, webcam_mode=True)
        main_window.video_loader_worker.webcam_thumbnail_ready.connect(partial(add_webcam_thumbnail_to_target_videos_list, main_window))
        main_window.video_loader_worker.start()
    else:
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, True)
        for _, target_video in main_window.target_videos.copy().items(): #Use a copy of the dict to prevent Dictionary changed during iteration exceptions
            if target_video.file_type == 'webcam':
                target_video.remove_target_media_from_list()
                if target_video == main_window.selected_video_button:
                    main_window.selected_video_button = False
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)

def clear_stop_loading_input_media(main_window: 'MainWindow'):
    if main_window.input_faces_loader_worker:
        main_window.input_faces_loader_worker.stop()
        main_window.input_faces_loader_worker.terminate()
        main_window.input_faces_loader_worker = False
        time.sleep(0.5)
        main_window.inputFacesList.clear()

@QtCore.Slot()
def select_input_face_images(main_window: 'MainWindow', source_type='folder', folder_name=False, files_list=None):
    files_list = files_list or []
    if source_type=='folder':
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(dir=main_window.last_input_media_folder_path)
        if not folder_name:
            return
        main_window.labelInputFacesPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelInputFacesPath.setToolTip(folder_name)
        main_window.last_input_media_folder_path = folder_name

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

def set_up_list_widget_placeholder(main_window: 'MainWindow', list_widget: QtWidgets.QListWidget):
    # Placeholder label
    placeholder_label = QtWidgets.QLabel(list_widget)
    placeholder_label.setText(
        "<html><body style='text-align:center;'>"
        "<p>Drop Files</p>"
        "<p><b>or</b></p>"
        "<p>Click here to Select a Folder</p>"
        "</body></html>"
    )
    # placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    placeholder_label.setStyleSheet("color: gray; font-size: 15px; font-weight: bold;")
    
    # Center the label inside the QListWidget
    # placeholder_label.setGeometry(list_widget.rect())  # Match QListWidget's size
    placeholder_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)  # Allow interactions to pass through
    placeholder_label.setVisible(not list_widget.count())  # Show if the list is empty

    # Use a QVBoxLayout to center the placeholder label
    layout = QtWidgets.QVBoxLayout(list_widget)
    layout.addWidget(placeholder_label)
    layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Center the label vertically and horizontally
    layout.setContentsMargins(0, 0, 0, 0)  # Remove margins to ensure full coverage

    # Keep a reference for toggling visibility later
    list_widget.placeholder_label = placeholder_label
    # Set default cursor as PointingHand
    list_widget.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

def select_output_media_folder(main_window: 'MainWindow'):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory()
    if folder_name:
        main_window.outputFolderLineEdit.setText(folder_name)
        common_widget_actions.create_control(main_window, 'OutputMediaFolder', folder_name)
