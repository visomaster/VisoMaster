
import PySide6.QtCore as qtc
from PySide6 import QtWidgets, QtGui
import time
import App.Helpers.Misc_Helpers as misc_helpers 
import App.Workers.UI_Workers as ui_workers
from App.UI.Widgets.WidgetComponents import TargetMediaCardButton, ProgressDialog
import App.UI.Widgets.WidgetActions as ui_helpers 
from functools import partial
import cv2
from App.UI.Core import media_rc
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

def clear_stop_loading_media(main_window):
    if main_window.video_loader_worker:
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        time.sleep(0.5)
        main_window.targetVideosList.clear()

@qtc.Slot()
def onClickSelectTargetVideosFolder(main_window):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory()
    main_window.selected_video_buttons = []
    main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
    main_window.labelTargetVideosPath.setToolTip(folder_name)
    clear_stop_loading_media(main_window)
    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(folder_name=folder_name)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_list, main_window))
    main_window.video_loader_worker.start()

@qtc.Slot()
def onClickSelectTargetVideosFiles(main_window):
    files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
    main_window.selected_video_buttons = []
    main_window.labelTargetVideosPath.setText('Selected Files')
    main_window.labelTargetVideosPath.setToolTip('Selected Files')
    clear_stop_loading_media(main_window)
    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(files_list=files_list)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_list, main_window))
    main_window.video_loader_worker.start()
    
@qtc.Slot()
def OnChangeSlider(main_window, new_position=0):
    video_processor = main_window.video_processor
    video_processor.stop_processing()
    video_processor.current_frame_number = new_position
    if video_processor.media_capture:
        video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
    video_processor.processing = True
    video_processor.process_next_frame()
    video_processor.processing = False
    ui_helpers.resetMediaButtons(main_window)


@qtc.Slot(str, QtGui.QPixmap)
def add_media_thumbnail_to_list(main_window, media_path, pixmap, file_type):
    button_size = qtc.QSize(70, 70)  # Set a fixed size for the buttons
    button = TargetMediaCardButton(media_path=media_path, file_type=file_type)
    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(button_size - qtc.QSize(3, 3))  # Slightly smaller than the button size to add some margin
    button.setFixedSize(button_size)

    targetVideosList = main_window.targetVideosList
    # Create a QListWidgetItem and set the button as its widget
    list_item = QtWidgets.QListWidgetItem(main_window.targetVideosList)
    list_item.setSizeHint(button_size)

    # Align the item to center
    list_item.setTextAlignment(qtc.Qt.AlignmentFlag.AlignCenter)

    targetVideosList.setItemWidget(list_item, button)
    # Adjust the QListWidget properties to handle the grid layout
    grid_size_with_padding = button_size + qtc.QSize(4, 4)  # Add padding around the buttons
    targetVideosList.setGridSize(grid_size_with_padding)  # Set grid size with padding
    targetVideosList.setWrapping(True)  # Enable wrapping to have items in rows
    targetVideosList.setFlow(QtWidgets.QListView.LeftToRight)  # Set flow direction
    targetVideosList.setResizeMode(QtWidgets.QListView.Adjust)  # Adjust layout automatically
    # Optionally, you can hide the scrollbars for a cleaner look
    # targetVideosList.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    # targetVideosList.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)


# from App.UI.MainUI import Ui_MainWindow
def update_graphics_view(main_window , pixmap, current_frame_number):

    main_window.videoSeekSlider.blockSignals(True)
    main_window.videoSeekSlider.setValue(current_frame_number)
    main_window.videoSeekSlider.blockSignals(False)
    # print(main_window.graphicsViewFrame.scene, pixmap)
    main_window.graphicsViewFrame.scene().clear()
    pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
    main_window.graphicsViewFrame.scene().addItem(pixmap_item)
    # Optionally fit the image to the view
    ui_helpers.fit_image_to_view(main_window, pixmap_item)


def get_pixmap_from_frame(main_window, frame):
    height, width, channel = frame.shape
    bytes_per_line = 3 * width
    q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(q_img)
    scaled_pixmap = ui_helpers.scale_pixmap_to_view(main_window.graphicsViewFrame, pixmap)
    return scaled_pixmap

def resetMediaButtons(main_window):
    main_window.buttonMediaPlay.setChecked(False)
    setPlayButtonIcon(main_window)

def setPlayButtonIcon(main_window):
    if main_window.buttonMediaPlay.isChecked(): 
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_on.png"))
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_off.png"))

def OnClickPlayButton(main_window):

    # progress_dialog = ProgressDialog("Processing...", "Cancel", 0, 100, main_window)
    # progress_dialog.setWindowModality(qtc.Qt.ApplicationModal)
    # progress_dialog.setMinimumDuration(10000)
    # progress_dialog.setWindowTitle("Progress")
    # progress_dialog.setAutoClose(True)  # Close the dialog when finished
    # progress_dialog.exec_()
    setPlayButtonIcon(main_window)



    main_window.video_processor.process_video()