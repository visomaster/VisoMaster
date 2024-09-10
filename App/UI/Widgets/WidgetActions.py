from PySide6 import QtWidgets,QtGui
from App.UI.Core.MainWindow import Ui_MainWindow as MainWindow
from App.Helpers import MiscHelpers as misc_helpers
from App.Workers import ThreadWorkers
from App.UI.Widgets.WidgetComponents import TargetMediaCardButton
from PySide6 import QtCore as qtc
from functools import partial
import os
import time

def clear_stop_loading_media(main_window: MainWindow):
    if main_window.video_loader_worker:
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        main_window.frames = []
        time.sleep(0.5)
        main_window.targetVideosList.clear()

@qtc.Slot()
def onClickSelectTargetVideos(main_window: MainWindow):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory()

    main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
    main_window.labelTargetVideosPath.setToolTip(folder_name)

    clear_stop_loading_media(main_window)

    # if folder_name:
    #     # Load all frames from the selected directory
    #     main_window.frames = [os.path.join(folder_name, file) for file in sorted(os.listdir(folder_name))]
    #     main_window.current_frame_index = 0
    #     main_window.videoSeekSlider.setMaximum(len(main_window.frames))
    #     main_window.timer.start(24)

    #     # Create and start the video loader worker
    main_window.video_loader_worker = ThreadWorkers.TargetMediaLoaderWorker(folder_name)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_video_thumbnail_to_list, main_window))
        # main_window.video_loader_worker.finished.connect(partial(on_load_finished, main_window))
    main_window.video_loader_worker.start()

@qtc.Slot()
def getSliderCurrentPos(main_window: MainWindow, temp=False):
    # print("cur pos", main_window.videoSeekSlider.value())
    main_window.current_frame_index = main_window.videoSeekSlider.value()
    # if len(main_window.frames) !=0 and len(main_window.frames)>=main_window.videoSeekSlider.value():

@qtc.Slot(str, QtGui.QPixmap)
def add_video_thumbnail_to_list(main_window: MainWindow, media_path, pixmap):
    button = TargetMediaCardButton(media_path=media_path)
    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(pixmap.size())
    button.setFixedSize(pixmap.size())

    # Create a QListWidgetItem and set the button as its widget
    list_item = QtWidgets.QListWidgetItem(main_window.targetVideosList)
    list_item.setSizeHint(pixmap.size() + qtc.QSize(10, 10))  # Add padding for spacing
    main_window.targetVideosList.setItemWidget(list_item, button)

def on_load_finished(main_window: MainWindow):
    print("Loading finished")
    main_window.video_loader_worker.terminate()
    main_window.timer.stop()


@qtc.Slot()
def update_frame(main_window: MainWindow):
    if main_window.frames:
        # Create a new QGraphicsPixmapItem for the current frame
        pixmap = QtGui.QPixmap(main_window.frames[main_window.current_frame_index])
        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        
        # Clear the scene and add the new pixmap item
        main_window.scene.clear()
        main_window.scene.addItem(pixmap_item)
        
        # Fit the image to the view
        fit_image_to_view(main_window, pixmap_item)
        
        # Move to the next frame
        main_window.current_frame_index = (main_window.current_frame_index + 1)
        main_window.videoSeekSlider.setValue(main_window.current_frame_index)
        if main_window.current_frame_index >= len(main_window.frames):
            main_window.timer.stop()
    else:
        main_window.timer.stop()

def fit_image_to_view(main_window, pixmap_item):
    # Reset the transform to ensure no previous transformations affect the new fit
    main_window.graphicsViewFrame.resetTransform()

    # Set the scene rectangle to the bounding rectangle of the pixmap item
    main_window.graphicsViewFrame.setSceneRect(pixmap_item.boundingRect())

    # Calculate the view's size
    view_rect = main_window.graphicsViewFrame.viewport().rect()

    # Fit the image to the view, keeping the aspect ratio
    main_window.graphicsViewFrame.fitInView(pixmap_item, qtc.Qt.AspectRatioMode.KeepAspectRatio)

    # Get the scaled size of the pixmap item
    scaled_pixmap_size = pixmap_item.pixmap().scaled(
        view_rect.size(),
        aspectRatioMode=qtc.Qt.AspectRatioMode.KeepAspectRatio
    ).size()

    # Calculate the position to center the image
    x_center = (view_rect.width() - scaled_pixmap_size.width()) / 2
    y_center = (view_rect.height() - scaled_pixmap_size.height()) / 2

    # Apply the position to the pixmap item to center it
    pixmap_item.setPos(x_center, y_center)

    # Update the view to ensure the image is correctly displayed
    main_window.graphicsViewFrame.update()

