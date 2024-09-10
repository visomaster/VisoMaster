from PySide6 import QtWidgets,QtGui
from App.UI.Core.MainWindow import Ui_MainWindow as MainWindow
from App.Helpers import MiscHelpers as misc_helpers
from App.Workers import ThreadWorkers
from App.UI.Widgets.WidgetComponents import ToggleButton
from PySide6 import QtCore as qtc
from functools import partial
import os

@qtc.Slot()
def onClickSelectTargetVideos(main_window: MainWindow):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory()

    if main_window.video_loader_worker:
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        main_window.frames = []
        import time
        time.sleep(0.5)
        main_window.targetVideosList.clear()

    main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
    main_window.labelTargetVideosPath.setToolTip(folder_name)

    if folder_name:
        # Load all frames from the selected directory
        main_window.frames = [os.path.join(folder_name, file) for file in sorted(os.listdir(folder_name))]
        main_window.current_frame_index = 0
        main_window.videoSeekSlider.setMaximum(len(main_window.frames))
        main_window.timer.start(24)

        # Create and start the video loader worker
        main_window.video_loader_worker = ThreadWorkers.VideoLoaderWorker(folder_name)
        main_window.video_loader_worker.thumbnail_ready.connect(partial(add_video_thumbnail_to_list, main_window))
        # main_window.video_loader_worker.finished.connect(partial(on_load_finished, main_window))
        main_window.video_loader_worker.start()

@qtc.Slot()
def getSliderCurrentPos(main_window: MainWindow, temp=False):
    # print("cur pos", main_window.videoSeekSlider.value())
    main_window.current_frame_index = main_window.videoSeekSlider.value()
    # if len(main_window.frames) !=0 and len(main_window.frames)>=main_window.videoSeekSlider.value():

@qtc.Slot(str, QtGui.QPixmap)
def add_video_thumbnail_to_list(main_window: MainWindow, video_path, pixmap):
    button = ToggleButton(media_path=video_path)
    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(pixmap.size())
    button.setFixedSize(pixmap.size())
    button.clicked.connect(partial(print, button.media_path))

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

def fit_image_to_view(main_window: MainWindow, pixmap_item):
    # Fit the image to the view, keeping the aspect ratio
    main_window.graphicsViewFrame.fitInView(pixmap_item, qtc.Qt.AspectRatioMode.KeepAspectRatio)
    
    # Set the alignment to center the image within the view
    main_window.graphicsViewFrame.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
    
    # Make sure the view updates its display correctly
    main_window.graphicsViewFrame.update()