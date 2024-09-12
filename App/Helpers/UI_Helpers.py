
import PySide6.QtCore as qtc
from PySide6 import QtWidgets, QtGui
import time
import App.Helpers.Misc_Helpers as misc_helpers 
import App.Workers.UI_Workers as ui_workers
from App.UI.Widgets.WidgetComponents import TargetMediaCardButton
import App.Helpers.UI_Helpers as ui_helpers 
from functools import partial
import cv2
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
    # Reset the transform to ensure no previous transformations affect the new fit
    main_window.graphicsViewFrame.resetTransform()
    # Set the scene rectangle to the bounding rectangle of the pixmap item
    main_window.graphicsViewFrame.setSceneRect(pixmap_item.boundingRect())

    # Fit the image to the view, keeping the aspect ratio
    main_window.graphicsViewFrame.fitInView(pixmap_item, qtc.Qt.AspectRatioMode.KeepAspectRatio)
    main_window.graphicsViewFrame.update()

def clear_stop_loading_media(main_window):
    if main_window.video_loader_worker:
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        time.sleep(0.5)
        main_window.targetVideosList.clear()

@qtc.Slot()
def onClickSelectTargetVideos(main_window):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory()

    main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
    main_window.labelTargetVideosPath.setToolTip(folder_name)
    clear_stop_loading_media(main_window)
    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(folder_name)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_video_thumbnail_to_list, main_window))
        # main_window.video_loader_worker.finished.connect(partial(on_load_finished, main_window))
    main_window.video_loader_worker.start()
@qtc.Slot()
def OnChangeSlider(main_window, new_position=0):
    main_window.video_processor.stop_processing()
    main_window.video_processor.current_frame_number = new_position
    if main_window.video_processor.media_capture:
        main_window.video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
    main_window.video_processor.processing = True
    main_window.video_processor.process_next_frame()
    main_window.video_processor.processing = False


@qtc.Slot(str, QtGui.QPixmap)
def add_video_thumbnail_to_list(main_window, media_path, pixmap):
    button = TargetMediaCardButton(media_path=media_path)
    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(pixmap.size())
    button.setFixedSize(pixmap.size())

    # Create a QListWidgetItem and set the button as its widget
    list_item = QtWidgets.QListWidgetItem(main_window.targetVideosList)
    list_item.setSizeHint(pixmap.size() + qtc.QSize(10, 10))  # Add padding for spacing
    main_window.targetVideosList.setItemWidget(list_item, button)


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