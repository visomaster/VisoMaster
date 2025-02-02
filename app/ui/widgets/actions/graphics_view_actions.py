from PySide6 import QtWidgets, QtGui, QtCore
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# @misc_helpers.benchmark  (Keep this decorator if you have it)
def update_graphics_view(main_window: 'MainWindow', pixmap: QtGui.QPixmap, current_frame_number, reset_fit=False):
    # print('(update_graphics_view) current_frame_number', current_frame_number)

    # Update the video seek slider and line edit
    if main_window.videoSeekSlider.value() != current_frame_number:
        main_window.videoSeekSlider.blockSignals(True)
        main_window.videoSeekSlider.setValue(current_frame_number)
        main_window.videoSeekSlider.blockSignals(False)

    current_text = main_window.videoSeekLineEdit.text()
    if current_text != str(current_frame_number):
        main_window.videoSeekLineEdit.setText(str(current_frame_number))

    # Preserve the current transform (zoom and pan state) - No longer needed if we are not clearing scene every time
    # current_transform = main_window.graphicsViewFrame.transform()

    # Get the scene and existing pixmap item
    scene = main_window.graphicsViewFrame.scene()
    pixmap_item = None
    previous_items = scene.items()
    if previous_items:
        pixmap_item = previous_items[0]  # Assume pixmap is the first item

    # Resize the pixmap if necessary (only if pixmap_item exists)
    if pixmap_item:
        bounding_rect = pixmap_item.boundingRect()
        # If the old pixmap is smaller than the new pixmap (ie, due to the face compare or mask compare), scale is to the size of the old one
        if bounding_rect.width() > pixmap.width() and bounding_rect.height() > pixmap.height():
            pixmap = pixmap.scaled(bounding_rect.width(), bounding_rect.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    # Update or create pixmap item
    if pixmap_item:
        pixmap_item.setPixmap(pixmap)  # Update the pixmap of the existing item
    else:
        pixmap_item_new = QtWidgets.QGraphicsPixmapItem(pixmap) # Create a new pixmap item only if it doesn't exist
        scene.addItem(pixmap_item_new)
        pixmap_item = pixmap_item_new # Use the newly created item for fitting view

    # Set the scene rectangle to the bounding rectangle of the pixmap
    scene_rect = pixmap_item.boundingRect()
    main_window.graphicsViewFrame.setSceneRect(scene_rect)

    # Reset the view or restore the previous transform
    if reset_fit:
        fit_image_to_view(main_window, pixmap_item, scene_rect) # Pass pixmap_item here
    # else: # No longer need to restore transform if we are not clearing scene
    #     zoom_andfit_image_to_view_onchange(main_window, current_transform) # No longer needed


def zoom_andfit_image_to_view_onchange(main_window: 'MainWindow', new_transform):
    """Restore the previous transform (zoom and pan state) and update the view."""
    # print("Called zoom_andfit_image_to_view_onchange()")
    main_window.graphicsViewFrame.setTransform(new_transform, combine=False)


def fit_image_to_view(main_window: 'MainWindow', pixmap_item: QtWidgets.QGraphicsPixmapItem, scene_rect):
    """Reset the view and fit the image to the view, keeping the aspect ratio."""
    # print("Called fit_image_to_view()")
    graphicsViewFrame = main_window.graphicsViewFrame
    # Reset the transform and set the scene rectangle
    graphicsViewFrame.resetTransform()
    graphicsViewFrame.setSceneRect(scene_rect)
    # Fit the image to the view, keeping the aspect ratio
    graphicsViewFrame.fitInView(pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)