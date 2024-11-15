from PySide6 import QtWidgets, QtGui, QtCore
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

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
    graphicsViewFrame.fitInView(pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    graphicsViewFrame.update()