from PySide6.QtWidgets import QPushButton, QGraphicsPixmapItem
from PySide6.QtGui import QImage, QPixmap
import PySide6.QtCore as qtc
import cv2

class TargetMediaCardButton(QPushButton):
    def __init__(self, media_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.media_path = media_path
        self.setCheckable(True)
        self.setStyleSheet("background-color: yellow;")  # Default color for unselected state
        self.setToolTip(media_path)
        self.clicked.connect(self.loadMediaOnClick)
        self.setStyleSheet("""QToolTip { color: #ffffff; background-color: #000000; border: 0px; }");""")

    def toggle_state(self):
        if self.isChecked():
            self.setStyleSheet("""
                QPushButton {
                    background-color: lightblue;
                    border: 2px solid blue;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border: 2px solid gray;
                    border-radius: 5px;
                }
            """)

    def loadMediaOnClick(self):
        main_window = self.window()
        print(self.media_path)
        main_window.media_capture = cv2.VideoCapture(self.media_path)
        main_window.media_capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame = main_window.media_capture.read()
        if ret:
            # Convert the frame to QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)

            # Scale the pixmap if necessary
            scaled_pixmap = scale_pixmap_to_view(main_window.graphicsViewFrame, pixmap)
            pixmap_item = QGraphicsPixmapItem(scaled_pixmap)

            main_window.scene.clear()
            main_window.scene.addItem(pixmap_item)
            
            # Fit the image to the view
            fit_image_to_view(main_window, pixmap_item)
        main_window.videoSeekSlider.setValue(1)
        main_window.media_capture.release()  # Release the video capture object


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
