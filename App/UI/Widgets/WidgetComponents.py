from PySide6.QtWidgets import QPushButton, QGraphicsPixmapItem
from PySide6.QtGui import QImage, QPixmap
import App.Helpers.UI_Helpers as ui_helpers
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

    # def toggle_state(self):
    #     if self.isChecked():
    #         self.setStyleSheet("""
    #             QPushButton {
    #                 background-color: lightblue;
    #                 border: 2px solid blue;
    #                 border-radius: 5px;
    #             }
    #         """)
    #     else:
    #         self.setStyleSheet("""
    #             QPushButton {
    #                 background-color: lightgray;
    #                 border: 2px solid gray;
    #                 border-radius: 5px;
    #             }
    #         """)

    def loadMediaOnClick(self):
        main_window = self.window()
        print(self.media_path)
        media_capture = cv2.VideoCapture(self.media_path)
        media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # main_window.video_processor.media_capture = cv2.VideoCapture(self.media_path)
        # main_window.video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
        # ret, frame = main_window.video_processor.media_capture.read()
        ret, frame = media_capture.read()
        if ret:
            # Convert the frame to QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)

            # Scale the pixmap if necessary
            scaled_pixmap = ui_helpers.scale_pixmap_to_view(main_window.graphicsViewFrame, pixmap)
            pixmap_item = QGraphicsPixmapItem(scaled_pixmap)

            main_window.scene.clear()
            main_window.scene.addItem(pixmap_item)
            
            # Fit the image to the view
            ui_helpers.fit_image_to_view(main_window, pixmap_item)
        main_window.videoSeekSlider.setValue(0)
        # main_window.video_processor.media_capture.release()  # Release the video capture object
