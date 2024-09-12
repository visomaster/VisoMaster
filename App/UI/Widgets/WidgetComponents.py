from PySide6.QtWidgets import QPushButton, QGraphicsPixmapItem
from PySide6.QtGui import QImage, QPixmap
import App.Helpers.UI_Helpers as ui_helpers
import PySide6.QtCore as qtc
import cv2

from PySide6.QtWidgets import QPushButton

class TargetMediaCardButton(QPushButton):
    def __init__(self, media_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.media_path = media_path
        self.setCheckable(True)
        self.setToolTip(media_path)
        self.clicked.connect(self.loadMediaOnClick)

    def loadMediaOnClick(self):
        main_window = self.window()
        if main_window.parent():
            main_window = main_window.parent()
        if main_window.selected_video_buttons:
            main_window.selected_video_buttons[0].toggle()
            main_window.selected_video_buttons.pop(0)
        main_window.video_processor.stop_processing()
        main_window.video_processor.current_frame_number = 0
        print(self.media_path)
        main_window.video_processor.media_path = self.media_path

        if main_window.video_processor.media_capture:
            main_window.video_processor.media_capture.release()  # Release the video capture object


        media_capture = cv2.VideoCapture(self.media_path)
        media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        max_frames_number = int(media_capture.get(cv2.CAP_PROP_FRAME_COUNT))
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
        main_window.video_processor.media_capture = media_capture
        main_window.videoSeekSlider.setMaximum(max_frames_number)
        main_window.videoSeekSlider.setValue(0)
        # Append video button to main_window selected videos list
        main_window.selected_video_buttons.append(self)


class GraphicsViewEventFilter(qtc.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, graphics_object, event):
        if event.type() == qtc.QEvent.Type.MouseButtonPress:
            if event.button() == qtc.Qt.MouseButton.LeftButton:
                # Check if it is a docked window of main_window
                if graphics_object.window().parent():
                    video_processor = graphics_object.window().parent().video_processor
                else:
                    video_processor = graphics_object.window().video_processor
                video_processor.process_video()
                # You can emit a signal or call another function here
                return True  # Mark the event as handled
        return False  # Pass the event to the original handler
    
class SliderEventFilter(qtc.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, slider_object, event):
        if event.type() == qtc.QEvent.Type.MouseButtonPress:
            if event.button() == qtc.Qt.MouseButton.LeftButton:
                super().eventFilter(slider_object, event)
                ui_helpers.OnChangeSlider(slider_object.window())
                # You can emit a signal or call another function here
                return True  # Mark the event as handled
        return False  # Pass the event to the original handler