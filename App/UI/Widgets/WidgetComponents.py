from PySide6.QtWidgets import QPushButton, QGraphicsPixmapItem, QVBoxLayout, QProgressDialog, QHBoxLayout, QWidget, QStyle, QApplication
from PySide6.QtGui import QImage, QPixmap
import App.UI.Widgets.WidgetActions as widget_actions
import PySide6.QtCore as qtc
import cv2

from PySide6.QtWidgets import QPushButton

class TargetMediaCardButton(QPushButton):
    def __init__(self, media_path, file_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_type = file_type
        self.media_path = media_path
        self.list_item = None
        self.setCheckable(True)
        self.setToolTip(media_path)
        self.clicked.connect(self.loadMediaOnClick)


    def loadMediaOnClick(self):
        main_window = self.window()
        # Check if it is docked or not 
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

        frame = False
        if self.file_type=='video':
            media_capture = cv2.VideoCapture(self.media_path)
            media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            max_frames_number = int(media_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame = media_capture.read()
            main_window.video_processor.media_capture = media_capture
            main_window.video_processor.max_frame_number = max_frames_number


        elif self.file_type=='image':
            frame = cv2.imread(self.media_path)
            max_frames_number = 1
        if not isinstance(frame,bool):
            # Convert the frame to QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)

            # Scale the pixmap if necessary
            scaled_pixmap = widget_actions.scale_pixmap_to_view(main_window.graphicsViewFrame, pixmap)
            pixmap_item = QGraphicsPixmapItem(scaled_pixmap)

            main_window.scene.clear()
            main_window.scene.addItem(pixmap_item)
            
            # Fit the image to the view
            widget_actions.fit_image_to_view(main_window, pixmap_item)
        widget_actions.resetMediaButtons(main_window)
        main_window.video_processor.file_type = self.file_type
        main_window.videoSeekSlider.setMaximum(max_frames_number)
        main_window.videoSeekSlider.setValue(0)
        # Append video button to main_window selected videos list
        main_window.selected_video_buttons.append(self)

class TargetFaceCardButton(QPushButton):
    def __init__(self, cropped_face, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cropped_face = cropped_face
        self.embedding = embedding
        self.setCheckable(True)
        self.clicked.connect(self.loadTargetFace)

    def loadTargetFace(self):
        main_window = self.window()
        # Check if it is docked or not 
        if main_window.parent():
            main_window = main_window.parent()

        if main_window.selected_target_face_buttons:
            if main_window.selected_target_face_buttons[0]!=self:
                main_window.selected_target_face_buttons[0].toggle()
            main_window.selected_target_face_buttons.pop(0)
        if self.isChecked():
            main_window.selected_target_face_buttons.append(self)


class InputFaceCardButton(QPushButton):
    def __init__(self, cropped_face, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cropped_face = cropped_face
        self.embedding = embedding
        self.setCheckable(True)
        self.clicked.connect(self.loadInputFace)

    def loadInputFace(self):
        main_window = self.window()
        # Check if it is docked or not 
        if main_window.parent():
            main_window = main_window.parent()
        # When not holding ctrl key
        if not QApplication.keyboardModifiers() == qtc.Qt.ControlModifier:
            for i in range(len(main_window.selected_input_face_buttons)-1, -1, -1):
                if main_window.selected_input_face_buttons[i]!=self:
                    main_window.selected_input_face_buttons[i].toggle()
                main_window.selected_input_face_buttons.pop(i)
        if self.isChecked():
            main_window.selected_input_face_buttons.append(self)
# Custom progress dialog
class ProgressDialog(QProgressDialog):
    pass


class GraphicsViewEventFilter(qtc.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, graphics_object, event):
        if event.type() == qtc.QEvent.Type.MouseButtonPress:
            if event.button() == qtc.Qt.MouseButton.LeftButton:
                # Check if it is a docked window of main_window
                if graphics_object.window().parent():
                    main_window = graphics_object.window().parent()
                else:
                    main_window = graphics_object.window()
                main_window.buttonMediaPlay.click()
                # You can emit a signal or call another function here
                return True  # Mark the event as handled
        return False  # Pass the event to the original handler