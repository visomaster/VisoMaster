from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QImage, QPixmap
import App.UI.Widgets.WidgetActions as widget_actions
import PySide6.QtCore as qtc
import cv2

from PySide6.QtWidgets import QPushButton
from App.UI.Widgets.LayoutData import SWAPPER_LAYOUT_DATA
from functools import partial
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

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
            pixmap_item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)

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
    def __init__(self, media_path, cropped_face, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.media_path = media_path
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
    def __init__(self, media_path, cropped_face, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cropped_face = cropped_face
        self.embedding = embedding
        self.media_path = media_path

        self.setCheckable(True)
        self.setToolTip(media_path)
        self.clicked.connect(self.loadInputFace)

    def loadInputFace(self):
        main_window = self.window()
        # Check if it is docked or not 
        if main_window.parent():
            main_window = main_window.parent()
        # When not holding ctrl key
        if not QtWidgets.QApplication.keyboardModifiers() == qtc.Qt.ControlModifier:
            for i in range(len(main_window.selected_input_face_buttons)-1, -1, -1):
                if main_window.selected_input_face_buttons[i]!=self:
                    main_window.selected_input_face_buttons[i].toggle()
                main_window.selected_input_face_buttons.pop(i)
        if self.isChecked():
            main_window.selected_input_face_buttons.append(self)
# Custom progress dialog
class ProgressDialog(QtWidgets.QProgressDialog):
    pass


class GraphicsViewEventFilter(qtc.QObject):
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def eventFilter(self, graphics_object: QtWidgets.QGraphicsView, event):
        if event.type() == qtc.QEvent.Type.MouseButtonPress:
            if event.button() == qtc.Qt.MouseButton.LeftButton:
                self.main_window.buttonMediaPlay.click()
                # You can emit a signal or call another function here
                return True  # Mark the event as handled
        return False  # Pass the event to the original handler
    


class ParametersWidget:
    def __init__(self, *args, **kwargs):
        self.default_value = kwargs.get('default_value', False)
        self.group_layout_data = kwargs.get('group_layout_data', {})
        self.widget_name = kwargs.get('widget_name', False)
        self.label_widget = kwargs.get('label_widget', False)
        self.group_widget = kwargs.get('group_widget', False)
        self.main_window = kwargs.get('main_window', False)
        self.line_edit = False #Only sliders have textbox currently

class ToggleSwitchButton(QtWidgets.QPushButton, ParametersWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)

class SelectionBox(QtWidgets.QComboBox, ParametersWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.currentTextChanged.connect(partial(widget_actions.show_hide_related_widgets, self.main_window, self, self.widget_name, ))
    
class ToggleButton(QtWidgets.QCheckBox, ParametersWidget):
    _circle_position = None

    def __init__(self, bg_color="#000000", circle_color="#ffffff", active_color="#16a085", *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        width = 30
        height = 15
        self.animation_curve = QtCore.QEasingCurve.Linear
        self.setFixedSize(width, height)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color
        self._circle_position = 1 
        self.stateChanged.connect(partial(widget_actions.show_hide_related_widgets, self.main_window, self, self.widget_name, None))

        
    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        
        rect = QtCore.QRect(0, 0, self.width(), self.height())
        
        if self.isChecked():
            p.setBrush(QtGui.QColor(self._active_color))
            p.drawRoundedRect(0, 0, rect.width(), self.height(), self.height() / 2, self.height() / 2)
            
            p.setBrush(QtGui.QColor(self._circle_color))
            # p.drawEllipse(self._circle_position, 1, 10, 10)
            p.drawEllipse(15, 1, 13, 13)

        else:
            p.setBrush(QtGui.QColor(self._bg_color))
            p.drawRoundedRect(0, 0, rect.width(), self.height(), self.height() / 2, self.height() / 2)
            
            p.setBrush(QtGui.QColor(self._circle_color))
            p.drawEllipse(self._circle_position, 1, 13, 13)
        
        p.end()

class ParameterSlider(QtWidgets.QSlider, ParametersWidget):
    def __init__(self, min_value=0, max_value=0, default_value=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.setMinimum(int(min_value))
        self.setMaximum(int(max_value))
        self.setValue(int(default_value))
        self.setOrientation(qtc.Qt.Orientation.Horizontal)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.setMinimumWidth(130)
        # Set a fixed width for the slider

class FormGroupBox(QtWidgets.QGroupBox):
    def __init__(self, main_window:'MainWindow', title="Form Group", parent=None,):
        super().__init__(title, parent)
        self.main_window = main_window
        self.setFlat(True)
