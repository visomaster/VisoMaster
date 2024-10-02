from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QImage, QPixmap
import App.UI.Widgets.WidgetActions as widget_actions
import PySide6.QtCore as qtc
import cv2
import numpy as np

from PySide6.QtWidgets import QPushButton
from App.UI.Widgets.LayoutData import SWAPPER_LAYOUT_DATA
from functools import partial
from typing import TYPE_CHECKING, Dict, List
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

class CardButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.main_window: 'MainWindow' = kwargs.get('main_window', False)
        self.list_item  = None

class TargetMediaCardButton(CardButton):
    def __init__(self, media_path: str, file_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_type = file_type
        self.media_path = media_path
        self.setCheckable(True)
        self.setToolTip(media_path)
        self.clicked.connect(self.loadMediaOnClick)

    def loadMediaOnClick(self):

        main_window = self.main_window
        # Deselect the currently selected video
        if main_window.selected_video_buttons:
            main_window.selected_video_buttons[0].toggle()  # Deselect the previous video
            main_window.selected_video_buttons.pop(0)
        
        # Stop the current video processing
        processing = main_window.video_processor.stop_processing()
        if processing:
            main_window.wait_for_processing_to_finish()
            qtc.QThread.msleep(200)  # Sleep per 200 ms

        # Reset the frame counter
        main_window.video_processor.current_frame_number = 0
        main_window.video_processor.media_path = self.media_path

        # Release the previous media_capture if it exists
        if main_window.video_processor.media_capture:
            main_window.video_processor.media_capture.release()

        frame = False
        max_frames_number = 0  # Initialize max_frames_number for either video or image
        
        if self.file_type == 'video':
            media_capture = cv2.VideoCapture(self.media_path)
            if not media_capture.isOpened():
                print(f"Error opening video {self.media_path}")
                return  # If the video cannot be opened, exit the function

            media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            max_frames_number = int(media_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            ret, frame = media_capture.read()
            main_window.video_processor.media_capture = media_capture
            main_window.video_processor.max_frame_number = max_frames_number

        elif self.file_type == 'image':
            frame = cv2.imread(self.media_path)
            max_frames_number = 0  # For an image, there is only one "frame"
            main_window.video_processor.max_frame_number = max_frames_number

        if frame is not None:
            if self.file_type == 'video':
                # restore initial video position after reading. == 0
                media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Convert the frame to QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)

            # Scale the pixmap if necessary
            scaled_pixmap = widget_actions.scale_pixmap_to_view(main_window.graphicsViewFrame, pixmap)
            pixmap_item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)

            # Clear the scene and add the new frame
            main_window.scene.clear()
            main_window.scene.addItem(pixmap_item)

            # Fit the image to the view
            widget_actions.fit_image_to_view(main_window, pixmap_item)

            # Immediately update the graphics view
            main_window.graphicsViewFrame.update()

        # Clear current target faces
        widget_actions.clear_target_faces(main_window)
        # Uncheck input faces
        widget_actions.uncheck_all_input_faces(main_window)

        # Reset buttons and slider
        widget_actions.resetMediaButtons(main_window)
        main_window.video_processor.file_type = self.file_type
        main_window.videoSeekSlider.blockSignals(True)  # Block signals to prevent unnecessary updates
        main_window.videoSeekSlider.setMaximum(max_frames_number)
        main_window.videoSeekSlider.setValue(0)  # Set the slider to 0 for the new video
        main_window.videoSeekSlider.blockSignals(False)  # Unblock signals

        # Append the selected video button to the list
        main_window.selected_video_buttons.append(self)

        # Update the graphics frame after the reset
        main_window.graphicsViewFrame.update()

class TargetFaceCardButton(CardButton):
    def __init__(self, media_path, cropped_face, embedding: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.media_path = media_path
        self.cropped_face = cropped_face
        self.embedding = embedding
        self.assigned_input_face_buttons: Dict[InputFaceCardButton, np.ndarray] = {} # Key: InputFaceCardButton, Value: Face Embedding
        self.assigned_embed_buttons: Dict[EmbeddingCardButton, np.ndarray] = {} # Key: EmbeddingCardButton, Value: Face Embedding
        self.assigned_input_embedding = np.array([])
        self.setCheckable(True)
        self.clicked.connect(self.loadTargetFace)

        # Set the context menu policy to trigger the custom context menu on right-click
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Connect the custom context menu request signal to the custom slot
        self.customContextMenuRequested.connect(self.on_context_menu)
        self.create_context_menu()

    def loadTargetFace(self):
        main_window = self.main_window
        main_window.cur_selected_target_face_button = self
        self.setChecked(True)
        for target_face_button in main_window.target_faces:
            # Uncheck all other target faces
            if target_face_button!=self:
                target_face_button.setChecked(False)

        widget_actions.uncheck_all_input_faces(main_window)

        for input_face_button in self.assigned_input_face_buttons.keys():
            input_face_button.setChecked(True)

        widget_actions.refresh_frame(main_window)

    def calculateAssignedInputEmbedding(self,):
        parameters = self.main_window.parameters.copy()
        input_face_embeddings = [embedding for embedding in self.assigned_input_face_buttons.values()]
        merged_embeddings = [embedding for embedding in self.assigned_embed_buttons.values()]
        all_input_embeddings = input_face_embeddings + merged_embeddings
        if len(all_input_embeddings)>0:
            if parameters['EmbMergeMethodSelection'] == 'Mean':
                self.assigned_input_embedding = np.mean(all_input_embeddings, 0)
            elif parameters['EmbMergeMethodSelection'] == 'Median':
                self.assigned_input_embedding = np.median(all_input_embeddings, 0)
        else:
            self.assigned_input_embedding = np.array([])

    def create_context_menu(self):
        # create context menu
        self.popMenu = QtWidgets.QMenu(self)
        remove_action = QtGui.QAction('Remove from List', self)
        remove_action.triggered.connect(self.remove_target_face_from_list)
        self.popMenu.addAction(remove_action)

    def on_context_menu(self, point):
        # show context menu
        self.popMenu.exec_(self.mapToGlobal(point))

    def remove_target_face_from_list(self):
        main_window = self.main_window
        for i in range(main_window.targetFacesList.count()):
            list_item = main_window.targetFacesList.item(i)
            if list_item.listWidget().itemWidget(list_item) == self:
                main_window.targetFacesList.takeItem(i)   
                main_window.target_faces.pop(i)
        widget_actions.refresh_frame(self.main_window)
        del self


class InputFaceCardButton(CardButton):
    def __init__(self, media_path, cropped_face, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cropped_face = cropped_face
        self.embedding = embedding
        self.media_path = media_path

        self.setCheckable(True)
        self.setToolTip(media_path)
        self.clicked.connect(self.loadInputFace)

        # Set the context menu policy to trigger the custom context menu on right-click
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Connect the custom context menu request signal to the custom slot
        self.customContextMenuRequested.connect(self.on_context_menu)
        self.create_context_menu()

    def loadInputFace(self):
        main_window = self.main_window

        if main_window.cur_selected_target_face_button:
            cur_selected_target_face_button = main_window.cur_selected_target_face_button
            if not QtWidgets.QApplication.keyboardModifiers() == qtc.Qt.ControlModifier:
                for input_face_button in cur_selected_target_face_button.assigned_input_face_buttons.keys():
                    if input_face_button!=self:
                        input_face_button.setChecked(False)
                cur_selected_target_face_button.assigned_input_face_buttons = {}

            cur_selected_target_face_button.assigned_input_face_buttons[self] = self.embedding

            if not self.isChecked():
                cur_selected_target_face_button.assigned_input_face_buttons.pop(self)
            cur_selected_target_face_button.calculateAssignedInputEmbedding()
        else:
            if not QtWidgets.QApplication.keyboardModifiers() == qtc.Qt.ControlModifier:
                # If there is no target face selected, uncheck all other input faces
                for input_face_button in main_window.input_faces:
                    if input_face_button!=self:
                        input_face_button.setChecked(False)

        widget_actions.refresh_frame(main_window)

    def create_context_menu(self):
        # create context menu
        self.popMenu = QtWidgets.QMenu(self)
        remove_action = QtGui.QAction('Create embedding from selected faces', self)
        remove_action.triggered.connect(self.create_embedding_from_selected_faces)
        self.popMenu.addAction(remove_action)

    def on_context_menu(self, point):
        # show context menu
        self.popMenu.exec_(self.mapToGlobal(point))

    def create_embedding_from_selected_faces(self):
        selected_faces_embeddings = [input_face.embedding for input_face in self.main_window.input_faces if input_face.isChecked()]
        if len(selected_faces_embeddings)==0:
            widget_actions.create_and_show_messagebox(self.main_window, "No Faces Selected!", "You need to select atleast one face to create a merged embedding!", self)
        else:
            embed_create_dialog = CreateEmbeddingDialog(self.main_window, selected_faces_embeddings)
            embed_create_dialog.exec_()

class EmbeddingCardButton(CardButton):
    def __init__(self, embedding_name, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding
        self.embedding_name = embedding_name
        self.setCheckable(True)
        self.setText(embedding_name)
        self.setToolTip(embedding_name)
        self.clicked.connect(self.loadEmbedding)

    def loadEmbedding(self):
        main_window = self.main_window
        if main_window.cur_selected_target_face_button:
            
            cur_selected_target_face_button = main_window.cur_selected_target_face_button
            if not QtWidgets.QApplication.keyboardModifiers() == qtc.Qt.ControlModifier:
                for embed_button in cur_selected_target_face_button.assigned_embed_buttons.keys():
                    if embed_button!=self:
                        embed_button.setChecked(False)
                cur_selected_target_face_button.assigned_embed_buttons = {}

            cur_selected_target_face_button.assigned_embed_buttons[self] = self.embedding

            if not self.isChecked():
                cur_selected_target_face_button.assigned_embed_buttons.pop(self)
            cur_selected_target_face_button.calculateAssignedInputEmbedding()
        else:
            if not QtWidgets.QApplication.keyboardModifiers() == qtc.Qt.ControlModifier:
                # If there is no target face selected, uncheck all other input faces
                for embed_button in main_window.merged_embeddings:
                    if embed_button!=self:
                        embed_button.setChecked(False)

        widget_actions.refresh_frame(main_window)


class CreateEmbeddingDialog(QtWidgets.QDialog):
    def __init__(self, main_window: 'MainWindow', embeddings: list = []):
        super().__init__()
        self.main_window = main_window
        self.embeddings = embeddings
        self.embedding_name = ''
        self.merge_type = ''
        self.setWindowTitle("Create Embedding")

        # Create widgets
        self.embed_name_edit = QtWidgets.QLineEdit(self)
        self.embed_name_edit.setPlaceholderText("Enter embedding name")

        self.merge_type_selection = QtWidgets.QComboBox(self)
        self.merge_type_selection.addItems(['Mean', 'Median'])
        self.merge_type_selection.setCurrentText(main_window.parameters['EmbMergeMethodSelection'])

        # Create button box
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.create_embedding)
        self.buttonBox.rejected.connect(self.reject)

        # Create layout and add widgets
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Embedding Name:"))
        layout.addWidget(self.embed_name_edit)
        layout.addWidget(QtWidgets.QLabel("Merge Type:"))
        layout.addWidget(self.merge_type_selection)
        layout.addWidget(self.buttonBox)

        # Set dialog layout
        self.setLayout(layout)

    def create_embedding(self):
        if self.embeddings:
            self.embedding_name = self.embed_name_edit.text().strip()
            self.merge_type = self.merge_type_selection.currentText()
            if self.embedding_name == '':
                widget_actions.create_and_show_messagebox(self.main_window, 'Empty Embedding Name!', 'Embedding Name cannot be empty!', self)
            else:
                if self.merge_type == 'Mean':
                    merged_embedding = np.mean(self.embeddings, 0)
                elif self.merge_type == 'Median':
                    merged_embedding = np.median(self.embeddings, 0)
                widget_actions.create_and_add_embed_button_to_list(main_window=self.main_window, embedding_name=self.embedding_name, merged_embedding = merged_embedding)
                self.accept()

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
        self.min_value = kwargs.get('min_value',False)
        self.max_value = kwargs.get('max_value',False)
        self.group_layout_data: Dict[str, dict]  = kwargs.get('group_layout_data', {})
        self.widget_name = kwargs.get('widget_name', False)
        self.label_widget: QtWidgets.QLabel = kwargs.get('label_widget', False)
        self.group_widget: QtWidgets.QGroupBox = kwargs.get('group_widget', False)
        self.main_window: 'MainWindow' = kwargs.get('main_window', False)
        self.line_edit: QtWidgets.QLineEdit = False #Only sliders have textbox currently
        self.reset_default_button: QPushButton = False

class ToggleSwitchButton(QtWidgets.QPushButton, ParametersWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)

class SelectionBox(QtWidgets.QComboBox, ParametersWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.currentTextChanged.connect(partial(widget_actions.show_hide_related_widgets, self.main_window, self, self.widget_name, ))

    def reset_to_default_value(self):
        self.setCurrentText(self.default_value)
    
class ToggleButton(QtWidgets.QPushButton, ParametersWidget):
    _circle_position = None

    def __init__(self, bg_color="#000000", circle_color="#ffffff", active_color="#16a085", default_value=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)

        self.setFixedSize(30, 15)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setCheckable(True)
        
        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color
        self.default_value = bool(default_value)
        self._circle_position = 1  # Start position of the circle
        self.animation_curve = QtCore.QEasingCurve.OutCubic
        
        # Animation
        self.animation = QtCore.QPropertyAnimation(self, b"circle_position", self)
        self.animation.setDuration(300)  # Animation duration in milliseconds
        self.animation.setEasingCurve(self.animation_curve)
        
        self.clicked.connect(partial(widget_actions.show_hide_related_widgets, self.main_window, self, self.widget_name, None))
        self.toggled.connect(self.start_animation)
        
    # Property for animation
    @QtCore.Property(int)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()  # Update the widget to trigger paintEvent

    def start_animation(self):
        # Animate circle position when toggled
        start_pos = 1 if self.isChecked() else 15
        end_pos = 15 if self.isChecked() else 1
        
        self.animation.setStartValue(start_pos)
        self.animation.setEndValue(end_pos)
        self.animation.start()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        
        rect = QtCore.QRect(0, 0, self.width(), self.height())
        
        if self.isChecked():
            p.setBrush(QtGui.QColor(self._active_color))
            p.drawRoundedRect(0, 0, rect.width(), self.height(), self.height() / 2, self.height() / 2)
        else:
            p.setBrush(QtGui.QColor(self._bg_color))
            p.drawRoundedRect(0, 0, rect.width(), self.height(), self.height() / 2, self.height() / 2)
        
        # Draw the circle at the animated position
        p.setBrush(QtGui.QColor(self._circle_color))
        p.drawEllipse(self._circle_position, 1, 13, 13)
        
        p.end()

    def reset_to_default_value(self):
        self.setChecked(bool(self.default_value))

class ParameterSlider(QtWidgets.QSlider, ParametersWidget):
    def __init__(self, min_value=0, max_value=0, default_value=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        self.default_value = int(default_value)
        self.setMinimum(int(min_value))
        self.setMaximum(int(max_value))
        self.setValue(self.default_value)
        self.setOrientation(qtc.Qt.Orientation.Horizontal)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.setFixedWidth(130)
        # Set a fixed width for the slider

    def reset_to_default_value(self):
        self.setValue(int(self.default_value))

class ParameterDecimalSlider(QtWidgets.QSlider, ParametersWidget):
    def __init__(self, min_value=0.0, max_value=1.0, default_value=0.0, decimals=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)

        # Ensure min, max, and default are floats
        min_value = float(min_value)
        max_value = float(max_value)
        default_value = float(default_value)

        # Store the number of decimals and calculate the scale factor
        self.decimals = decimals
        self.scale_factor = 10 ** self.decimals

        # Convert min, max, and default to scaled integers for QSlider
        self.min_value = int(min_value * self.scale_factor)
        self.max_value = int(max_value * self.scale_factor)
        self.default_value = int(default_value * self.scale_factor)

        # Set the slider's integer range and default value
        self.setMinimum(self.min_value)
        self.setMaximum(self.max_value)
        self.setValue(self.default_value)
        print(self.default_value)

        self.setOrientation(qtc.Qt.Orientation.Horizontal)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.setFixedWidth(130)
    def reset_to_default_value(self):
        """Reset the slider to its default value."""
        self.setValue(self.default_value / self.scale_factor)

    def value(self):
        """Return the slider value as a float, scaled by the decimals."""
        return super().value() / self.scale_factor

    def setValue(self, value):
        """Set the slider value, scaling it from a float to the internal integer."""
        scaled_value = int(float(value) * self.scale_factor)
        super().setValue(scaled_value)

class ParameterLineEdit(QtWidgets.QLineEdit):
    def __init__(self, min_value: int, max_value: int, default_value: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(38)  # Make the line edit narrower
        self.setMaxLength(3)
        self.setValidator(QtGui.QIntValidator(min_value, max_value))  # Restrict input to numbers
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(default_value)

class ParameterLineDecimalEdit(QtWidgets.QLineEdit):
    def __init__(self, min_value: float, max_value: float, default_value: str, decimals: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(50)  # Adjust the width for decimal numbers
        self.decimals = decimals
        self.setMaxLength(5)
        self.setValidator(QtGui.QDoubleValidator(min_value, max_value, decimals))
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(default_value)

    def set_value(self, value: float):
        """Set the line edit's value."""
        self.setText(f"{value:.{self.decimals}f}")

    def get_value(self) -> float:
        """Get the current value from the line edit."""
        return float(self.text())

class ParameterResetDefaultButton(QtWidgets.QPushButton):
    def __init__(self, related_widget: ParameterSlider | SelectionBox, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.related_widget = related_widget
        button_icon = QtGui.QIcon(QtGui.QPixmap(':/media/Media/reset_default.png'))
        self.setIcon(button_icon)
        self.setFixedWidth(30)  # Make the line edit narrower
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip('Reset to default value')

        self.clicked.connect(related_widget.reset_to_default_value)


class FormGroupBox(QtWidgets.QGroupBox):
    def __init__(self, main_window:'MainWindow', title="Form Group", parent=None,):
        super().__init__(title, parent)
        self.main_window = main_window
        self.setFlat(True)
