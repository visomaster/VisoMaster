from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QImage, QPixmap
import App.UI.Widgets.WidgetActions as widget_actions
import PySide6.QtCore as qtc
import cv2
import numpy as np

from PySide6.QtWidgets import QPushButton

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
        # Imposta lo stylesheet solo per questo pulsante
        self.setStyleSheet("""
        CardButton:checked {
            background-color: #555555;
            border: 2px solid #1abc9c;
        }
        """)

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
        main_window.parameters = {}

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
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)

            # Clear the scene and add the new frame
            main_window.scene.clear()
            main_window.scene.addItem(pixmap_item)

            # Fit the image to the view
            widget_actions.fit_image_to_view(main_window, pixmap_item)

            # Immediately update the graphics view
            main_window.graphicsViewFrame.update()

        # Set up videoSeekLineEdit
        widget_actions.set_up_video_seek_line_edit(main_window)
        # Clear current target faces
        widget_actions.clear_target_faces(main_window)
        # Uncheck input faces
        widget_actions.uncheck_all_input_faces(main_window)
        # Uncheck merged embeddings
        widget_actions.uncheck_all_merged_embeddings(main_window)

        main_window.cur_selected_target_face_button = False

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

        # Set Parameter widget values to default
        widget_actions.set_widgets_values_using_face_id_parameters(main_window=main_window, face_id=False)

class TargetFaceCardButton(CardButton):
    def __init__(self, media_path, cropped_face, embedding: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.main_window.target_faces:
            self.face_id = max([target_face.face_id for target_face in self.main_window.target_faces]) + 1
        else:
            self.face_id = 0
        self.media_path = media_path
        self.cropped_face = cropped_face
        self.embedding = embedding
        self.assigned_input_face_buttons: Dict[InputFaceCardButton, np.ndarray] = {} # Key: InputFaceCardButton, Value: Face Embedding
        self.assigned_embed_buttons: Dict[EmbeddingCardButton, np.ndarray] = {} # Key: EmbeddingCardButton, Value: Face Embedding
        self.assigned_input_embedding = np.array([])
        self.setCheckable(True)
        self.clicked.connect(self.loadTargetFace)

        # Imposta lo stylesheet solo per questo pulsante
        self.setStyleSheet("""
        CardButton:checked {
            background-color: #555555;
            border: 2px solid #1abc9c;
        }
        """)
        
        # Set the context menu policy to trigger the custom context menu on right-click
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Connect the custom context menu request signal to the custom slot
        self.customContextMenuRequested.connect(self.on_context_menu)
        self.create_context_menu()

        # Create parameter dict for the target
        if not self.main_window.parameters.get(self.face_id):
            widget_actions.create_parameter_dict_for_face_id(self.main_window, self.face_id)

    def loadTargetFace(self):
        main_window = self.main_window
        main_window.cur_selected_target_face_button = self
        self.setChecked(True)
        for target_face_button in main_window.target_faces:
            # Uncheck all other target faces
            if target_face_button!=self:
                target_face_button.setChecked(False)

        widget_actions.uncheck_all_input_faces(main_window)
        widget_actions.uncheck_all_merged_embeddings(main_window)

        for input_face_button in self.assigned_input_face_buttons.keys():
            input_face_button.setChecked(True)
        for embed_button in self.assigned_embed_buttons.keys():
            embed_button.setChecked(True)
        
        main_window.selected_target_face_id = self.face_id
        print('main_window.selected_target_face_id', main_window.selected_target_face_id)     
        widget_actions.set_widgets_values_using_face_id_parameters(main_window=main_window, face_id=self.face_id)      

        # widget_actions.refresh_frame(main_window)

    def calculateAssignedInputEmbedding(self,):
        control = self.main_window.control.copy()
        input_face_embeddings = [embedding for embedding in self.assigned_input_face_buttons.values()]
        merged_embeddings = [embedding for embedding in self.assigned_embed_buttons.values()]
        all_input_embeddings = input_face_embeddings + merged_embeddings
        if len(all_input_embeddings)>0:
            if control['EmbMergeMethodSelection'] == 'Mean':
                self.assigned_input_embedding = np.mean(all_input_embeddings, 0)
            elif control['EmbMergeMethodSelection'] == 'Median':
                self.assigned_input_embedding = np.median(all_input_embeddings, 0)
        else:
            self.assigned_input_embedding = np.array([])

    def create_context_menu(self):
        # create context menu
        self.popMenu = QtWidgets.QMenu(self)
        parameters_copy_action = QtGui.QAction('Copy Parameters', self)
        parameters_copy_action.triggered.connect(self.copy_parameters)
        parameters_paste_action = QtGui.QAction('Apply Copied Parameters', self)
        parameters_paste_action.triggered.connect(self.paste_and_apply_parameters)
        remove_action = QtGui.QAction('Remove from List', self)
        remove_action.triggered.connect(self.remove_target_face_from_list)
        self.popMenu.addAction(parameters_copy_action)
        self.popMenu.addAction(parameters_paste_action)
        self.popMenu.addAction(remove_action)

    def on_context_menu(self, point):
        # show context menu
        self.popMenu.exec_(self.mapToGlobal(point))

    def remove_target_face_from_list(self):
        main_window = self.main_window
        for i in range(main_window.targetFacesList.count()-1, -1, -1):
            list_item = main_window.targetFacesList.item(i)
            if list_item:
                if list_item.listWidget().itemWidget(list_item) == self:
                    main_window.targetFacesList.takeItem(i)   
                    main_window.target_faces.pop(i)
                    # Pop parameters using the target's face_id
                    main_window.parameters.pop(self.face_id)
        # Click and Select the first target face if target_faces are not empty
        if main_window.target_faces:
            main_window.target_faces[0].click()
        # Otherwise reset parameter widgets value to the default
        else:
            widget_actions.set_widgets_values_using_face_id_parameters(main_window, face_id=False)
        widget_actions.refresh_frame(self.main_window)
        self.deleteLater()

    def copy_parameters(self):

        self.main_window.copied_parameters = self.main_window.parameters[self.face_id].copy()

    def paste_and_apply_parameters(self):
        if not self.main_window.copied_parameters:
            widget_actions.create_and_show_messagebox(self.main_window, 'No parameters found in Clipboard', 'You need to copy parameters from any of the target face before pasting it!', parent_widget=self)
        else:
            self.main_window.parameters[self.face_id] = self.main_window.copied_parameters.copy()
            widget_actions.set_widgets_values_using_face_id_parameters(self.main_window, face_id=self.face_id)
            widget_actions.refresh_frame(main_window=self.main_window)

class InputFaceCardButton(CardButton):
    def __init__(self, media_path, cropped_face, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cropped_face = cropped_face
        self.embedding = embedding
        self.media_path = media_path

        self.setCheckable(True)
        self.setToolTip(media_path)
        self.clicked.connect(self.loadInputFace)

        # Imposta lo stylesheet solo per questo pulsante
        self.setStyleSheet("""
        CardButton:checked {
            background-color: #555555;
            border: 2px solid #1abc9c;
        }
        """)

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
    def __init__(self, embedding_name: str, embedding: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding
        self.embedding_name = embedding_name
        self.setCheckable(True)
        self.setText(embedding_name)
        self.setToolTip(embedding_name)
        self.clicked.connect(self.loadEmbedding)

        # Imposta lo stylesheet solo per questo pulsante
        self.setStyleSheet("""
        CardButton:checked {
            background-color: #555555;
            border: 2px solid #1abc9c;
        }
        """)

        # Set the context menu policy to trigger the custom context menu on right-click
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # Connect the custom context menu request signal to the custom slot
        self.customContextMenuRequested.connect(self.on_context_menu)
        self.create_context_menu()

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

    def create_context_menu(self):
        # create context menu
        self.popMenu = QtWidgets.QMenu(self)
        remove_action = QtGui.QAction('Remove Embedding', self)
        remove_action.triggered.connect(self.remove_embedding_from_list)
        self.popMenu.addAction(remove_action)

    def on_context_menu(self, point):
        # show context menu
        self.popMenu.exec_(self.mapToGlobal(point))

    def remove_embedding_from_list(self):
        main_window = self.main_window
        for i in range(main_window.inputEmbeddingsList.count()):
            list_item = main_window.inputEmbeddingsList.item(i)
            if list_item.listWidget().itemWidget(list_item) == self:
                main_window.inputEmbeddingsList.takeItem(i)   
                main_window.merged_embeddings.pop(i)
        widget_actions.refresh_frame(self.main_window)
        self.deleteLater()        


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
        self.merge_type_selection.setCurrentText(main_window.control['EmbMergeMethodSelection'])

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
    
class videoSeekSliderLineEditEventFilter(qtc.QObject):
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window
    
    def eventFilter(self, line_edit: QtWidgets.QLineEdit, event):
        if event.type() == qtc.QEvent.KeyPress:
            # Check if the pressed key is Enter/Return
            if event.key() in (qtc.Qt.Key_Enter, qtc.Qt.Key_Return):            
                new_value = line_edit.text()
                # Reset the line edit value to the slider value if the user input an empty text
                if new_value=='':
                    new_value = str(self.main_window.videoSeekSlider.value())
                else:
                    new_value = int(new_value)
                    max_frame_number = self.main_window.video_processor.max_frame_number
                    # If the value entered by user if greater than the max no of frames in the video, set the new value to the max_frame_number
                    if new_value > max_frame_number:
                        new_value = max_frame_number
                # Update values of line edit and slider
                line_edit.setText(str(new_value))
                self.main_window.videoSeekSlider.setValue(new_value)
                return True
        return False


class ParametersWidget:
    def __init__(self, *args, **kwargs):
        self.default_value = kwargs.get('default_value', False)
        self.min_value = kwargs.get('min_value',False)
        self.max_value = kwargs.get('max_value',False)
        self.group_layout_data: Dict[str, Dict[str, str|int|float|bool]]  = kwargs.get('group_layout_data', {})
        self.widget_name = kwargs.get('widget_name', False)
        self.label_widget: QtWidgets.QLabel = kwargs.get('label_widget', False)
        self.group_widget: QtWidgets.QGroupBox = kwargs.get('group_widget', False)
        self.main_window: 'MainWindow' = kwargs.get('main_window', False)
        self.line_edit: QtWidgets.QLineEdit = False #Only sliders have textbox currently
        self.reset_default_button: QPushButton = False
        self.enable_refresh_frame = True #This flag can be used to temporarily disable refreshing the frame when the widget value is changed

class ToggleSwitchButton(QtWidgets.QPushButton, ParametersWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)

class SelectionBox(QtWidgets.QComboBox, ParametersWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.selection_values = kwargs.get('selection_values', [])
        self.currentTextChanged.connect(partial(widget_actions.show_hide_related_widgets, self.main_window, self, self.widget_name, ))

    def reset_to_default_value(self):
        # Check if selection values are dynamically retrieved
        if callable(self.selection_values) and callable(self.default_value):
            self.clear()
            self.addItems(self.selection_values())
            self.setCurrentText(self.default_value())
        else:
            self.setCurrentText(self.default_value)

    def set_value(self, value):
        if callable(value):
            self.setCurrentText(value())
        else:
            self.setCurrentText(value)
    
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
        
        self.toggled.connect(partial(widget_actions.show_hide_related_widgets, self.main_window, self, self.widget_name, None))
        
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

    # Custom method in all parameter widgets to set value
    def set_value(self, value):
        self.setChecked(value)

class ParameterSlider(QtWidgets.QSlider, ParametersWidget):
    def __init__(self, min_value=0, max_value=0, default_value=0, step_size=1, fixed_width = 130, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        self.step_size = int(step_size)
        self.default_value = int(default_value)

        # Debounce timer for handle_slider_moved
        self.debounce_timer = qtc.QTimer()
        self.debounce_timer.setSingleShot(True)  # Assicura che il timer scatti una sola volta
        self.debounce_timer.timeout.connect(self.handle_slider_moved)  # Collega il timeout al metodo

        self.setMinimum(int(min_value))
        self.setMaximum(int(max_value))
        self.setValue(self.default_value)
        self.setOrientation(qtc.Qt.Orientation.Horizontal)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        # Set a fixed width for the slider
        self.setFixedWidth(fixed_width)

        # Connect sliderMoved with debounce
        self.sliderMoved.connect(self.start_debounce)

    def start_debounce(self):
        """Start debounce timer for slider movements."""
        self.debounce_timer.start(300)  # Attendi 300ms dopo lo spostamento dello slider

    def handle_slider_moved(self):
        """Handle the slider movement after debounce."""
        position = self.sliderPosition()  # Ottieni la posizione attuale dello slider
        """Handle the slider movement (dragging) and set the correct value."""
        new_value = round(position / self.step_size) * self.step_size

        # Set the scaled value
        self.setValue(new_value)
        if hasattr(self, 'line_edit'):
            self.line_edit.set_value(new_value)  # Aggiorna immediatamente il valore nel line edit
        print(f"Slider moved to: {new_value}")  # Debugging: log the final value

    def reset_to_default_value(self):
        self.setValue(int(self.default_value))

        # Aggiorna il line edit o altre componenti associate immediatamente
        if hasattr(self, 'line_edit'):
            self.line_edit.set_value(int(self.default_value))  # Aggiorna immediatamente il valore nel line edit

    def value(self):
        """Return the slider value as a float, scaled by the decimals."""
        return super().value()

    def setValue(self, value):
        """Set the slider value, scaling it from a float to the internal integer."""
        super().setValue(int(value))

    def wheelEvent(self, event):
        """Override wheel event to define custom increments/decrements with the mouse wheel."""
        num_steps = event.angleDelta().y() / 120  # 120 is one step of the wheel

        # Adjust the current value based on the number of steps
        current_value = self.value()

        # Calculate the new value based on the step size and num_steps
        new_value = current_value + (self.step_size * num_steps)

        # Ensure the new value is within the valid range
        new_value = min(max(new_value, self.min_value), self.max_value)

        # Update the slider's internal value (ensuring precision)
        self.setValue(new_value)

        # Aggiorna il line edit o altre componenti associate immediatamente
        if hasattr(self, 'line_edit'):
            self.line_edit.set_value(new_value)  # Aggiorna immediatamente il valore nel line edit

        # Accept the event
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Override key press event to handle arrow key increments/decrements."""
        # Get the current value of the slider
        current_value = self.value()

        # Check which key is pressed
        if event.key() == QtCore.Qt.Key_Right:
            # Increment value by step_size when right arrow is pressed
            new_value = current_value + self.step_size
        elif event.key() == QtCore.Qt.Key_Left:
            # Decrement value by step_size when left arrow is pressed
            new_value = current_value - self.step_size
        else:
            # Pass the event to the base class if it's not an arrow key
            super().keyPressEvent(event)
            return

        # Ensure the new value is within the valid range
        new_value = min(max(new_value, self.min_value), self.max_value)

        # Set the new value to the slider
        self.setValue(new_value)

        # Esegui immediatamente onchange_slider o l'aggiornamento necessario
        if hasattr(self, 'line_edit'):
            self.line_edit.set_value(new_value)  # Aggiorna il line edit con il nuovo valore

        # Accept the event
        event.accept()

    def mousePressEvent(self, event):
        """Handle the mouse press event to update the slider value immediately."""
        if event.button() == QtCore.Qt.LeftButton:  # Verifica che sia il pulsante sinistro del mouse
            # Calcola la posizione cliccata lungo la barra dello slider
            new_position = QtWidgets.QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), event.pos().x(), self.width()
            )
            # Applica lo step size, arrotondando il valore allo step più vicino
            new_value = round(new_position / self.step_size) * self.step_size

            # Aggiorna immediatamente il valore dello slider
            self.setValue(new_value)
            
            # Esegui immediatamente onchange_slider o l'aggiornamento necessario
            if hasattr(self, 'line_edit'):
                self.line_edit.set_value(new_value)  # Aggiorna il line edit con il nuovo valore

        # Chiama il metodo della classe base per gestire il resto dell'evento
        super().mousePressEvent(event)

    def set_value(self, value):
        self.setValue(value)
    
class ParameterDecimalSlider(QtWidgets.QSlider, ParametersWidget):
    def __init__(self, min_value=0.0, max_value=1.0, default_value=0.00, decimals=2, step_size=0.01, fixed_width = 130, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)

        # Ensure min, max, and default are floats
        min_value = float(min_value)
        max_value = float(max_value)
        default_value = float(default_value)

        # Store step size and decimal precision
        self.step_size = step_size
        self.decimals = decimals

        # Debounce timer for handle_slider_moved
        self.debounce_timer = qtc.QTimer()
        self.debounce_timer.setSingleShot(True)  # Assicura che il timer scatti una sola volta
        self.debounce_timer.timeout.connect(self.handle_slider_moved)  # Collega il timeout al metodo

        # Scale values for internal handling (to manage decimals)
        self.scale_factor = 10 ** self.decimals
        self.min_value = int(min_value * self.scale_factor)
        self.max_value = int(max_value * self.scale_factor)
        self.default_value = int(default_value * self.scale_factor)

        # Set slider properties
        self.setMinimum(self.min_value)
        self.setMaximum(self.max_value)
        self.setValue(float(self.default_value) / self.scale_factor)
        self.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.setFixedWidth(fixed_width)

        # Connect sliderMoved with debounce
        self.sliderMoved.connect(self.start_debounce)

    def start_debounce(self):
        """Start debounce timer for slider movements."""
        self.debounce_timer.start(300)  # Attendi 300ms dopo lo spostamento dello slider

    def handle_slider_moved(self):
        """Handle the slider movement after debounce."""
        position = self.sliderPosition()  # Ottieni la posizione attuale dello slider
        new_value = position / self.scale_factor
        new_value = round(new_value / self.step_size) * self.step_size

        # Imposta il nuovo valore
        self.setValue(new_value)
        if hasattr(self, 'line_edit'):
            self.line_edit.set_value(new_value)  # Aggiorna immediatamente il valore nel line edit
        print(f"Slider moved to: {new_value}")  # Debugging: log the final value

    def reset_to_default_value(self):
        """Reset the slider to its default value."""
        self.setValue(float(self.default_value) / self.scale_factor)

        # Aggiorna il line edit o altre componenti associate immediatamente
        if hasattr(self, 'line_edit'):
            self.line_edit.set_value(float(self.default_value) / self.scale_factor)  # Aggiorna immediatamente il valore nel line edit

    def value(self):
        """Return the slider value as a float, scaled by the decimals."""
        return super().value() / self.scale_factor

    def setValue(self, value):
        """Set the slider value, scaling it from a float to the internal integer."""
        # Arrotonda il valore a 2 decimali, come specificato in decimals
        value = round(value, self.decimals)
        
        # Moltiplica per il fattore di scala e arrotonda prima di convertirlo in intero
        scaled_value = int(round(float(value) * float(self.scale_factor)))

        super().setValue(scaled_value)

    def wheelEvent(self, event):
        """Override wheel event to define custom increments/decrements with the mouse wheel."""
        num_steps = event.angleDelta().y() / 120  # 120 is one step of the wheel

        # Adjust the current value based on the number of steps
        current_value = self.value()

        # Calculate the new value based on the step size and num_steps
        new_value = current_value + (self.step_size * num_steps)

        # Ensure the new value is within the valid range
        new_value = min(max(round(new_value, self.decimals), self.min_value / self.scale_factor), self.max_value / self.scale_factor)

        # Update the slider's internal value (ensuring precision)
        self.setValue(new_value)

        # Esegui immediatamente onchange_slider o l'aggiornamento necessario
        if hasattr(self, 'line_edit'):
            self.line_edit.set_value(new_value)  # Aggiorna il line edit con il nuovo valore
        
        # Accept the event
        event.accept()

    def keyPressEvent(self, event):
        """Override key press event to handle arrow key increments/decrements."""
        # Get the current value of the slider
        current_value = self.value()

        # Check which key is pressed
        if event.key() == QtCore.Qt.Key_Right:
            # Increment value by step_size when right arrow is pressed
            new_value = current_value + self.step_size
        elif event.key() == QtCore.Qt.Key_Left:
            # Decrement value by step_size when left arrow is pressed
            new_value = current_value - self.step_size
        else:
            # Pass the event to the base class if it's not an arrow key
            super().keyPressEvent(event)
            return

        # Ensure the new value is within the valid range
        new_value = min(max(round(new_value, self.decimals), self.min_value / self.scale_factor), self.max_value / self.scale_factor)

        # Set the new value to the slider
        self.setValue(new_value)

        # Esegui immediatamente onchange_slider o l'aggiornamento necessario
        if hasattr(self, 'line_edit'):
            self.line_edit.set_value(new_value)  # Aggiorna il line edit con il nuovo valore

        # Accept the event
        event.accept()

    def mousePressEvent(self, event):
        """Handle the mouse press event to update the slider value immediately."""
        if event.button() == QtCore.Qt.LeftButton:  # Verifica che sia il pulsante sinistro del mouse
            # Calcola la posizione cliccata lungo la barra dello slider
            new_position = QtWidgets.QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), event.pos().x(), self.width()
            )

            # Converti la nuova posizione nello spazio decimale
            new_value = new_position / self.scale_factor

            # Applica lo step size, arrotondando il valore allo step più vicino
            new_value = round(new_value / self.step_size) * self.step_size

            # Imposta il nuovo valore con la precisione corretta
            new_value = round(new_value, self.decimals)

            # Aggiorna immediatamente il valore dello slider
            self.setValue(new_value)

            # Esegui immediatamente onchange_slider o l'aggiornamento necessario
            if hasattr(self, 'line_edit'):
                self.line_edit.set_value(new_value)  # Aggiorna il line edit con il nuovo valore

        # Chiama il metodo della classe base per gestire il resto dell'evento
        super().mousePressEvent(event)

    def set_value(self, value):
        self.setValue(value)

class ParameterLineEdit(QtWidgets.QLineEdit):
    def __init__(self, min_value: int, max_value: int, default_value: str, fixed_width: int = 38, max_length: int = 3, alignment: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(fixed_width)  # Make the line edit narrower
        self.setMaxLength(max_length)
        self.setValidator(QtGui.QIntValidator(min_value, max_value))  # Restrict input to numbers

        # Optional: Align text to the right for better readability
        if alignment == 0:
            self.setAlignment(QtGui.Qt.AlignLeft)
        elif alignment == 1:
            self.setAlignment(QtGui.Qt.AlignCenter)
        else:
            self.setAlignment(QtGui.Qt.AlignRight)

        self.setText(default_value)

    def set_value(self, value: int):
        """Set the line edit's value."""
        self.setText(str(value))

class ParameterLineDecimalEdit(QtWidgets.QLineEdit):
    def __init__(self, min_value: float, max_value: float, default_value: str, decimals: int = 2, step_size=0.01, fixed_width: int = 38, max_length: int = 5, alignment: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(fixed_width)  # Adjust the width for decimal numbers
        self.decimals = decimals
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value
        default_value = float(default_value)
        self.setMaxLength(max_length)
        self.setValidator(QtGui.QDoubleValidator(min_value, max_value, decimals))
        # Optional: Align text to the right for better readability
        if alignment == 0:
            self.setAlignment(QtGui.Qt.AlignLeft)
        elif alignment == 1:
            self.setAlignment(QtGui.Qt.AlignCenter)
        else:
            self.setAlignment(QtGui.Qt.AlignRight)
        self.setText(f"{default_value:.{self.decimals}f}")

    def set_value(self, value: float):
        """Set the line edit's value with proper handling for step size and rounding."""
        # Clamp the value to ensure it's within min and max range
        new_value = max(min(value, self.max_value), self.min_value)

        # Round the value to the nearest step size
        rounded_value = round(new_value / self.step_size) * self.step_size

        # Ensure the value is rounded to the specified number of decimals
        rounded_value = round(rounded_value, self.decimals)

        # Ensure the formatted value has exactly 'self.decimals' decimal places, even for negative numbers
        format_string = f"{{:.{self.decimals}f}}"
        formatted_value = format_string.format(rounded_value)

        # Set the text with the correct number of decimal places
        self.setText(formatted_value)

    def get_value(self) -> float:
        """Get the current value from the line edit."""
        return float(self.text())

class ParameterText(QtWidgets.QLineEdit, ParametersWidget):
    def __init__(self, default_value: str, fixed_width: int = 130, max_length: int = 500, alignment: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.data_type = kwargs.get('data_type')
        self.exec_function = kwargs.get('exec_function')
        self.exec_function_args = kwargs.get('exec_function_args', [])

        self.setFixedWidth(fixed_width)  # Make the line edit narrower
        self.setMaxLength(max_length)
        self.default_value = default_value

        # Optional: Align text to the right for better readability
        if alignment == 0:
            self.setAlignment(QtGui.Qt.AlignLeft)
        elif alignment == 1:
            self.setAlignment(QtGui.Qt.AlignCenter)
        else:
            self.setAlignment(QtGui.Qt.AlignRight)

        # Set the initial text to the default value
        self.setText(self.default_value)

    def reset_to_default_value(self):
        """Reset the line edit to its default value."""
        self.setText(self.default_value)
        if self.data_type == 'parameter':
            widget_actions.update_parameter(self.main_window, self.widget_name, self.text(), enable_refresh_frame=self.enable_refresh_frame)
        else:
            widget_actions.update_control(self.main_window, self.widget_name, self.text(), exec_function=self.exec_function, exec_function_args=self.exec_function_args)

    def focusOutEvent(self, event):
        """Handle the focus out event (when the QLineEdit loses focus)."""
        if self.data_type == 'parameter':
            widget_actions.update_parameter(self.main_window, self.widget_name, self.text(), enable_refresh_frame=self.enable_refresh_frame)
        else:
            widget_actions.update_control(self.main_window, self.widget_name, self.text(), exec_function=self.exec_function, exec_function_args=self.exec_function_args)

        # Call the base class method to ensure normal behavior
        super().focusOutEvent(event)

    def set_value(self, value):
        self.setText(value)
class ParameterResetDefaultButton(QtWidgets.QPushButton):
    def __init__(self, related_widget: ParameterSlider | ParameterDecimalSlider | SelectionBox, *args, **kwargs):
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
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.setFlat(True)
