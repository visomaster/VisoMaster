# pylint: disable=keyword-arg-before-vararg
import os
from functools import partial
import uuid
from typing import TYPE_CHECKING, Dict

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QPushButton
import cv2
import numpy as np

import app.ui.widgets.actions.common_actions as common_widget_actions
import app.ui.widgets.actions.layout_actions as layout_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import save_load_actions
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

class CardButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.main_window: 'MainWindow' = kwargs.get('main_window', False)
        self.list_item  = None

class TargetMediaCardButton(CardButton):
    def __init__(self, media_path: str, file_type: str, media_id:str, is_webcam=False, webcam_index=-1, webcam_backend=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.media_id = media_id
        self.file_type = file_type
        self.media_path = media_path
        self.is_webcam = is_webcam
        self.webcam_index = webcam_index
        self.webcam_backend = webcam_backend
        self.media_capture: cv2.VideoCapture|bool = False
        self.setCheckable(True)
        self.setToolTip(media_path)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)  # Space between icon and label
        filename = os.path.basename(media_path)
        text_label = QtWidgets.QLabel(filename, self)
        text_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)
        text_label.setStyleSheet("font-size: 8px; font-weight:bold;")  # Style for the label
        layout.addWidget(text_label)
        self.clicked.connect(self.load_media)
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

    def reset_media_state(self):
        main_window = self.main_window
        # Deselect the currently selected video
        if main_window.selected_video_button:
            main_window.selected_video_button.toggle()  # Deselect the previous video
            main_window.selected_video_button = False
        
        # Stop the current video processing
        main_window.video_processor.stop_processing()

    def reset_related_widgets_and_values(self):
        main_window = self.main_window

        # Set up videoSeekLineEdit
        video_control_actions.set_up_video_seek_line_edit(main_window)
        # Clear current target faces
        card_actions.clear_target_faces(main_window, refresh_frame=False)
        # Uncheck input faces
        card_actions.uncheck_all_input_faces(main_window)
        # Uncheck merged embeddings
        card_actions.uncheck_all_merged_embeddings(main_window)
        # Remove all markers
        video_control_actions.remove_all_markers(main_window)

        main_window.cur_selected_target_face_button = False

        # Reset buttons and slider
        video_control_actions.reset_media_buttons(main_window)

    def load_media(self):

        main_window = self.main_window
        # Deselect the currently selected video
        if main_window.selected_video_button:
            main_window.selected_video_button.toggle()  # Deselect the previous video
            main_window.selected_video_button = False
        
        # Stop the current video processing
        main_window.video_processor.stop_processing()

        if main_window.selected_target_face_id:
            main_window.current_widget_parameters = main_window.parameters[main_window.selected_target_face_id].copy()

        if main_window.control.get('AutoSwapToggle'):
            prev_selected_input_faces = [face for _,face in main_window.input_faces.items() if face.isChecked()]
            prev_selected_embeddings = [embed for _,embed in main_window.merged_embeddings.items() if embed.isChecked()]
        # Reset the frame counter
        main_window.video_processor.current_frame_number = 0
        main_window.video_processor.media_path = self.media_path
        main_window.parameters = {}
        main_window.selected_target_face_id = False

        # Release the previous media_capture if it exists
        if main_window.video_processor.media_capture:
            main_window.video_processor.media_capture.release()

        frame = None
        max_frames_number = 0  # Initialize max_frames_number for either video or image
        
        if self.file_type == 'video':
            media_capture = cv2.VideoCapture(self.media_path)
            if not media_capture.isOpened():
                print(f"Error opening video {self.media_path}")
                return  # If the video cannot be opened, exit the function

            media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            max_frames_number = int(media_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            _, frame = misc_helpers.read_frame(media_capture)
            main_window.video_processor.media_capture = media_capture
            self.media_capture = media_capture
            main_window.video_processor.fps = media_capture.get(cv2.CAP_PROP_FPS)
            main_window.video_processor.max_frame_number = max_frames_number

        elif self.file_type == 'image':
            frame = misc_helpers.read_image_file(self.media_path)
            max_frames_number = 0  # For an image, there is only one "frame"
            main_window.video_processor.max_frame_number = max_frames_number

        elif self.file_type == 'webcam':
            res_width, res_height = self.main_window.control['WebcamMaxResSelection'].split('x')

            media_capture = cv2.VideoCapture(self.webcam_index, self.webcam_backend)
            media_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(res_width))
            media_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(res_height))
            max_frames_number = 999999
            _, frame = misc_helpers.read_frame(media_capture)
            main_window.video_processor.media_capture = media_capture
            self.media_capture = media_capture
            main_window.video_processor.fps = media_capture.get(cv2.CAP_PROP_FPS)
            main_window.video_processor.max_frame_number = max_frames_number

        if frame is not None:
            main_window.scene.clear()
            if self.file_type == 'video':
                # restore initial video position after reading. == 0
                media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            pixmap = common_widget_actions.get_pixmap_from_frame(main_window, frame)
            graphics_view_actions.update_graphics_view(main_window, pixmap, 0, reset_fit=True)

        self.reset_related_widgets_and_values()

        main_window.video_processor.file_type = self.file_type
        main_window.videoSeekSlider.blockSignals(True)  # Block signals to prevent unnecessary updates
        main_window.videoSeekSlider.setMaximum(max_frames_number)
        main_window.videoSeekSlider.setValue(0)  # Set the slider to 0 for the new video

        main_window.videoSeekSlider.blockSignals(False)  # Unblock signals

        # Append the selected video button to the list
        main_window.selected_video_button = self

        # Update the graphics frame after the reset
        main_window.graphicsViewFrame.update()

        # Set Parameter widget values to default
        common_widget_actions.set_widgets_values_using_face_id_parameters(main_window=main_window, face_id=False)
        
        main_window.loading_new_media = True
        common_widget_actions.refresh_frame(main_window)

        if main_window.control.get('AutoSwapToggle'):
            card_actions.find_target_faces(main_window)
            for _, target_face in main_window.target_faces.items():
                for input_face in prev_selected_input_faces:
                    target_face.assigned_input_faces[input_face.face_id] = input_face.embedding_store
                for embedding in prev_selected_embeddings:
                    target_face.assigned_merged_embeddings[embedding.embedding_id] = embedding.embedding_store
                target_face.calculate_assigned_input_embedding()
            if main_window.target_faces:
                list(main_window.target_faces.values())[0].click()
            common_widget_actions.refresh_frame(main_window)
            layout_actions.fit_image_to_view_onchange(main_window)


        # list_view_actions.find_target_faces(main_window)

    def remove_target_media_from_list(self):
        main_window = self.main_window

        # Deselect the currently selected video
        if main_window.selected_video_button == self:
            self.reset_media_state()
        
            # Reset the frame counter
            main_window.video_processor.current_frame_number = 0
            main_window.video_processor.media_path = False
            main_window.parameters = {}
            main_window.selected_target_face_id = False

            main_window.video_processor.media_capture = False
            main_window.video_processor.fps = 0
            main_window.video_processor.max_frame_number = 0

            self.main_window.scene.clear()

            self.reset_related_widgets_and_values()

            main_window.videoSeekSlider.blockSignals(True)  # Block signals to prevent unnecessary updates
            main_window.videoSeekSlider.setMaximum(1)
            main_window.videoSeekSlider.setValue(0)  # Set the slider to 0 for the new video
            main_window.videoSeekSlider.blockSignals(False)  # Unblock signals
            # Append the selected video button to the list
            main_window.selected_video_button = False


            # Update the graphics frame after the reset
            main_window.graphicsViewFrame.update()

            main_window.video_processor.file_type = None

            if self.media_capture:
                self.media_capture.release()
                self.media_capture = False

        for i in range(main_window.targetVideosList.count()-1, -1, -1):
            list_item = main_window.targetVideosList.item(i)
            if list_item:
                if list_item.listWidget().itemWidget(list_item) == self:
                    main_window.targetVideosList.takeItem(i)   
                    main_window.target_videos.pop(self.media_id)

        # If the target media list is empty, show the placeholder text
        if not main_window.target_videos:
            main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

        self.deleteLater()

    def create_context_menu(self):
        self.popMenu = QtWidgets.QMenu(self)
        remove_action = QtGui.QAction('Remove from list', self)
        remove_action.triggered.connect(self.remove_target_media_from_list)
        self.popMenu.addAction(remove_action)

    def on_context_menu(self, point):
        # show context menu
        self.popMenu.exec_(self.mapToGlobal(point))

class TargetFaceCardButton(CardButton):
    def __init__(self, media_path, cropped_face, embedding_store: Dict[str, np.ndarray], face_id:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if self.main_window.target_faces:
        #     self.face_id = max([target_face.face_id for target_face in self.main_window.target_faces]) + 1
        # else:
        #     self.face_id = 0
        self.face_id = face_id
        self.media_path = media_path
        self.cropped_face = cropped_face

        self.embedding_store = embedding_store  # Key: embedding_swap_model, Value: embedding

        self.assigned_input_faces: Dict[str, Dict[str, np.ndarray]] = {}  # Inside Dict (key - input face_id): {Key: embedding_swap_model, Value: InputFaceCardButton.embedding_store}
        self.assigned_merged_embeddings: Dict[str, Dict[str, np.ndarray]] = {}  # Key: embedding_swap_model, Value: EmbeddingCardButton.embedding_store
        self.assigned_input_embedding = {}  # Key: embedding_swap_model, Value: np.ndarray
        
        self.setCheckable(True)
        self.clicked.connect(self.load_target_face)

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
            common_widget_actions.create_parameter_dict_for_face_id(self.main_window, self.face_id)

    def set_embedding(self, embedding_swap_model: str, embedding: np.ndarray):
        self.embedding_store[embedding_swap_model] = embedding

    def get_embedding(self, embedding_swap_model: str) -> np.ndarray:
        return self.embedding_store.get(embedding_swap_model, np.array([]))

    def load_target_face(self):
        main_window = self.main_window
        main_window.cur_selected_target_face_button = self
        self.setChecked(True)
        for _, target_face_button in main_window.target_faces.items():
            # Uncheck all other target faces
            if target_face_button!=self:
                target_face_button.setChecked(False)

        card_actions.uncheck_all_input_faces(main_window)
        card_actions.uncheck_all_merged_embeddings(main_window)

        for input_face_id in self.assigned_input_faces.keys():
            main_window.input_faces[input_face_id].setChecked(True)
        for embedding_id in self.assigned_merged_embeddings.keys():
            main_window.merged_embeddings[embedding_id].setChecked(True)
        
        main_window.selected_target_face_id = self.face_id

        # print('main_window.selected_target_face_id', main_window.selected_target_face_id)     
        common_widget_actions.set_widgets_values_using_face_id_parameters(main_window=main_window, face_id=self.face_id)      
        # common_widget_actions.refresh_frame(main_window)

        main_window.current_widget_parameters = main_window.parameters[self.face_id]

    def calculate_assigned_input_embedding(self):
        control = self.main_window.control.copy()

        all_input_embeddings = []
        all_embedding_swap_models = set()

        # Itera su `assigned_input_faces` e raccogli gli embedding e i modelli
        for _, embedding_store in self.assigned_input_faces.items():
            if embedding_store:  # Verifica se l'embedding_store non è vuoto
                all_embedding_swap_models.update(embedding_store.keys())
                all_input_embeddings.append(embedding_store)  # Aggiungi l'intero store
        
        # Itera su `assigned_merged_embeddings` e raccogli gli embedding e i modelli
        for _, embedding_store in self.assigned_merged_embeddings.items():
            if embedding_store:  # Verifica se l'embedding_store non è vuoto
                all_embedding_swap_models.update(embedding_store.keys())
                all_input_embeddings.append(embedding_store)  # Aggiungi l'intero store

        # Calcolo degli embedding se presenti
        if len(all_input_embeddings) > 0:
            if control['EmbMergeMethodSelection'] == 'Mean':
                self.assigned_input_embedding = {
                    model: np.mean([store[model] for store in all_input_embeddings if model in store], axis=0)
                    for model in all_embedding_swap_models
                }
            elif control['EmbMergeMethodSelection'] == 'Median':
                self.assigned_input_embedding = {
                    model: np.median([store[model] for store in all_input_embeddings if model in store], axis=0)
                    for model in all_embedding_swap_models
                }

        else:
            self.assigned_input_embedding = {}

    def create_context_menu(self):
        # create context menu
        self.popMenu = QtWidgets.QMenu(self)
        parameters_copy_action = QtGui.QAction('Copy Parameters', self)
        parameters_copy_action.triggered.connect(self.copy_parameters)
        parameters_paste_action = QtGui.QAction('Apply Copied Parameters', self)
        parameters_paste_action.triggered.connect(self.paste_and_apply_parameters)
        save_parameters_action = QtGui.QAction('Save Current Parameters and Settings', self)
        save_parameters_action.triggered.connect(partial(save_load_actions.save_current_parameters_and_control, self.main_window, self.face_id))
        load_parameters_action = QtGui.QAction('Load Parameters', self)
        load_parameters_action.triggered.connect(partial(save_load_actions.load_parameters_and_settings, self.main_window, self.face_id))
        load_parameters_and_settings_action = QtGui.QAction('Load Parameters and Settings', self)
        load_parameters_and_settings_action.triggered.connect(partial(save_load_actions.load_parameters_and_settings, self.main_window, self.face_id, True))
        remove_action = QtGui.QAction('Remove from List', self)
        remove_action.triggered.connect(self.remove_target_face_from_list)
        self.popMenu.addAction(parameters_copy_action)
        self.popMenu.addAction(parameters_paste_action)
        self.popMenu.addAction(save_parameters_action)
        self.popMenu.addAction(load_parameters_action)
        self.popMenu.addAction(load_parameters_and_settings_action)
        self.popMenu.addAction(remove_action)

    def on_context_menu(self, point):
        # show context menu
        self.popMenu.exec_(self.mapToGlobal(point))

    def remove_target_face_from_list(self):
        main_window = self.main_window

        if main_window.video_processor.processing:
            main_window.video_processor.stop_processing()
            
        for i in range(main_window.targetFacesList.count()-1, -1, -1):
            list_item = main_window.targetFacesList.item(i)
            if list_item:
                if list_item.listWidget().itemWidget(list_item) == self:
                    main_window.targetFacesList.takeItem(i)   
                    main_window.target_faces.pop(self.face_id)
                    # Pop parameters using the target's face_id
                    main_window.parameters.pop(self.face_id)
        # Click and Select the first target face if target_faces are not empty
        if main_window.target_faces:
            list(main_window.target_faces.values())[0].click()
        # Otherwise reset parameter widgets value to the default
        else:
            common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, face_id=False)
        video_control_actions.remove_face_parameters_and_control_from_markers(main_window, self.face_id) #Remove parameters for the face from all markers
        common_widget_actions.refresh_frame(self.main_window)
        self.deleteLater()

    def remove_assigned_input_face(self, input_face_id):
        if self.assigned_input_faces.get(input_face_id):
            self.assigned_input_faces.pop(input_face_id)
            self.calculate_assigned_input_embedding()


    def remove_assigned_merged_embedding(self, embedding_id):
        if self.assigned_merged_embeddings.get(embedding_id):
            self.assigned_merged_embeddings.pop(embedding_id)
            self.calculate_assigned_input_embedding()

    def copy_parameters(self):

        self.main_window.copied_parameters = self.main_window.parameters[self.face_id].copy()

    def paste_and_apply_parameters(self):
        if not self.main_window.copied_parameters:
            common_widget_actions.create_and_show_messagebox(self.main_window, 'No parameters found in Clipboard', 'You need to copy parameters from any of the target face before pasting it!', parent_widget=self)
        else:
            self.main_window.parameters[self.face_id] = self.main_window.copied_parameters.copy()
            common_widget_actions.set_widgets_values_using_face_id_parameters(self.main_window, face_id=self.face_id)
            common_widget_actions.refresh_frame(main_window=self.main_window)

class InputFaceCardButton(CardButton):
    def __init__(self, media_path, cropped_face, embedding_store: Dict[str, np.ndarray], face_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_id = face_id
        self.cropped_face = cropped_face
        self.embedding_store = embedding_store  # Key: embedding_swap_model, Value: embedding
        self.media_path = media_path

        self.setCheckable(True)
        self.setToolTip(media_path)
        self.clicked.connect(self.load_input_face)

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

    def set_embedding(self, embedding_swap_model: str, embedding: np.ndarray):
        self.embedding_store[embedding_swap_model] = embedding

    def get_embedding(self, embedding_swap_model: str) -> np.ndarray:
        return self.embedding_store.get(embedding_swap_model, np.array([]))

    def load_input_face(self):
        main_window = self.main_window

        if main_window.cur_selected_target_face_button:
            cur_selected_target_face_button = main_window.cur_selected_target_face_button
            if not QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
                for input_face_id in cur_selected_target_face_button.assigned_input_faces.keys():
                    input_face_button = main_window.input_faces[input_face_id]
                    if input_face_button!=self:
                        input_face_button.setChecked(False)
                cur_selected_target_face_button.assigned_input_faces = {}

            cur_selected_target_face_button.assigned_input_faces[self.face_id] = self.embedding_store

            if not self.isChecked():
                cur_selected_target_face_button.assigned_input_faces.pop(self.face_id)
            cur_selected_target_face_button.calculate_assigned_input_embedding()
        else:
            if not QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
                # If there is no target face selected, uncheck all other input faces
                for _, input_face_button in main_window.input_faces.items():
                    if input_face_button!=self:
                        input_face_button.setChecked(False)

        common_widget_actions.refresh_frame(main_window)
        
    def remove_input_face_from_list(self):
        main_window = self.main_window
        for i in range(main_window.inputFacesList.count()-1, -1, -1):
            list_item = main_window.inputFacesList.item(i)
            if list_item.listWidget().itemWidget(list_item) == self:
                main_window.inputFacesList.takeItem(i)   
                main_window.input_faces.pop(self.face_id)
                for target_face_id in main_window.target_faces:
                    main_window.target_faces[target_face_id].remove_assigned_input_face(self.face_id)
        common_widget_actions.refresh_frame(self.main_window)
        self.deleteLater()
        # If the input faces list is empty, show the placeholder text
        if not main_window.input_faces:
            main_window.placeholder_update_signal.emit(self.main_window.inputFacesList, False)

    def create_context_menu(self):
        # create context menu
        self.popMenu = QtWidgets.QMenu(self)
        create_embed_action = QtGui.QAction('Create embedding from selected faces', self)
        create_embed_action.triggered.connect(self.create_embedding_from_selected_faces)
        self.popMenu.addAction(create_embed_action)

        remove_action = QtGui.QAction('Remove from list', self)
        remove_action.triggered.connect(self.remove_input_face_from_list)
        self.popMenu.addAction(remove_action)
    def on_context_menu(self, point):
        # show context menu
        self.popMenu.exec_(self.mapToGlobal(point))

    def create_embedding_from_selected_faces(self):
        # Raccogli l'intero embedding_store dalle facce selezionate
        selected_faces_embeddings_store = [
            input_face.embedding_store 
            for _, input_face in self.main_window.input_faces.items() 
            if input_face.isChecked()
        ]

        # Controlla se ci sono facce selezionate
        if len(selected_faces_embeddings_store) == 0:
            common_widget_actions.create_and_show_messagebox(
                self.main_window, 
                "No Faces Selected!", 
                "You need to select at least one face to create a merged embedding!", 
                self
            )
        else:
            # Passa l'intero embedding_store al dialogo per la creazione dell'embedding
            embed_create_dialog = CreateEmbeddingDialog(self.main_window, selected_faces_embeddings_store)
            embed_create_dialog.exec_()

class EmbeddingCardButton(CardButton):
    def __init__(self, embedding_name: str, embedding_store: Dict[str, np.ndarray], embedding_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_id = embedding_id
        self.embedding_store = embedding_store  # Key: embedding_swap_model, Value: embedding
        self.embedding_name = embedding_name
        self.setCheckable(True)
        self.setText(embedding_name)
        self.setToolTip(embedding_name)
        self.clicked.connect(self.load_embedding)

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

    def set_embedding(self, embedding_swap_model: str, embedding: np.ndarray):
        self.embedding_store[embedding_swap_model] = embedding

    def get_embedding(self, embedding_swap_model: str):
        """Restituisce l'embedding associato a un embedding_swap_model, se esiste."""
        return self.embedding_store.get(embedding_swap_model, None)

    def load_embedding(self):
        main_window = self.main_window
        if main_window.cur_selected_target_face_button:
            
            cur_selected_target_face_button = main_window.cur_selected_target_face_button
            if not QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
                for embedding_id in cur_selected_target_face_button.assigned_merged_embeddings.keys():
                    embed_button = main_window.merged_embeddings[embedding_id]
                    if embed_button!=self:
                        embed_button.setChecked(False)
                cur_selected_target_face_button.assigned_merged_embeddings = {}

            cur_selected_target_face_button.assigned_merged_embeddings[self.embedding_id] = self.embedding_store

            if not self.isChecked():
                cur_selected_target_face_button.assigned_merged_embeddings.pop(self.embedding_id)
            cur_selected_target_face_button.calculate_assigned_input_embedding()
        else:
            if not QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
                # If there is no target face selected, uncheck all other input faces
                for embedding_id, embed_button in main_window.merged_embeddings.items():
                    if embed_button!=self:
                        embed_button.setChecked(False)

        common_widget_actions.refresh_frame(main_window)

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
        for i in range(main_window.inputEmbeddingsList.count()-1, -1, -1):
            list_item = main_window.inputEmbeddingsList.item(i)
            if list_item.listWidget().itemWidget(list_item) == self:
                main_window.inputEmbeddingsList.takeItem(i)   
                main_window.merged_embeddings.pop(self.embedding_id)
                for target_face_id in main_window.target_faces:
                    main_window.target_faces[target_face_id].remove_assigned_merged_embedding(self.embedding_id)
        common_widget_actions.refresh_frame(self.main_window)
        self.deleteLater()

class CreateEmbeddingDialog(QtWidgets.QDialog):
    def __init__(self, main_window: 'MainWindow', embedding_stores: list=None):
        super().__init__()
        self.embedding_stores = embedding_stores or []
        self.main_window = main_window
        self.embedding_name = ''
        self.merge_type = ''
        self.setWindowTitle("Create Embedding")
        self.setWindowIcon(QtGui.QIcon(u":/media/media/visomaster_small.png"))

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
        self.embedding_name = self.embed_name_edit.text().strip()
        self.merge_type = self.merge_type_selection.currentText()

        if self.embedding_name == '':
            common_widget_actions.create_and_show_messagebox(self.main_window, 'Empty Embedding Name!', 'Embedding Name cannot be empty!', self)
        else:
            # Estrai tutti gli embedding per ogni embedding_swap_model
            merged_embedding_store = {}
            
            for embedding_store in self.embedding_stores:
                for embedding_swap_model, embedding in embedding_store.items():
                    if embedding_swap_model not in merged_embedding_store:
                        merged_embedding_store[embedding_swap_model] = []
                    merged_embedding_store[embedding_swap_model].append(embedding)

            # Calcola l'embedding unito per ciascun embedding_swap_model
            final_embedding_store = {}
            for swap_model, embeddings in merged_embedding_store.items():
                if self.merge_type == 'Mean':
                    final_embedding_store[swap_model] = np.mean(embeddings, axis=0)
                elif self.merge_type == 'Median':
                    final_embedding_store[swap_model] = np.median(embeddings, axis=0)

            # Crea e aggiungi il nuovo embedding_store con tutti i modelli di swap
            list_view_actions.create_and_add_embed_button_to_list(
                main_window=self.main_window, 
                embedding_name=self.embedding_name, 
                embedding_store=final_embedding_store,  # Passa l'intero embedding_store
                embedding_id=str(uuid.uuid1().int)
            )
            self.accept()



class LoadingDialog(QtWidgets.QDialog):
    def __init__(self, message="Loading Models, please wait...\nDon't panic if it looks stuck!"):
        super().__init__()
        self.setWindowTitle("Loading Models")
        self.setWindowIcon(QtGui.QIcon(u":/media/media/visomaster_small.png"))
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.setModal(True)  # Block interaction with other windows
        self.setFixedSize(225, 125)  # Increased size for better layout

        # Create main layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)  # Add some padding
        layout.setSpacing(8)  # Add spacing between elements

        # Icon Label
        self.icon_label = QtWidgets.QLabel()
        self.icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setPixmap(
            QtGui.QPixmap(":/media/media/repeat.png").scaled(
                30, 30, 
                QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
        )

        # Message Label
        self.label = QtWidgets.QLabel(message)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)  # Allow text to wrap within the dialog
        self.label.setStyleSheet("""
            font-size: 12px;  /* Set font size */
            font-weight: bold;  /* Make the text bold */
        """)

        # Add widgets to layout
        layout.addWidget(self.icon_label)
        layout.addWidget(self.label)
        self.setLayout(layout)

# Custom progress dialog
class ProgressDialog(QtWidgets.QProgressDialog):
    pass

class LoadLastWorkspaceDialog(QtWidgets.QDialog):
    def __init__(self, main_window: 'MainWindow',):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Load Last Workspace")
        self.setWindowIcon(QtGui.QIcon(u":/media/media/visomaster_small.png"))

        # Create button box
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.load_workspace)
        self.buttonBox.rejected.connect(self.reject)

        # Create layout and add widgets
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Do you want to load your last workspace?"))
        layout.addWidget(self.buttonBox)

        # Set dialog layout
        self.setLayout(layout)

    def load_workspace(self):
        self.accept()
        save_load_actions.load_saved_workspace(self.main_window, 'last_workspace.json')    

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
        self.line_edit: ParameterLineEdit|ParameterLineDecimalEdit = False #Only sliders have textbox currently
        self.reset_default_button: QPushButton = False
        self.enable_refresh_frame = True #This flag can be used to temporarily disable refreshing the frame when the widget value is changed

class SelectionBox(QtWidgets.QComboBox, ParametersWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ParametersWidget.__init__(self, *args, **kwargs)
        self.selection_values = kwargs.get('selection_values', [])
        self.currentTextChanged.connect(partial(common_widget_actions.show_hide_related_widgets, self.main_window, self, self.widget_name, ))

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

    def __init__(self, bg_color="#000000", circle_color="#ffffff", active_color="#4facc9", default_value=False, *args, **kwargs):
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
        
        self.toggled.connect(partial(common_widget_actions.show_hide_related_widgets, self.main_window, self, self.widget_name, None))
        
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
        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(True)  # Assicura che il timer scatti una sola volta
        self.debounce_timer.timeout.connect(self.handle_slider_moved)  # Collega il timeout al metodo

        self.setMinimum(int(min_value))
        self.setMaximum(int(max_value))
        self.setValue(self.default_value)
        self.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        # Set a fixed width for the slider
        self.setFixedWidth(fixed_width)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

        # Connect sliderMoved with debounce
        self.sliderMoved.connect(self.start_debounce)

    def start_debounce(self):
        """Start debounce timer for slider movements."""
        self.debounce_timer.start(300)  # Attendi 300ms dopo lo spostamento dello slider

    def handle_slider_moved(self):
        """Handle the slider movement after debounce."""
        position = self.sliderPosition()  # Ottieni la posizione attuale dello slider
        # """Handle the slider movement (dragging) and set the correct value."""
        new_value = round(position / self.step_size) * self.step_size

        # Set the scaled value
        self.setValue(new_value)

        # print(f"Slider moved to: {new_value}")  # Debugging: log the final value

    def reset_to_default_value(self):
        self.setValue(int(self.default_value))

    # def value(self):
    #     # """Return the slider value as a float, scaled by the decimals."""
    #     return super().value()

    def setValue(self, value):
        """Set the slider value, scaling it from a float to the internal integer."""
        super().setValue(int(value))
        if self.line_edit:
            self.line_edit.set_value(int(value))  # Aggiorna immediatamente il valore nel line edit

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

        # Accept the event
        event.accept()

    def mousePressEvent(self, event):
        """Handle the mouse press event to update the slider value immediately."""
        if event.button() == QtCore.Qt.LeftButton:  # Verifica che sia il pulsante sinistro del mouse
            self.setValue(self.pos_to_value(event.pos().x()))

        # Chiama il metodo della classe base per gestire il resto dell'evento
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        new_value = self.pos_to_value(event.pos().x())
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f'{new_value}')
        super().mouseMoveEvent(event)

    def set_value(self, value):
        self.setValue(value)

    def pos_to_value(self, x) -> float:
        # Calcola la posizione cliccata lungo la barra dello slider
        new_position = QtWidgets.QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), x, self.width()
        )
        # Applica lo step size, arrotondando il valore allo step più vicino
        return round(new_position / self.step_size) * self.step_size

    
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
        self.debounce_timer = QtCore.QTimer()
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
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

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

        # print(f"Slider moved to: {new_value}")  # Debugging: log the final value

    def reset_to_default_value(self):
        """Reset the slider to its default value."""
        self.setValue(float(self.default_value) / self.scale_factor)

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
        if self.line_edit:
            self.line_edit.set_value(float(value))

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

        # Accept the event
        event.accept()

    def mousePressEvent(self, event):
        """Handle the mouse press event to update the slider value immediately."""
        if event.button() == QtCore.Qt.LeftButton:  # Verifica che sia il pulsante sinistro del mouse
            # Aggiorna immediatamente il valore dello slider
            self.setValue(self.pos_to_value(event.pos().x()))

        # Chiama il metodo della classe base per gestire il resto dell'evento
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        new_value = self.pos_to_value(event.pos().x())
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f'{new_value}')
        super().mouseMoveEvent(event)

    def set_value(self, value):
        self.setValue(value)

    def pos_to_value(self, x) -> float:
        new_position = QtWidgets.QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), x, self.width()
        )

        # Converti la nuova posizione nello spazio decimale
        new_value = new_position / self.scale_factor

        # Applica lo step size, arrotondando il valore allo step più vicino
        new_value = round(new_value / self.step_size) * self.step_size

        # Imposta il nuovo valore con la precisione corretta
        return round(new_value, self.decimals)


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
            common_widget_actions.update_parameter(self.main_window, self.widget_name, self.text(), enable_refresh_frame=self.enable_refresh_frame)
        else:
            common_widget_actions.update_control(self.main_window, self.widget_name, self.text(), exec_function=self.exec_function, exec_function_args=self.exec_function_args)

    def focusOutEvent(self, event):
        """Handle the focus out event (when the QLineEdit loses focus)."""
        if self.data_type == 'parameter':
            common_widget_actions.update_parameter(self.main_window, self.widget_name, self.text(), enable_refresh_frame=self.enable_refresh_frame)
        else:
            common_widget_actions.update_control(self.main_window, self.widget_name, self.text(), exec_function=self.exec_function, exec_function_args=self.exec_function_args)

        # Call the base class method to ensure normal behavior
        super().focusOutEvent(event)

    def set_value(self, value):
        self.setText(value)
class ParameterResetDefaultButton(QtWidgets.QPushButton):
    def __init__(self, related_widget: ParameterSlider | ParameterDecimalSlider | SelectionBox, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.related_widget = related_widget
        button_icon = QtGui.QIcon(QtGui.QPixmap(':/media/media/reset_default.png'))
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
