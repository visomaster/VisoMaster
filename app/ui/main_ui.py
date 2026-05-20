from typing import Dict
from pathlib import Path
from functools import partial
import copy

from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore

from app.ui.core.main_window import Ui_MainWindow
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import graphics_view_actions

from app.processors.video_processor import VideoProcessor
from app.processors.models_processor import ModelsProcessor
from app.ui.widgets import widget_components
from app.ui.widgets.event_filters import GraphicsViewEventFilter, VideoSeekSliderEventFilter, videoSeekSliderLineEditEventFilter, ListWidgetEventFilter
from app.ui.widgets import ui_workers
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
from app.helpers.miscellaneous import DFM_MODELS_DATA, ParametersDict
from app.helpers.typing_helper import FacesParametersTypes, ParametersTypes, ControlTypes, MarkerTypes

ParametersWidgetTypes = Dict[str, widget_components.ToggleButton|widget_components.SelectionBox|widget_components.ParameterDecimalSlider|widget_components.ParameterSlider|widget_components.ParameterText]

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = QtCore.Signal(int, int)
    model_loading_signal = QtCore.Signal()
    model_loaded_signal = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)
    def initialize_variables(self):
        self.video_loader_worker: ui_workers.TargetMediaLoaderWorker|bool = False
        self.webrtc_server_process = None  # multiprocessing.Process for WebRTC server
        self.input_faces_loader_worker: ui_workers.InputFacesLoaderWorker|bool = False
        self.target_videos_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='target_videos')
        self.input_faces_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='input_faces')
        self.merged_embeddings_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='merged_embeddings')
        self.video_processor = VideoProcessor(self)
        self.models_processor = ModelsProcessor(self)
        self.target_videos: Dict[int, widget_components.TargetMediaCardButton] = {} #Contains button objects of target videos (Set as list instead of single video to support batch processing in future)
        self.target_faces: Dict[int, widget_components.TargetFaceCardButton] = {} #Contains button objects of target faces
        self.input_faces: Dict[int, widget_components.InputFaceCardButton] = {} #Contains button objects of source faces (images)
        self.merged_embeddings: Dict[int, widget_components.EmbeddingCardButton] = {}
        self.cur_selected_target_face_button: widget_components.TargetFaceCardButton = False
        self.selected_video_button: widget_components.TargetMediaCardButton = False
        self.selected_target_face_id = False
        # '''
            # self.parameters dict have the following structure:
            # {
                # face_id (int): 
                # {
                    # parameter_name: parameter_value,
                    # ------
                # }
                # -----
            # }
        # '''
        self.parameters: FacesParametersTypes = {} 

        self.default_parameters: ParametersTypes = {}
        self.copied_parameters: ParametersTypes = {}
        self.current_widget_parameters: ParametersTypes = {}

        self.markers: MarkerTypes = {} #Video Markers (Contains parameters for each face)
        self.parameters_list = {}
        self.control: ControlTypes = {}
        self.parameter_widgets: ParametersWidgetTypes = {}
        self.loaded_embedding_filename: str = ''
        
        self.last_target_media_folder_path = ''
        self.last_input_media_folder_path = ''

        self.is_full_screen = False
        self.dfm_models_data = DFM_MODELS_DATA
        # This flag is used to make sure new loaded media is properly fit into the graphics frame on the first load
        self.loading_new_media = False

        # Streaming transform state — separate per source
        self.webcam_rotation = 0   # 0, 90, 180, 270 degrees clockwise
        self.webcam_flip_h = False
        self.webcam_flip_v = False
        self.webrtc_rotation = 0
        self.webrtc_flip_h = False
        self.webrtc_flip_v = False

        # Output window for OBS capture (initialized lazily)
        self._output_window = None

        self.gpu_memory_update_signal.connect(partial(common_widget_actions.set_gpu_memory_progressbar_value, self))
        self.placeholder_update_signal.connect(partial(common_widget_actions.update_placeholder_visibility, self))
        self.model_loading_signal.connect(partial(common_widget_actions.show_model_loading_dialog, self))
        self.model_loaded_signal.connect(partial(common_widget_actions.hide_model_loading_dialog, self))
        self.display_messagebox_signal.connect(partial(common_widget_actions.create_and_show_messagebox, self))
    def initialize_widgets(self):
        # Initialize QListWidget for target media (Media tab)
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidgets for streaming tabs
        self.webcamList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.webcamList.setWrapping(True)
        self.webcamList.setResizeMode(QtWidgets.QListWidget.Adjust)

        self.webrtcList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.webrtcList.setWrapping(True)
        self.webrtcList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidget for face images
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Set up Menu Actions
        layout_actions.set_up_menu_actions(self)

        # Set up placeholder texts in ListWidgets
        list_view_actions.set_up_list_widget_placeholder(self, self.targetVideosList)
        list_view_actions.set_up_list_widget_placeholder(self, self.webcamList)
        list_view_actions.set_up_list_widget_placeholder(self, self.webrtcList)
        list_view_actions.set_up_list_widget_placeholder(self, self.inputFacesList)

        # Set up click to select and drop action on ListWidgets
        self.targetVideosList.setAcceptDrops(True)
        self.targetVideosList.viewport().setAcceptDrops(False)
        self.inputFacesList.setAcceptDrops(True)
        self.inputFacesList.viewport().setAcceptDrops(False)
        list_widget_event_filter = ListWidgetEventFilter(self, self)
        self.targetVideosList.installEventFilter(list_widget_event_filter)
        self.targetVideosList.viewport().installEventFilter(list_widget_event_filter)
        self.webcamList.installEventFilter(list_widget_event_filter)
        self.webcamList.viewport().installEventFilter(list_widget_event_filter)
        self.webrtcList.installEventFilter(list_widget_event_filter)
        self.webrtcList.viewport().installEventFilter(list_widget_event_filter)
        self.inputFacesList.installEventFilter(list_widget_event_filter)
        self.inputFacesList.viewport().installEventFilter(list_widget_event_filter)

        # Set up folder open buttons for Target and Input
        self.buttonTargetVideosPath.clicked.connect(partial(list_view_actions.select_target_medias, self, 'folder'))
        self.buttonInputFacesPath.clicked.connect(partial(list_view_actions.select_input_face_images, self, 'folder'))

        # Initialize graphics frame to view frames
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        # Event filter to start playing when clicking on frame
        graphics_event_filter = GraphicsViewEventFilter(self, self.graphicsViewFrame,)
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)

        video_control_actions.enable_zoom_and_pan(self.graphicsViewFrame)

        video_slider_event_filter = VideoSeekSliderEventFilter(self, self.videoSeekSlider)
        self.videoSeekSlider.installEventFilter(video_slider_event_filter)
        self.videoSeekSlider.valueChanged.connect(partial(video_control_actions.on_change_video_seek_slider, self))
        self.videoSeekSlider.sliderPressed.connect(partial(video_control_actions.on_slider_pressed, self))
        self.videoSeekSlider.sliderReleased.connect(partial(video_control_actions.on_slider_released, self))
        video_control_actions.set_up_video_seek_slider(self)
        self.frameAdvanceButton.clicked.connect(partial(video_control_actions.advance_video_slider_by_n_frames, self))
        self.frameRewindButton.clicked.connect(partial(video_control_actions.rewind_video_slider_by_n_frames, self))

        self.addMarkerButton.clicked.connect(partial(video_control_actions.add_video_slider_marker, self))
        self.removeMarkerButton.clicked.connect(partial(video_control_actions.remove_video_slider_marker, self))
        self.nextMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_next_nearest_marker, self))
        self.previousMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_previous_nearest_marker, self))

        self.viewFullScreenButton.clicked.connect(partial(video_control_actions.view_fullscreen, self))
        # Set up videoSeekLineEdit and add the event filter to handle changes
        video_control_actions.set_up_video_seek_line_edit(self)
        video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(self, self.videoSeekLineEdit)
        self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)

        # Connect the Play/Stop button to the play_video method
        self.buttonMediaPlay.toggled.connect(partial(video_control_actions.play_video, self))
        self.buttonMediaRecord.toggled.connect(partial(video_control_actions.record_video, self))
        self.findTargetFacesButton.clicked.connect(partial(card_actions.find_target_faces, self))
        self.clearTargetFacesButton.clicked.connect(partial(card_actions.clear_target_faces, self))
        self.targetVideosSearchBox.textChanged.connect(partial(filter_actions.filter_target_videos, self))
        self.filterImagesCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
        self.filterVideosCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))

        # Input source tab switching (Media / Streaming)
        self.inputSourceTabWidget.currentChanged.connect(partial(self.on_input_source_tab_changed))
        # Streaming sub-tab switching (Webcam / WebRTC)
        self.streamingSubTabWidget.currentChanged.connect(partial(self.on_streaming_sub_tab_changed))

        # Webcam transform buttons
        self.webcamBtnRotateCCW.clicked.connect(partial(self._on_webcam_rotate_ccw))
        self.webcamBtnRotateCW.clicked.connect(partial(self._on_webcam_rotate_cw))
        self.webcamBtnFlipH.toggled.connect(partial(self._on_webcam_flip_h))
        self.webcamBtnFlipV.toggled.connect(partial(self._on_webcam_flip_v))

        # WebRTC transform buttons
        self.webrtcBtnRotateCCW.clicked.connect(partial(self._on_webrtc_rotate_ccw))
        self.webrtcBtnRotateCW.clicked.connect(partial(self._on_webrtc_rotate_cw))
        self.webrtcBtnFlipH.toggled.connect(partial(self._on_webrtc_flip_h))
        self.webrtcBtnFlipV.toggled.connect(partial(self._on_webrtc_flip_v))

        self.inputFacesSearchBox.textChanged.connect(partial(filter_actions.filter_input_faces, self))
        self.inputEmbeddingsSearchBox.textChanged.connect(partial(filter_actions.filter_merged_embeddings, self))
        self.openEmbeddingButton.clicked.connect(partial(save_load_actions.open_embeddings_from_file, self))
        self.saveEmbeddingButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self))
        self.saveEmbeddingAsButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self, True))

        self.swapfacesButton.clicked.connect(partial(video_control_actions.process_swap_faces, self))
        self.editFacesButton.clicked.connect(partial(video_control_actions.process_edit_faces, self))

        self.saveImageButton.clicked.connect(partial(video_control_actions.save_current_frame_to_file, self))
        self.clearMemoryButton.clicked.connect(partial(common_widget_actions.clear_gpu_memory, self))

        self.parametersPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_parameters_panel, self))
        self.facesPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_faces_panel, self))
        self.mediaPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_input_target_media_panel, self))

        self.faceMaskCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))
        self.faceCompareCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))

        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=COMMON_LAYOUT_DATA, layoutWidget=self.commonWidgetsLayout, data_type='parameter')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout, data_type='parameter')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA, layoutWidget=self.settingsWidgetsLayout, data_type='control')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA, layoutWidget=self.faceEditorWidgetsLayout, data_type='parameter')

        # Set up output folder select button (It is inside the settings tab Widget)
        self.outputFolderButton.clicked.connect(partial(list_view_actions.select_output_media_folder, self))
        # Create a control value for OutputMediaFolder
        common_widget_actions.create_control(self, 'OutputMediaFolder', '')

        # Initialize current_widget_parameters with default values
        self.current_widget_parameters = ParametersDict(copy.deepcopy(self.default_parameters), self.default_parameters)

        # Initialize the button states
        video_control_actions.reset_media_buttons(self)

        #Set GPU Memory Progressbar
        font = self.vramProgressBar.font()
        font.setBold(True)
        self.vramProgressBar.setFont(font)
        common_widget_actions.update_gpu_memory_progressbar(self)
        # Set face_swap_tab as the default focused tab
        self.tabWidget.setCurrentIndex(0)

        self.video_processor.fps_update_signal.connect(self._on_fps_update)


    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_variables()
        self.initialize_widgets()
        self.load_last_workspace()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        # print("Called resizeEvent()")
        super().resizeEvent(event)
        # Call the method to fit the image to the view whenever the window resizes
        if self.scene.items():
            pixmap_item = self.scene.items()[0]
            # Set the scene rectangle to the bounding rectangle of the pixmap
            scene_rect = pixmap_item.boundingRect()
            self.graphicsViewFrame.setSceneRect(scene_rect)
            graphics_view_actions.fit_image_to_view(self, pixmap_item, scene_rect )

    def keyPressEvent(self, event):
        match event.key():
            case QtCore.Qt.Key_F11:
                video_control_actions.view_fullscreen(self)
            case QtCore.Qt.Key_V:
                video_control_actions.advance_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_C:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_D:
                video_control_actions.advance_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_A:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_Z:
                self.videoSeekSlider.setValue(0)
            case QtCore.Qt.Key_Space:
                self.buttonMediaPlay.click()
            case QtCore.Qt.Key_R:
                self.buttonMediaRecord.click()
            case QtCore.Qt.Key_F:
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    video_control_actions.remove_video_slider_marker(self)
                else:
                    video_control_actions.add_video_slider_marker(self)
            case QtCore.Qt.Key_W:
                video_control_actions.move_slider_to_nearest_marker(self, 'next')
            case QtCore.Qt.Key_Q:
                video_control_actions.move_slider_to_nearest_marker(self, 'previous')
            case QtCore.Qt.Key_S:
                self.swapfacesButton.click()

    def closeEvent(self, event):
        print("MainWindow: closeEvent called.")

        self.video_processor.stop_processing()
        list_view_actions.clear_stop_loading_input_media(self)
        list_view_actions.clear_stop_loading_target_media(self)

        # Close output window if open
        if self._output_window is not None:
            self._output_window.close()
            self._output_window = None

        # Stop WebRTC server subprocess if running
        if self.webrtc_server_process and self.webrtc_server_process.is_alive():
            print("Stopping WebRTC server process...")
            self.webrtc_server_process.terminate()
            self.webrtc_server_process.join(timeout=3)
            self.webrtc_server_process = None

        save_load_actions.save_current_workspace(self, 'last_workspace.json')
        # Optionally handle the event if needed
        event.accept()

    def load_last_workspace(self):
        # Show the load workspace dialog if the file exists
        if Path('last_workspace.json').is_file():
            load_dialog = widget_components.LoadLastWorkspaceDialog(self)
            load_dialog.exec_()

    def on_input_source_tab_changed(self, tab_index):
        """Handle switching between Media (0) and Streaming (1) tabs."""
        self.video_processor.stop_processing()

        # Release webcam if switching away
        if self.selected_video_button and hasattr(self.selected_video_button, 'file_type') and self.selected_video_button.file_type == 'webcam':
            if self.selected_video_button.media_capture:
                self.selected_video_button.media_capture.release()
                self.selected_video_button.media_capture = None
            if self.video_processor.media_capture:
                self.video_processor.media_capture.release()
                self.video_processor.media_capture = None
            import time
            time.sleep(0.2)

        self.video_processor.file_type = None
        self.video_processor.media_capture = None
        self.video_processor.webrtc_shm = None
        self.video_processor.current_frame = []
        self.selected_video_button = False
        self.scene.clear()

        if tab_index == 0:  # Media tab — show folder group
            self.groupBox_TargetVideos_Select.setVisible(True)
            # Stop WebRTC server if running
            if self.webrtc_server_process and self.webrtc_server_process.is_alive():
                print("[WebRTC] Stopping WebRTC server (switched to Media tab)...")
                self.webrtc_server_process.terminate()
                self.webrtc_server_process.join(timeout=3)
                self.webrtc_server_process = None
            self.streamingFpsLabel.setText("FPS: --")
            self.streamingFpsLabel.setStyleSheet("")

        elif tab_index == 1:  # Streaming tab — hide folder group, trigger active sub-tab
            self.groupBox_TargetVideos_Select.setVisible(False)
            self.on_streaming_sub_tab_changed(self.streamingSubTabWidget.currentIndex())

    def on_streaming_sub_tab_changed(self, sub_index):
        """Handle switching between Webcam (0) and WebRTC (1) sub-tabs."""
        if self.inputSourceTabWidget.currentIndex() != 1:
            return

        self.video_processor.stop_processing()

        # Release webcam if switching away
        if self.selected_video_button and hasattr(self.selected_video_button, 'file_type') and self.selected_video_button.file_type == 'webcam':
            if self.selected_video_button.media_capture:
                self.selected_video_button.media_capture.release()
                self.selected_video_button.media_capture = None
            if self.video_processor.media_capture:
                self.video_processor.media_capture.release()
                self.video_processor.media_capture = None
            import time
            time.sleep(0.2)

        # Stop WebRTC server if switching away from WebRTC
        if sub_index != 1:
            if self.webrtc_server_process and self.webrtc_server_process.is_alive():
                print("[WebRTC] Stopping WebRTC server...")
                self.webrtc_server_process.terminate()
                self.webrtc_server_process.join(timeout=3)
                self.webrtc_server_process = None

        self.video_processor.file_type = None
        self.video_processor.media_capture = None
        self.video_processor.webrtc_shm = None
        self.video_processor.current_frame = []
        self.selected_video_button = False
        self.scene.clear()
        self.streamingFpsLabel.setText("FPS: --")
        self.streamingFpsLabel.setStyleSheet("")
        if sub_index == 0:  # Webcam
            list_view_actions.clear_stop_loading_target_media_streaming(self, 'webcam')
            list_view_actions.load_target_webcams(self)
        elif sub_index == 1:  # WebRTC — auto-start server
            list_view_actions.clear_stop_loading_target_media_streaming(self, 'webrtc')
            list_view_actions.load_target_webrtc(self)

    # ── Webcam transform handlers ────────────────────────────────────────────
    def _on_webcam_rotate_ccw(self):
        self.webcam_rotation = (self.webcam_rotation - 90) % 360
        self.webcamRotationLabel.setText(f"{self.webcam_rotation}°")

    def _on_webcam_rotate_cw(self):
        self.webcam_rotation = (self.webcam_rotation + 90) % 360
        self.webcamRotationLabel.setText(f"{self.webcam_rotation}°")

    def _on_webcam_flip_h(self, checked):
        self.webcam_flip_h = checked

    def _on_webcam_flip_v(self, checked):
        self.webcam_flip_v = checked

    # ── WebRTC transform handlers ────────────────────────────────────────────
    def _on_webrtc_rotate_ccw(self):
        self.webrtc_rotation = (self.webrtc_rotation - 90) % 360
        self.webrtcRotationLabel.setText(f"{self.webrtc_rotation}°")

    def _on_webrtc_rotate_cw(self):
        self.webrtc_rotation = (self.webrtc_rotation + 90) % 360
        self.webrtcRotationLabel.setText(f"{self.webrtc_rotation}°")

    def _on_webrtc_flip_h(self, checked):
        self.webrtc_flip_h = checked

    def _on_webrtc_flip_v(self, checked):
        self.webrtc_flip_v = checked

    @QtCore.Slot(float)
    def _on_fps_update(self, fps_value):
        """Update the FPS corner label with color coding based on value."""
        if self.inputSourceTabWidget.currentIndex() != 1:
            return
        self.streamingFpsLabel.setText(f"FPS: {fps_value:.1f}")
        if fps_value >= 20:
            color = "#4caf50"   # green — smooth
        elif fps_value >= 10:
            color = "#ff9800"   # orange — acceptable
        else:
            color = "#f44336"   # red — poor
        self.streamingFpsLabel.setStyleSheet(f"color: {color}; font-weight: bold;")

    def on_media_source_changed(self, source_index):
        """Legacy stub — kept for workspace save/load compatibility."""
        pass

    def save_last_workspace(self):
        pass