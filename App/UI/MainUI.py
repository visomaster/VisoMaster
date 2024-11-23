import cv2
import threading
import queue
from App.UI.Core.MainWindow import Ui_MainWindow
from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore
import App.UI.Widgets.Actions.LayoutActions as layout_actions

import App.UI.Widgets.Actions.VideoControlActions as video_control_actions
from App.UI.Widgets.Actions import FilterActions as filter_actions
from App.UI.Widgets.Actions import EmbeddingActions as embedding_actions
from App.UI.Widgets.Actions import ListViewActions as list_view_actions

from functools import partial
from App.Processors.VideoProcessor import VideoProcessor
from App.Processors.ModelsProcessor import ModelsProcessor
from App.UI.Widgets.WidgetComponents import *
from App.UI.Widgets.EventFilters import GraphicsViewEventFilter, VideoSeekSliderEventFilter, videoSeekSliderLineEditEventFilter, ListWidgetEventFilter
from App.UI.Widgets.UI_Workers import *
from App.UI.Widgets.SwapperLayoutData import SWAPPER_LAYOUT_DATA
from App.UI.Widgets.SettingsLayoutData import SETTINGS_LAYOUT_DATA
from App.UI.Widgets.FaceEditorLayoutData import FACE_EDITOR_LAYOUT_DATA
from typing import Dict, List
from App.Helpers.Misc_Helpers import DFM_MODELS_DATA


ParametersWidgetTypes = Dict[str, ToggleButton|SelectionBox|ParameterDecimalSlider|ParameterSlider|ParameterText]

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    placeholder_update_signal = qtc.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = qtc.Signal(int, int)
    def initialize_variables(self):
        self.video_loader_worker: TargetMediaLoaderWorker|bool = False
        self.input_faces_loader_worker: InputFacesLoaderWorker|bool = False
        self.target_videos_filter_worker = FilterWorker(main_window=self, search_text='', filter_list='target_videos')
        self.input_faces_filter_worker = FilterWorker(main_window=self, search_text='', filter_list='input_faces')
        self.merged_embeddings_filter_worker = FilterWorker(main_window=self, search_text='', filter_list='merged_embeddings')
        self.video_processor = VideoProcessor(self)
        self.models_processor = ModelsProcessor(self)
        self.target_videos: List[TargetMediaCardButton] = [] #Contains button objects of target videos (Set as list instead of single video to support batch processing in future)
        self.target_faces: List[TargetFaceCardButton] = [] #Contains button objects of target faces
        self.input_faces: List[InputFaceCardButton] = [] #Contains button objects of source faces (images)
        self.merged_embeddings: List[EmbeddingCardButton] = []
        self.cur_selected_target_face_button: TargetFaceCardButton = False
        self.selected_video_button: TargetMediaCardButton = False
        self.selected_target_face_id = False
        '''
            self.parameters dict have the following structure:
            {
                face_id (int): 
                {
                    parameter_name: parameter_value,
                    ------
                }
                -----
            }
        '''
        self.parameters: Dict[int, Dict[str, bool|int|float|str]] = {} 

        self.default_parameters: Dict[str, bool|int|float|str] = {}
        self.copied_parameters: Dict[str, bool|int|float|str] = {}

        self.markers: Dict[int, Dict[int, Dict[str, bool|int|float|str]]] = {} #Video Markers (Contains parameters for each face)
        self.parameters_list = {}
        self.control = {}
        self.parameter_widgets: ParametersWidgetTypes = {}
        self.loaded_embedding_filename: str = ''
        
        self.is_full_screen = False
        self.dfm_models_data = DFM_MODELS_DATA
        # This flag is used to make sure new loaded media is properly fit into the graphics frame on the first load
        self.loading_new_media = False

        self.gpu_memory_update_signal.connect(partial(common_widget_actions.set_gpu_memory_progressbar_value, self))
        self.placeholder_update_signal.connect(partial(common_widget_actions.update_placeholder_visibility, self))

    def initialize_widgets(self):
        # Initialize QListWidget for target media
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidget for face images
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Set up Menu Actions
        layout_actions.set_up_menu_actions(self)

        # Set up placeholder texts in ListWidgets (Target Videos and Input Faces)
        list_view_actions.set_up_list_widget_placeholder(self, self.targetVideosList)
        list_view_actions.set_up_list_widget_placeholder(self, self.inputFacesList)

        # Set up drop action on ListWidgets
        self.targetVideosList.setAcceptDrops(True)
        listWidgetEventFilter = ListWidgetEventFilter(self, print, self.targetVideosList)
        self.targetVideosList.installEventFilter(listWidgetEventFilter)
        self.inputFacesList.installEventFilter(listWidgetEventFilter)

        # Initialize graphics frame to view frames
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        # Event filter to start playing when clicking on frame
        graphics_event_filter = GraphicsViewEventFilter(self, self.graphicsViewFrame,)
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)

        video_control_actions.enable_zoom_and_pan(self.graphicsViewFrame)

        video_slider_event_filter = VideoSeekSliderEventFilter(self, self.videoSeekSlider)
        self.videoSeekSlider.installEventFilter(video_slider_event_filter)
        self.videoSeekSlider.valueChanged.connect(partial(video_control_actions.OnChangeSlider, self))
        self.videoSeekSlider.sliderPressed.connect(partial(video_control_actions.on_slider_pressed, self))
        self.videoSeekSlider.sliderReleased.connect(partial(video_control_actions.on_slider_released, self))
        self.videoSeekSlider.sliderMoved.connect(partial(print, 'Slider Moved ()'))
        video_control_actions.set_up_video_seek_slider(self)
        self.addMarkerButton.clicked.connect(partial(video_control_actions.add_video_slider_marker, self))
        self.removeMarkerButton.clicked.connect(partial(video_control_actions.remove_video_slider_marker, self))
        self.nextMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_next_nearest_marker, self))
        self.previousMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_previous_nearest_marker, self))

        self.viewFullScreenButton.clicked.connect(partial(video_control_actions.view_fullscreen, self))
        # Set up videoSeekLineEdit and add the event filter to handle changes
        video_control_actions.set_up_video_seek_line_edit(self)
        video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(self, self.videoSeekLineEdit)
        self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)

        # Connect the Play/Stop button to the OnClickPlayButton method
        self.buttonMediaPlay.toggled.connect(partial(video_control_actions.OnClickPlayButton, self))
        self.buttonMediaRecord.toggled.connect(partial(video_control_actions.OnClickRecordButton, self))
        # self.buttonMediaStop.clicked.connect(partial(self.video_processor.stop_processing))
        self.findTargetFacesButton.clicked.connect(partial(list_view_actions.find_target_faces, self))
        self.clearTargetFacesButton.clicked.connect(partial(card_actions.clear_target_faces, self))
        self.targetVideosSearchBox.textChanged.connect(partial(filter_actions.filterTargetVideos, self))
        self.filterImagesCheckBox.clicked.connect(partial(filter_actions.filterTargetVideos, self))
        self.filterVideosCheckBox.clicked.connect(partial(filter_actions.filterTargetVideos, self))
        self.filterWebcamsCheckBox.clicked.connect(partial(filter_actions.filterTargetVideos, self))
        self.filterWebcamsCheckBox.clicked.connect(partial(list_view_actions.onClickLoadWebcams, self))

        self.inputFacesSearchBox.textChanged.connect(partial(filter_actions.filterInputFaces, self))
        self.inputEmbeddingsSearchBox.textChanged.connect(partial(filter_actions.filterMergedEmbeddings, self))
        self.openEmbeddingButton.clicked.connect(partial(embedding_actions.open_embeddings_from_file, self))
        self.saveEmbeddingButton.clicked.connect(partial(embedding_actions.save_embeddings_to_file, self))
        self.saveEmbeddingAsButton.clicked.connect(partial(embedding_actions.save_embeddings_to_file, self, True))

        self.swapfacesButton.clicked.connect(partial(video_control_actions.process_swap_faces, self))
        self.editFacesButton.clicked.connect(partial(video_control_actions.process_edit_faces, self))

        self.saveImageButton.clicked.connect(partial(video_control_actions.save_current_frame_to_file, self))
        self.clearMemoryButton.clicked.connect(partial(common_widget_actions.clear_gpu_memory, self))

        self.parametersPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_parameters_panel, self))
        self.facesPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_faces_panel, self))
        self.mediaPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_input_target_media_panel, self))

        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout, data_type='parameter')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA, layoutWidget=self.settingsWidgetsLayout, data_type='control')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA, layoutWidget=self.faceEditorWidgetsLayout, data_type='parameter')

        # Initialize the button states
        video_control_actions.resetMediaButtons(self)

        #Set GPU Memory Progressbar
        font = self.vramProgressBar.font()
        font.setBold(True)
        self.vramProgressBar.setFont(font)
        common_widget_actions.update_gpu_memory_progressbar(self)
        # widget_actions.add_groupbox_and_widgets_from_layout_map(self)
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_variables()
        self.initialize_widgets()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        print("Called resizeEvent()")
        super().resizeEvent(event)
        # Call the method to fit the image to the view whenever the window resizes
        if self.scene.items():
            graphics_view_actions.fit_image_to_view(self, self.scene.items()[0])

    def keyPressEvent(self, event):
        # Toggle full screen when F11 is pressed
        if event.key() == QtCore.Qt.Key_F11:
            video_control_actions.view_fullscreen(self)

    def closeEvent(self, event):
        print("MainWindow: closeEvent called.")

        self.video_processor.stop_processing()
        list_view_actions.clear_stop_loading_input_media(self)
        list_view_actions.clear_stop_loading_target_media(self)

        # Optionally handle the event if needed
        event.accept()