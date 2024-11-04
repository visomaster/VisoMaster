import cv2
import threading
import queue
from App.UI.Core.MainWindow import Ui_MainWindow
from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore
import App.UI.Widgets.WidgetActions as widget_actions
from functools import partial
from App.Processors.VideoProcessor import VideoProcessor
from App.Processors.ModelsProcessor import ModelsProcessor
from App.UI.Widgets.WidgetComponents import *
from App.UI.Widgets.UI_Workers import *
from App.UI.Widgets.SwapperLayoutData import SWAPPER_LAYOUT_DATA
from App.UI.Widgets.SettingsLayoutData import SETTINGS_LAYOUT_DATA
from App.UI.Widgets.FaceEditorLayoutData import FACE_EDITOR_LAYOUT_DATA
from typing import Dict, List
from App.Helpers.Misc_Helpers import DFM_MODELS_DATA


ParametersWidgetTypes = Dict[str, ToggleButton|SelectionBox|ParameterDecimalSlider|ParameterSlider|ParameterText]

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
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
        self.selected_video_buttons: List[TargetMediaCardButton] = [] #Contains list of buttons linked to videos/images
        self.selected_target_face_id = 0
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

        self.parameters_list = {}
        self.control = {}
        self.parameter_widgets: ParametersWidgetTypes = {}
        self.loaded_embedding_filename: str = ''
        self.processed_frames = {}
        self.next_frame_to_display = -1  # Index of the next frame to display
        self._is_slider_pressed = threading.Event()
        self.is_full_screen = False
        self.dfm_models_data = DFM_MODELS_DATA

    def initialize_widgets(self):
        # Initialize QListWidget for target media
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidget for face images
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize graphics frame to view frames
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        # Event filter to start playing when clicking on frame
        graphics_event_filter = GraphicsViewEventFilter(self, self.graphicsViewFrame,)
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)

        self.buttonSelectTargetVideos.clicked.connect(partial(widget_actions.onClickSelectTargetVideos, self, 'folder'))
        self.buttonSelectTargetVideoFiles.clicked.connect(partial(widget_actions.onClickSelectTargetVideos, self, 'files'))
        self.buttonSelectInputFaces.clicked.connect(partial(widget_actions.onClickSelectInputImages, self, 'folder'))
        self.buttonSelectInputFacesFiles.clicked.connect(partial(widget_actions.onClickSelectInputImages, self, 'files'))

        self.videoSeekSlider.valueChanged.connect(partial(widget_actions.OnChangeSlider, self))
        self.videoSeekSlider.sliderPressed.connect(partial(widget_actions.on_slider_pressed, self))
        self.videoSeekSlider.sliderReleased.connect(partial(widget_actions.on_slider_released, self))

        self.viewFullScreenButton.clicked.connect(partial(widget_actions.view_fullscreen, self))
        # Set up videoSeekLineEdit and add the event filter to handle changes
        widget_actions.set_up_video_seek_line_edit(self)
        video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(self, self.videoSeekLineEdit)
        self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)

        # Connect the Play/Stop button to the OnClickPlayButton method
        self.buttonMediaPlay.toggled.connect(partial(widget_actions.OnClickPlayButton, self))
        # self.buttonMediaStop.clicked.connect(partial(self.video_processor.stop_processing))
        self.findTargetFacesButton.clicked.connect(partial(widget_actions.find_target_faces, self))
        self.clearTargetFacesButton.clicked.connect(partial(widget_actions.clear_target_faces, self))
        self.targetVideosSearchBox.textChanged.connect(partial(widget_actions.filterTargetVideos, self))
        self.filterImagesCheckBox.clicked.connect(partial(widget_actions.filterTargetVideos, self))
        self.filterVideosCheckBox.clicked.connect(partial(widget_actions.filterTargetVideos, self))
        self.inputFacesSearchBox.textChanged.connect(partial(widget_actions.filterInputFaces, self))
        self.inputEmbeddingsSearchBox.textChanged.connect(partial(widget_actions.filterMergedEmbeddings, self))
        self.openEmbeddingButton.clicked.connect(partial(widget_actions.open_embeddings_from_file, self))
        self.saveEmbeddingButton.clicked.connect(partial(widget_actions.save_embeddings_to_file, self))
        self.saveEmbeddingAsButton.clicked.connect(partial(widget_actions.save_embeddings_to_file, self, True))

        self.swapfacesButton.clicked.connect(partial(widget_actions.process_swap_faces, self))
        self.editFacesButton.clicked.connect(partial(widget_actions.process_edit_faces, self))

        widget_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout, data_type='parameter')
        widget_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA, layoutWidget=self.settingsWidgetsLayout, data_type='control')
        widget_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA, layoutWidget=self.faceEditorWidgetsLayout, data_type='parameter')

        # Initialize the button states
        widget_actions.resetMediaButtons(self)

        # widget_actions.add_groupbox_and_widgets_from_layout_map(self)
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_variables()
        self.initialize_widgets()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Call the method to fit the image to the view whenever the window resizes
        if self.scene.items():
            widget_actions.fit_image_to_view(self, self.scene.items()[0])

    def reset_frame_counter(self):
        self.processed_frames.clear()
        self.next_frame_to_display = self.video_processor.current_frame_number or 0

    def keyPressEvent(self, event):
        # Toggle full screen when F11 is pressed
        if event.key() == QtCore.Qt.Key_F11:
            widget_actions.view_fullscreen(self)

    def closeEvent(self, event):
        print("MainWindow: closeEvent called.")

        self.video_processor.stop_processing()
        widget_actions.clear_stop_loading_input_media(self)
        widget_actions.clear_stop_loading_target_media(self)

        # Optionally handle the event if needed
        event.accept()