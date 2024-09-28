import cv2
from App.UI.Core.MainWindow import Ui_MainWindow
from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore
import App.UI.Widgets.WidgetActions as widget_actions
from functools import partial
from App.Processors.VideoProcessor import VideoProcessor
from App.Processors.ModelsProcessor import ModelsProcessor
from App.UI.Widgets.WidgetComponents import GraphicsViewEventFilter
from App.UI.Widgets.LayoutData import SWAPPER_LAYOUT_DATA

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    update_frame_signal = QtCore.Signal(int, QtGui.QPixmap)

    def initialize_variables(self):
        self.video_loader_worker = False
        self.input_faces_loader_worker = False
        self.video_processor = VideoProcessor(self)
        self.models_processor = ModelsProcessor(self)
        self.target_videos = [] #Contains button objects of target videos (Set as list instead of single video to support batch processing in future)
        self.target_faces = [] #Contains button objects of target faces
        self.input_faces = [] #Contains button objects of source faces (images)
        self.selected_target_face_buttons = [] 
        self.selected_input_face_buttons = []
        self.selected_input_emb_buttons = []
        self.selected_video_buttons = [] #Contains list of buttons linked to videos/images
        self.parameters = {}
        self.parameters_list = {}
        self.control = {}
        self.parameter_widgets = {}
        self.processed_frames = {}
        self.next_frame_to_display = None  # Index of the next frame to display

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
        self.videoSeekSlider.sliderReleased.connect(self.on_slider_released)

        # Connect the Play/Stop button to the OnClickPlayButton method
        self.buttonMediaPlay.toggled.connect(partial(widget_actions.OnClickPlayButton, self))
        # self.buttonMediaStop.clicked.connect(partial(self.video_processor.stop_processing))
        self.findTargetFacesButton.clicked.connect(partial(widget_actions.find_target_faces, self))
        self.clearTargetFacesButton.clicked.connect(partial(widget_actions.clear_target_faces, self))
        self.targetVideosSearchBox.textChanged.connect(partial(widget_actions.filterTargetVideos, self))
        self.inputFacesSearchBox.textChanged.connect(partial(widget_actions.filterInputFaces, self))

        self.update_frame_signal.connect(self.handle_processed_frame)

        widget_actions.initializeModelLoadDialog(self)
        widget_actions.add_parameter_widgets(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout)

        # Initialize the button states
        widget_actions.resetMediaButtons(self)

    def on_slider_released(self):
        new_position = self.videoSeekSlider.value()
        print(f"Slider released. New position: {new_position}")
    
        # Perform the update to the new frame
        video_processor = self.video_processor
        if video_processor.media_capture:
            video_processor.process_current_frame()  # Process the current frame

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

    @QtCore.Slot(int, QtGui.QPixmap)
    def handle_processed_frame(self, frame_number, pixmap):
        self.processed_frames[frame_number] = pixmap
        self.display_frames_in_order()

    def display_frames_in_order(self):
        if self.next_frame_to_display is None:
            if self.processed_frames:
                self.next_frame_to_display = min(self.processed_frames.keys())
            else:
                return  # No frames processed yet

        while self.next_frame_to_display in self.processed_frames:
            pixmap = self.processed_frames.pop(self.next_frame_to_display)
            widget_actions.update_graphics_view(self, pixmap, self.next_frame_to_display)
            self.next_frame_to_display += 1

    def reset_frame_counter(self):
        self.processed_frames.clear()
        self.next_frame_to_display = None

    def closeEvent(self, event):
        print("MainWindow: closeEvent called.")
        self.video_processor.stop_processing()
        widget_actions.clear_stop_loading_input_media(self)
        widget_actions.clear_stop_loading_target_media(self)

        # Optionally handle the event if needed
        event.accept()