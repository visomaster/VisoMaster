from App.UI.Core.MainWindow import Ui_MainWindow
from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore as qtc
import App.UI.Widgets.WidgetActions as widget_actions
from functools import partial
from App.Processors.VideoProcessor import VideoProcessor
from App.UI.Widgets.WidgetComponents import GraphicsViewEventFilter


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    update_frame_signal = qtc.Signal(Ui_MainWindow, QtGui.QPixmap, int)

    def initialize_variables(self):
        self.video_loader_worker = False
        self.video_processor = VideoProcessor(self)
        self.thread_pool = qtc.QThreadPool()
        self.selected_video_buttons = []

    def initialize_widgets(self):

        # self.input_Target_DockWidget.setWindowTitle('Target Videos and Input Faces')
        # self.controlOptionsDockWidget.setWindowTitle('Control Options')
        # Initialize QListWidget for target media
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidget for face images
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        graphics_event_filter = GraphicsViewEventFilter(self.graphicsViewFrame)
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)

        self.buttonSelectTargetVideos.clicked.connect(partial(widget_actions.onClickSelectTargetVideosFolder, self))
        self.buttonSelectTargetVideoFiles.clicked.connect(partial(widget_actions.onClickSelectTargetVideosFiles, self))

        self.videoSeekSlider.valueChanged.connect(partial(widget_actions.OnChangeSlider, self))

        self.buttonMediaPlay.clicked.connect(partial(widget_actions.OnClickPlayButton, self))
        # self.buttonMediaStop.clicked.connect(partial(self.video_processor.stop_processing))

        self.update_frame_signal.connect(widget_actions.update_graphics_view)
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