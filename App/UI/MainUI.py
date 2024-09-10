import sys
from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore as qtc
from App.UI.Core.MainWindow import Ui_MainWindow
from App.UI.Widgets import WidgetComponents, WidgetActions
from App.Workers import ThreadWorkers
import os
import cv2
from functools import partial

import qdarktheme


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def initialize_variables(self):
        self.frames = []
        self.current_frame_index = 0
        self.video_loader_worker = False

        self.timer = qtc.QTimer()
        self.timer.timeout.connect(partial(WidgetActions.update_frame, self))


        self.media_capture = False

    def initialize_widgets(self):
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

        self.buttonSelectTargetVideos.clicked.connect(partial(WidgetActions.onClickSelectTargetVideos, self))

        self.videoSeekSlider.valueChanged.connect(partial(WidgetActions.getSliderCurrentPos, self))


    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_variables()
        self.initialize_widgets()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Call the method to fit the image to the view whenever the window resizes
        print('self.scene.items()',self.scene.items())
        if self.scene.items():
            WidgetActions.fit_image_to_view(self, self.scene.items()[0])
            print("HEllo")