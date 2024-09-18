# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QDockWidget, QGraphicsView, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QSizePolicy, QSlider, QSpacerItem, QTabWidget,
    QVBoxLayout, QWidget)
import media_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1081, 586)
        font = QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        icon = QIcon()
        icon.addFile(u":/media/Media/rope_logo.jpg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.graphicsViewFrame = QGraphicsView(self.widget)
        self.graphicsViewFrame.setObjectName(u"graphicsViewFrame")

        self.verticalLayout.addWidget(self.graphicsViewFrame)

        self.verticalLayoutMediaControls = QVBoxLayout()
        self.verticalLayoutMediaControls.setObjectName(u"verticalLayoutMediaControls")
        self.videoSeekSlider = QSlider(self.widget)
        self.videoSeekSlider.setObjectName(u"videoSeekSlider")
        self.videoSeekSlider.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayoutMediaControls.addWidget(self.videoSeekSlider)

        self.horizontalLayoutMediaButtons = QHBoxLayout()
        self.horizontalLayoutMediaButtons.setObjectName(u"horizontalLayoutMediaButtons")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer)

        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy1)
        self.pushButton.setMaximumSize(QSize(100, 16777215))
        icon1 = QIcon()
        icon1.addFile(u":/media/Media/previous_marker_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton.setIcon(icon1)
        self.pushButton.setFlat(True)

        self.horizontalLayoutMediaButtons.addWidget(self.pushButton)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer_3)

        self.buttonMediaPlay = QPushButton(self.widget)
        self.buttonMediaPlay.setObjectName(u"buttonMediaPlay")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.buttonMediaPlay.sizePolicy().hasHeightForWidth())
        self.buttonMediaPlay.setSizePolicy(sizePolicy2)
        self.buttonMediaPlay.setMaximumSize(QSize(100, 16777215))
        icon2 = QIcon()
        icon2.addFile(u":/media/Media/play_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.buttonMediaPlay.setIcon(icon2)
        self.buttonMediaPlay.setCheckable(True)
        self.buttonMediaPlay.setFlat(True)

        self.horizontalLayoutMediaButtons.addWidget(self.buttonMediaPlay)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer_4)

        self.pushButton_2 = QPushButton(self.widget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        sizePolicy1.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy1)
        self.pushButton_2.setMaximumSize(QSize(100, 16777215))
        icon3 = QIcon()
        icon3.addFile(u":/media/Media/next_marker_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton_2.setIcon(icon3)
        self.pushButton_2.setFlat(True)

        self.horizontalLayoutMediaButtons.addWidget(self.pushButton_2)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer_2)


        self.verticalLayoutMediaControls.addLayout(self.horizontalLayoutMediaButtons)


        self.verticalLayout.addLayout(self.verticalLayoutMediaControls)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.gridGroupBox = QGroupBox(self.widget)
        self.gridGroupBox.setObjectName(u"gridGroupBox")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.gridGroupBox.sizePolicy().hasHeightForWidth())
        self.gridGroupBox.setSizePolicy(sizePolicy3)
        self.gridGroupBox.setMaximumSize(QSize(16777215, 180))
        self.gridGroupBox.setAutoFillBackground(False)
        self.gridGroupBox.setFlat(True)
        self.gridGroupBox.setCheckable(False)
        self.gridLayout_2 = QGridLayout(self.gridGroupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.widget_2 = QWidget(self.gridGroupBox)
        self.widget_2.setObjectName(u"widget_2")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy4)
        self.verticalLayout_8 = QVBoxLayout(self.widget_2)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.findTargetFacesButton = QPushButton(self.widget_2)
        self.findTargetFacesButton.setObjectName(u"findTargetFacesButton")
        self.findTargetFacesButton.setMinimumSize(QSize(100, 0))
        self.findTargetFacesButton.setCheckable(False)
        self.findTargetFacesButton.setFlat(True)

        self.verticalLayout_8.addWidget(self.findTargetFacesButton)

        self.clearTargetFacesButton = QPushButton(self.widget_2)
        self.clearTargetFacesButton.setObjectName(u"clearTargetFacesButton")
        self.clearTargetFacesButton.setCheckable(False)
        self.clearTargetFacesButton.setFlat(True)

        self.verticalLayout_8.addWidget(self.clearTargetFacesButton)

        self.pushButton_4 = QPushButton(self.widget_2)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setCheckable(True)
        self.pushButton_4.setFlat(True)

        self.verticalLayout_8.addWidget(self.pushButton_4)

        self.pushButton_6 = QPushButton(self.widget_2)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setCheckable(True)
        self.pushButton_6.setFlat(True)

        self.verticalLayout_8.addWidget(self.pushButton_6)


        self.gridLayout_2.addWidget(self.widget_2, 1, 0, 1, 1)

        self.targetFacesList = QListWidget(self.gridGroupBox)
        self.targetFacesList.setObjectName(u"targetFacesList")
        self.targetFacesList.setAutoFillBackground(True)

        self.gridLayout_2.addWidget(self.targetFacesList, 1, 1, 1, 1)

        self.inputEmbeddingsList = QListWidget(self.gridGroupBox)
        self.inputEmbeddingsList.setObjectName(u"inputEmbeddingsList")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.inputEmbeddingsList.sizePolicy().hasHeightForWidth())
        self.inputEmbeddingsList.setSizePolicy(sizePolicy5)

        self.gridLayout_2.addWidget(self.inputEmbeddingsList, 1, 2, 1, 1)

        self.inputEmbeddingsSearchBox = QLineEdit(self.gridGroupBox)
        self.inputEmbeddingsSearchBox.setObjectName(u"inputEmbeddingsSearchBox")

        self.gridLayout_2.addWidget(self.inputEmbeddingsSearchBox, 0, 2, 1, 1)


        self.verticalLayout.addWidget(self.gridGroupBox)


        self.horizontalLayout.addWidget(self.widget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.input_Target_DockWidget = QDockWidget(MainWindow)
        self.input_Target_DockWidget.setObjectName(u"input_Target_DockWidget")
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.gridLayout_4 = QGridLayout(self.dockWidgetContents)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.vboxLayout = QVBoxLayout()
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.groupBox_TargetVideos_Select = QGroupBox(self.dockWidgetContents)
        self.groupBox_TargetVideos_Select.setObjectName(u"groupBox_TargetVideos_Select")
        self.gridLayout_3 = QGridLayout(self.groupBox_TargetVideos_Select)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.labelTargetVideosPath = QLabel(self.groupBox_TargetVideos_Select)
        self.labelTargetVideosPath.setObjectName(u"labelTargetVideosPath")
        self.labelTargetVideosPath.setWordWrap(False)

        self.gridLayout_3.addWidget(self.labelTargetVideosPath, 1, 0, 1, 1)

        self.buttonSelectTargetVideos = QPushButton(self.groupBox_TargetVideos_Select)
        self.buttonSelectTargetVideos.setObjectName(u"buttonSelectTargetVideos")

        self.gridLayout_3.addWidget(self.buttonSelectTargetVideos, 0, 0, 1, 1)

        self.buttonSelectTargetVideoFiles = QPushButton(self.groupBox_TargetVideos_Select)
        self.buttonSelectTargetVideoFiles.setObjectName(u"buttonSelectTargetVideoFiles")

        self.gridLayout_3.addWidget(self.buttonSelectTargetVideoFiles, 0, 1, 1, 1)


        self.vboxLayout.addWidget(self.groupBox_TargetVideos_Select)

        self.targetVideosSearchBox = QLineEdit(self.dockWidgetContents)
        self.targetVideosSearchBox.setObjectName(u"targetVideosSearchBox")

        self.vboxLayout.addWidget(self.targetVideosSearchBox)

        self.targetVideosList = QListWidget(self.dockWidgetContents)
        self.targetVideosList.setObjectName(u"targetVideosList")

        self.vboxLayout.addWidget(self.targetVideosList)

        self.groupBox_InputFaces_Select = QGroupBox(self.dockWidgetContents)
        self.groupBox_InputFaces_Select.setObjectName(u"groupBox_InputFaces_Select")
        self.gridLayout = QGridLayout(self.groupBox_InputFaces_Select)
        self.gridLayout.setObjectName(u"gridLayout")
        self.labelInputFacesPath = QLabel(self.groupBox_InputFaces_Select)
        self.labelInputFacesPath.setObjectName(u"labelInputFacesPath")

        self.gridLayout.addWidget(self.labelInputFacesPath, 1, 0, 1, 1)

        self.buttonSelectInputFaces = QPushButton(self.groupBox_InputFaces_Select)
        self.buttonSelectInputFaces.setObjectName(u"buttonSelectInputFaces")

        self.gridLayout.addWidget(self.buttonSelectInputFaces, 0, 0, 1, 1)

        self.buttonSelectInputFacesFiles = QPushButton(self.groupBox_InputFaces_Select)
        self.buttonSelectInputFacesFiles.setObjectName(u"buttonSelectInputFacesFiles")

        self.gridLayout.addWidget(self.buttonSelectInputFacesFiles, 0, 1, 1, 1)


        self.vboxLayout.addWidget(self.groupBox_InputFaces_Select)

        self.inputFacesSearchBox = QLineEdit(self.dockWidgetContents)
        self.inputFacesSearchBox.setObjectName(u"inputFacesSearchBox")

        self.vboxLayout.addWidget(self.inputFacesSearchBox)

        self.inputFacesList = QListWidget(self.dockWidgetContents)
        self.inputFacesList.setObjectName(u"inputFacesList")

        self.vboxLayout.addWidget(self.inputFacesList)


        self.gridLayout_4.addLayout(self.vboxLayout, 0, 0, 1, 1)

        self.input_Target_DockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.input_Target_DockWidget)
        self.controlOptionsDockWidget = QDockWidget(MainWindow)
        self.controlOptionsDockWidget.setObjectName(u"controlOptionsDockWidget")
        self.dockWidgetContents_2 = QWidget()
        self.dockWidgetContents_2.setObjectName(u"dockWidgetContents_2")
        self.gridLayout_5 = QGridLayout(self.dockWidgetContents_2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.tabWidget = QTabWidget(self.dockWidgetContents_2)
        self.tabWidget.setObjectName(u"tabWidget")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(1)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy6)
        font1 = QFont()
        font1.setFamilies([u"Segoe UI Semibold"])
        font1.setPointSize(10)
        font1.setBold(True)
        self.tabWidget.setFont(font1)
        self.tabWidget.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.tabWidget.setTabPosition(QTabWidget.TabPosition.North)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(True)
        self.tabWidget.setTabBarAutoHide(False)
        self.face_swap_tab = QWidget()
        self.face_swap_tab.setObjectName(u"face_swap_tab")
        self.verticalLayout_4 = QVBoxLayout(self.face_swap_tab)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_tab3 = QLabel(self.face_swap_tab)
        self.label_tab3.setObjectName(u"label_tab3")
        self.label_tab3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_tab3)

        self.tabWidget.addTab(self.face_swap_tab, "")
        self.face_editor_tab = QWidget()
        self.face_editor_tab.setObjectName(u"face_editor_tab")
        self.verticalLayout_3 = QVBoxLayout(self.face_editor_tab)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_tab2 = QLabel(self.face_editor_tab)
        self.label_tab2.setObjectName(u"label_tab2")
        self.label_tab2.setEnabled(True)
        sizePolicy4.setHeightForWidth(self.label_tab2.sizePolicy().hasHeightForWidth())
        self.label_tab2.setSizePolicy(sizePolicy4)
        self.label_tab2.setMaximumSize(QSize(16777215, 16777215))
        self.label_tab2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_tab2)

        self.tabWidget.addTab(self.face_editor_tab, "")
        self.settings_tab = QWidget()
        self.settings_tab.setObjectName(u"settings_tab")
        self.verticalLayout_2 = QVBoxLayout(self.settings_tab)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_tab1 = QLabel(self.settings_tab)
        self.label_tab1.setObjectName(u"label_tab1")
        self.label_tab1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_tab1)

        self.tabWidget.addTab(self.settings_tab, "")

        self.gridLayout_5.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.controlOptionsDockWidget.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.controlOptionsDockWidget)

        self.retranslateUi(MainWindow)

        self.pushButton_6.setDefault(False)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Rope-Live 0.1a", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.pushButton.setText("")
        self.buttonMediaPlay.setText("")
        self.pushButton_2.setText("")
        self.findTargetFacesButton.setText(QCoreApplication.translate("MainWindow", u"Find Faces", None))
        self.clearTargetFacesButton.setText(QCoreApplication.translate("MainWindow", u"Clear Faces", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.inputEmbeddingsSearchBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search Embeddings", None))
        self.input_Target_DockWidget.setWindowTitle(QCoreApplication.translate("MainWindow", u"Target Videos and Input Faces", None))
        self.groupBox_TargetVideos_Select.setTitle(QCoreApplication.translate("MainWindow", u"Target Videos/Images", None))
        self.labelTargetVideosPath.setText(QCoreApplication.translate("MainWindow", u"Select Videos/Images Path", None))
        self.buttonSelectTargetVideos.setText(QCoreApplication.translate("MainWindow", u"Select Folder", None))
        self.buttonSelectTargetVideoFiles.setText(QCoreApplication.translate("MainWindow", u"   Select Files     ", None))
        self.targetVideosSearchBox.setText("")
        self.targetVideosSearchBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search Videos/Images", None))
        self.groupBox_InputFaces_Select.setTitle(QCoreApplication.translate("MainWindow", u"Input Faces", None))
        self.labelInputFacesPath.setText(QCoreApplication.translate("MainWindow", u"Select Face Images Path", None))
        self.buttonSelectInputFaces.setText(QCoreApplication.translate("MainWindow", u"         Select Folder       ", None))
        self.buttonSelectInputFacesFiles.setText(QCoreApplication.translate("MainWindow", u"Select Files", None))
        self.inputFacesSearchBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search Faces", None))
        self.controlOptionsDockWidget.setWindowTitle(QCoreApplication.translate("MainWindow", u"Control Options", None))
        self.label_tab3.setText(QCoreApplication.translate("MainWindow", u"Face Swap", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.face_swap_tab), QCoreApplication.translate("MainWindow", u"Face Swap", None))
        self.label_tab2.setText(QCoreApplication.translate("MainWindow", u"Face Editor", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.face_editor_tab), QCoreApplication.translate("MainWindow", u"Face Editor", None))
        self.label_tab1.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.settings_tab), QCoreApplication.translate("MainWindow", u"Settings", None))
    # retranslateUi

