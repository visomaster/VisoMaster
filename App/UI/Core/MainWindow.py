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
from PySide6.QtWidgets import (QApplication, QGraphicsView, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QSizePolicy, QSlider,
    QSpacerItem, QTabWidget, QVBoxLayout, QWidget)
from App.UI.Core import media_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(878, 586)
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
        self.group_Input_FIles = QGroupBox(self.centralwidget)
        self.group_Input_FIles.setObjectName(u"group_Input_FIles")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.group_Input_FIles.sizePolicy().hasHeightForWidth())
        self.group_Input_FIles.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setFamilies([u"Segoe UI Semibold"])
        font1.setPointSize(10)
        font1.setBold(True)
        self.group_Input_FIles.setFont(font1)
        self.gridLayout_2 = QGridLayout(self.group_Input_FIles)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.groupBox_TargetVideos_Select = QGroupBox(self.group_Input_FIles)
        self.groupBox_TargetVideos_Select.setObjectName(u"groupBox_TargetVideos_Select")
        self.gridLayout_3 = QGridLayout(self.groupBox_TargetVideos_Select)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.buttonSelectTargetVideos = QPushButton(self.groupBox_TargetVideos_Select)
        self.buttonSelectTargetVideos.setObjectName(u"buttonSelectTargetVideos")

        self.gridLayout_3.addWidget(self.buttonSelectTargetVideos, 0, 0, 1, 1)

        self.labelTargetVideosPath = QLabel(self.groupBox_TargetVideos_Select)
        self.labelTargetVideosPath.setObjectName(u"labelTargetVideosPath")
        self.labelTargetVideosPath.setWordWrap(False)

        self.gridLayout_3.addWidget(self.labelTargetVideosPath, 1, 0, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox_TargetVideos_Select, 0, 0, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.targetVideosList = QListWidget(self.group_Input_FIles)
        self.targetVideosList.setObjectName(u"targetVideosList")

        self.gridLayout_2.addWidget(self.targetVideosList, 1, 0, 1, 1)

        self.groupBox_InputFaces_Select = QGroupBox(self.group_Input_FIles)
        self.groupBox_InputFaces_Select.setObjectName(u"groupBox_InputFaces_Select")
        self.gridLayout = QGridLayout(self.groupBox_InputFaces_Select)
        self.gridLayout.setObjectName(u"gridLayout")
        self.buttonSelectInputFaces = QPushButton(self.groupBox_InputFaces_Select)
        self.buttonSelectInputFaces.setObjectName(u"buttonSelectInputFaces")

        self.gridLayout.addWidget(self.buttonSelectInputFaces, 0, 0, 1, 1)

        self.labelInputFacesPath = QLabel(self.groupBox_InputFaces_Select)
        self.labelInputFacesPath.setObjectName(u"labelInputFacesPath")

        self.gridLayout.addWidget(self.labelInputFacesPath, 1, 0, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox_InputFaces_Select, 0, 1, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.inputFacesList = QListWidget(self.group_Input_FIles)
        self.inputFacesList.setObjectName(u"inputFacesList")

        self.gridLayout_2.addWidget(self.inputFacesList, 1, 1, 1, 1)


        self.horizontalLayout.addWidget(self.group_Input_FIles)

        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(2)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy1)
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
        self.buttonMediaPlay = QPushButton(self.widget)
        self.buttonMediaPlay.setObjectName(u"buttonMediaPlay")
        icon1 = QIcon()
        icon1.addFile(u":/media/Media/play_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.buttonMediaPlay.setIcon(icon1)
        self.buttonMediaPlay.setFlat(True)

        self.horizontalLayoutMediaButtons.addWidget(self.buttonMediaPlay)

        self.buttonMediaStop = QPushButton(self.widget)
        self.buttonMediaStop.setObjectName(u"buttonMediaStop")
        icon2 = QIcon()
        icon2.addFile(u":/media/Media/stop_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.buttonMediaStop.setIcon(icon2)
        self.buttonMediaStop.setFlat(True)

        self.horizontalLayoutMediaButtons.addWidget(self.buttonMediaStop)


        self.verticalLayoutMediaControls.addLayout(self.horizontalLayoutMediaButtons)


        self.verticalLayout.addLayout(self.verticalLayoutMediaControls)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout.addWidget(self.widget)

        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setFont(font1)
        self.tabWidget.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.tabWidget.setTabPosition(QTabWidget.TabPosition.North)
        self.settings_tab = QWidget()
        self.settings_tab.setObjectName(u"settings_tab")
        self.verticalLayout_2 = QVBoxLayout(self.settings_tab)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_tab1 = QLabel(self.settings_tab)
        self.label_tab1.setObjectName(u"label_tab1")
        self.label_tab1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_tab1)

        self.tabWidget.addTab(self.settings_tab, "")
        self.face_editor_tab = QWidget()
        self.face_editor_tab.setObjectName(u"face_editor_tab")
        self.verticalLayout_3 = QVBoxLayout(self.face_editor_tab)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_tab2 = QLabel(self.face_editor_tab)
        self.label_tab2.setObjectName(u"label_tab2")
        self.label_tab2.setEnabled(True)
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_tab2.sizePolicy().hasHeightForWidth())
        self.label_tab2.setSizePolicy(sizePolicy2)
        self.label_tab2.setMaximumSize(QSize(16777215, 16777215))
        self.label_tab2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_tab2)

        self.tabWidget.addTab(self.face_editor_tab, "")
        self.face_swap_tab = QWidget()
        self.face_swap_tab.setObjectName(u"face_swap_tab")
        self.verticalLayout_4 = QVBoxLayout(self.face_swap_tab)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_tab3 = QLabel(self.face_swap_tab)
        self.label_tab3.setObjectName(u"label_tab3")
        self.label_tab3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_tab3)

        self.tabWidget.addTab(self.face_swap_tab, "")

        self.horizontalLayout.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Rope-Live 0.1a", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.group_Input_FIles.setTitle("")
        self.groupBox_TargetVideos_Select.setTitle(QCoreApplication.translate("MainWindow", u"Target Videos", None))
        self.buttonSelectTargetVideos.setText(QCoreApplication.translate("MainWindow", u"Select Folder", None))
        self.labelTargetVideosPath.setText(QCoreApplication.translate("MainWindow", u"Select Videos Path", None))
        self.groupBox_InputFaces_Select.setTitle(QCoreApplication.translate("MainWindow", u"Input Faces", None))
        self.buttonSelectInputFaces.setText(QCoreApplication.translate("MainWindow", u"Select Folder", None))
        self.labelInputFacesPath.setText(QCoreApplication.translate("MainWindow", u"Select Faces Path ", None))
        self.buttonMediaPlay.setText("")
        self.buttonMediaStop.setText("")
        self.label_tab1.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.settings_tab), QCoreApplication.translate("MainWindow", u"Settings", None))
        self.label_tab2.setText(QCoreApplication.translate("MainWindow", u"Face Editor", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.face_editor_tab), QCoreApplication.translate("MainWindow", u"Face Editor", None))
        self.label_tab3.setText(QCoreApplication.translate("MainWindow", u"Face Swap", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.face_swap_tab), QCoreApplication.translate("MainWindow", u"Face Swap", None))
    # retranslateUi

