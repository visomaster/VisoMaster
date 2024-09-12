from App.UI import MainUI
from PySide6 import QtWidgets 
import sys

QSS = """
/* QSlider --------------------------------------  */
QSlider::groove:horizontal {
    border-radius: 4px;
    height: 8px; /* Increase groove thickness */
    margin: 0px;
    background-color: rgb(85, 91, 94); /* Background groove color */
}
QSlider::groove:horizontal:hover {
    background-color: rgb(85, 91, 94); /* Groove color on hover */
}
QSlider::sub-page:horizontal {
    background-color: rgb(0, 145, 0); /* Bright green for completed part */
    border-radius: 4px;
    height: 8px;
}
QSlider::sub-page:horizontal:hover {
    background-color: rgb(0, 200, 0); /* Slightly darker green on hover */
}
QSlider::add-page:horizontal {
    background-color: rgb(85, 91, 94); /* Uncompleted part color */
    border-radius: 4px;
    height: 8px;
}
QSlider::handle:horizontal {
    background-color: rgb(214, 208, 208);
    border: none;
    height: 20px; /* Handle height */
    width: 20px;  /* Handle width */
    margin: -6px 0; /* Adjust handle position */
    border-radius: 10px; /* Adjust handle radius */
}
QSlider::handle:horizontal:hover {
    background-color: rgb(144, 238, 144); /* Light green on hover */
}
QSlider::handle:horizontal:pressed {
    background-color: rgb(0, 128, 0); /* Darker green when pressed */
}
"""


if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(QSS)
    window = MainUI.MainWindow()
    window.show()
    app.exec()