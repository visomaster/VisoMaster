from App.UI import MainUI
from PySide6 import QtWidgets 
import sys
import qdarkstyle


if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)

    app.setStyleSheet(qdarkstyle.load_stylesheet())

    window = MainUI.MainWindow()
    window.show()
    app.exec()