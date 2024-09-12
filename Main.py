from App.UI import MainUI
from PySide6 import QtWidgets 
import sys


if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)
    with open("App/UI/Styles/styles.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)
    window = MainUI.MainWindow()
    window.show()
    app.exec()