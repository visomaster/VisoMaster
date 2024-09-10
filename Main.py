from App.UI import MainUI
from PySide6 import QtWidgets 
import sys

if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = MainUI.MainWindow()
    window.show()
    app.exec()