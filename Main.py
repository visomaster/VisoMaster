from App.UI import MainUI
from PySide6 import QtWidgets 
import sys

import qdarktheme

if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)
    with open("App/UI/Styles/dark_styles.qss", "r") as f:
        _style = f.read()
        _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"})+'\n'+_style
        app.setStyleSheet(_style)
    window = MainUI.MainWindow()
    window.show()
    app.exec()