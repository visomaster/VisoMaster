from app.ui import main_ui
from PySide6 import QtWidgets 
import sys

import qdarktheme

if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)
    with open("app/ui/styles/dark_styles.qss", "r") as f:
        _style = f.read()
        _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"})+'\n'+_style
        app.setStyleSheet(_style)
    window = main_ui.MainWindow()
    window.show()
    app.exec()