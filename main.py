from app.ui import main_ui
from PySide6 import QtWidgets, QtCore
import sys
import signal

import qdarktheme
from app.ui.core.proxy_style import ProxyStyle

if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)

    # Allow Ctrl+C to close the application gracefully.
    # closeAllWindows triggers closeEvent on MainWindow, ensuring proper cleanup.
    def handle_sigint(*args):
        app.closeAllWindows()

    signal.signal(signal.SIGINT, handle_sigint)
    # Timer gives Python a chance to run signal handlers during Qt event loop
    _signal_timer = QtCore.QTimer()
    _signal_timer.start(1000)
    _signal_timer.timeout.connect(lambda: None)

    app.setStyle(ProxyStyle())
    with open("app/ui/styles/dark_styles.qss", "r") as f:
        _style = f.read()
        _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"})+'\n'+_style
        app.setStyleSheet(_style)
    window = main_ui.MainWindow()
    window.show()
    sys.exit(app.exec())
