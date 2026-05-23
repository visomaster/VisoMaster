import sys
import signal
from pathlib import Path

# Bootstrap streamrelay package path so it resolves without a pip install.
# If the user has done `pip install -e packages/streamrelay` this is a no-op.
_streamrelay_src = Path(__file__).parent / "packages" / "streamrelay" / "src"
if _streamrelay_src.is_dir() and str(_streamrelay_src) not in sys.path:
    sys.path.insert(0, str(_streamrelay_src))

from app.ui import main_ui
from PySide6 import QtWidgets, QtCore

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
