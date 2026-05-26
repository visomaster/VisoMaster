"""
web_main.py
───────────
Entry point for the new Qt + WebEngine UI.

Usage:
    python web_main.py
    python web_main.py --skip-workspace
    python web_main.py --auto-last-workspace
    python web_main.py --workspace path/to/workspace.json

Flags:
    --skip-workspace       Skip the "Load last workspace?" dialog and start
                           with an empty session.
    --auto-last-workspace  Silently load last_workspace.json on startup
                           (no dialog).
    --workspace <path>     Load the given workspace JSON on startup without
                           prompting. Useful for scripted launches.

The three flags are mutually exclusive.

Make sure the Vite dev server is running first:
    cd visomaster-ui && bun run dev
"""
import argparse
import signal
import sys
from pathlib import Path

# Bootstrap streamrelay package path
_streamrelay_src = Path(__file__).parent / "packages" / "streamrelay" / "src"
if _streamrelay_src.is_dir() and str(_streamrelay_src) not in sys.path:
    sys.path.insert(0, str(_streamrelay_src))

from PySide6 import QtWidgets, QtCore
import qdarktheme

from app.ui.core.proxy_style import ProxyStyle
from app.ui.web_main import WebMainWindow


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="web_main.py",
        description="VisoMaster Qt + WebEngine UI launcher",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-workspace",
        "--no-workspace",
        action="store_true",
        dest="skip_workspace",
        help="Skip the 'load last workspace' dialog and start from scratch.",
    )
    group.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Load the given workspace JSON on startup (bypasses the dialog).",
    )
    group.add_argument(
        "--auto-last-workspace",
        "--load-last-workspace",
        action="store_true",
        dest="auto_last_workspace",
        help="Silently load last_workspace.json on startup without prompting.",
    )
    # parse_known_args so any Qt-specific args (e.g. -platform) pass through
    return parser.parse_known_args(argv)


if __name__ == "__main__":
    args, qt_argv = _parse_args(sys.argv[1:])

    app = QtWidgets.QApplication([sys.argv[0], *qt_argv])

    # Allow Ctrl+C to close gracefully
    def handle_sigint(*_):
        app.closeAllWindows()

    signal.signal(signal.SIGINT, handle_sigint)
    _signal_timer = QtCore.QTimer()
    _signal_timer.start(500)
    _signal_timer.timeout.connect(lambda: None)

    app.setStyle(ProxyStyle())
    with open("app/ui/styles/dark_styles.qss", "r") as f:
        _style = f.read()
        _style = (
            qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"})
            + "\n"
            + _style
        )
        app.setStyleSheet(_style)

    window = WebMainWindow(
        skip_workspace=args.skip_workspace,
        workspace_path=args.workspace,
        auto_last_workspace=args.auto_last_workspace,
    )
    window.show()
    sys.exit(app.exec())
