"""
app/ui/web_main.py
──────────────────
New Qt main window that hosts:
  • QWebEngineView  — React UI (TopBar + panels) via QWebChannel
  • QGraphicsView   — native video preview (zero-copy, no JPEG encoding)

Entry point: web_main.py at the project root calls WebMainWindow().
"""
from __future__ import annotations

import copy
import json
import signal
from functools import partial
from pathlib import Path
from typing import Dict

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineScript, QWebEngineUrlRequestInterceptor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, Qt

from app.processors.models_processor import ModelsProcessor
from app.processors.video_processor import VideoProcessor
from app.ui.bridge import BackendBridge
from app.ui.widgets import widget_components, ui_workers
from app.ui.widgets.actions import (
    card_actions,
    common_actions as common_widget_actions,
    layout_actions,
    list_view_actions,
    save_load_actions,
    video_control_actions,
)
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.helpers.miscellaneous import DFM_MODELS_DATA, ParametersDict
from app.helpers.typing_helper import (
    ControlTypes,
    FacesParametersTypes,
    MarkerTypes,
    ParametersTypes,
)

# ── Dev server URL (Vite) ─────────────────────────────────────────────────────
VITE_URL = "http://localhost:5173"


class WebMainWindow(QtWidgets.QMainWindow):
    """
    Main window for the new Qt + WebEngine UI.

    The QWebEngineView fills the entire window.
    A hidden QGraphicsView stub is kept for compatibility with
    VideoProcessor helpers that call update_graphics_view().
    """

    # Signals reused from the original MainWindow interface so VideoProcessor
    # and other helpers that reference main_window.* still work.
    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal  = QtCore.Signal(int, int)
    model_loading_signal      = QtCore.Signal()
    model_loaded_signal       = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)

    def __init__(
        self,
        *,
        skip_workspace: bool = False,
        workspace_path: str | Path | None = None,
        auto_last_workspace: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        skip_workspace
            If True, skip the "Load last workspace?" dialog and start fresh.
            Takes precedence over the other workspace flags.
        workspace_path
            Optional path to a workspace JSON file to load on startup,
            bypassing the dialog. If the file does not exist, falls back
            to the normal dialog behaviour.
        auto_last_workspace
            If True, silently load ``last_workspace.json`` on startup
            (no dialog). Ignored when ``workspace_path`` is set.
        """
        super().__init__()
        self.setWindowTitle("VisoMaster")
        self.resize(1400, 860)

        self._skip_workspace = skip_workspace
        self._auto_last_workspace = auto_last_workspace
        self._workspace_path: Path | None = (
            Path(workspace_path) if workspace_path else None
        )

        self._init_state()
        self._build_ui()
        self._init_processors()
        self._init_bridge()
        self._connect_signals()
        self._load_last_workspace()

    # ── State (mirrors MainWindow attributes used by actions / FrameWorker) ──

    def _init_state(self) -> None:
        self.video_loader_worker: ui_workers.TargetMediaLoaderWorker | bool = False
        self.input_faces_loader_worker: ui_workers.InputFacesLoaderWorker | bool = False
        self.webrtc_server_process = None

        self.target_videos:      Dict[int, widget_components.TargetMediaCardButton] = {}
        self.target_faces:       Dict[int, widget_components.TargetFaceCardButton]  = {}
        self.input_faces:        Dict[int, widget_components.InputFaceCardButton]   = {}
        self.merged_embeddings:  Dict[int, widget_components.EmbeddingCardButton]   = {}

        self.cur_selected_target_face_button: widget_components.TargetFaceCardButton | bool = False
        self.selected_video_button:           widget_components.TargetMediaCardButton | bool = False
        self.selected_target_face_id = False

        self.parameters:               FacesParametersTypes = {}
        self.default_parameters:       ParametersTypes      = {}
        self.copied_parameters:        ParametersTypes      = {}
        self.current_widget_parameters: ParametersTypes     = {}
        self.markers:                  MarkerTypes          = {}
        self.parameters_list                                = {}
        self.control:                  ControlTypes         = {}
        self.parameter_widgets                              = {}
        self.loaded_embedding_filename: str                 = ""

        self.last_target_media_folder_path = ""
        self.last_input_media_folder_path  = ""
        self.is_full_screen                = False
        self.dfm_models_data               = DFM_MODELS_DATA
        self.loading_new_media             = False

        self.webcam_rotation = 0
        self.webcam_flip_h   = False
        self.webcam_flip_v   = False
        self.webrtc_rotation = 0
        self.webrtc_flip_h   = False
        self.webrtc_flip_v   = False
        self.media_rotation  = 0
        self.media_flip_h    = False
        self.media_flip_v    = False

        self._output_window = None
        self._preview_window = None

        # Hidden list widgets — needed by list_view_actions helpers
        self.targetVideosList    = QtWidgets.QListWidget()
        self.targetFacesList     = QtWidgets.QListWidget()
        self.webcamList          = QtWidgets.QListWidget()
        self.webrtcList          = QtWidgets.QListWidget()
        self.inputFacesList      = QtWidgets.QListWidget()
        self.inputEmbeddingsList = QtWidgets.QListWidget()

        for lw in (self.targetVideosList, self.webcamList,
                   self.webrtcList, self.inputFacesList):
            lw.setFlow(QtWidgets.QListWidget.Flow.LeftToRight)
            lw.setWrapping(True)
            lw.setResizeMode(QtWidgets.QListWidget.ResizeMode.Adjust)

        # Stub widgets referenced by helpers
        self.targetVideosSearchBox   = QtWidgets.QLineEdit()
        self.inputFacesSearchBox     = QtWidgets.QLineEdit()
        self.inputEmbeddingsSearchBox = QtWidgets.QLineEdit()
        self.filterImagesCheckBox    = QtWidgets.QCheckBox()
        self.filterVideosCheckBox    = QtWidgets.QCheckBox()
        self.outputFolderLineEdit    = QtWidgets.QLineEdit()
        self.vramProgressBar         = QtWidgets.QProgressBar()
        self.streamingFpsLabel       = QtWidgets.QLabel("FPS: --")

        # Stub tab widgets (used by on_input_source_tab_changed logic)
        self.inputSourceTabWidget   = QtWidgets.QTabWidget()
        self.streamingSubTabWidget  = QtWidgets.QTabWidget()

        # Stub buttons referenced by video_control_actions
        self.buttonMediaPlay   = QtWidgets.QPushButton()
        self.buttonMediaRecord = QtWidgets.QPushButton()
        self.buttonMediaPlay.setCheckable(True)
        self.buttonMediaRecord.setCheckable(True)

        # Stub parameter layout widgets (needed by layout_actions)
        self.commonWidgetsLayout    = QtWidgets.QVBoxLayout()
        self.swapWidgetsLayout      = QtWidgets.QVBoxLayout()
        self.settingsWidgetsLayout  = QtWidgets.QVBoxLayout()
        self.faceEditorWidgetsLayout = QtWidgets.QVBoxLayout()
        self.tabWidget              = QtWidgets.QTabWidget()

        # Stub seek slider (used by video_control_actions)
        self.videoSeekSlider  = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.videoSeekLineEdit = QtWidgets.QLineEdit()
        self.frameAdvanceButton = QtWidgets.QPushButton()
        self.frameRewindButton  = QtWidgets.QPushButton()
        self.addMarkerButton    = QtWidgets.QPushButton()
        self.removeMarkerButton = QtWidgets.QPushButton()
        self.nextMarkerButton   = QtWidgets.QPushButton()
        self.previousMarkerButton = QtWidgets.QPushButton()
        self.viewFullScreenButton = QtWidgets.QPushButton()
        self.swapfacesButton    = QtWidgets.QPushButton()
        self.editFacesButton    = QtWidgets.QPushButton()
        self.saveImageButton    = QtWidgets.QPushButton()
        self.clearMemoryButton  = QtWidgets.QPushButton()
        self.findTargetFacesButton  = QtWidgets.QPushButton()
        self.clearTargetFacesButton = QtWidgets.QPushButton()
        self.openEmbeddingButton    = QtWidgets.QPushButton()
        self.saveEmbeddingButton    = QtWidgets.QPushButton()
        self.saveEmbeddingAsButton  = QtWidgets.QPushButton()
        self.outputFolderButton     = QtWidgets.QPushButton()
        self.parametersPanelCheckBox = QtWidgets.QCheckBox()
        self.facesPanelCheckBox      = QtWidgets.QCheckBox()
        self.mediaPanelCheckBox      = QtWidgets.QCheckBox()
        self.faceMaskCheckBox        = QtWidgets.QCheckBox()
        self.faceCompareCheckBox     = QtWidgets.QCheckBox()
        self.webcamBtnRotateCCW = QtWidgets.QPushButton()
        self.webcamBtnRotateCW  = QtWidgets.QPushButton()
        self.webcamBtnFlipH     = QtWidgets.QPushButton()
        self.webcamBtnFlipV     = QtWidgets.QPushButton()
        self.webrtcBtnRotateCCW = QtWidgets.QPushButton()
        self.webrtcBtnRotateCW  = QtWidgets.QPushButton()
        self.webrtcBtnFlipH     = QtWidgets.QPushButton()
        self.webrtcBtnFlipV     = QtWidgets.QPushButton()
        self.webcamRotationLabel = QtWidgets.QLabel("0°")
        self.webrtcRotationLabel = QtWidgets.QLabel("0°")
        self.groupBox_TargetVideos_Select = QtWidgets.QGroupBox()

    # ── UI layout ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── QWebEngineView fills the entire window ────────────────────────
        self._webview = QWebEngineView()
        root_layout.addWidget(self._webview, 1)

        # Hidden QGraphicsView — kept as a stub so update_graphics_view()
        # and resizeEvent() don't crash, but never shown to the user.
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame = QtWidgets.QGraphicsView(self.scene)
        self.graphicsViewFrame.hide()

    # ── Processors ────────────────────────────────────────────────────────

    def _init_processors(self) -> None:
        # Build default parameters from layout data
        def _collect(layout_data: dict, target: dict) -> None:
            for _, widgets in layout_data.items():
                for name, cfg in widgets.items():
                    default = cfg.get("default")
                    if callable(default):
                        default = default()
                    target[name] = default

        _collect(COMMON_LAYOUT_DATA,      self.default_parameters)
        _collect(SWAPPER_LAYOUT_DATA,     self.default_parameters)
        _collect(FACE_EDITOR_LAYOUT_DATA, self.default_parameters)
        _collect(SETTINGS_LAYOUT_DATA,    self.control)

        self.current_widget_parameters = ParametersDict(
            copy.deepcopy(self.default_parameters), self.default_parameters
        )

        self.models_processor = ModelsProcessor(self)
        self.video_processor  = VideoProcessor(self)

        # Wire VideoProcessor callbacks
        def _on_state_change(event: str, **kwargs):
            if event == "stopped":
                video_control_actions.reset_media_buttons(self)
            elif event == "error":
                msg = kwargs.get("message", "Unknown error")
                self.display_messagebox_signal.emit("Error", msg, self)

        def _on_frame_done(frame_number: int, frame_bgr, is_single_frame: bool):
            pixmap = common_widget_actions.get_pixmap_from_frame(self, frame_bgr)
            if self.video_processor.file_type in ("webcam", "webrtc") and not is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(pixmap, frame_bgr)
            elif not is_single_frame:
                self.video_processor.frame_processed_signal.emit(frame_number, pixmap, frame_bgr)
            else:
                self.video_processor.single_frame_processed_signal.emit(frame_number, pixmap, frame_bgr)

        def _on_fps_update(fps: float):
            self.video_processor.fps_update_signal.emit(fps)

        self.video_processor.on_frame_done   = _on_frame_done
        self.video_processor.on_state_change = _on_state_change
        self.video_processor.on_fps_update   = _on_fps_update

        # Connect GPU memory signal
        self.gpu_memory_update_signal.connect(
            partial(common_widget_actions.set_gpu_memory_progressbar_value, self)
        )
        self.display_messagebox_signal.connect(
            partial(common_widget_actions.create_and_show_messagebox, self)
        )
        self.model_loading_signal.connect(
            partial(common_widget_actions.show_model_loading_dialog, self)
        )
        self.model_loaded_signal.connect(
            partial(common_widget_actions.hide_model_loading_dialog, self)
        )

        # Set up seek slider (needed by video_control_actions internals)
        video_control_actions.set_up_video_seek_slider(self)
        video_control_actions.set_up_video_seek_line_edit(self)

        # Build parameter widgets (hidden, but needed for defaults)
        layout_actions.add_widgets_to_tab_layout(
            self, LAYOUT_DATA=COMMON_LAYOUT_DATA,
            layoutWidget=self.commonWidgetsLayout, data_type="parameter"
        )
        layout_actions.add_widgets_to_tab_layout(
            self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA,
            layoutWidget=self.swapWidgetsLayout, data_type="parameter"
        )
        layout_actions.add_widgets_to_tab_layout(
            self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA,
            layoutWidget=self.settingsWidgetsLayout, data_type="control"
        )
        layout_actions.add_widgets_to_tab_layout(
            self, LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA,
            layoutWidget=self.faceEditorWidgetsLayout, data_type="parameter"
        )

        common_widget_actions.create_control(self, "OutputMediaFolder", "")
        video_control_actions.reset_media_buttons(self)
        common_widget_actions.update_gpu_memory_progressbar(self)

    # ── QWebChannel + bridge ──────────────────────────────────────────────

    def _init_bridge(self) -> None:
        self._bridge  = BackendBridge(self)
        self._channel = QWebChannel(self._webview.page())
        self._channel.registerObject("backend", self._bridge)
        self._webview.page().setWebChannel(self._channel)

        # Inject qwebchannel.js before the page loads
        script = QWebEngineScript()
        script.setName("qwebchannel")
        script.setSourceUrl(QUrl("qrc:///qtwebchannel/qwebchannel.js"))
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        self._webview.page().scripts().insert(script)

        self._webview.load(QUrl(VITE_URL))
        self._webview.loadFinished.connect(self._on_load_finished)

    def _on_load_finished(self, ok: bool) -> None:
        if not ok:
            print(f"[WebView] Failed to load {VITE_URL} — is Vite running?")
        else:
            print(f"[WebView] Loaded {VITE_URL}")

    # ── Signal connections ────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.video_processor.fps_update_signal.connect(self._on_fps_update)

    @QtCore.Slot(float)
    def _on_fps_update(self, fps_value: float) -> None:
        self.streamingFpsLabel.setText(f"FPS: {fps_value:.1f}")

    # ── Workspace ─────────────────────────────────────────────────────────

    def _load_last_workspace(self) -> None:
        # --no-workspace / start-from-scratch: skip everything
        if self._skip_workspace:
            print("[Workspace] --skip-workspace set; starting from scratch")
            return

        # --workspace <path>: load a specific workspace without the dialog
        if self._workspace_path is not None:
            if self._workspace_path.is_file():
                print(f"[Workspace] Auto-loading {self._workspace_path}")
                save_load_actions.load_saved_workspace(
                    self, str(self._workspace_path)
                )
                if self._bridge is not None:
                    QtCore.QTimer.singleShot(50, self._bridge._emit_workspace_loaded)
                return
            print(
                f"[Workspace] {self._workspace_path} not found; "
                "falling back to default behaviour"
            )

        # --auto-last-workspace: silently load last_workspace.json if it exists
        if self._auto_last_workspace:
            last = Path("last_workspace.json")
            if last.is_file():
                print("[Workspace] Auto-loading last_workspace.json (no prompt)")
                save_load_actions.load_saved_workspace(self, str(last))
                if self._bridge is not None:
                    QtCore.QTimer.singleShot(50, self._bridge._emit_workspace_loaded)
            else:
                print("[Workspace] --auto-last-workspace set but no "
                      "last_workspace.json found; starting fresh")
            return

        # Default: prompt the user if last_workspace.json exists
        if Path("last_workspace.json").is_file():
            dialog = widget_components.LoadLastWorkspaceDialog(self)
            result = dialog.exec_()
            # If the user accepted, the dialog has already invoked
            # load_saved_workspace(). Notify the React UI so it can re-pull
            # state and update its folder paths / face lists.
            if result == QtWidgets.QDialog.DialogCode.Accepted and self._bridge is not None:
                # Defer the emission a tick so any pending Qt loading is
                # flushed before the React UI calls getState().
                QtCore.QTimer.singleShot(50, self._bridge._emit_workspace_loaded)

    # ── Qt event overrides ────────────────────────────────────────────────

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        print("WebMainWindow: closeEvent")
        self.video_processor.stop_processing()
        list_view_actions.clear_stop_loading_input_media(self)
        list_view_actions.clear_stop_loading_target_media(self)

        if self._output_window is not None:
            self._output_window.close()
            self._output_window = None

        if self._preview_window is not None:
            try:
                self._preview_window.close()
            except RuntimeError:
                pass
            self._preview_window = None

        if self.webrtc_server_process and self.webrtc_server_process.is_alive():
            self.webrtc_server_process.terminate()
            self.webrtc_server_process.join(timeout=3)
            self.webrtc_server_process = None

        save_load_actions.save_current_workspace(self, "last_workspace.json")
        event.accept()

    # ── Stubs for compatibility with existing action helpers ──────────────

    def on_media_source_changed(self, source_index: int) -> None:
        pass

    def save_last_workspace(self) -> None:
        pass
