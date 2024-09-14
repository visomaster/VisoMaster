from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, QTimer, QCoreApplication
import PySide6.QtCore as qtc
from PySide6.QtWidgets import QProgressDialog
import onnxruntime
from functools import partial
from threading import Thread
from App.UI.Widgets import WidgetActions as widget_actions
from App.UI.Widgets.WidgetComponents import ProgressDialog
import threading

class ModelsProcessor(QObject):
    processing_complete = Signal()
    model_loaded = qtc.Signal()  # Signal emitted with Onnx InferenceSession

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.providers = [
            ('TensorrtExecutionProvider', {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': "tensorrt-engines",
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': "tensorrt-engines",
                'trt_dump_ep_context_model': True,
                'trt_ep_context_file_path': "tensorrt-engines",
                'trt_layer_norm_fp32_fallback': True,
                'trt_builder_optimization_level': 5,
            }),
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')
        ]
        self.models = {
            'inswapper_128': {'model_path': './App/ONNXModels/inswapper_128.fp16.onnx', 'model_instance': None}
        }
        self.current_loading_model = False

    def test_run_model_function(self, model_name, *args):
        if not self.models[model_name]['model_instance']:
            self.load_model_and_exec(model_name, self.test_run_model_function, model_name, *args)
        else:
            self.model_load_timer.stop()
            print("Success", self.models['inswapper_128'])

    def load_model_and_exec(self, model_name, exec_func: callable, *args):
        self.showModelLoadingProgressBar()
        self.main_window.model_load_thread = qtc.QThread(self)
        self.main_window.model_load_thread.started.connect(partial(self.load_model, model_name))
        self.main_window.model_load_thread.finished.connect(self.hideModelLoadProgressBar)
        self.main_window.model_load_thread.start()
        # Initialize and start the timer for checking the loading status
        self.model_load_timer = QTimer()
        self.model_load_timer.timeout.connect(partial(self.check_model_loaded, model_name, exec_func, *args))
        self.model_load_timer.start(100)  # Check every 500ms

    def load_model(self, model_name):
        # Load the model in a separate thread
        self.models[model_name]['model_instance'] = onnxruntime.InferenceSession(
            self.models[model_name]['model_path'], providers=self.providers)

    def check_model_loaded(self, model_name, exec_func: callable, *args):
        if not self.models[model_name]['model_instance']:
            self.main_window.model_load_dialog.setValue(10)
            QCoreApplication.processEvents()
        else:
            self.model_load_timer.stop()  # Stop checking once the model is loaded
            self.model_loaded.emit()
            exec_func(*args)  # Execute the next function

    def showModelLoadingProgressBar(self):
        self.main_window.model_load_dialog = ProgressDialog("Processing...", "Cancel", 0, 100, self.main_window)
        self.main_window.model_load_dialog.setWindowModality(qtc.Qt.ApplicationModal)
        self.main_window.model_load_dialog.setMinimumDuration(2000)
        self.main_window.model_load_dialog.setWindowTitle("Progress")
        self.main_window.model_load_dialog.setAutoClose(True)  # Close the dialog when finished
        self.model_loaded.connect(self.hideModelLoadProgressBar)
        self.main_window.model_load_dialog.show()

    def hideModelLoadProgressBar(self):
        self.main_window.model_load_dialog.close()