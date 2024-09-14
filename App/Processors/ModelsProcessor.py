from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, QTimer
import PySide6.QtCore as qtc
# from App.Workers.UI_Workers import ModelLoaderWorker
import cv2
import onnxruntime
from functools import partial
from App.UI.Widgets import WidgetActions as widget_actions

class ModelLoaderWorker(qtc.QRunnable):
    def __init__(self, model_name=False, models_processor=None, providers=[], parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.providers = providers
        self.models_processor = models_processor

    def run(self):
        if self.model_name:
            self.models_processor.models[self.model_name]['model_instance'] = onnxruntime.InferenceSession(self.models_processor.models[self.model_name]['model_path'], providers=self.providers)
            self.models_processor.model_loaded.emit()

class ModelsProcessor(QObject):
    processing_complete = Signal()
    model_loaded = qtc.Signal()  # Signal with Onnx InferenceSession

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.providers =  [
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
            'inswapper_128': {'model_path': './App/ONNXModels/inswapper_128.fp16.onnx', 'model_instance':None}
        }
        self.current_loading_model = False
        self.thread_pool = QThreadPool()

    def test_run_model_function(self, *args):
        self.load_model_and_show_progress_bar('inswapper_128')
        print(self.models['inswapper_128'])
    
    def load_model_and_show_progress_bar(self, model_name):
        model_load_worker = ModelLoaderWorker(model_name, self, self.providers)
        self.model_loaded.connect(partial(widget_actions.hideModelLoadProgressBar, self.main_window))
        self.thread_pool.start(model_load_worker)
        widget_actions.showModelLoadingProgressBar(self.main_window)


