import threading
import os
import subprocess as sp
import gc
from typing import Dict, TYPE_CHECKING

from packaging import version
import numpy as np
import onnxruntime
import torch
from torchvision.transforms import v2
from PySide6 import QtCore
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    print("No TensorRT Found")
    TENSORRT_AVAILABLE = False

from app.processors.utils.engine_builder import onnx_to_trt as onnx2trt
from app.processors.utils.tensorrt_predictor import TensorRTPredictor
from app.processors.face_detectors import FaceDetectors
from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.face_masks import FaceMasks
from app.processors.face_restorers import FaceRestorers
from app.processors.face_swappers import FaceSwappers
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.face_editors import FaceEditors
from app.processors.utils.dfm_model import DFMModel
from app.processors.models_data import models_list, arcface_mapping_model_dict, models_trt_list

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1
lock = threading.Lock()

class ModelsProcessor(QtCore.QObject):
    processing_complete = QtCore.Signal()
    model_loaded = QtCore.Signal()  # Signal emitted with Onnx InferenceSession

    def __init__(self, main_window: 'MainWindow', device='cuda'):
        super().__init__()
        self.main_window = main_window
        self.provider_name = 'TensorRT'
        self.device = device
        self.model_lock = threading.RLock()  # Reentrant lock for model access
        self.trt_ep_options = {
            'trt_max_workspace_size': 3 << 30,  # Dimensione massima dello spazio di lavoro in bytes
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': "tensorrt-engines",
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': "tensorrt-engines",
            'trt_dump_ep_context_model': True,
            'trt_ep_context_file_path': "tensorrt-engines",
            'trt_layer_norm_fp32_fallback': True,
            'trt_builder_optimization_level': 5,
        }
        self.providers = [
            ('TensorrtExecutionProvider', self.trt_ep_options),
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')
        ]       
        self.nThreads = 5
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=self.device)

        # Initialize models and models_path
        self.models: Dict[str, onnxruntime.InferenceSession] = {}
        self.models_path = {}
        for model_data in models_list:
            model_name, model_path = model_data['model_name'], model_data['local_path']
            self.models[model_name] = None #Model Instance
            self.models_path[model_name] = model_path

        self.dfm_models: Dict[str, DFMModel] = {}

        if TENSORRT_AVAILABLE:
            # Initialize models_trt and models_trt_path
            self.models_trt = {}
            self.models_trt_path = {}
            for model_data in models_trt_list:
                model_name, model_path = model_data['model_name'], model_data['local_path']
                self.models_trt[model_name] = None #Model Instance
                self.models_trt_path[model_name] = model_path

        self.face_detectors = FaceDetectors(self)
        self.face_landmark_detectors = FaceLandmarkDetectors(self)
        self.face_masks = FaceMasks(self)
        self.face_restorers = FaceRestorers(self)
        self.face_swappers = FaceSwappers(self)
        self.frame_enhancers = FrameEnhancers(self)
        self.face_editors = FaceEditors(self)

        self.clip_session = []
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        self.mean_lmk = []
        self.anchors  = []
        self.emap = []
        self.LandmarksSubsetIdxs = [
            0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
            40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
            81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
            136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
            161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
            249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
            296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
            336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
            384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
            466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
        ]

        self.normalize = v2.Normalize(mean = [ 0., 0., 0. ],
                                      std = [ 1/1.0, 1/1.0, 1/1.0 ])
        
        self.lp_mask_crop = self.face_editors.lp_mask_crop
        self.lp_lip_array = self.face_editors.lp_lip_array

    def load_model(self, model_name, session_options=None):
        with self.model_lock:
            self.main_window.model_loading_signal.emit()
            # QApplication.processEvents()
            if session_options is None:
                model_instance = onnxruntime.InferenceSession(self.models_path[model_name], providers=self.providers)
            else:
                model_instance = onnxruntime.InferenceSession(self.models_path[model_name], sess_options=session_options, providers=self.providers)

            # Check if another thread has already loaded an instance for this model, if yes then delete the current one and return that instead
            if self.models[model_name]:
                del model_instance
                gc.collect()
                return self.models[model_name]
            self.main_window.model_loaded_signal.emit()

            return model_instance

    def load_dfm_model(self, dfm_model):
        with self.model_lock:
            if not self.dfm_models.get(dfm_model):
                self.main_window.model_loading_signal.emit()
                max_models_to_keep = self.main_window.control['MaxDFMModelsSlider']
                total_loaded_models = len(self.dfm_models)
                if total_loaded_models==max_models_to_keep:
                    print("Clearing DFM Model")
                    model_name, model_instance = list(self.dfm_models.items())[0]
                    del model_instance
                    self.dfm_models.pop(model_name)
                    gc.collect()
                self.dfm_models[dfm_model] = DFMModel(self.main_window.dfm_models_data[dfm_model], self.providers, self.device)
            self.main_window.model_loaded_signal.emit()
            return self.dfm_models[dfm_model]


    def load_model_trt(self, model_name, custom_plugin_path=None, precision='fp16', debug=False):
        # self.showModelLoadingProgressBar()
        #time.sleep(0.5)
        self.main_window.model_loading_signal.emit()

        if not os.path.exists(self.models_trt_path[model_name]):
            onnx2trt(onnx_model_path=self.models_path[model_name],
                     trt_model_path=self.models_trt_path[model_name],
                     precision=precision,
                     custom_plugin_path=custom_plugin_path,
                     verbose=False
                    )
        model_instance = TensorRTPredictor(model_path=self.models_trt_path[model_name], custom_plugin_path=custom_plugin_path, pool_size=self.nThreads, device=self.device, debug=debug)

        self.main_window.model_loaded_signal.emit()
        return model_instance

    def delete_models(self):
        for model_name, model_instance in self.models.items():
            del model_instance
            self.models[model_name] = None
        self.clip_session = []
        gc.collect()

    def delete_models_trt(self):
        if TENSORRT_AVAILABLE:
            for model_data in models_trt_list:
                model_name = model_data['model_name']
                if isinstance(self.models_trt[model_name], TensorRTPredictor):
                    # È un'istanza di TensorRTPredictor
                    self.models_trt[model_name].cleanup()
                    del self.models_trt[model_name]
                    self.models_trt[model_name] = None #Model Instance
            gc.collect()

    def delete_models_dfm(self):
        keys_to_remove = []
        for model_name, model_instance in self.dfm_models.items():
            del model_instance
            keys_to_remove.append(model_name)
        
        for model_name in keys_to_remove:
            self.dfm_models.pop(model_name)
        
        self.clip_session = []
        gc.collect()

    def showModelLoadingProgressBar(self):
        self.main_window.model_load_dialog.show()

    def hideModelLoadProgressBar(self):
        if self.main_window.model_load_dialog:
            self.main_window.model_load_dialog.close()

    def switch_providers_priority(self, provider_name):
        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                providers = [
                                ('TensorrtExecutionProvider', self.trt_ep_options),
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
                if version.parse(trt.__version__) < version.parse("10.2.0") and provider_name == "TensorRT-Engine":
                    print("TensorRT-Engine provider cannot be used when TensorRT version is lower than 10.2.0.")
                    provider_name = "TensorRT"

            case "CPU":
                providers = [
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cpu'
            case "CUDA":
                providers = [
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
            #case _:

        self.providers = providers
        self.provider_name = provider_name
        self.lp_mask_crop = self.lp_mask_crop.to(self.device)

        return self.provider_name

    def set_number_of_threads(self, value):
        self.nThreads = value
        self.delete_models_trt()

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        memory_used = memory_total[0] - memory_free[0]

        return memory_used, memory_total[0]
    
    def clear_gpu_memory(self):
        self.delete_models()
        self.delete_models_dfm()
        self.delete_models_trt()
        torch.cuda.empty_cache()

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5, input_size=(512, 512), use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        return self.face_detectors.run_detect(img, detect_mode, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles)
    
    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        return self.face_landmark_detectors.run_detect_landmark(img, bbox, det_kpss, detect_mode, score, from_points)

    def get_arcface_model(self, face_swapper_model): 
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        return self.face_swappers.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def calc_inswapper_latent(self, source_embedding):
        return self.face_swappers.calc_inswapper_latent(source_embedding)

    def run_inswapper(self, image, embedding, output):
        self.face_swappers.run_inswapper(image, embedding, output)

    def calc_fsis_latent(self, source_embedding):
        return self.face_swappers.calc_fsis_latent(source_embedding)

    def run_fsiswapper(self, image, embedding, output):
        self.face_swappers.run_fsiswapper(image, embedding, output)

    def calc_swapper_latent_simswap512(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_simswap512(source_embedding)

    def run_swapper_simswap512(self, image, embedding, output):
        self.face_swappers.run_swapper_simswap512(image, embedding, output)

    def calc_swapper_latent_ghost(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_ghost(source_embedding)

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        self.face_swappers.run_swapper_ghostface(image, embedding, output, swapper_model)

    def calc_swapper_latent_cscs(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_cscs(source_embedding)

    def run_swapper_cscs(self, image, embedding, output):
        self.face_swappers.run_swapper_cscs(image, embedding, output)

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
        return self.frame_enhancers.run_enhance_frame_tile_process(img, enhancer_type, tile_size, scale)

    def run_deoldify_artistic(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)

    def run_deoldify_stable(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)
    
    def run_deoldify_video(self, image, output):
        return self.frame_enhancers.run_deoldify_video(image, output)

    def run_occluder(self, image, output):
        self.face_masks.run_occluder(image, output)

    def run_dfl_xseg(self, image, output):
        self.face_masks.run_dfl_xseg(image, output)

    def run_faceparser(self, image, output):
        self.face_masks.run_faceparser(image, output)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        return self.face_masks.run_CLIPs(img, CLIPText, CLIPAmount)
    
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        return self.face_editors.lp_motion_extractor(img, face_editor_type, **kwargs)

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        return self.face_editors.lp_appearance_feature_extractor(img, face_editor_type)

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_eye(kp_source, eye_close_ratio, face_editor_type)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_lip(kp_source, lip_close_ratio, face_editor_type)

    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitch(kp_source, kp_driving, face_editor_type)

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitching(kp_source, kp_driving, face_editor_type)

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_warp_decode(feature_3d, kp_source, kp_driving, face_editor_type)

    def findCosineDistance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0
        return 100-cos_dist*50

    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score):
        return self.face_restorers.apply_facerestorer(swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score)

    def apply_occlusion(self, img, amount):
        return self.face_masks.apply_occlusion(img, amount)
    
    def apply_dfl_xseg(self, img, amount):
        return self.face_masks.apply_dfl_xseg(img, amount)
    
    def apply_face_parser(self, img, parameters):
        return self.face_masks.apply_face_parser(img, parameters)
    
    def apply_face_makeup(self, img, parameters):
        return self.face_editors.apply_face_makeup(img, parameters)
    
    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        return self.face_masks.restore_mouth(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset)

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        return self.face_masks.restore_eyes(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset, eye_spacing_offset)

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        return self.face_masks.apply_fake_diff(swapped_face, original_face, DiffAmount)
