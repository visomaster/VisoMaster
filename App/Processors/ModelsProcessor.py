from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, QTimer, QCoreApplication, QThread
import PySide6.QtCore as qtc

from PySide6.QtWidgets import QProgressDialog, QApplication
import onnxruntime
from functools import partial
from threading import Thread
from App.UI.Widgets.WidgetComponents import ProgressDialog
import App.UI.Widgets.Actions.CommonActions as common_widget_actions
import time
import threading
lock = threading.Lock()
import torch
from App.Processors.Utils import FaceUtil as faceutil
from skimage import transform as trans
import torchvision
from torchvision.transforms import v2
from torchvision import transforms

import numpy as np
from numpy.linalg import norm as l2norm
from packaging import version
import subprocess as sp
import pickle
from itertools import product as product
from App.Processors.Utils.EngineBuilder import onnx_to_trt as onnx2trt
from App.Processors.Utils.TensorRTPredictor import TensorRTPredictor
import App.Processors.FaceDetectors as face_detectors
import App.Processors.FaceLandmarkDetectors as face_landmark_detectors
import App.Processors.FaceRestorers as face_restorers
import App.Processors.FaceMasks as face_masks
import App.Processors.FrameEnhancers as frame_enhancers
import App.Processors.FaceSwappers as face_swappers

import cv2
import os
import math
from App.Processors.Utils.DFMModel import DFMModel
from App.Processors.ModelsData import models_dir, models_list, arcface_mapping_model_dict, models_trt_list
from typing import Dict, TYPE_CHECKING
import gc
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
try:
    from torch.cuda import nvtx
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    print("No TensorRT Found")
    TENSORRT_AVAILABLE = False
import onnx

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1

def load_lip_array():
    with open(f'{models_dir}/liveportrait_onnx/lip_array.pkl', 'rb') as f:
        return pickle.load(f)

class ModelsProcessor(QObject):
    processing_complete = Signal()
    model_loaded = Signal()  # Signal emitted with Onnx InferenceSession

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
        self.lp_mask_crop = faceutil.create_faded_inner_mask(size=(512, 512), border_thickness=5, fade_thickness=15, blur_radius=5, device=self.device)
        self.lp_mask_crop = torch.unsqueeze(self.lp_mask_crop, 0)
        self.lp_lip_array = np.array(load_lip_array())

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

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5, input_size=(512, 512), use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles:list[int]=[0]):
        return face_detectors.run_detect(self, img, detect_mode, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles)
    
    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        return face_landmark_detectors.run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False)

    def get_arcface_model(self, face_swapper_model): 
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        return face_swappers.run_recognize_direct(self, img, kps, similarity_type, arcface_model)

    def calc_inswapper_latent(self, source_embedding):
        return face_swappers.calc_inswapper_latent(self, source_embedding)

    def run_inswapper(self, image, embedding, output):
        face_swappers.run_inswapper(self, image, embedding, output)

    def calc_fsis_latent(self, source_embedding):
        return face_swappers.calc_fsis_latent(self, source_embedding)

    def run_fsiswapper(self, image, embedding, output):
        face_swappers.run_fsiswapper(self, image, embedding, output)

    def calc_swapper_latent_simswap512(self, source_embedding):
        return face_swappers.calc_swapper_latent_simswap512(self, source_embedding)

    def run_swapper_simswap512(self, image, embedding, output):
        face_swappers.run_swapper_simswap512(self, image, embedding, output)

    def calc_swapper_latent_ghost(self, source_embedding):
        return face_swappers.calc_swapper_latent_ghost(self, source_embedding)

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        face_swappers.run_swapper_ghostface(self, image, embedding, output, swapper_model)

    def calc_swapper_latent_cscs(self, source_embedding):
        return face_swappers.calc_swapper_latent_cscs(self, source_embedding)

    def run_swapper_cscs(self, image, embedding, output):
        face_swappers.run_swapper_cscs(self, image, embedding, output)

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
        return frame_enhancers.run_enhance_frame_tile_process(self, img, enhancer_type, tile_size, scale)

    def run_occluder(self, image, output):
        face_masks.run_occluder(self, image, output)

    def run_dfl_xseg(self, image, output):
        face_masks.run_dfl_xseg(self, image, output)

    def run_faceparser(self, image, output):
        face_masks.run_faceparser(self, image, output)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        return face_masks.run_CLIPs(self, img, CLIPText, CLIPAmount)
    
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        kp_info = {}
        with torch.no_grad():
            # We force to use TensorRT because it doesn't work well in trt
            #if self.provider_name == "TensorRT-Engine":
            if self.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_trt['LivePortraitMotionExtractor']:
                        self.models_trt['LivePortraitMotionExtractor'] = self.load_model_trt('LivePortraitMotionExtractor', custom_plugin_path=None, precision="fp32")

                motion_extractor_model = self.models_trt['LivePortraitMotionExtractor']
                input_spec = motion_extractor_model.input_spec()
                output_spec = motion_extractor_model.output_spec()

                # prepare_source
                I_s = torch.div(img.type(torch.float32), 255.)
                I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
                I_s = torch.unsqueeze(I_s, 0).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["img"] = I_s
                #stream = torch.cuda.Stream()
                #preds_dict = motion_extractor_model.predict_async(feed_dict, stream)
                preds_dict = motion_extractor_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = motion_extractor_model.predict(feed_dict)

                kp_info = {
                    'pitch': preds_dict["pitch"],
                    'yaw': preds_dict["yaw"],
                    'roll': preds_dict["roll"],
                    't': preds_dict["t"],
                    'exp': preds_dict["exp"],
                    'scale': preds_dict["scale"],
                    'kp': preds_dict["kp"]
                }

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models['LivePortraitMotionExtractor']:
                        self.models['LivePortraitMotionExtractor'] = self.load_model('LivePortraitMotionExtractor')

                motion_extractor_model = self.models['LivePortraitMotionExtractor']

                # prepare_source
                I_s = torch.div(img.type(torch.float32), 255.)
                I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
                I_s = torch.unsqueeze(I_s, 0).contiguous()

                pitch = torch.empty((1,66), dtype=torch.float32, device=self.device).contiguous()
                yaw = torch.empty((1,66), dtype=torch.float32, device=self.device).contiguous()
                roll = torch.empty((1,66), dtype=torch.float32, device=self.device).contiguous()
                t = torch.empty((1,3), dtype=torch.float32, device=self.device).contiguous()
                exp = torch.empty((1,63), dtype=torch.float32, device=self.device).contiguous()
                scale = torch.empty((1,1), dtype=torch.float32, device=self.device).contiguous()
                kp = torch.empty((1,63), dtype=torch.float32, device=self.device).contiguous()

                io_binding = motion_extractor_model.io_binding()
                io_binding.bind_input(name='img', device_type=self.device, device_id=0, element_type=np.float32, shape=I_s.size(), buffer_ptr=I_s.data_ptr())
                io_binding.bind_output(name='pitch', device_type=self.device, device_id=0, element_type=np.float32, shape=pitch.size(), buffer_ptr=pitch.data_ptr())
                io_binding.bind_output(name='yaw', device_type=self.device, device_id=0, element_type=np.float32, shape=yaw.size(), buffer_ptr=yaw.data_ptr())
                io_binding.bind_output(name='roll', device_type=self.device, device_id=0, element_type=np.float32, shape=roll.size(), buffer_ptr=roll.data_ptr())
                io_binding.bind_output(name='t', device_type=self.device, device_id=0, element_type=np.float32, shape=t.size(), buffer_ptr=t.data_ptr())
                io_binding.bind_output(name='exp', device_type=self.device, device_id=0, element_type=np.float32, shape=exp.size(), buffer_ptr=exp.data_ptr())
                io_binding.bind_output(name='scale', device_type=self.device, device_id=0, element_type=np.float32, shape=scale.size(), buffer_ptr=scale.data_ptr())
                io_binding.bind_output(name='kp', device_type=self.device, device_id=0, element_type=np.float32, shape=kp.size(), buffer_ptr=kp.data_ptr())

                if self.device == "cuda":
                    torch.cuda.synchronize()
                elif self.device != "cpu":
                    self.syncvec.cpu()
                motion_extractor_model.run_with_iobinding(io_binding)

                kp_info = {
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    't': t,
                    'exp': exp,
                    'scale': scale,
                    'kp': kp
                }

            flag_refine_info: bool = kwargs.get('flag_refine_info', True)
            if flag_refine_info:
                bs = kp_info['kp'].shape[0]
                kp_info['pitch'] = faceutil.headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
                kp_info['yaw'] = faceutil.headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
                kp_info['roll'] = faceutil.headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
                kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
                kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        with torch.no_grad():
            # We force to use TensorRT. 
            #if self.provider_name == "TensorRT-Engine":
            if self.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_trt['LivePortraitAppearanceFeatureExtractor']:
                        self.models_trt['LivePortraitAppearanceFeatureExtractor'] = self.load_model_trt('LivePortraitAppearanceFeatureExtractor', custom_plugin_path=None, precision="fp16")

                appearance_feature_extractor_model = self.models_trt['LivePortraitAppearanceFeatureExtractor']

                # prepare_source
                I_s = torch.div(img.type(torch.float32), 255.)
                I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
                I_s = torch.unsqueeze(I_s, 0).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["img"] = I_s
                preds_dict = appearance_feature_extractor_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = appearance_feature_extractor_model.predict(feed_dict)

                output = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models['LivePortraitAppearanceFeatureExtractor']:
                        self.models['LivePortraitAppearanceFeatureExtractor'] = self.load_model('LivePortraitAppearanceFeatureExtractor')

                appearance_feature_extractor_model = self.models['LivePortraitAppearanceFeatureExtractor']

                # prepare_source
                I_s = torch.div(img.type(torch.float32), 255.)
                I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
                I_s = torch.unsqueeze(I_s, 0).contiguous()

                output = torch.empty((1,32,16,64,64), dtype=torch.float32, device=self.device).contiguous()

                io_binding = appearance_feature_extractor_model.io_binding()
                io_binding.bind_input(name='img', device_type=self.device, device_id=0, element_type=np.float32, shape=I_s.size(), buffer_ptr=I_s.data_ptr())
                io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

                if self.device == "cuda":
                    torch.cuda.synchronize()
                elif self.device != "cpu":
                    self.syncvec.cpu()
                appearance_feature_extractor_model.run_with_iobinding(io_binding)

        return output

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp)
        """
        with torch.no_grad():
            # We force to use TensorRT. 
            #if self.provider_name == "TensorRT-Engine":
            if self.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_trt['LivePortraitStitchingEye']:
                        self.models_trt['LivePortraitStitchingEye'] = self.load_model_trt('LivePortraitStitchingEye', custom_plugin_path=None, precision="fp16")

                stitching_eye_model = self.models_trt['LivePortraitStitchingEye']

                feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["input"] = feat_eye
                preds_dict = stitching_eye_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = stitching_eye_model.predict(feed_dict)

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models['LivePortraitStitchingEye']:
                        self.models['LivePortraitStitchingEye'] = self.load_model('LivePortraitStitchingEye')

                stitching_eye_model = self.models['LivePortraitStitchingEye']

                feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).contiguous()
                delta = torch.empty((1,63), dtype=torch.float32, device=self.device).contiguous()

                io_binding = stitching_eye_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=feat_eye.size(), buffer_ptr=feat_eye.data_ptr())
                io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                if self.device == "cuda":
                    torch.cuda.synchronize()
                elif self.device != "cpu":
                    self.syncvec.cpu()
                stitching_eye_model.run_with_iobinding(io_binding)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        Return: Bx(3*num_kp)
        """
        with torch.no_grad():
            # We force to use TensorRT. 
            #if self.provider_name == "TensorRT-Engine":
            if self.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_trt['LivePortraitStitchingLip']:
                        self.models_trt['LivePortraitStitchingLip'] = self.load_model_trt('LivePortraitStitchingLip', custom_plugin_path=None, precision="fp16")

                stitching_lip_model = self.models_trt['LivePortraitStitchingLip']

                feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["input"] = feat_lip
                preds_dict = stitching_lip_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = stitching_lip_model.predict(feed_dict)

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models['LivePortraitStitchingLip']:
                        self.models['LivePortraitStitchingLip'] = self.load_model('LivePortraitStitchingLip')

                stitching_lip_model = self.models['LivePortraitStitchingLip']

                feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).contiguous()
                delta = torch.empty((1,63), dtype=torch.float32, device=self.device).contiguous()

                io_binding = stitching_lip_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=feat_lip.size(), buffer_ptr=feat_lip.data_ptr())
                io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                if self.device == "cuda":
                    torch.cuda.synchronize()
                elif self.device != "cpu":
                    self.syncvec.cpu()
                stitching_lip_model.run_with_iobinding(io_binding)

        return delta.reshape(-1, kp_source.shape[1], 3)
    
    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        with torch.no_grad():
            # We force to use TensorRT. 
            #if self.provider_name == "TensorRT-Engine":
            if self.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_trt['LivePortraitStitching']:
                        self.models_trt['LivePortraitStitching'] = self.load_model_trt('LivePortraitStitching', custom_plugin_path=None, precision="fp16")

                stitching_model = self.models_trt['LivePortraitStitching']

                feat_stiching = faceutil.concat_feat(kp_source, kp_driving).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["input"] = feat_stiching
                preds_dict = stitching_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = stitching_model.predict(feed_dict)

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models['LivePortraitStitching']:
                        self.models['LivePortraitStitching'] = self.load_model('LivePortraitStitching')

                stitching_model = self.models['LivePortraitStitching']

                feat_stiching = faceutil.concat_feat(kp_source, kp_driving).contiguous()
                delta = torch.empty((1,65), dtype=torch.float32, device=self.device).contiguous()

                io_binding = stitching_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=feat_stiching.size(), buffer_ptr=feat_stiching.data_ptr())
                io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                if self.device == "cuda":
                    torch.cuda.synchronize()
                elif self.device != "cpu":
                    self.syncvec.cpu()
                stitching_model.run_with_iobinding(io_binding)

        return delta

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """
        bs, num_kp = kp_source.shape[:2]

        # calculate default delta from kp_source (using kp_source as default)
        kp_driving_default = kp_source.clone()

        default_delta = self.lp_stitch(kp_source, kp_driving_default, face_editor_type=face_editor_type)

        # Clone default delta values for expression and translation/rotation
        default_delta_exp = default_delta[..., :3*num_kp].reshape(bs, num_kp, 3).clone()  # 1x20x3
        default_delta_tx_ty = default_delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2).clone()  # 1x1x2

        # Debug: Print default delta values (should be close to zero)
        #print("default_delta_exp:", default_delta_exp)
        #print("default_delta_tx_ty:", default_delta_tx_ty)

        kp_driving_new = kp_driving.clone()

        # calculate new delta based on kp_driving
        delta = self.lp_stitch(kp_source, kp_driving_new, face_editor_type=face_editor_type)

        # Clone new delta values for expression and translation/rotation
        delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3).clone()  # 1x20x3
        delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2).clone()  # 1x1x2

        # Debug: Print new delta values
        #print("delta_exp:", delta_exp)
        #print("delta_tx_ty:", delta_tx_ty)

        # Calculate the difference between new and default delta
        delta_exp_diff = delta_exp - default_delta_exp
        delta_tx_ty_diff = delta_tx_ty - default_delta_tx_ty

        # Debug: Print the delta differences
        #print("delta_exp_diff:", delta_exp_diff)
        #print("delta_tx_ty_diff:", delta_tx_ty_diff)

        # Apply delta differences to the keypoints only if significant differences are found
        kp_driving_new += delta_exp_diff
        kp_driving_new[..., :2] += delta_tx_ty_diff

        return kp_driving_new

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """

        with torch.no_grad():
            if self.provider_name == "TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_trt['LivePortraitWarpingSpadeFix']:
                        self.models_trt['LivePortraitWarpingSpadeFix'] = self.load_model_trt('LivePortraitWarpingSpadeFix', custom_plugin_path=f'{models_dir}/grid_sample_3d_plugin.dll', precision="fp16")

                warping_spade_model = self.models_trt['LivePortraitWarpingSpadeFix']

                feature_3d = feature_3d.contiguous()
                kp_source = kp_source.contiguous()
                kp_driving = kp_driving.contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["feature_3d"] = feature_3d
                feed_dict["kp_source"] = kp_source
                feed_dict["kp_driving"] = kp_driving
                stream = torch.cuda.Stream()
                preds_dict = warping_spade_model.predict_async(feed_dict, stream)
                #preds_dict = warping_spade_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = warping_spade_model.predict(feed_dict)

                out = preds_dict["out"]

                nvtx.range_pop()
            else:
                if face_editor_type == 'Human-Face':
                    if not self.models['LivePortraitWarpingSpade']:
                        self.models['LivePortraitWarpingSpade'] = self.load_model('LivePortraitWarpingSpade')

                warping_spade_model = self.models['LivePortraitWarpingSpade']

                feature_3d = feature_3d.contiguous()
                kp_source = kp_source.contiguous()
                kp_driving = kp_driving.contiguous()

                out = torch.empty((1,3,512,512), dtype=torch.float32, device=self.device).contiguous()
                io_binding = warping_spade_model.io_binding()
                io_binding.bind_input(name='feature_3d', device_type=self.device, device_id=0, element_type=np.float32, shape=feature_3d.size(), buffer_ptr=feature_3d.data_ptr())
                io_binding.bind_input(name='kp_driving', device_type=self.device, device_id=0, element_type=np.float32, shape=kp_driving.size(), buffer_ptr=kp_driving.data_ptr())
                io_binding.bind_input(name='kp_source', device_type=self.device, device_id=0, element_type=np.float32, shape=kp_source.size(), buffer_ptr=kp_source.data_ptr())
                io_binding.bind_output(name='out', device_type=self.device, device_id=0, element_type=np.float32, shape=out.size(), buffer_ptr=out.data_ptr())

                if self.device == "cuda":
                    torch.cuda.synchronize()
                elif self.device != "cpu":
                    self.syncvec.cpu()
                warping_spade_model.run_with_iobinding(io_binding)

        return out

    def findCosineDistance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0
        return 100-cos_dist*50

    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score):
        return face_restorers.apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score)

    def apply_occlusion(self, img, amount):
        return face_masks.apply_occlusion(self, img, amount)
    
    def apply_dfl_xseg(self, img, amount):
        return face_masks.apply_dfl_xseg(self, img, amount)
    
    def apply_face_parser(self, img, parameters):
        return face_masks.apply_face_parser(self, img, parameters)

    def face_parser_makeup_direct_rgb(self, img, parsing, part=(17,), color=[230, 50, 20], blend_factor=0.2):
        device = img.device  # Ensure we use the same device

        # Clamp blend factor to ensure it stays between 0 and 1
        blend_factor = min(max(blend_factor, 0.0), 1.0)

        # Normalize the target RGB color to [0, 1]
        r, g, b = [x / 255.0 for x in color]
        tar_color = torch.tensor([r, g, b], dtype=torch.float32).view(3, 1, 1).to(device)

        #print(f"Target RGB color: {tar_color}")

        # Create hair mask based on parsing for multiple parts
        if isinstance(part, tuple):
            hair_mask = torch.zeros_like(parsing, dtype=torch.bool)
            for p in part:
                hair_mask |= (parsing == p)  # Accumulate masks for all parts in the tuple
        else:
            hair_mask = (parsing == part)

        #print(f"Hair mask shape: {hair_mask.shape}, Non-zero elements in mask: {hair_mask.sum().item()}")

        # Expand mask to match the image dimensions
        mask = hair_mask.unsqueeze(0).expand_as(img)
        #print(f"Expanded mask shape: {mask.shape}, Non-zero elements: {mask.sum().item()}")

        # Ensure that the image is normalized to [0, 1]
        image_normalized = img.float() / 255.0

        # Perform the color blending for the target region
        # (1 - blend_factor) * original + blend_factor * target_color
        changed = torch.where(
            mask,
            (1 - blend_factor) * image_normalized + blend_factor * tar_color,
            image_normalized
        )

        # Scale back to [0, 255] for final output
        changed = torch.clamp(changed * 255, 0, 255).to(torch.uint8)

        return changed

    def apply_face_makeup(self, img, parameters):
        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

        # Normalize the image and perform parsing
        temp = torch.div(img, 255)
        temp = v2.functional.normalize(temp, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        temp = torch.reshape(temp, (1, 3, 512, 512))
        outpred = torch.empty((1, 19, 512, 512), dtype=torch.float32, device=self.device).contiguous()

        self.run_faceparser(temp, outpred)

        # Perform parsing prediction
        outpred = torch.squeeze(outpred)
        outpred = torch.argmax(outpred, 0)

        # Clone the image for modifications
        out = img.clone()

        # Apply makeup for each face part
        if parameters['FaceMakeupEnableToggle']:
            color = [parameters['FaceMakeupRedSlider'], parameters['FaceMakeupGreenSlider'], parameters['FaceMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(img=out, parsing=outpred, part=(1, 7, 8, 10), color=color, blend_factor=parameters['FaceMakeupBlendAmountDecimalSlider'])

        if parameters['HairMakeupEnableToggle']:
            color = [parameters['HairMakeupRedSlider'], parameters['HairMakeupGreenSlider'], parameters['HairMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(img=out, parsing=outpred, part=17, color=color, blend_factor=parameters['HairMakeupBlendAmountDecimalSlider'])

        if parameters['EyeBrowsMakeupEnableToggle']:
            color = [parameters['EyeBrowsMakeupRedSlider'], parameters['EyeBrowsMakeupGreenSlider'], parameters['EyeBrowsMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(img=out, parsing=outpred, part=(2, 3), color=color, blend_factor=parameters['EyeBrowsMakeupBlendAmountDecimalSlider'])

        if parameters['LipsMakeupEnableToggle']:
            color = [parameters['LipsMakeupRedSlider'], parameters['LipsMakeupGreenSlider'], parameters['LipsMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(img=out, parsing=outpred, part=(12, 13), color=color, blend_factor=parameters['LipsMakeupBlendAmountDecimalSlider'])

        # Define the different face attributes to apply makeup on
        face_attributes = {
            1: parameters['FaceMakeupEnableToggle'],  # Face
            2: parameters['EyeBrowsMakeupEnableToggle'],  # Left Eyebrow
            3: parameters['EyeBrowsMakeupEnableToggle'],  # Right Eyebrow
            7: parameters['FaceMakeupEnableToggle'],  # Left Ear
            8: parameters['FaceMakeupEnableToggle'],  # Right Ear
            10: parameters['FaceMakeupEnableToggle'],  # Nose
            12: parameters['LipsMakeupEnableToggle'],  # Upper Lip
            13: parameters['LipsMakeupEnableToggle'],  # Lower Lip
            17: parameters['HairMakeupEnableToggle'],  # Hair
        }

        # Pre-calculated kernel per dilatazione (kernel 3x3)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=self.device)

        # Apply blur if blur kernel size is greater than 1
        blur_kernel_size = parameters['FaceEditorBlurAmountSlider'] * 2 + 1
        if blur_kernel_size > 1:
            gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['FaceEditorBlurAmountSlider'] + 1) * 0.2)

        # Generate masks for each face attribute
        face_parses = []
        for attribute in face_attributes.keys():
            if face_attributes[attribute]:  # Se l'attributo è abilitato
                attribute_idxs = torch.tensor([attribute], device=self.device)

                # Create the mask: white for the part, black for the rest
                attribute_parse = torch.isin(outpred, attribute_idxs).float()
                attribute_parse = torch.clamp(attribute_parse, 0, 1)  # Manteniamo i valori tra 0 e 1
                attribute_parse = torch.reshape(attribute_parse, (1, 1, 512, 512))

                # Dilate the mask (if necessary)
                for i in range(1):  # One pass, modify if needed
                    attribute_parse = torch.nn.functional.conv2d(attribute_parse, kernel, padding=(1, 1))
                    attribute_parse = torch.clamp(attribute_parse, 0, 1)

                # Squeeze to restore dimensions
                attribute_parse = torch.squeeze(attribute_parse)

                # Apply blur if required
                if blur_kernel_size > 1:
                    attribute_parse = gauss(attribute_parse.unsqueeze(0)).squeeze(0)

            else:
                # If the attribute is not enabled, use a black mask
                attribute_parse = torch.zeros((512, 512), dtype=torch.float32, device=self.device)
            
            # Add the mask to the list
            face_parses.append(attribute_parse)

        # Create a final mask to combine all the individual masks
        combined_mask = torch.zeros((512, 512), dtype=torch.float32, device=self.device)
        for face_parse in face_parses:
            # Add batch and channel dimensions for interpolation
            face_parse = face_parse.unsqueeze(0).unsqueeze(0)  # From (512, 512) to (1, 1, 512, 512)
            
            # Apply bilinear interpolation for anti-aliasing
            face_parse = torch.nn.functional.interpolate(face_parse, size=(512, 512), mode='bilinear', align_corners=True)
            
            # Remove the batch and channel dimensions
            face_parse = face_parse.squeeze(0).squeeze(0)  # Back to (512, 512)
            combined_mask = torch.max(combined_mask, face_parse)  # Combine the masks

        # Final application of the makeup mask on the original image
        out = img * (1 - combined_mask.unsqueeze(0)) + out * combined_mask.unsqueeze(0)

        return out, combined_mask.unsqueeze(0)
    
    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        return face_masks.restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset)

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        return face_masks.restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset, eye_spacing_offset)

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        return face_masks.apply_fake_diff(self, swapped_face, original_face, DiffAmount)
