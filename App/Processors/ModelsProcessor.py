from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, QTimer, QCoreApplication, QThread
import PySide6.QtCore as qtc
from PySide6.QtWidgets import QProgressDialog
import onnxruntime
from functools import partial
from threading import Thread
from App.UI.Widgets import WidgetActions as widget_actions
from App.UI.Widgets.WidgetComponents import ProgressDialog
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
from App.Processors.external.clipseg import CLIPDensePredT
import cv2
import os
import math
from App.Processors.Utils.DFMModel import DFMModel
from typing import Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
try:
    from torch.cuda import nvtx
    import tensorrt as trt
except ModuleNotFoundError:
    print("No TensorRT Found")

import onnx
models_dir = './App/ONNXModels'


onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1

arcface_mapping_model_dict = {
    'Inswapper128': 'Inswapper128ArcFace',
    'DeepFaceLive (DFM)': 'Inswapper128ArcFace',
    'SimSwap512': 'SimSwapArcFace',
    'GhostFace-v1': 'GhostArcFace',
    'GhostFace-v2': 'GhostArcFace',
    'GhostFace-v3': 'GhostArcFace',
    'CSCS': 'CSCSArcFace',
}

models_list = [
    {'Inswapper128': f'{models_dir}/inswapper_128.fp16.onnx',},
    {'SimSwap512': f'{models_dir}/simswap_512_unoff.onnx',},
    {'GhostFacev1': f'{models_dir}/ghost_unet_1_block.onnx',},
    {'GhostFacev2': f'{models_dir}/ghost_unet_2_block.onnx',},
    {'GhostFacev3': f'{models_dir}/ghost_unet_3_block.onnx',},
    {'CSCS': f'{models_dir}/cscs_256.onnx',},
    {'RetinaFace': f'{models_dir}/det_10g.onnx',},
    {'SCRFD2.5g': f'{models_dir}/scrfd_2.5g_bnkps.onnx',},
    {'YoloFace8n': f'{models_dir}/yoloface_8n.onnx',},
    {'YunetN': f'{models_dir}/yunet_n_640_640.onnx',},
    {'FaceLandmark5': f'{models_dir}/res50.onnx',},
    {'FaceLandmark68': f'{models_dir}/2dfan4.onnx',},
    {'FaceLandmark3d68': f'{models_dir}/1k3d68.onnx',},
    {'FaceLandmark98': f'{models_dir}/peppapig_teacher_Nx3x256x256.onnx',},
    {'FaceLandmark106': f'{models_dir}/2d106det.onnx',},
    {'FaceLandmark203': f'{models_dir}/landmark.onnx',},
    {'FaceLandmark478': f'{models_dir}/face_landmarks_detector_Nx3x256x256.onnx',},
    {'FaceBlendShapes': f'{models_dir}/face_blendshapes_Nx146x2.onnx',},
    {'Inswapper128ArcFace': f'{models_dir}/w600k_r50.onnx',},
    {'SimSwapArcFace': f'{models_dir}/simswap_arcface_model.onnx',},
    {'GhostArcFace': f'{models_dir}/ghost_arcface_backbone.onnx',},
    {'CSCSArcFace': f'{models_dir}/cscs_arcface_model.onnx',},
    {'CSCSIDArcFace': f'{models_dir}/cscs_id_adapter.onnx',},
    {'GFPGANv1.4': f'{models_dir}/GFPGANv1.4.onnx',},
    {'GPENBFR256': f'{models_dir}/GPEN-BFR-256.onnx',},
    {'GPENBFR512': f'{models_dir}/GPEN-BFR-512.onnx',},
    {'GPENBFR1024': f'{models_dir}/GPEN-BFR-1024.onnx',},
    {'GPENBFR2048': f'{models_dir}/GPEN-BFR-2048.onnx',},
    {'CodeFormer': f'{models_dir}/codeformer_fp16.onnx',},
    {'VQFRv2': f'{models_dir}/VQFRv2.fp16.onnx',},
    {'RestoreFormerPlusPlus': f'{models_dir}/RestoreFormerPlusPlus.fp16.onnx',},
    {'RealEsrganx2Plus': f'{models_dir}/RealESRGAN_x2plus.fp16.onnx',},
    {'RealEsrganx4Plus': f'{models_dir}/RealESRGAN_x4plus.fp16.onnx',},
    {'RealEsrx4v3': f'{models_dir}/realesr-general-x4v3.onnx',},
    {'BSRGANx2': f'{models_dir}/BSRGANx2.fp16.onnx',},
    {'BSRGANx4': f'{models_dir}/BSRGANx4.fp16.onnx',},
    {'UltraSharpx4': f'{models_dir}/4x-UltraSharp.fp16.onnx',},
    {'UltraMixx4': f'{models_dir}/4x-UltraMix_Smooth.fp16.onnx',},
    {'DeoldifyArt': f'{models_dir}/ColorizeArtistic.fp16.onnx',},
    {'DeoldifyStable': f'{models_dir}/ColorizeStable.fp16.onnx',},
    {'DeoldifyVideo': f'{models_dir}/ColorizeVideo.fp16.onnx',},
    {'DDColorArt': f'{models_dir}/ddcolor_artistic.onnx',},
    {'DDcolor': f'{models_dir}/ddcolor.onnx',},
    {'Occluder': f'{models_dir}/occluder.onnx',},
    {'XSeg': f'{models_dir}/XSeg_model.onnx',},
    {'FaceParser': f'{models_dir}/faceparser_resnet34.onnx',},
    {'LivePortraitMotionExtractor': f'{models_dir}/liveportrait_onnx/motion_extractor.onnx',},
    {'LivePortraitAppearanceFeatureExtractor': f'{models_dir}/liveportrait_onnx/appearance_feature_extractor.onnx',},
    {'LivePortraitStitchingEye': f'{models_dir}/liveportrait_onnx/stitching_eye.onnx',},
    {'LivePortraitStitchingLip': f'{models_dir}/liveportrait_onnx/stitching_lip.onnx',},
    {'LivePortraitStitching': f'{models_dir}/liveportrait_onnx/stitching.onnx',},
    {'LivePortraitWarpingSpade': f'{models_dir}/liveportrait_onnx/warping_spade.onnx',},
    {'LivePortraitWarpingSpadeFix': f'{models_dir}/liveportrait_onnx/warping_spade-fix.onnx',}
]

def load_lip_array():
    with open(f'{models_dir}/liveportrait_onnx/lip_array.pkl', 'rb') as f:
        return pickle.load(f)

if 'trt' in globals():
    models_trt_list = [
        {'LivePortraitMotionExtractor': f'{models_dir}/liveportrait_onnx/motion_extractor.' + trt.__version__ + '.trt',},
        {'LivePortraitAppearanceFeatureExtractor': f'{models_dir}/liveportrait_onnx/appearance_feature_extractor.' + trt.__version__ + '.trt',},
        {'LivePortraitStitchingEye': f'{models_dir}/liveportrait_onnx/stitching_eye.' + trt.__version__ + '.trt',},
        {'LivePortraitStitchingLip': f'{models_dir}/liveportrait_onnx/stitching_lip.' + trt.__version__ + '.trt',},
        {'LivePortraitStitching': f'{models_dir}/liveportrait_onnx/stitching.' + trt.__version__ + '.trt',},
        {'LivePortraitWarpingSpadeFix': f'{models_dir}/liveportrait_onnx/warping_spade-fix.' + trt.__version__ + '.trt',}
    ]

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
            model_name, model_path = list(model_data.items())[0]
            self.models[model_name] = None #Model Instance
            self.models_path[model_name] = model_path

        self.dfm_models: Dict[str, DFMModel] = {}

        if 'trt' in globals():
            # Initialize models_trt and models_trt_path
            self.models_trt = {}
            self.models_trt_path = {}
            for model_data in models_trt_list:
                model_name, model_path = list(model_data.items())[0]
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
            # self.showModelLoadingProgressBar()
            #time.sleep(0.5)
            if session_options is None:
                model_instance = onnxruntime.InferenceSession(self.models_path[model_name], providers=self.providers)
            else:
                model_instance = onnxruntime.InferenceSession(self.models_path[model_name], sess_options=session_options, providers=self.providers)
            # model_instance = 'FAsfd'
            # self.hideModelLoadProgressBar()
            return model_instance
    
    def load_dfm_model(self, dfm_model):
        if not self.dfm_models.get(dfm_model):
            self.dfm_models[dfm_model] = DFMModel(self.main_window.dfm_models_data[dfm_model], self.providers, self.device)
        return self.dfm_models[dfm_model]


    def load_model_trt(self, model_name, custom_plugin_path=None, precision='fp16', debug=False):
        # self.showModelLoadingProgressBar()
        #time.sleep(0.5)
        if not os.path.exists(self.models_trt_path[model_name]):
            onnx2trt(onnx_model_path=self.models_path[model_name],
                     trt_model_path=self.models_trt_path[model_name],
                     precision=precision,
                     custom_plugin_path=custom_plugin_path,
                     verbose=False
                    )
        model_instance = TensorRTPredictor(model_path=self.models_trt_path[model_name], custom_plugin_path=custom_plugin_path, pool_size=self.nThreads, device=self.device, debug=debug)

        # model_instance = 'FAsfd'
        # self.hideModelLoadProgressBar()
        return model_instance

    def delete_models(self):
        for model_data in models_list:
            model_name, _ = list(model_data.items())[0]
            self.models[model_name] = None #Model Instance

        self.clip_session = []

    def delete_models_trt(self):
        if 'trt' in globals():
            for model_data in models_trt_list:
                model_name, _ = list(model_data.items())[0]
                if isinstance(self.models_trt[model_name], TensorRTPredictor):
                    # È un'istanza di TensorRTPredictor
                    self.models_trt[model_name].cleanup()
                    del self.models_trt[model_name]
                    self.models_trt[model_name] = None #Model Instance

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

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5, use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles:list[int]=[0]):
        bboxes = []
        kpss_5 = []
        kpss = []

        if detect_mode=='Retinaface':
            if not self.models['RetinaFace']:
                self.models['RetinaFace'] = self.load_model('RetinaFace')

            bboxes, kpss_5, kpss = self.detect_retinaface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='SCRFD':
            if not self.models['SCRFD2.5g']:
                self.models['SCRFD2.5g'] = self.load_model('SCRFD2.5g')

            bboxes, kpss_5, kpss = self.detect_scrdf(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yolov8':
            if not self.models['YoloFace8n']:
                self.models['YoloFace8n'] = self.load_model('YoloFace8n')

            bboxes, kpss_5, kpss = self.detect_yoloface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yunet':
            if not self.models['YunetN']:
                self.models['YunetN'] = self.load_model('YunetN')

            bboxes, kpss_5, kpss = self.detect_yunet(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        return bboxes, kpss_5, kpss

    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        kpss_5 = []
        kpss = []
        scores = []

        if detect_mode=='5':
            if not self.models['FaceLandmark5']:
                self.models['FaceLandmark5'] = self.load_model('FaceLandmark5')

                feature_maps = [[64, 64], [32, 32], [16, 16]]
                min_sizes = [[16, 32], [64, 128], [256, 512]]
                steps = [8, 16, 32]
                image_size = 512
                # re-initialize self.anchors due to clear_mem function
                self.anchors  = []

                for k, f in enumerate(feature_maps):
                    min_size_array = min_sizes[k]
                    for i, j in product(range(f[0]), range(f[1])):
                        for min_size in min_size_array:
                            s_kx = min_size / image_size
                            s_ky = min_size / image_size
                            dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                            dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                            for cy, cx in product(dense_cy, dense_cx):
                                self.anchors += [cx, cy, s_kx, s_ky]

            kpss_5, kpss, scores = self.detect_face_landmark_5(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='68':
            if not self.models['FaceLandmark68']:
                self.models['FaceLandmark68'] = self.load_model('FaceLandmark68')

            kpss_5, kpss, scores = self.detect_face_landmark_68(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='3d68':
            if not self.models['FaceLandmark3d68']:
                self.models['FaceLandmark3d68'] = self.load_model('FaceLandmark3d68')
                with open(f'{models_dir}/meanshape_68.pkl', 'rb') as f:
                    self.mean_lmk = pickle.load(f)

            kpss_5, kpss, scores = self.detect_face_landmark_3d68(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='98':
            if not self.models['FaceLandmark98']:
                self.models['FaceLandmark98'] = self.load_model('FaceLandmark98')

            kpss_5, kpss, scores = self.detect_face_landmark_98(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='106':
            if not self.models['FaceLandmark106']:
                self.models['FaceLandmark106'] = self.load_model('FaceLandmark106')

            kpss_5, kpss, scores = self.detect_face_landmark_106(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='203':
            if not self.models['FaceLandmark203']:
                self.models['FaceLandmark203'] = self.load_model('FaceLandmark203')

            kpss_5, kpss, scores = self.detect_face_landmark_203(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='478':
            if not self.models['FaceLandmark478']:
                self.models['FaceLandmark478'] = self.load_model('FaceLandmark478')

            if not self.models['FaceBlendShapes']:
                self.models['FaceBlendShapes'] = self.load_model('FaceBlendShapes')

            kpss_5, kpss, scores = self.detect_face_landmark_478(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        if len(kpss_5) > 0:
            if len(scores) > 0:
                if np.mean(scores) >= score:
                    return kpss_5, kpss, scores
            else:
                return kpss_5, kpss, scores

        return [], [], []

    def detect_retinaface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device=self.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        #det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.models['RetinaFace'].io_binding()
            io_binding.bind_input(name='input.1', device_type=self.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            io_binding.bind_output('448', self.device)
            io_binding.bind_output('471', self.device)
            io_binding.bind_output('494', self.device)
            io_binding.bind_output('451', self.device)
            io_binding.bind_output('474', self.device)
            io_binding.bind_output('497', self.device)
            io_binding.bind_output('454', self.device)
            io_binding.bind_output('477', self.device)
            io_binding.bind_output('500', self.device)

            # Sync and run model
            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device != "cpu":
                self.syncvec.cpu()
            self.models['RetinaFace'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            input_height = aimg.shape[2]
            input_width = aimg.shape[3]

            fmc = 3
            center_cache = {}
            for idx, stride in enumerate([8, 16, 32]):
                scores = net_outs[idx]
                bbox_preds = net_outs[idx+fmc]
                bbox_preds = bbox_preds * stride

                kps_preds = net_outs[idx+fmc*2] * stride
                height = input_height // stride
                width = input_width // stride
                key = (height, width, stride)
                if key in center_cache:
                    anchor_centers = center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                    if len(center_cache)<100:
                        center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=score)[0]

                x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
                y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
                x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
                y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                preds = []
                for i in range(0, kps_preds.shape[1], 2):
                    px = anchor_centers[:, i%2] + kps_preds[:, i]
                    py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                    preds.append(px)
                    preds.append(py)
                kpss = np.stack(preds, axis=-1)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                # Always add to kpss, regardless of the length of landmark_kpss.
                kpss.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(kpss, dtype=object)

        return det, kpss_5, kpss

    def detect_scrdf(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device=self.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        #det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        input_name = self.models['SCRFD2.5g'].get_inputs()[0].name
        outputs = self.models['SCRFD2.5g'].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.models['SCRFD2.5g'].io_binding()
            io_binding.bind_input(name=input_name, device_type=self.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], self.device)

            # Sync and run model
            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device != "cpu":
                self.syncvec.cpu()
            self.models['SCRFD2.5g'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            input_height = aimg.shape[2]
            input_width = aimg.shape[3]

            fmc = 3
            center_cache = {}
            for idx, stride in enumerate([8, 16, 32]):
                scores = net_outs[idx]
                bbox_preds = net_outs[idx+fmc]
                bbox_preds = bbox_preds * stride

                kps_preds = net_outs[idx+fmc*2] * stride
                height = input_height // stride
                width = input_width // stride
                key = (height, width, stride)
                if key in center_cache:
                    anchor_centers = center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                    if len(center_cache)<100:
                        center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=score)[0]

                x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
                y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
                x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
                y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                preds = []
                for i in range(0, kps_preds.shape[1], 2):
                    px = anchor_centers[:, i%2] + kps_preds[:, i]
                    py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                    preds.append(px)
                    preds.append(py)
                kpss = np.stack(preds, axis=-1)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                # Always add to kpss, regardless of the length of landmark_kpss.
                kpss.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(kpss, dtype=object)

        return det, kpss_5, kpss

    def detect_yoloface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        # model_ratio = float(input_size[1]) / input_size[0]
        model_ratio = 1.0
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device=self.device)
        det_img[:new_height,:new_width,  :] = img

        det_img = det_img.permute(2, 0, 1)

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = aimg.permute(1, 2, 0)
                aimg = torch.div(aimg, 255.0)
                aimg = aimg.permute(2, 0, 1)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                aimg = det_img.permute(1, 2, 0)
                aimg = torch.div(aimg, 255.0)
                aimg = aimg.permute(2, 0, 1)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
                IM = None

            io_binding = self.models['YoloFace8n'].io_binding()
            io_binding.bind_input(name='images', device_type=self.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())
            io_binding.bind_output('output0', self.device)

            # Sync and run model
            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device != "cpu":
                self.syncvec.cpu()
            self.models['YoloFace8n'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            outputs = np.squeeze(net_outs).T

            bbox_raw, score_raw, kps_raw = np.split(outputs, [4, 5], axis=1)

            keep_indices = np.where(score_raw > score)[0]

            if keep_indices.any():
                bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[keep_indices], score_raw[keep_indices]

                # Compute the transformed bounding box coordinates
                x1 = bbox_raw[:, 0] - bbox_raw[:, 2] / 2
                y1 = bbox_raw[:, 1] - bbox_raw[:, 3] / 2
                x2 = bbox_raw[:, 0] + bbox_raw[:, 2] / 2
                y2 = bbox_raw[:, 1] + bbox_raw[:, 3] / 2

                # Stack the results into a single array
                bboxes_raw = np.stack((x1, y1, x2, y2), axis=-1)

                # bboxes
                if angle != 0:
                    if len(bboxes_raw) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = bboxes_raw[:, :2]  # (x1, y1)
                        points2 = bboxes_raw[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        bboxes_raw = np.hstack((points1, points2))

                kps_list = []
                for kps in kps_raw:
                    indexes = np.arange(0, len(kps), 3)
                    temp_kps = []
                    for index in indexes:
                        temp_kps.append([kps[index], kps[index + 1]])
                    kps_list.append(np.array(temp_kps))

                kpss_raw = np.stack(kps_list)

                if do_rotation:
                    for i in range(len(kpss_raw)):
                        face_size = max(bboxes_raw[i][2] - bboxes_raw[i][0], bboxes_raw[i][3] - bboxes_raw[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, kpss_raw[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            score_raw[i] = 0.0

                        if angle != 0:
                            kpss_raw[i] = faceutil.trans_points2d(kpss_raw[i], IM)

                    keep_indices = np.where(score_raw>=score)[0]
                    score_raw = score_raw[keep_indices]
                    bboxes_raw = bboxes_raw[keep_indices]
                    kpss_raw = kpss_raw[keep_indices]

                kpss_list.append(kpss_raw)
                bboxes_list.append(bboxes_raw)
                scores_list.append(score_raw)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                # Always add to kpss, regardless of the length of landmark_kpss.
                kpss.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(kpss, dtype=object)

        return det, kpss_5, kpss

    def detect_yunet(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=False)
        img = resize(img)

        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device=self.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to BGR
        det_img = det_img[:, :, [2,1,0]]

        det_img = det_img.permute(2, 0, 1) #3,640,640

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        input_name = self.models['YunetN'].get_inputs()[0].name
        outputs = self.models['YunetN'].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()
            aimg = aimg.to(dtype=torch.float32)

            io_binding = self.models['YunetN'].io_binding()
            io_binding.bind_input(name=input_name, device_type=self.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], self.device)

            # Sync and run model
            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device != "cpu":
                self.syncvec.cpu()
            self.models['YunetN'].run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()

            strides = [8, 16, 32]
            for idx, stride in enumerate(strides):
                cls_pred = net_outs[idx].reshape(-1, 1)
                obj_pred = net_outs[idx + len(strides)].reshape(-1, 1)
                reg_pred = net_outs[idx + len(strides) * 2].reshape(-1, 4)
                kps_pred = net_outs[idx + len(strides) * 3].reshape(
                    -1, 5 * 2)

                anchor_centers = np.stack(
                    np.mgrid[:(input_size[1] // stride), :(input_size[0] //
                                                           stride)][::-1],
                    axis=-1)
                anchor_centers = (anchor_centers * stride).astype(
                    np.float32).reshape(-1, 2)

                scores = (cls_pred * obj_pred)
                pos_inds = np.where(scores>=score)[0]

                bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[:]
                bbox_wh = np.exp(reg_pred[:, 2:]) * stride
                tl_x = (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.)
                tl_y = (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.)
                br_x = (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.)
                br_y = (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.)

                bboxes = np.stack([tl_x, tl_y, br_x, br_y], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                kpss = np.concatenate(
                    [((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers)
                     for i in range(5)],
                    axis=-1)

                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                # Always add to kpss, regardless of the length of landmark_kpss.
                kpss.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(kpss, dtype=object)

        return det, kpss_5, kpss

    def detect_face_landmark_5(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 512.0  / (max(w, h)*1.5)
            image, M = faceutil.transform(img, center, 512, _scale, rotate)
        else:
            image, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 512, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        image = image.permute(1,2,0)

        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device=self.device)
        image = torch.sub(image, mean)

        image = image.permute(2,0,1)
        image = torch.reshape(image, (1, 3, 512, 512))

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device=self.device)

        conf = torch.empty((1,10752,2), dtype=torch.float32, device=self.device).contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device=self.device).contiguous()

        io_binding = self.models['FaceLandmark5'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['FaceLandmark5'].run_with_iobinding(io_binding)

        scores = torch.squeeze(conf)[:, 1]
        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to(self.device)

        pre = torch.squeeze(landmarks, 0)

        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        score=.1
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()
        scores = scores.cpu().numpy()

        landmarks, scores = landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]

        if len(order) > 0:
            landmarks = landmarks[order][0]
            scores = scores[order][0]

            landmarks = np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])

            IM = faceutil.invertAffineTransform(M)
            landmarks = faceutil.trans_points2d(landmarks, IM)
            scores = np.array([scores])

            return landmarks, landmarks, scores

        return [], [], []

    def detect_face_landmark_68(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            crop_image, affine_matrix = faceutil.warp_face_by_bounding_box_for_landmark_68(img, bbox, (256, 256))
        else:
            crop_image, affine_matrix = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 256, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        crop_image = crop_image.to(dtype=torch.float32)
        crop_image = torch.div(crop_image, 255.0)
        crop_image = torch.unsqueeze(crop_image, 0).contiguous()

        io_binding = self.models['FaceLandmark68'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32,  shape=crop_image.size(), buffer_ptr=crop_image.data_ptr())

        io_binding.bind_output('landmarks_xyscore', self.device)
        io_binding.bind_output('heatmaps', self.device)

        # Sync and run model
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['FaceLandmark68'].run_with_iobinding(io_binding)
        net_outs = io_binding.copy_outputs_to_cpu()
        face_landmark_68 = net_outs[0]
        face_heatmap = net_outs[1]

        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64.0
        face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256.0
        face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))

        face_landmark_68 = face_landmark_68.reshape(-1, 2)
        face_landmark_68_score = np.amax(face_heatmap, axis = (2, 3))
        face_landmark_68_score = face_landmark_68_score.reshape(-1, 1)

        face_landmark_68_5, face_landmark_68_score = faceutil.convert_face_landmark_68_to_5(face_landmark_68, face_landmark_68_score)

        return face_landmark_68_5, face_landmark_68, face_landmark_68_score

    def detect_face_landmark_3d68(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 192  / (max(w, h)*1.5)
            aimg, M = faceutil.transform(img, center, 192, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=192, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = self.normalize(aimg)
        io_binding = self.models['FaceLandmark3d68'].io_binding()
        io_binding.bind_input(name='data', device_type=self.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('fc1', self.device)

        # Sync and run model
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['FaceLandmark3d68'].run_with_iobinding(io_binding)
        pred = io_binding.copy_outputs_to_cpu()[0][0]

        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if 68 < pred.shape[0]:
            pred = pred[68*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (192 // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (192 // 2)

        IM = faceutil.invertAffineTransform(M)
        pred = faceutil.trans_points3d(pred, IM)

        # at moment we don't use 3d points
        '''
        P = faceutil.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
        s, R, t = faceutil.P2sRt(P)
        rx, ry, rz = faceutil.matrix2angle(R)
        pose = np.array( [rx, ry, rz], dtype=np.float32 ) #pitch, yaw, roll
        '''

        # convert from 3d68 to 2d68 keypoints
        landmark2d68 = np.array(pred[:, [0, 1]])

        # convert from 68 to 5 keypoints
        landmark2d68_5, _ = faceutil.convert_face_landmark_68_to_5(landmark2d68, [])

        return landmark2d68_5, landmark2d68, []

    def detect_face_landmark_98(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            crop_image, detail = faceutil.warp_face_by_bounding_box_for_landmark_98(img, bbox, (256, 256))
        else:
            crop_image, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=256, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
            h, w = (crop_image.size(dim=1), crop_image.size(dim=2))

        landmark = []
        landmark_5 = []
        landmark_score = []
        if crop_image is not None:
            crop_image = crop_image.to(dtype=torch.float32)
            crop_image = torch.div(crop_image, 255.0)
            crop_image = torch.unsqueeze(crop_image, 0).contiguous()

            io_binding = self.models['FaceLandmark98'].io_binding()
            io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32,  shape=crop_image.size(), buffer_ptr=crop_image.data_ptr())

            io_binding.bind_output('landmarks_xyscore', self.device)

            # Sync and run model
            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device != "cpu":
                self.syncvec.cpu()
            self.models['FaceLandmark98'].run_with_iobinding(io_binding)
            landmarks_xyscore = io_binding.copy_outputs_to_cpu()[0]

            if len(landmarks_xyscore) > 0:
                for one_face_landmarks in landmarks_xyscore:
                    landmark_score = one_face_landmarks[:, [2]].reshape(-1)
                    landmark = one_face_landmarks[:, [0, 1]].reshape(-1,2)

                    ##recover, and grouped as [98,2]
                    if from_points == False:
                        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
                        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]
                    else:
                        landmark[:, 0] = landmark[:, 0] * w
                        landmark[:, 1] = landmark[:, 1] * h

                        IM = faceutil.invertAffineTransform(M)
                        landmark = faceutil.trans_points2d(landmark, IM)

                    landmark_5, landmark_score = faceutil.convert_face_landmark_98_to_5(landmark, landmark_score)

        return landmark_5, landmark, landmark_score

    def detect_face_landmark_106(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 192  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 192, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=192, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = self.normalize(aimg)
        io_binding = self.models['FaceLandmark106'].io_binding()
        io_binding.bind_input(name='data', device_type=self.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('fc1', self.device)

        # Sync and run model
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['FaceLandmark106'].run_with_iobinding(io_binding)
        pred = io_binding.copy_outputs_to_cpu()[0][0]

        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))

        if 106 < pred.shape[0]:
            pred = pred[106*-1:,:]

        pred[:, 0:2] += 1
        pred[:, 0:2] *= (192 // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (192 // 2)

        IM = faceutil.invertAffineTransform(M)
        pred = faceutil.trans_points(pred, IM)

        pred_5 = []
        if pred is not None:
            # convert from 106 to 5 keypoints
            pred_5 = faceutil.convert_face_landmark_106_to_5(pred)

        return pred_5, pred, []

    def detect_face_landmark_203(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 224  / (max(w, h)*1.5)

            aimg, M = faceutil.transform(img, center, 224, _scale, rotate)
        elif len(det_kpss) == 0:
            return [], [], []
        else:
            if det_kpss.shape[0] == 5:
                aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=224, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
            else:
                aimg, M, IM = faceutil.warp_face_by_face_landmark_x(img, det_kpss, dsize=224, scale=1.5, vy_ratio=-0.1, interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = torch.div(aimg, 255.0)
        io_binding = self.models['FaceLandmark203'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('output', self.device)
        io_binding.bind_output('853', self.device)
        io_binding.bind_output('856', self.device)

        # Sync and run model
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['FaceLandmark203'].run_with_iobinding(io_binding)
        out_lst = io_binding.copy_outputs_to_cpu()
        out_pts = out_lst[2]

        out_pts = out_pts.reshape((-1, 2)) * 224.0

        if len(det_kpss) == 0 or det_kpss.shape[0] == 5:
            IM = faceutil.invertAffineTransform(M)

        out_pts = faceutil.trans_points(out_pts, IM)

        out_pts_5 = []
        if out_pts is not None:
            # convert from 203 to 5 keypoints
            out_pts_5 = faceutil.convert_face_landmark_203_to_5(out_pts)

        return out_pts_5, out_pts, []

    def detect_face_landmark_478(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 256.0  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 256, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 256, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = torch.div(aimg, 255.0)
        io_binding = self.models['FaceLandmark478'].io_binding()
        io_binding.bind_input(name='input_12', device_type=self.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('Identity', self.device)
        io_binding.bind_output('Identity_1', self.device)
        io_binding.bind_output('Identity_2', self.device)

        # Sync and run model
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['FaceLandmark478'].run_with_iobinding(io_binding)
        landmarks, faceflag, blendshapes = io_binding.copy_outputs_to_cpu()
        landmarks = landmarks.reshape( (1,478,3))

        landmark = []
        landmark_5 = []
        landmark_score = []
        if len(landmarks) > 0:
            for one_face_landmarks in landmarks:
                landmark = one_face_landmarks
                IM = faceutil.invertAffineTransform(M)
                landmark = faceutil.trans_points3d(landmark, IM)
                '''
                P = faceutil.estimate_affine_matrix_3d23d(self.mean_lmk, landmark)
                s, R, t = faceutil.P2sRt(P)
                rx, ry, rz = faceutil.matrix2angle(R)
                pose = np.array( [rx, ry, rz], dtype=np.float32 ) #pitch, yaw, roll
                '''
                landmark = landmark[:, [0, 1]].reshape(-1,2)

                #get scores
                landmark_for_score = landmark[self.LandmarksSubsetIdxs]
                landmark_for_score = landmark_for_score[:, :2]
                landmark_for_score = np.expand_dims(landmark_for_score, axis=0)
                landmark_for_score = landmark_for_score.astype(np.float32)
                landmark_for_score = torch.from_numpy(landmark_for_score).to(self.device)

                io_binding_bs = self.models['FaceBlendShapes'].io_binding()
                io_binding_bs.bind_input(name='input_points', device_type=self.device, device_id=0, element_type=np.float32,  shape=tuple(landmark_for_score.shape), buffer_ptr=landmark_for_score.data_ptr())
                io_binding_bs.bind_output('output', self.device)

                # Sync and run model
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elif self.device != "cpu":
                    self.syncvec.cpu()
                self.models['FaceBlendShapes'].run_with_iobinding(io_binding_bs)
                landmark_score = io_binding_bs.copy_outputs_to_cpu()[0]

                # convert from 478 to 5 keypoints
                landmark_5 = faceutil.convert_face_landmark_478_to_5(landmark)

        #return landmark, landmark_score
        return landmark_5, landmark, []

    def get_arcface_model(self, face_swapper_model): 
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        if not self.models[arcface_model]:
            self.models[arcface_model] = self.load_model(arcface_model)

        if arcface_model == 'CSCSArcFace':
            embedding, cropped_image = self.recognize_cscs(img, kps)
        else:
            embedding, cropped_image = self.recognize(arcface_model, img, kps, similarity_type=similarity_type)

        return embedding, cropped_image
        
    def run_recognize(self, img, kps, similarity_type='Opal', face_swapper_model='Inswapper128'):
        arcface_model = self.get_arcface_model(face_swapper_model)
        if not self.models[arcface_model]:
            self.models[arcface_model] = self.load_model(arcface_model)
        
        if arcface_model == 'CSCSArcFace':
            embedding, cropped_image = self.recognize_cscs(img, kps)
        else:
            embedding, cropped_image = self.recognize(arcface_model, img, kps, similarity_type=similarity_type)

        return embedding, cropped_image

    def recognize(self, arcface_model, img, face_kps, similarity_type):
        if similarity_type == 'Optimal':
            # Find transform & Transform
            img, _ = faceutil.warp_face_by_face_landmark_5(img, face_kps, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)
        elif similarity_type == 'Pearl':
            # Find transform
            dst = self.arcface_dst.copy()
            dst[:, 0] += 8.0

            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 128, 128)
            img = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(img)
        else:
            # Find transform
            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, self.arcface_dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 112, 112)

        if arcface_model == 'Inswapper128ArcFace':
            cropped_image = img.permute(1, 2, 0).clone()
            if img.dtype == torch.uint8:
                img = img.to(torch.float32)  # Convert to float32 if uint8
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5)
        elif arcface_model == 'SimSwapArcFace':
            cropped_image = img.permute(1, 2, 0).clone()
            if img.dtype == torch.uint8:
                img = torch.div(img.to(torch.float32), 255.0)
            img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False)
        else:
            cropped_image = img.permute(1,2,0).clone() #112,112,3
            if img.dtype == torch.uint8:
                img = img.to(torch.float32)  # Convert to float32 if uint8
            # Normalize
            img = torch.div(img, 127.5)
            img = torch.sub(img, 1)

        # Prepare data and find model parameters
        img = torch.unsqueeze(img, 0).contiguous()
        input_name = self.models[arcface_model].get_inputs()[0].name

        outputs = self.models[arcface_model].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = self.models[arcface_model].io_binding()
        io_binding.bind_input(name=input_name, device_type=self.device, device_id=0, element_type=np.float32,  shape=img.size(), buffer_ptr=img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], self.device)

        # Sync and run model
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models[arcface_model].run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def preprocess_image_cscs(self, img, face_kps):
        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, self.FFHQ_kps)

        temp = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
        temp = v2.functional.crop(temp, 0,0, 512, 512)
        
        image = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(temp)
        
        cropped_image = image.permute(1, 2, 0).clone()
        if image.dtype == torch.uint8:
            image = torch.div(image.to(torch.float32), 255.0)

        image = v2.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)

        # Ritorna l'immagine e l'immagine ritagliata
        return torch.unsqueeze(image, 0).contiguous(), cropped_image  # (C, H, W) e (H, W, C)
    
    def recognize_cscs(self, img, face_kps):
        # Usa la funzione di preprocessamento
        img, cropped_image = self.preprocess_image_cscs(img, face_kps)

        io_binding = self.models['CSCSArcFace'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=img.size(), buffer_ptr=img.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()

        self.models['CSCSArcFace'].run_with_iobinding(io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]
        embedding = torch.from_numpy(output).to('cpu')
        embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
        embedding = embedding.numpy().flatten()

        embedding_id = self.recognize_cscs_id_adapter(img, None)
        embedding = embedding + embedding_id

        return embedding, cropped_image

    def recognize_cscs_id_adapter(self, img, face_kps):
        if not self.models['CSCSIDArcFace']:
            self.models['CSCSIDArcFace'] = self.load_model('CSCSIDArcFace')

        # Use preprocess_image_cscs when face_kps is not None. When it is None img is already preprocessed.
        if face_kps is not None:
            img, _ = self.preprocess_image_cscs(img, face_kps)

        io_binding = self.models['CSCSIDArcFace'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=img.size(), buffer_ptr=img.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
            
        self.models['CSCSIDArcFace'].run_with_iobinding(io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]
        embedding_id = torch.from_numpy(output).to('cpu')
        embedding_id = torch.nn.functional.normalize(embedding_id, dim=-1, p=2)

        return embedding_id.numpy().flatten()
    
    def calc_swapper_latent(self, source_embedding):
        if not self.models['Inswapper128']:
            graph = onnx.load(self.models_path['Inswapper128']).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])

        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_swapper(self, image, embedding, output):
        if not self.models['Inswapper128']:
            self.models['Inswapper128'] = self.load_model('Inswapper128')

        io_binding = self.models['Inswapper128'].io_binding()
        io_binding.bind_input(name='target', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['Inswapper128'].run_with_iobinding(io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        #latent /= np.linalg.norm(latent)
        latent = latent/np.linalg.norm(latent,axis=1,keepdims=True)
        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        if not self.models['SimSwap512']:
            self.models['SimSwap512'] = self.load_model('SimSwap512')

        io_binding = self.models['SimSwap512'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='onnx::Gemm_1', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['SimSwap512'].run_with_iobinding(io_binding)

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1,-1))

        return latent

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        if swapper_model == 'GhostFace-v1':
            if not self.models['GhostFacev1']:
                self.models['GhostFacev1'] = self.load_model('GhostFacev1')

            ghostfaceswap_model = self.models['GhostFacev1']
            output_name = '781'

        elif swapper_model == 'GhostFace-v2':
            if not self.models['GhostFacev2']:
                self.models['GhostFacev2'] = self.load_model('GhostFacev2')

            ghostfaceswap_model = self.models['GhostFacev2']
            output_name = '1165'

        elif swapper_model == 'GhostFace-v3':
            if not self.models['GhostFacev3']:
                self.models['GhostFacev3'] = self.load_model('GhostFacev3')

            ghostfaceswap_model = self.models['GhostFacev3']
            output_name = '1549'

        io_binding = ghostfaceswap_model.io_binding()
        io_binding.bind_input(name='target', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name=output_name, device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        ghostfaceswap_model.run_with_iobinding(io_binding)

    def calc_swapper_latent_cscs(self, source_embedding):
        latent = source_embedding.reshape((1,-1))

        return latent

    def run_swapper_cscs(self, image, embedding, output):
        if not self.models['CSCS']:
            self.models['CSCS'] = self.load_model('CSCS')

        io_binding = self.models['CSCS'].io_binding()
        io_binding.bind_input(name='input_1', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='input_2', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['CSCS'].run_with_iobinding(io_binding)

    def run_GFPGAN(self, image, output):
        if not self.models['GFPGANv1.4']:
            self.models['GFPGANv1.4'] = self.load_model('GFPGANv1.4')

        io_binding = self.models['GFPGANv1.4'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['GFPGANv1.4'].run_with_iobinding(io_binding)

    def run_GPEN_256(self, image, output):
        if not self.models['GPENBFR256']:
            self.models['GPENBFR256'] = self.load_model('GPENBFR256')

        io_binding = self.models['GPENBFR256'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['GPENBFR256'].run_with_iobinding(io_binding)

    def run_GPEN_512(self, image, output):
        if not self.models['GPENBFR512']:
            self.models['GPENBFR512'] = self.load_model('GPENBFR512')

        io_binding = self.models['GPENBFR512'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['GPENBFR512'].run_with_iobinding(io_binding)

    def run_GPEN_1024(self, image, output):
        if not self.models['GPENBFR1024']:
            self.models['GPENBFR1024'] = self.load_model('GPENBFR1024')

        io_binding = self.models['GPENBFR1024'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['GPENBFR1024'].run_with_iobinding(io_binding)

    def run_GPEN_2048(self, image, output):
        if not self.models['GPENBFR2048']:
            self.models['GPENBFR2048'] = self.load_model('GPENBFR2048')

        io_binding = self.models['GPENBFR2048'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['GPENBFR2048'].run_with_iobinding(io_binding)

    def run_codeformer(self, image, output, fidelity_weight_value=0.9):
        if not self.models['CodeFormer']:
            self.models['CodeFormer'] = self.load_model('CodeFormer')

        io_binding = self.models['CodeFormer'].io_binding()
        io_binding.bind_input(name='x', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        w = np.array([fidelity_weight_value], dtype=np.double)
        io_binding.bind_cpu_input('w', w)
        io_binding.bind_output(name='y', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['CodeFormer'].run_with_iobinding(io_binding)

    def run_VQFR_v2(self, image, output, fidelity_ratio_value):
        if not self.models['VQFRv2']:
            self.models['VQFRv2'] = self.load_model('VQFRv2')

        assert fidelity_ratio_value >= 0.0 and fidelity_ratio_value <= 1.0, 'fidelity_ratio must in range[0,1]'
        fidelity_ratio = torch.tensor(fidelity_ratio_value).to(self.device)

        io_binding = self.models['VQFRv2'].io_binding()
        io_binding.bind_input(name='x_lq', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='fidelity_ratio', device_type=self.device, device_id=0, element_type=np.float32, shape=fidelity_ratio.size(), buffer_ptr=fidelity_ratio.data_ptr())
        io_binding.bind_output('enc_feat', self.device)
        io_binding.bind_output('quant_logit', self.device)
        io_binding.bind_output('texture_dec', self.device)
        io_binding.bind_output(name='main_dec', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['VQFRv2'].run_with_iobinding(io_binding)

    def run_RestoreFormerPlusPlus(self, image, output):
        if not self.models['RestoreFormerPlusPlus']:
            self.models['RestoreFormerPlusPlus'] = self.load_model('RestoreFormerPlusPlus')

        io_binding = self.models['RestoreFormerPlusPlus'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='2359', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())
        io_binding.bind_output('1228', self.device)
        io_binding.bind_output('1238', self.device)
        io_binding.bind_output('onnx::MatMul_1198', self.device)
        io_binding.bind_output('onnx::Shape_1184', self.device)
        io_binding.bind_output('onnx::ArgMin_1182', self.device)
        io_binding.bind_output('input.1', self.device)
        io_binding.bind_output('x', self.device)
        io_binding.bind_output('x.3', self.device)
        io_binding.bind_output('x.7', self.device)
        io_binding.bind_output('x.11', self.device)
        io_binding.bind_output('x.15', self.device)
        io_binding.bind_output('input.252', self.device)
        io_binding.bind_output('input.280', self.device)
        io_binding.bind_output('input.288', self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['RestoreFormerPlusPlus'].run_with_iobinding(io_binding)

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
        _, _, height, width = img.shape

        # Calcolo del numero di tile necessari
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # Calcolo del padding necessario per adattare l'immagine alle dimensioni dei tile
        pad_right = (tile_size - (width % tile_size)) % tile_size
        pad_bottom = (tile_size - (height % tile_size)) % tile_size

        # Padding dell'immagine se necessario
        if pad_right != 0 or pad_bottom != 0:
            img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom), 'constant', 0)

        # Creazione di un output tensor vuoto
        b, c, h, w = img.shape
        output = torch.empty((b, c, h * scale, w * scale), dtype=torch.float32, device=self.device).contiguous()

        # Selezione della funzione di upscaling in base al tipo
        upscaler_functions = {
            'RealEsrgan-x2-Plus': self.run_realesrganx2,
            'RealEsrgan-x4-Plus': self.run_realesrganx4,
            'BSRGan-x2': self.run_bsrganx2,
            'BSRGan-x4': self.run_bsrganx4,
            'UltraSharp-x4': self.run_ultrasharpx4,
            'UltraMix-x4': self.run_ultramixx4,
            'RealEsr-General-x4v3': self.run_realesrx4v3
        }

        fn_upscaler = upscaler_functions.get(enhancer_type)

        if not fn_upscaler:  # Se il tipo di enhancer non è valido
            if pad_right != 0 or pad_bottom != 0:
                img = v2.functional.crop(img, 0, 0, height, width)
            return img

        with torch.no_grad():  # Disabilita il calcolo del gradiente
            # Elaborazione dei tile
            for j in range(tiles_y):
                for i in range(tiles_x):
                    x_start, y_start = i * tile_size, j * tile_size
                    x_end, y_end = x_start + tile_size, y_start + tile_size

                    # Estrazione del tile di input
                    input_tile = img[:, :, y_start:y_end, x_start:x_end].contiguous()
                    output_tile = torch.empty((input_tile.shape[0], input_tile.shape[1], input_tile.shape[2] * scale, input_tile.shape[3] * scale), dtype=torch.float32, device=self.device).contiguous()

                    # Upscaling del tile
                    fn_upscaler(input_tile, output_tile)

                    # Inserimento del tile upscalato nel tensor di output
                    output_y_start, output_x_start = y_start * scale, x_start * scale
                    output_y_end, output_x_end = output_y_start + output_tile.shape[2], output_x_start + output_tile.shape[3]
                    output[:, :, output_y_start:output_y_end, output_x_start:output_x_end] = output_tile

            # Ritaglio dell'output per rimuovere il padding aggiunto
            if pad_right != 0 or pad_bottom != 0:
                output = v2.functional.crop(output, 0, 0, height * scale, width * scale)

        return output

    def run_realesrganx2(self, image, output):
        if not self.models['RealEsrganx2Plus']:
            self.models['RealEsrganx2Plus'] = self.load_model('RealEsrganx2Plus')

        io_binding = self.models['RealEsrganx2Plus'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['RealEsrganx2Plus'].run_with_iobinding(io_binding)

    def run_realesrganx4(self, image, output):
        if not self.models['RealEsrganx4Plus']:
            self.models['RealEsrganx4Plus'] = self.load_model('RealEsrganx4Plus')

        io_binding = self.models['RealEsrganx4Plus'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['RealEsrganx4Plus'].run_with_iobinding(io_binding)

    def run_realesrx4v3(self, image, output):
        if not self.models['RealEsrx4v3']:
            self.models['RealEsrx4v3'] = self.load_model('RealEsrx4v3')

        io_binding = self.models['RealEsrx4v3'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['RealEsrx4v3'].run_with_iobinding(io_binding)

    def run_bsrganx2(self, image, output):
        if not self.models['BSRGANx2']:
            self.models['BSRGANx2'] = self.load_model('BSRGANx2')

        io_binding = self.models['BSRGANx2'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['BSRGANx2'].run_with_iobinding(io_binding)

    def run_bsrganx4(self, image, output):
        if not self.models['BSRGANx4']:
            self.models['BSRGANx4'] = self.load_model('BSRGANx4')

        io_binding = self.models['BSRGANx4'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['BSRGANx4'].run_with_iobinding(io_binding)

    def run_ultrasharpx4(self, image, output):
        if not self.models['UltraSharpx4']:
            self.models['UltraSharpx4'] = self.load_model('UltraSharpx4')

        io_binding = self.models['UltraSharpx4'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['UltraSharpx4'].run_with_iobinding(io_binding)

    def run_ultramixx4(self, image, output):
        if not self.models['UltraMixx4']:
            self.models['UltraMixx4'] = self.load_model('UltraMixx4')

        io_binding = self.models['UltraMixx4'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['UltraMixx4'].run_with_iobinding(io_binding)

    def run_deoldify_artistic(self, image, output):
        if not self.models['DeoldifyArt']:
            self.models['DeoldifyArt'] = self.load_model('DeoldifyArt')

        io_binding = self.models['DeoldifyArt'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['DeoldifyArt'].run_with_iobinding(io_binding)

    def run_deoldify_stable(self, image, output):
        if not self.models['DeoldifyStable']:
            self.models['DeoldifyStable'] = self.load_model('DeoldifyStable')

        io_binding = self.models['DeoldifyStable'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['DeoldifyStable'].run_with_iobinding(io_binding)

    def run_deoldify_video(self, image, output):
        if not self.models['DeoldifyVideo']:
            self.models['DeoldifyVideo'] = self.load_model('DeoldifyVideo')

        io_binding = self.models['DeoldifyVideo'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['DeoldifyVideo'].run_with_iobinding(io_binding)

    def run_ddcolor_artistic(self, image, output):
        if not self.models['DDColorArt']:
            self.models['DDColorArt'] = self.load_model('DDColorArt')

        io_binding = self.models['DDColorArt'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['DDColorArt'].run_with_iobinding(io_binding)

    def run_ddcolor(self, image, output):
        if not self.models['DDcolor']:
            self.models['DDcolor'] = self.load_model('DDcolor')

        io_binding = self.models['DDcolor'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['DDcolor'].run_with_iobinding(io_binding)

    def run_occluder(self, image, output):
        if not self.models['Occluder']:
            self.models['Occluder'] = self.load_model('Occluder')

        io_binding = self.models['Occluder'].io_binding()
        io_binding.bind_input(name='img', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['Occluder'].run_with_iobinding(io_binding)

    def run_dfl_xseg(self, image, output):
        if not self.models['XSeg']:
            self.models['XSeg'] = self.load_model('XSeg')

        io_binding = self.models['XSeg'].io_binding()
        io_binding.bind_input(name='in_face:0', device_type=self.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='out_mask:0', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['XSeg'].run_with_iobinding(io_binding)

    # https://github.com/yakhyo/face-parsing
    def run_faceparser(self, image, output):
        if not self.models['FaceParser']:
            self.models['FaceParser'] = self.load_model('FaceParser')

        image = image.contiguous()
        io_binding = self.models['FaceParser'].io_binding()
        io_binding.bind_input(name='input', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.device, device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models['FaceParser'].run_with_iobinding(io_binding)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        # Ottieni il dispositivo su cui si trova l'immagine
        device = img.device

        # Controllo se la sessione CLIP è già stata inizializzata
        if not self.clip_session:
            self.clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            self.clip_session.eval()
            self.clip_session.load_state_dict(torch.load(f'{models_dir}/rd64-uni-refined.pth', weights_only=True), strict=False)
            self.clip_session.to(device)  # Sposta il modello sul dispositivo dell'immagine

        # Crea un mask tensor direttamente sul dispositivo dell'immagine
        clip_mask = torch.ones((352, 352), device=device)

        # L'immagine è già un tensore, quindi la converto a float32 e la normalizzo nel range [0, 1]
        img = img.float() / 255.0  # Conversione in float32 e normalizzazione

        # Rimuovi la parte ToTensor(), dato che img è già un tensore.
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352))
        ])

        # Applica la trasformazione all'immagine
        CLIPimg = transform(img).unsqueeze(0).contiguous().to(device)

        # Se ci sono prompt CLIPText, esegui la predizione
        if CLIPText != "":
            prompts = CLIPText.split(',')

            with torch.no_grad():
                # Esegui la predizione sulla sessione CLIP
                preds = self.clip_session(CLIPimg.repeat(len(prompts), 1, 1, 1), prompts)[0]

            # Calcola la maschera CLIP usando la sigmoid e tieni tutto sul dispositivo
            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts) - 1):
                clip_mask *= 1 - torch.sigmoid(preds[i + 1][0])

            # Applica la soglia sulla maschera
            thresh = CLIPAmount / 100.0
            clip_mask = (clip_mask > thresh).float()

        return clip_mask.unsqueeze(0)  # Ritorna il tensore torch direttamente
    
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
        temp = swapped_face_upscaled
        t512 = v2.Resize((512, 512), antialias=False)
        t256 = v2.Resize((256, 256), antialias=False)
        t1024 = v2.Resize((1024, 1024), antialias=False)
        t2048 = v2.Resize((2048, 2048), antialias=False)

        # If using a separate detection mode
        if restorer_det_type == 'Blend' or restorer_det_type == 'Reference':
            if restorer_det_type == 'Blend':
                # Set up Transformation
                dst = self.arcface_dst * 4.0
                dst[:,0] += 32.0

            elif restorer_det_type == 'Reference':
                try:
                    dst, _, _ = self.run_detect_landmark(swapped_face_upscaled, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='5', score=detect_score/100.0, from_points=False)
                except Exception as e:
                    print(f"exception: {e}")
                    return swapped_face_upscaled

            tform = trans.SimilarityTransform()
            tform.estimate(dst, self.FFHQ_kps)

            # Transform, scale, and normalize
            temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            temp = v2.functional.crop(temp, 0,0, 512, 512)

        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)

        if restorer_type == 'GPEN-256':
            temp = t256(temp)

        temp = torch.unsqueeze(temp, 0).contiguous()

        # Bindings
        outpred = torch.empty((1,3,512,512), dtype=torch.float32, device=self.device).contiguous()

        if restorer_type == 'GFPGAN-v1.4':
            self.run_GFPGAN(temp, outpred)

        elif restorer_type == 'CodeFormer':
            self.run_codeformer(temp, outpred, fidelity_weight)

        elif restorer_type == 'GPEN-256':
            outpred = torch.empty((1,3,256,256), dtype=torch.float32, device=self.device).contiguous()
            self.run_GPEN_256(temp, outpred)

        elif restorer_type == 'GPEN-512':
            self.run_GPEN_512(temp, outpred)

        elif restorer_type == 'GPEN-1024':
            temp = t1024(temp)
            outpred = torch.empty((1, 3, 1024, 1024), dtype=torch.float32, device=self.device).contiguous()
            self.run_GPEN_1024(temp, outpred)

        elif restorer_type == 'GPEN-2048':
            temp = t2048(temp)
            outpred = torch.empty((1, 3, 2048, 2048), dtype=torch.float32, device=self.device).contiguous()
            self.run_GPEN_2048(temp, outpred)

        elif restorer_type == 'RestoreFormer++':
            self.run_RestoreFormerPlusPlus(temp, outpred)

        elif restorer_type == 'VQFR-v2':
            self.run_VQFR_v2(temp, outpred, fidelity_weight)

        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)

        if restorer_type == 'GPEN-256' or restorer_type == 'GPEN-1024' or restorer_type == 'GPEN-2048':
            outpred = t512(outpred)

        # Invert Transform
        if restorer_det_type == 'Blend' or restorer_det_type == 'Reference':
            outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # Blend
        alpha = float(restorer_blend)/100.0
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred

    def apply_occlusion(self, img, amount):
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.device).contiguous()

        self.run_occluder(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount >0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount <0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def apply_dfl_xseg(self, img, amount):
        img = img.type(torch.float32)
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.device).contiguous()

        self.run_dfl_xseg(img, outpred)

        outpred = torch.clamp(outpred, min=0.0, max=1.0)
        outpred[outpred < 0.1] = 0
        # invert values to mask areas to keep
        outpred = 1.0 - outpred
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount > 0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount < 0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred
    
    def apply_face_parser(self, img, parameters):
        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        FaceAmount = parameters["BackgroundParserSlider"]

        img = torch.div(img, 255)
        img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = torch.reshape(img, (1, 3, 512, 512))
        outpred = torch.empty((1,19,512,512), dtype=torch.float32, device=self.device).contiguous()

        self.run_faceparser(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = torch.argmax(outpred, 0)

        face_attributes = {
            1: parameters['FaceParserSlider'], #Face
            2: parameters['LeftEyebrowParserSlider'], #Left Eyebrow
            3: parameters['RightEyebrowParserSlider'], #Right Eyebrow
            4: parameters['LeftEyeParserSlider'], #Left Eye
            5: parameters['RightEyeParserSlider'], #Right Eye
            6: parameters['EyeGlassesParserSlider'], #EyeGlasses
            10: parameters['NoseParserSlider'], #Nose
            11: parameters['MouthParserSlider'], #Mouth
            12: parameters['UpperLipParserSlider'], #Upper Lip
            13: parameters['LowerLipParserSlider'], #Lower Lip
            14: parameters['NeckParserSlider'], #Neck
            17: parameters['HairParserSlider'], #Hair
        }
        
        # Pre-calculated kernel for dilation (3x3 kernel to reduce iterations)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=self.device)  # Kernel 3x3

        face_parses = []
        for attribute in face_attributes.keys():
            if face_attributes[attribute] > 0:
                attribute_idxs = torch.tensor( [attribute], device=self.device)
                iters = int(face_attributes[attribute])

                attribute_parse = torch.isin(outpred, attribute_idxs)
                attribute_parse = torch.clamp(~attribute_parse, 0, 1).type(torch.float32)
                attribute_parse = torch.reshape(attribute_parse, (1,1,512,512))
                attribute_parse = torch.neg(attribute_parse)
                attribute_parse = torch.add(attribute_parse, 1)

                for i in range(iters):
                    attribute_parse = torch.nn.functional.conv2d(attribute_parse, kernel, padding=(1, 1))
                    attribute_parse = torch.clamp(attribute_parse, 0, 1)

                attribute_parse = torch.squeeze(attribute_parse)
                attribute_parse = torch.neg(attribute_parse)
                attribute_parse = torch.add(attribute_parse, 1)
                attribute_parse = torch.reshape(attribute_parse, (1, 512, 512))

                # Apply Gaussian blur if needed
                blur_kernel_size = parameters['FaceBlurParserSlider'] * 2 + 1
                if blur_kernel_size > 1:
                    gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['FaceBlurParserSlider'] + 1) * 0.2)
                    attribute_parse = gauss(attribute_parse)
            else:
                attribute_parse = torch.ones((1, 512, 512), dtype=torch.float32, device=self.device)
            face_parses.append(attribute_parse)

        # BG Parse
        bg_idxs = torch.tensor([0, 14, 15, 16, 17, 18], device=self.device)

        # For bg_parse, invert the mask so that the black background expands
        bg_parse = 1 - torch.isin(outpred, bg_idxs).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

        if FaceAmount > 0:
            for i in range(int(FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))  # Padding (1, 1) for 3x3 kernel
                bg_parse = torch.clamp(bg_parse, 0, 1)

            blur_kernel_size = parameters['BackgroundBlurParserSlider'] * 2 + 1
            if blur_kernel_size > 1:
                gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['BackgroundBlurParserSlider'] + 1) * 0.2)
                bg_parse = gauss(bg_parse)

            bg_parse = torch.clamp(bg_parse, 0, 1)

        elif FaceAmount < 0:
            bg_parse = 1 - bg_parse  # Invert mask back
            for i in range(int(-FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))  # Padding (1, 1) for 3x3 kernel
                bg_parse = torch.clamp(bg_parse, 0, 1)

            bg_parse = 1 - bg_parse  # Re-invert back
            blur_kernel_size = parameters['BackgroundBlurParserSlider'] * 2 + 1
            if blur_kernel_size > 1:
                gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['BackgroundBlurParserSlider'] + 1) * 0.2)
                bg_parse = gauss(bg_parse)

            bg_parse = torch.clamp(bg_parse, 0, 1)

        else:
            # If FaceAmount is 0, use a fully white mask
            bg_parse = torch.ones((1, 512, 512), dtype=torch.float32, device=self.device)

        out_parse = bg_parse.squeeze(0)
        for face_parse in face_parses:
            out_parse = torch.mul(out_parse, face_parse)

        # Final clamping to ensure the output parse is valid
        out_parse = torch.clamp(out_parse, 0, 1)

        return out_parse

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

    def soft_oval_mask(self, height, width, center, radius_x, radius_y, feather_radius=None):
        """
        Create a soft oval mask with feathering effect using integer operations.

        Args:
            height (int): Height of the mask.
            width (int): Width of the mask.
            center (tuple): Center of the oval (x, y).
            radius_x (int): Radius of the oval along the x-axis.
            radius_y (int): Radius of the oval along the y-axis.
            feather_radius (int): Radius for feathering effect.

        Returns:
            torch.Tensor: Soft oval mask tensor of shape (H, W).
        """
        if feather_radius is None:
            feather_radius = max(radius_x, radius_y) // 2  # Integer division

        # Calculating the normalized distance from the center
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

        # Calculating the normalized distance from the center
        normalized_distance = torch.sqrt(((x - center[0]) / radius_x) ** 2 + ((y - center[1]) / radius_y) ** 2)

        # Creating the oval mask with a feathering effect
        mask = torch.clamp((1 - normalized_distance) * (radius_x / feather_radius), 0, 1)

        return mask

    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        """
        Extract mouth from img_orig using the provided keypoints and place it in img_swap.

        Args:
            img_orig (torch.Tensor): The original image tensor of shape (C, H, W) from which mouth is extracted.
            img_swap (torch.Tensor): The target image tensor of shape (C, H, W) where mouth is placed.
            kpss_orig (list): List of keypoints arrays for detected faces. Each keypoints array contains coordinates for 5 keypoints.
            radius_factor_x (float): Factor to scale the horizontal radius. 1.0 means circular, >1.0 means wider oval, <1.0 means narrower.
            radius_factor_y (float): Factor to scale the vertical radius. 1.0 means circular, >1.0 means taller oval, <1.0 means shorter.
            x_offset (int): Horizontal offset for shifting the mouth left (negative value) or right (positive value).
            y_offset (int): Vertical offset for shifting the mouth up (negative value) or down (positive value).

        Returns:
            torch.Tensor: The resulting image tensor with mouth from img_orig placed on img_swap.
        """
        left_mouth = np.array([int(val) for val in kpss_orig[3]])
        right_mouth = np.array([int(val) for val in kpss_orig[4]])

        mouth_center = (left_mouth + right_mouth) // 2
        mouth_base_radius = int(np.linalg.norm(left_mouth - right_mouth) * size_factor)

        # Calculate the scaled radii
        radius_x = int(mouth_base_radius * radius_factor_x)
        radius_y = int(mouth_base_radius * radius_factor_y)

        # Apply the x/y_offset to the mouth center
        mouth_center[0] += x_offset
        mouth_center[1] += y_offset

        # Calculate bounding box for mouth region
        ymin = max(0, mouth_center[1] - radius_y)
        ymax = min(img_orig.size(1), mouth_center[1] + radius_y)
        xmin = max(0, mouth_center[0] - radius_x)
        xmax = min(img_orig.size(2), mouth_center[0] + radius_x)

        mouth_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
        mouth_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                         (radius_x, radius_y),
                                         radius_x, radius_y,
                                         feather_radius).to(img_orig.device)

        target_ymin = ymin
        target_ymax = ymin + mouth_region_orig.size(1)
        target_xmin = xmin
        target_xmax = xmin + mouth_region_orig.size(2)

        img_swap_mouth = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
        blended_mouth = blend_alpha * img_swap_mouth + (1 - blend_alpha) * mouth_region_orig

        img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = mouth_mask * blended_mouth + (1 - mouth_mask) * img_swap_mouth
        return img_swap

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        """
        Extract eyes from img_orig using the provided keypoints and place them in img_swap.

        Args:
            img_orig (torch.Tensor): The original image tensor of shape (C, H, W) from which eyes are extracted.
            img_swap (torch.Tensor): The target image tensor of shape (C, H, W) where eyes are placed.
            kpss_orig (list): List of keypoints arrays for detected faces. Each keypoints array contains coordinates for 5 keypoints.
            radius_factor_x (float): Factor to scale the horizontal radius. 1.0 means circular, >1.0 means wider oval, <1.0 means narrower.
            radius_factor_y (float): Factor to scale the vertical radius. 1.0 means circular, >1.0 means taller oval, <1.0 means shorter.
            x_offset (int): Horizontal offset for shifting the eyes left (negative value) or right (positive value).
            y_offset (int): Vertical offset for shifting the eyes up (negative value) or down (positive value).
            eye_spacing_offset (int): Horizontal offset to move eyes closer together (negative value) or farther apart (positive value).

        Returns:
            torch.Tensor: The resulting image tensor with eyes from img_orig placed on img_swap.
        """
        # Extract original keypoints for left and right eye
        left_eye = np.array([int(val) for val in kpss_orig[0]])
        right_eye = np.array([int(val) for val in kpss_orig[1]])

        # Apply horizontal offset (x-axis)
        left_eye[0] += x_offset
        right_eye[0] += x_offset

        # Apply vertical offset (y-axis)
        left_eye[1] += y_offset
        right_eye[1] += y_offset

        # Calculate eye distance and radii
        eye_distance = np.linalg.norm(left_eye - right_eye)
        base_eye_radius = int(eye_distance / size_factor)

        # Calculate the scaled radii
        radius_x = int(base_eye_radius * radius_factor_x)
        radius_y = int(base_eye_radius * radius_factor_y)

        # Adjust for eye spacing (horizontal movement)
        left_eye[0] += eye_spacing_offset
        right_eye[0] -= eye_spacing_offset

        def extract_and_blend_eye(eye_center, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius):
            ymin = max(0, eye_center[1] - radius_y)
            ymax = min(img_orig.size(1), eye_center[1] + radius_y)
            xmin = max(0, eye_center[0] - radius_x)
            xmax = min(img_orig.size(2), eye_center[0] + radius_x)

            eye_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
            eye_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                           (radius_x, radius_y),
                                           radius_x, radius_y,
                                           feather_radius).to(img_orig.device)

            target_ymin = ymin
            target_ymax = ymin + eye_region_orig.size(1)
            target_xmin = xmin
            target_xmax = xmin + eye_region_orig.size(2)

            img_swap_eye = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
            blended_eye = blend_alpha * img_swap_eye + (1 - blend_alpha) * eye_region_orig

            img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = eye_mask * blended_eye + (1 - eye_mask) * img_swap_eye

        # Process both eyes with updated positions
        extract_and_blend_eye(left_eye, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius)
        extract_and_blend_eye(right_eye, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius)

        return img_swap

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        swapped_face = swapped_face.permute(1,2,0)
        original_face = original_face.permute(1,2,0)

        diff = swapped_face-original_face
        diff = torch.abs(diff)

        # Find the diffrence between the swap and original, per channel
        fthresh = DiffAmount*2.55

        # Bimodal
        diff[diff<fthresh] = 0
        diff[diff>=fthresh] = 1

        # If any of the channels exceeded the threshhold, them add them to the mask
        diff = torch.sum(diff, dim=2)
        diff = torch.unsqueeze(diff, 2)
        diff[diff>0] = 1

        diff = diff.permute(2,0,1)

        return diff