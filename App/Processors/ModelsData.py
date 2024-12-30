models_dir = './App/ONNXModels'

try:
    from torch.cuda import nvtx
    import tensorrt as trt
    models_trt_list = [
        {'model_name': 'LivePortraitMotionExtractor', 'local_path': f'{models_dir}/liveportrait_onnx/motion_extractor.' + trt.__version__ + '.trt'},
        {'model_name': 'LivePortraitAppearanceFeatureExtractor', 'local_path': f'{models_dir}/liveportrait_onnx/appearance_feature_extractor.' + trt.__version__ + '.trt'},
        {'model_name': 'LivePortraitStitchingEye', 'local_path': f'{models_dir}/liveportrait_onnx/stitching_eye.' + trt.__version__ + '.trt'},
        {'model_name': 'LivePortraitStitchingLip', 'local_path': f'{models_dir}/liveportrait_onnx/stitching_lip.' + trt.__version__ + '.trt'},
        {'model_name': 'LivePortraitStitching', 'local_path': f'{models_dir}/liveportrait_onnx/stitching.' + trt.__version__ + '.trt'},
        {'model_name': 'LivePortraitWarpingSpadeFix', 'local_path': f'{models_dir}/liveportrait_onnx/warping_spade-fix.' + trt.__version__ + '.trt'}
    ]
except ModuleNotFoundError:
    models_trt_list = []

arcface_mapping_model_dict = {
    'Inswapper128': 'Inswapper128ArcFace',
    'FaceStyleInswapper256': 'Inswapper128ArcFace',
    'DeepFaceLive (DFM)': 'Inswapper128ArcFace',
    'SimSwap512': 'SimSwapArcFace',
    'GhostFace-v1': 'GhostArcFace',
    'GhostFace-v2': 'GhostArcFace',
    'GhostFace-v3': 'GhostArcFace',
    'CSCS': 'CSCSArcFace',
}

models_list = [
    {'model_name': 'Inswapper128', 'local_path': f'{models_dir}/inswapper_128.fp16.onnx'},
    {'model_name': 'FaceStyleInswapper256', 'local_path': f'{models_dir}/FaceStyleInswapper_256.fp16.onnx'},
    {'model_name': 'SimSwap512', 'local_path': f'{models_dir}/simswap_512_unoff.onnx'},
    {'model_name': 'GhostFacev1', 'local_path': f'{models_dir}/ghost_unet_1_block.onnx'},
    {'model_name': 'GhostFacev2', 'local_path': f'{models_dir}/ghost_unet_2_block.onnx'},
    {'model_name': 'GhostFacev3', 'local_path': f'{models_dir}/ghost_unet_3_block.onnx'},
    {'model_name': 'CSCS', 'local_path': f'{models_dir}/cscs_256.onnx'},
    {'model_name': 'RetinaFace', 'local_path': f'{models_dir}/det_10g.onnx'},
    {'model_name': 'SCRFD2.5g', 'local_path': f'{models_dir}/scrfd_2.5g_bnkps.onnx'},
    {'model_name': 'YoloFace8n', 'local_path': f'{models_dir}/yoloface_8n.onnx'},
    {'model_name': 'YunetN', 'local_path': f'{models_dir}/yunet_n_640_640.onnx'},
    {'model_name': 'FaceLandmark5', 'local_path': f'{models_dir}/res50.onnx'},
    {'model_name': 'FaceLandmark68', 'local_path': f'{models_dir}/2dfan4.onnx'},
    {'model_name': 'FaceLandmark3d68', 'local_path': f'{models_dir}/1k3d68.onnx'},
    {'model_name': 'FaceLandmark98', 'local_path': f'{models_dir}/peppapig_teacher_Nx3x256x256.onnx'},
    {'model_name': 'FaceLandmark106', 'local_path': f'{models_dir}/2d106det.onnx'},
    {'model_name': 'FaceLandmark203', 'local_path': f'{models_dir}/landmark.onnx'},
    {'model_name': 'FaceLandmark478', 'local_path': f'{models_dir}/face_landmarks_detector_Nx3x256x256.onnx'},
    {'model_name': 'FaceBlendShapes', 'local_path': f'{models_dir}/face_blendshapes_Nx146x2.onnx'},
    {'model_name': 'Inswapper128ArcFace', 'local_path': f'{models_dir}/w600k_r50.onnx'},
    {'model_name': 'SimSwapArcFace', 'local_path': f'{models_dir}/simswap_arcface_model.onnx'},
    {'model_name': 'GhostArcFace', 'local_path': f'{models_dir}/ghost_arcface_backbone.onnx'},
    {'model_name': 'CSCSArcFace', 'local_path': f'{models_dir}/cscs_arcface_model.onnx'},
    {'model_name': 'CSCSIDArcFace', 'local_path': f'{models_dir}/cscs_id_adapter.onnx'},
    {'model_name': 'GFPGANv1.4', 'local_path': f'{models_dir}/GFPGANv1.4.onnx'},
    {'model_name': 'GPENBFR256', 'local_path': f'{models_dir}/GPEN-BFR-256.onnx'},
    {'model_name': 'GPENBFR512', 'local_path': f'{models_dir}/GPEN-BFR-512.onnx'},
    {'model_name': 'GPENBFR1024', 'local_path': f'{models_dir}/GPEN-BFR-1024.onnx'},
    {'model_name': 'GPENBFR2048', 'local_path': f'{models_dir}/GPEN-BFR-2048.onnx'},
    {'model_name': 'CodeFormer', 'local_path': f'{models_dir}/codeformer_fp16.onnx'},
    {'model_name': 'VQFRv2', 'local_path': f'{models_dir}/VQFRv2.fp16.onnx'},
    {'model_name': 'RestoreFormerPlusPlus', 'local_path': f'{models_dir}/RestoreFormerPlusPlus.fp16.onnx'},
    {'model_name': 'RealEsrganx2Plus', 'local_path': f'{models_dir}/RealESRGAN_x2plus.fp16.onnx'},
    {'model_name': 'RealEsrganx4Plus', 'local_path': f'{models_dir}/RealESRGAN_x4plus.fp16.onnx'},
    {'model_name': 'RealEsrx4v3', 'local_path': f'{models_dir}/realesr-general-x4v3.onnx'},
    {'model_name': 'BSRGANx2', 'local_path': f'{models_dir}/BSRGANx2.fp16.onnx'},
    {'model_name': 'BSRGANx4', 'local_path': f'{models_dir}/BSRGANx4.fp16.onnx'},
    {'model_name': 'UltraSharpx4', 'local_path': f'{models_dir}/4x-UltraSharp.fp16.onnx'},
    {'model_name': 'UltraMixx4', 'local_path': f'{models_dir}/4x-UltraMix_Smooth.fp16.onnx'},
    {'model_name': 'DeoldifyArt', 'local_path': f'{models_dir}/ColorizeArtistic.fp16.onnx'},
    {'model_name': 'DeoldifyStable', 'local_path': f'{models_dir}/ColorizeStable.fp16.onnx'},
    {'model_name': 'DeoldifyVideo', 'local_path': f'{models_dir}/ColorizeVideo.fp16.onnx'},
    {'model_name': 'DDColorArt', 'local_path': f'{models_dir}/ddcolor_artistic.onnx'},
    {'model_name': 'DDcolor', 'local_path': f'{models_dir}/ddcolor.onnx'},
    {'model_name': 'Occluder', 'local_path': f'{models_dir}/occluder.onnx'},
    {'model_name': 'XSeg', 'local_path': f'{models_dir}/XSeg_model.onnx'},
    {'model_name': 'FaceParser', 'local_path': f'{models_dir}/faceparser_resnet34.onnx'},
    {'model_name': 'LivePortraitMotionExtractor', 'local_path': f'{models_dir}/liveportrait_onnx/motion_extractor.onnx'},
    {'model_name': 'LivePortraitAppearanceFeatureExtractor', 'local_path': f'{models_dir}/liveportrait_onnx/appearance_feature_extractor.onnx'},
    {'model_name': 'LivePortraitStitchingEye', 'local_path': f'{models_dir}/liveportrait_onnx/stitching_eye.onnx'},
    {'model_name': 'LivePortraitStitchingLip', 'local_path': f'{models_dir}/liveportrait_onnx/stitching_lip.onnx'},
    {'model_name': 'LivePortraitStitching', 'local_path': f'{models_dir}/liveportrait_onnx/stitching.onnx'},
    {'model_name': 'LivePortraitWarpingSpade', 'local_path': f'{models_dir}/liveportrait_onnx/warping_spade.onnx'},
    {'model_name': 'LivePortraitWarpingSpadeFix', 'local_path': f'{models_dir}/liveportrait_onnx/warping_spade-fix.onnx'}
]
