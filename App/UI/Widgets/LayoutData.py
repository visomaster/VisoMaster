from PySide6.QtWidgets import QComboBox
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.Widgets.WidgetComponents import ToggleButton

# Widgets in Face Swap tab are created from this Layout
SWAPPER_LAYOUT_DATA = {
    'Swapper': {
        'SwapModelSelection': {
            'level': 1,
            'label': 'Swapper Model',
            'options': ['Inswapper128', 'SimSwap256', 'SimSwap512'],
            'default': 'Inswapper128',
            'help': 'Choose which swapper model to use for face swapping.'
        },
        'SwapperResSelection': {
            'level': 2,
            'label': 'Swapper Resolution',
            'options': ['128', '256', '384', '512'],
            'default': '128',
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Inswapper128',
            'help': 'Select the resolution for the swapped face in pixels. Higher values offer better quality but are slower to process.'
        },
    },
    'Detectors': {
        'DetectorModelSelection': {
            'level': 1,
            'label': 'Face Detect Model',
            'options': ['Retinaface', 'Yolov8', 'SCRFD', 'Yunet'],
            'default': 'Retinaface',
            'help': 'Select the face detection model to use for detecting faces in the input image or video.'
        },
        'DetectorScoreSlider': {
            'level': 1,
            'label': 'Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'help': 'Set the confidence score threshold for face detection. Higher values ensure more confident detections but may miss some faces.'
        },
        'AutoRotationToggle': {
            'level': 1,
            'label': 'Auto Rotation',
            'default': False,
            'help': 'Automatically rotate the input to detect faces in various orientations.'
        },
        'SimilarityThresholdSlider': {
            'level': 1,
            'label': 'Similarity Threshold',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
            'help': 'Set the similarity threshold to control how similar the detected face should be to the reference face.'
        },
        'SimilarityTypeSelection': {
            'level': 1,
            'label': 'Similarity Type',
            'options': ['Opal', 'Pearl', 'Optimal'],
            'default': 'Opal',
            'help': 'Choose the type of similarity calculation for face detection and matching.'
        },
        'LandmarkDetectToggle': {
            'level': 1,
            'label': 'Enable Landmark Detection',
            'default': False,
            'help': 'Enable or disable facial landmark detection, which is used to refine face alignment.'
        },
        'LandmarkDetectModelSelection': {
            'level': 2,
            'label': 'Landmark Detect Model',
            'options': ['5', '68', '3d68', '98', '106', '203', '478'],
            'default': '203',
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Select the landmark detection model, where different models detect varying numbers of facial landmarks.'
        },
        'LandmarkDetectScoreSlider': {
            'level': 2,
            'label': 'Landmark Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Set the confidence score threshold for facial landmark detection.'
        },
        'DetectFromPointsToggle': {
            'level': 2,
            'label': 'Detect From Points',
            'default': False,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': 'Enable detection of faces from specified landmark points.'
        },
    },
    'Webcam': {
        'WebcamEnableToggle': {
            'level': 1,
            'label': 'Enable Webcam',
            'default': False,
            'help': 'Enable the use of the webcam as the input source for face swapping.'
        },
        'WebcamMaxNoSelection': {
            'level': 2,
            'label': 'Webcam Max No',
            'options': ['1', '2', '3', '4', '5', '6'],
            'default': '1',
            'parentToggle': 'WebcamEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the maximum number of webcam streams to allow for face swapping.'
        },
        'WebcamBackendSelection': {
            'level': 2,
            'label': 'Webcam Backend',
            'options': ['Default', 'Directshow', 'MSMF', 'V4L', 'V4L2', 'GSTREAMER'],
            'default': 'Default',
            'parentToggle': 'WebcamEnableToggle',
            'requiredToggleValue': True,
            'help': 'Choose the backend for accessing webcam input.'
        },
        'WebcamMaxResSelection': {
            'level': 2,
            'label': 'Webcam Resolution',
            'options': ['480x360', '640x480', '1280x720', '1920x1080', '2560x1440', '3840x2160'],
            'default': '1280x720',
            'parentToggle': 'WebcamEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the maximum resolution for webcam input.'
        },
        'WebCamMaxFPSSelection': {
            'level': 2,
            'label': 'Webcam FPS',
            'options': ['23', '30', '60'],
            'default': '30',
            'parentToggle': 'WebcamEnableToggle',
            'requiredToggleValue': True,
            'help': 'Set the maximum frames per second (FPS) for webcam input.'
        },
    },
    'Face Restorer': {
        'FaceRestorerEnableToggle': {
            'level': 1,
            'label': 'Enable Face Restorer',
            'default': False,
            'help': 'Enable the use of a face restoration model to improve the quality of the face after swapping.'
        },
        'FaceRestorerTypeSelection': {
            'level': 2,
            'label': 'Restorer Type',
            'options': ['GFPGAN-v1.4', 'CodeFormer', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048', 'RestoreFormer++', 'VQFR-v2'],
            'default': 'GFPGAN-v1.4',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the model type for face restoration.'
        },
        'FaceRestorerDetTypeSelection': {
            'level': 2,
            'label': 'Alignment',
            'options': ['Original', 'Blend', 'Reference'],
            'default': 'Blend',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the alignment method for restoring the face to its original or blended position.'
        },
        'FaceFidelityWeightDecimalSlider': {
            'level': 2,
            'label': 'Fidelity Weight',
            'min_value': '0.0',
            'max_value': '1.0',
            'default': '0.9',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the fidelity weight to control how closely the restoration preserves the original face details.'
        },
        'FaceRestorerBlendSlider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Control the blend ratio between the restored face and the swapped face.'
        },
    },
    'Frame Enhancer':{
        'FrameEnhancerEnableToggle':{
            'level': 1,
            'label': 'Enable Frame Enhancer',
            'default': False,
            'help': 'Enable frame enhancement for video inputs to improve visual quality.'
        },
        'FrameEnhancerTypeSelection':{
            'level': 2,
            'label': 'Frame Enhancer Type',
            'options': ['RealEsrgan-x2-Plus', 'RealEsrgan-x4-Plus', 'RealEsr-General-x4v3', 'BSRGan-x2', 'BSRGan-x4', 'UltraSharp-x4', 'UltraMix-x4', 'DDColor-Artistic', 'DDColor', 'DeOldify-Artistic', 'DeOldify-Stable', 'DeOldify-Video'],
            'default': 'RealEsrgan-x2-Plus',
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the type of frame enhancement to apply, based on the content and resolution requirements.'
        },
        'FrameEnhancerBlendSlider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blends the enhanced results back into the original frame.'
        },
    },
    'Face Mask':{
        'BorderBottomSlider':{
            'level': 1,
            'label': 'Bottom Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderLeftSlider':{
            'level': 1,
            'label': 'Left Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderRightSlider':{
            'level': 1,
            'label': 'Right Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderTopSlider':{
            'level': 1,
            'label': 'Top Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderBlurSlider':{
            'level': 1,
            'label': 'Border Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'help': 'Border mask blending distance.'
        },
        'OccluderEnableToggle': {
            'level': 1,
            'label': 'Occlusion Mask',
            'default': False,
            'help': 'Allow objects occluding the face to show up in the swapped image.'
        },
        'OccluderSizeSlider': {
            'level': 2,
            'label': 'Size',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'parentToggle': 'OccluderEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region'
        },
        'DFLXSegEnableToggle': {
            'level': 1,
            'label': 'DFL XSeg Mask',
            'default': False,
            'help': 'Allow objects occluding the face to show up in the swapped image.'
        },
        'DFLXSegSizeSlider': {
            'level': 2,
            'label': 'Size',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'parentToggle': 'DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region'
        },
        'OccluderXSegBlurSlider': {
            'level': 1,
            'label': 'Occluder/DFL XSeg Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'parentToggle': 'OccluderEnableToggle, DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend value for Occluder and XSeg.'
        },
        
    },
    'Embedding Merge Method':{
        'EmbMergeMethodSelection':{
            'level': 1,
            'label': 'Embedding Merge Method',
            'options': ['Mean','Median'],
            'default': 'Mean',
            'help': 'Select the method to merge facial embeddings. "Mean" averages the embeddings, while "Median" selects the middle value, providing more robustness to outliers.'
        }
    }
}

SETTINGS_LAYOUT_DATA = {
    'General': {
        'ProvidersPrioritySelection': {
            'level': 1,
            'label': 'Providers Priority',
            'options': ['CUDA', 'TensorRT', 'TensorRT-Engine', 'CPU'],
            'default': 'CUDA',
            'help': 'Select the providers priority to be used with the system.'
        },
        'nThreadsSlider': {
            'level': 1,
            'label': 'Number of Threads',
            'min_value': '1',
            'max_value': '30',
            'default': '5',
            'help': 'Set number of execution threads while playing and recording. Depends strongly on GPU VRAM. 5 threads for 12GB.'
        },
    },
}
