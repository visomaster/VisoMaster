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
        },
        'SwapperResSelection': {
            'level': 2,
            'label': 'Swapper Resolution',
            'options': ['128', '256', '512'],
            'default': '128',
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Inswapper128',
        },
    },
    'Detectors': {
        'DetectorModelSelection': {
            'level': 1,
            'label': 'Face Detect Model',
            'options': ['Retinaface', 'Yolov8', 'SCRFD'],
            'default': 'Retinaface',
        },
        'DetectorScoreSlider': {
            'level': 1,
            'label': 'Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
        },
        'SimilarityThresholdSlider': {
            'level': 1,
            'label': 'Similarity Threshold',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
        },
        'SimilariyTypeSelection': {
            'level': 1,
            'label': 'Similarity Type',
            'options': ['Opal', 'Pearl', 'Optimal'],
            'default': 'Opal',
        },
        'LandmarkDetectToggle': {
            'level': 1,
            'label': 'Enable Landmark Detection',
            'default': False,
        },
        'LandmarkDetectModelSelection': {
            'level': 2,
            'label': 'Landmark Detect Model',
            'options': ['5', '68', '3d68', '98', '106', '203', '478'],
            'default': '203',
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
        },
        'LandmarkDetectScoreSlider': {
            'level': 2,
            'label': 'Landmark Detect Score',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
        },
    },
    'Webcam': {
        'WebcamEnableToggle': {
            'level': 1,
            'label': 'Enable Webcam',
            'default': False,
        },
        'WebcamMaxNoSelection': {
            'level': 2,
            'label': 'Webcam Max No',
            'options': ['1', '2', '3', '4', '5', '6'],
            'default': '1',
            'parentToggle': 'WebcamEnableToggle',
            'requiredToggleValue': True,
        },
        'WebcamBackendSelection': {
            'level': 2,
            'label': 'Webcam Backend',
            'options': ['Default', 'Directshow', 'MSMF', 'V4L', 'V4L2', 'GSTREAMER'],
            'default': 'Default',
            'parentToggle': 'WebcamEnableToggle',
            'requiredToggleValue': True,
        },
        'WebcamMaxResSelection': {
            'level': 2,
            'label': 'Webcam Resolution',
            'options': ['480x360', '640x480', '1280x720', '1920x1080', '2560x1440', '3840x2160'],
            'default': '1280x720',
            'parentToggle': 'WebcamEnableToggle',
            'requiredToggleValue': True,
        },
        'WebCamMaxFPSSelection': {
            'level': 2,
            'label': 'Webcam FPS',
            'options': ['23', '30', '60'],
            'default': '30',
            'parentToggle': 'WebcamEnableToggle',
            'requiredToggleValue': True,
        },
    },
    'Face Restorer': {
        'FaceRestorerEnableToggle': {
            'level': 1,
            'label': 'Enable Face Restorer',
            'default': False,
        },
        'FaceRestorerTypeSelection': {
            'level': 2,
            'label': 'Restorer Type',
            'options': ['GFPGAN-v1.4', 'CodeFormer', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048', 'RestoreFormer++', 'VQFR-v2'],
            'default': 'GFPGAN-v1.4',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
        },
        'FaceRestorerDetTypeSelection': {
            'level': 2,
            'label': 'Alignment',
            'options': ['Original', 'Blend', 'Reference'],
            'default': 'Blend',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
        },
    },
    'Frame Enhancer':{
        'FrameEnhancerEnableToggle':{
            'level': 1,
            'label': 'Enable Frame Enhancer',
            'default': False,
        },
        'FrameEnhancerTypeSelection':{
            'level': 2,
            'label': 'Frame Enhancer Type',
            'options': ['RealEsrgan-x2-Plus', 'RealEsrgan-x4-Plus', 'RealEsr-General-x4v3', 'BSRGan-x2', 'BSRGan-x4', 'UltraSharp-x4', 'UltraMix-x4', 'DDColor-Artistic', 'DDColor', 'DeOldify-Artistic', 'DeOldify-Stable', 'DeOldify-Video'],
            'default': 'RealEsrgan-x2-Plus',
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
        }
    },
    'Embedding Merge Method':{
        'EmbMergeMethodSelection':{
            'level': 1,
            'label': 'Embedding Merge Method',
            'options': ['Mean','Median'],
            'default': 'Mean',
        }
    }
}