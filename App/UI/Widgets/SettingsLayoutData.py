import App.UI.Widgets.ControlActions as control_actions
SETTINGS_LAYOUT_DATA = {
    'General': {
        'ProvidersPrioritySelection': {
            'level': 1,
            'label': 'Providers Priority',
            'options': ['CUDA', 'TensorRT', 'TensorRT-Engine', 'CPU'],
            'default': 'CUDA',
            'help': 'Select the providers priority to be used with the system.',
            'exec_function': control_actions.change_execution_provider,
            'exec_function_args': [],
        },
        'nThreadsSlider': {
            'level': 1,
            'label': 'Number of Threads',
            'min_value': '1',
            'max_value': '30',
            'default': '5',
            'step': 1,
            'help': 'Set number of execution threads while playing and recording. Depends strongly on GPU VRAM.',
            'exec_function': control_actions.change_threads_number,
            'exec_function_args': [],
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
            'step': 1,
            'help': 'Set the confidence score threshold for face detection. Higher values ensure more confident detections but may miss some faces.'
        },
        'MaxFacesToDetectSlider': {
            'level': 1,
            'label': 'Max No of Faces to Detect',
            'min_value': '1',
            'max_value': '50',
            'default': '20',
            'step': 1,     
            'help': 'Set the maximum number of faces to detect in a frame'
   
        },
        'AutoRotationToggle': {
            'level': 1,
            'label': 'Auto Rotation',
            'default': False,
            'help': 'Automatically rotate the input to detect faces in various orientations.'
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
            'step': 1,
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

    'Webcam Settings': {
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