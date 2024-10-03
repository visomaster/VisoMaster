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
            'help': 'Set number of execution threads while playing and recording. Depends strongly on GPU VRAM.',
            'exec_function': control_actions.change_threads_number,
            'exec_function_args': [],
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
}