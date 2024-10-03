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