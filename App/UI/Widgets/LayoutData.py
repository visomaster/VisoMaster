from PySide6.QtWidgets import QComboBox
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.Widgets.WidgetComponents import ToggleButton

SWAPPER_LAYOUT_DATA = {
    'Swapper':{
        'SwapModelSelection': { 'level': 1, 'label': 'Swapper Model', 'options': ['Inswapper128', 'SimSwap256', 'SimSwap512'], 'default': 'Inswapper128', },
            'SwapperResSelection': { 'level': 1, 'label': 'Swapper Resolution', 'options': ['128','256','512'], 'default':'128', 'parentSelection': 'SwapModelSelection', 'requiredSelectionValue': 'Inswapper128'},        
    },
    'Detectors': {
        'DetectorModelSelection': { 'level': 1, 'label': 'Face Detect Model', 'options': ['Retinaface', 'YoloV8', 'SCRFD'], 'default': 'Retinaface',},
        'LandmarkDetectToggle': {'level': 1, 'label': 'Enable Landmark Detection', 'default': False, },
            'LandmarkDetectModelSelection': {'level': 2, 'label': 'Landmark Detect Model', 'options': ['68', '3d68', '202'],  'default': '68', 'parentToggle': 'LandmarkDetectToggle', 'requiredToggleValue': True}
    }

}