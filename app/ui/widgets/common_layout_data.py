from app.helpers.typing import LayoutDictTypes
import app.ui.widgets.actions.layout_actions as layout_actions

COMMON_LAYOUT_DATA: LayoutDictTypes = {
    'Face Compare':{
        'ViewFaceMaskEnableToggle':{
            'level': 1,
            'label': 'View Face Mask',
            'default': False,
            'help': 'Show Face Mask',
            'exec_function': layout_actions.fit_image_to_view_onchange,
            'exec_function_args': [],
        },
        'ViewFaceCompareEnableToggle':{
            'level': 1,
            'label': 'View Face Compare',
            'default': False,
            'help': 'Show Face Compare',
            'exec_function': layout_actions.fit_image_to_view_onchange,
            'exec_function_args': [],
        },
    },
}