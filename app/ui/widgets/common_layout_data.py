from app.helpers.types import LayoutDictTypes
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
            'default': 'Reference',
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
            'decimals': 1,
            'step': 0.1,
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
            'step': 1,
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': 'Control the blend ratio between the restored face and the swapped face.'
        },
        'FaceRestorerEnable2Toggle': {
            'level': 1,
            'label': 'Enable Face Restorer 2',
            'default': False,
            'help': 'Enable the use of a face restoration model to improve the quality of the face after swapping.'
        },
        'FaceRestorerType2Selection': {
            'level': 2,
            'label': 'Restorer Type',
            'options': ['GFPGAN-v1.4', 'CodeFormer', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048', 'RestoreFormer++', 'VQFR-v2'],
            'default': 'GFPGAN-v1.4',
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': 'Select the model type for face restoration.'
        },
        'FaceRestorerDetType2Selection': {
            'level': 2,
            'label': 'Alignment',
            'options': ['Original', 'Blend', 'Reference'],
            'default': 'Original',
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': 'Select the alignment method for restoring the face to its original or blended position.'
        },
        'FaceFidelityWeight2DecimalSlider': {
            'level': 2,
            'label': 'Fidelity Weight',
            'min_value': '0.0',
            'max_value': '1.0',
            'default': '0.9',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': 'Adjust the fidelity weight to control how closely the restoration preserves the original face details.'
        },
        'FaceRestorerBlend2Slider': {
            'level': 2,
            'label': 'Blend',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': 'Control the blend ratio between the restored face and the swapped face.'
        },
        'FaceExpressionEnableToggle': {
            'level': 1,
            'label': 'Enable Face Expression Restorer',
            'default': False,
            'help': 'Enabled the use of the LivePortrait face expression model to restore facial expressions after swapping.'
        },
        'FaceExpressionCropScaleDecimalSlider': {
            'level': 2,
            'label': 'Crop Scale',
            'min_value': '1.80',
            'max_value': '3.00',
            'default': '2.30',
            'step': 0.05,
            'decimals': 2,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes swap crop scale. Increase the value to capture the face more distantly.'
        },
        'FaceExpressionVYRatioDecimalSlider': {
            'level': 2,
            'label': 'VY Ratio',
            'min_value': '-0.125',
            'max_value': '-0.100',
            'default': '-0.125',
            'step': 0.001,
            'decimals': 3,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the vy ratio for crop scale. Increase the value to capture the face more distantly.'
        },
        'FaceExpressionFriendlyFactorDecimalSlider': {
            'level': 2,
            'label': 'Expression Friendly Factor',
            'min_value': '0.0',
            'max_value': '1.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Control the expression similarity between the driving face and the swapped face.'
        },
        'FaceExpressionAnimationRegionSelection': {
            'level': 2,
            'label': 'Animation Region',
            'options': ['all', 'eyes', 'lips'],
            'default': 'all',
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'The facial region involved in the restoration process.'
        },
        'FaceExpressionNormalizeLipsEnableToggle': {
            'level': 2,
            'label': 'Normalize Lips',
            'default': True,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Normalize the lips during the facial restoration process.'
        },
        'FaceExpressionNormalizeLipsThresholdDecimalSlider': {
            'level': 3,
            'label': 'Normalize Lips Threshold',
            'min_value': '0.00',
            'max_value': '1.00',
            'default': '0.03',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceExpressionNormalizeLipsEnableToggle|FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Threshold value for Normalize Lips.'
        },
        'FaceExpressionRetargetingEyesEnableToggle': {
            'level': 2,
            'label': 'Retargeting Eyes',
            'default': False,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjusting or redirecting the gaze or movement of the eyes during the facial restoration process. It overrides the Animation Region settings, meaning that the Animation Region will be ignored.'
        },
        'FaceExpressionRetargetingEyesMultiplierDecimalSlider': {
            'level': 3,
            'label': 'Retargeting Eyes Multiplier',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceExpressionRetargetingEyesEnableToggle|FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Multiplier value for Retargeting Eyes.'
        },
        'FaceExpressionRetargetingLipsEnableToggle': {
            'level': 2,
            'label': 'Retargeting Lips',
            'default': False,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjusting or modifying the position, shape, or movement of the lips during the facial restoration process. It overrides the Animation Region settings, meaning that the Animation Region will be ignored.'
        },
        'FaceExpressionRetargetingLipsMultiplierDecimalSlider': {
            'level': 3,
            'label': 'Retargeting Lips Multiplier',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceExpressionRetargetingLipsEnableToggle|FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': 'Multiplier value for Retargeting Lips.'
        },
    },
}