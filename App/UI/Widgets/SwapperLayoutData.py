import App.Helpers.Miscellaneous as misc_helpers
from App.Helpers.Typing import LayoutDictTypes

# Widgets in Face Swap tab are created from this Layout
SWAPPER_LAYOUT_DATA: LayoutDictTypes = {
    'Swapper': {
        'SwapModelSelection': {
            'level': 1,
            'label': 'Swapper Model',
            'options': ['Inswapper128', 'ReSwapperEx128', 'DeepFaceLive (DFM)', 'SimSwap512', 'GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'],
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
        'DFMModelSelection': {
            'level': 2,
            'label': 'DFM Model',
            'options': misc_helpers.get_dfm_models_selection_values,
            'default': misc_helpers.get_dfm_models_default_value,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'Select which pretrained DeepFaceLive (DFM) Model to use for swapping.'
        },
        'DFMAmpMorphSlider': {
            'level': 2,
            'label': 'AMP Morph Factor',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'AMP Morph Factor for DFM AMP Models',
        },
        'DFMRCTColorToggle': {
            'level': 2,
            'label': 'RCT Color Transfer',
            'default': False,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'RCT Color Transfer for DFM Models',
        }
    },
    'Face Landmarks Correction': {
        'FaceAdjEnableToggle': {
            'level': 1,
            'label': 'Face Adjustments',
            'default': False,
            'help': 'This is an experimental feature to perform direct adjustments to the face landmarks found by the detector. There is also an option to adjust the scale of the swapped face.'
        },
        'KpsXSlider': {
            'level': 2,
            'label': 'Keypoints X-Axis',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the detection points left and right.'
        },
        'KpsYSlider': {
            'level': 2,
            'label': 'Keypoints Y-Axis',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the detection points up and down.'
        },
        'KpsScaleSlider': {
            'level': 2,
            'label': 'Keypoints Scale',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows and shrinks the detection point distances.'
        },
        'FaceScaleAmountSlider': {
            'level': 2,
            'label': 'Face Scale Amount',
            'min_value': '-20',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows and shrinks the entire face.'
        },
        'LandmarksPositionAdjEnableToggle': {
            'level': 1,
            'label': '5 - Keypoints Adjustments',
            'default': False,
            'help': 'This is an experimental feature to perform direct adjustments to the position of face landmarks found by the detector.'
        },
        'EyeLeftXAmountSlider': {
            'level': 2,
            'label': 'Left Eye:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye left detection point left and right.'
        },
        'EyeLeftYAmountSlider': {
            'level': 2,
            'label': 'Left Eye:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye left detection point up and down.'
        },
        'EyeRightXAmountSlider': {
            'level': 2,
            'label': 'Right Eye:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye right detection point left and right.'
        },
        'EyeRightYAmountSlider': {
            'level': 2,
            'label': 'Right Eye:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the eye right detection point up and down.'
        },
        'NoseXAmountSlider': {
            'level': 2,
            'label': 'Nose:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the nose detection point left and right.'
        },
        'NoseYAmountSlider': {
            'level': 2,
            'label': 'Nose:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the nose detection point up and down.'
        },
        'MouthLeftXAmountSlider': {
            'level': 2,
            'label': 'Left Mouth:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth left detection point left and right.'
        },
        'MouthLeftYAmountSlider': {
            'level': 2,
            'label': 'Left Mouth:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth left detection point up and down.'
        },
        'MouthRightXAmountSlider': {
            'level': 2,
            'label': 'Right Mouth:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth Right detection point left and right.'
        },
        'MouthRightYAmountSlider': {
            'level': 2,
            'label': 'Right Mouth:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Shifts the mouth Right detection point up and down.'
        },
    },
    'Face Similarity': {
        'SimilarityThresholdSlider': {
            'level': 1,
            'label': 'Similarity Threshold',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
            'step': 1,
            'help': 'Set the similarity threshold to control how similar the detected face should be to the reference (target) face.'
        },
        'StrengthEnableToggle': {
            'level': 1,
            'label': 'Strength',
            'default': False,
            'help': 'Apply additional swapping iterations to increase the strength of the result, which may increase likeness.'
        },
        'StrengthAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '500',
            'default': '100',
            'step': 25,
            'parentToggle': 'StrengthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase up to 5x additional swaps (500%). 200% is generally a good result. Set to 0 to turn off swapping but allow the rest of the pipeline to apply to the original image.'
        },
        'FaceLikenessEnableToggle': {
            'level': 1,
            'label': 'Face Likeness',
            'default': False,
            'help': 'This is a feature to perform direct adjustments to likeness of faces.'
        },
        'FaceLikenessFactorDecimalSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '-1.00',
            'max_value': '1.00',
            'default': '0.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'FaceLikenessEnableToggle',
            'requiredToggleValue': True,
            'help': 'Determines the factor of likeness between the source and assigned faces.'
        },
        'DifferencingEnableToggle': {
            'level': 1,
            'label': 'Differencing',
            'default': False,
            'help': 'Allow some of the original face to show in the swapped result when the difference between the two images is small. Can help bring back some texture to the swapped face.'
        },
        'DifferencingAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '4',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Higher values relaxes the similarity constraint.'
        },
        'DifferencingBlendAmountSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend differecing value.'
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
    'Face Mask':{
        'BorderBottomSlider':{
            'level': 1,
            'label': 'Bottom Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderLeftSlider':{
            'level': 1,
            'label': 'Left Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderRightSlider':{
            'level': 1,
            'label': 'Right Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderTopSlider':{
            'level': 1,
            'label': 'Top Border',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': 'A rectangle with adjustable bottom, left, right, top, and sides that masks the swapped face result back into the original image.'
        },
        'BorderBlurSlider':{
            'level': 1,
            'label': 'Border Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
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
            'step': 1,
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
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Grows or shrinks the occluded region.'
        },
        'OccluderXSegBlurSlider': {
            'level': 1,
            'label': 'Occluder/DFL XSeg Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle, DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend value for Occluder and XSeg.'
        },
        'ClipEnableToggle': {
            'level': 1,
            'label': 'Text Masking',
            'default': False,
            'help': 'Use descriptions to identify objects that will be present in the final swapped image.'
        },
        'ClipText': {
            'level': 2,
            'label': 'Text Masking Entry',
            'min_value': '0',
            'max_value': '1000',
            'default': '',
            'width': 130,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': 'To use, type a word(s) in the box separated by commas and press <enter>.'
        },
        'ClipAmountSlider': {
            'level': 2,
            'label': 'Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase to strengthen the effect.'
        },
        'FaceParserEnableToggle': {
            'level': 1,
            'label': 'Face Parser Mask',
            'default': False,
            'help': 'Allow the unprocessed background from the orginal image to show in the final swap.'
        },
        'BackgroundParserSlider': {
            'level': 2,
            'label': 'Background',
            'min_value': '-50',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Negative/Positive values shrink and grow the mask.'
        },
        'FaceParserSlider': {
            'level': 2,
            'label': 'Face',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the entire face.'
        },
        'LeftEyebrowParserSlider': {
            'level': 2,
            'label': 'Left Eyebrow',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the left eyebrow.'
        },
        'RightEyebrowParserSlider': {
            'level': 2,
            'label': 'Right Eyebrow',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the right eyebrow.'
        },
        'LeftEyeParserSlider': {
            'level': 2,
            'label': 'Left Eye',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the left eye.'
        },
        'RightEyeParserSlider': {
            'level': 2,
            'label': 'Right Eye',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the right eye.'
        },
        'EyeGlassesParserSlider': {
            'level': 2,
            'label': 'EyeGlasses',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the eyeglasses.'
        },
        'NoseParserSlider': {
            'level': 2,
            'label': 'Nose',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the nose.'
        },
        'MouthParserSlider': {
            'level': 2,
            'label': 'Mouth',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the inside of the mouth, including the tongue.'
        },
        'UpperLipParserSlider': {
            'level': 2,
            'label': 'Upper Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the upper lip.'
        },
        'LowerLipParserSlider': {
            'level': 2,
            'label': 'Lower Lip',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the lower lip.'
        },
        'NeckParserSlider': {
            'level': 2,
            'label': 'Neck',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the neck.'
        },
        'HairParserSlider': {
            'level': 2,
            'label': 'Hair',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the size of the Mask. Mast the hair.'
        },
        'BackgroundBlurParserSlider': {
            'level': 2,
            'label': 'Background Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Background Parser'
        },
        'FaceBlurParserSlider': {
            'level': 2,
            'label': 'Face Blur',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value for Face Parser'
        },
        'FaceParserHairMakeupEnableToggle': {
            'level': 2,
            'label': 'Hair Makeup',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Enable hair makeup'
        },
        'FaceParserHairMakeupRedSlider': {
            'level': 3,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'FaceParserHairMakeupGreenSlider': {
            'level': 3,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'FaceParserHairMakeupBlueSlider': {
            'level': 3,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'FaceParserHairMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': 'Blend Amount',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.0 represents the original color, 1.0 represents the full target color.'
        },
        'FaceParserLipsMakeupEnableToggle': {
            'level': 2,
            'label': 'Lips Makeup',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': 'Enable lips makeup'
        },
        'FaceParserLipsMakeupRedSlider': {
            'level': 3,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'FaceParserLipsMakeupGreenSlider': {
            'level': 3,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'FaceParserLipsMakeupBlueSlider': {
            'level': 3,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'FaceParserLipsMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': 'Blend Amount',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.0 represents the original color, 1.0 represents the full target color.'
        },
        'RestoreEyesEnableToggle': {
            'level': 1,
            'label': 'Restore Eyes',
            'default': False,
            'help': 'Restore eyes from the original face.'
        },
        'RestoreEyesBlendAmountSlider': {
            'level': 2,
            'label': 'Eyes Blend Amount',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this to show more of the swapped eyes. Decrease it to show more of the original eyes.'
        },
        'RestoreEyesSizeFactorDecimalSlider': {
            'level': 2,
            'label': 'Eyes Size Factor',
            'min_value': '2.0',
            'max_value': '4.0',
            'default': '3.0',
            'decimals': 1,
            'step': 0.5,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Reduce this when swapping faces zoomed out of the frame.'
        },
        'RestoreEyesFeatherBlendSlider': {
            'level': 2,
            'label': 'Eyes Feather Blend',
            'min_value': '1',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blending of eyes border. Increase this to show more of the original eyes. Decrease this to show more of the swapped eyes.'
        },
        'RestoreXEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X Eyes Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreYEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y Eyes Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreXEyesOffsetSlider': {
            'level': 2,
            'label': 'X Eyes Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Eyes Mask on the X Axis.'
        },
        'RestoreYEyesOffsetSlider': {
            'level': 2,
            'label': 'Y Eyes Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Eyes Mask on the Y Axis.'
        },
        'RestoreEyesSpacingOffsetSlider': {
            'level': 2,
            'label': 'Eyes Spacing Offset',
            'min_value': '-200',
            'max_value': '200',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': 'Change the Eyes Spacing distance.'
        },
        'RestoreMouthEnableToggle': {
            'level': 1,
            'label': 'Restore Mouth',
            'default': False,
            'help': 'Restore mouth from the original face.'
        },
        'RestoreMouthBlendAmountSlider': {
            'level': 2,
            'label': 'Mouth Blend Amount',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this to show more of the swapped Mouth. Decrease it to show more of the original Mouth.'
        },       
        'RestoreMouthSizeFactorSlider': {
            'level': 2,
            'label': 'Mouth Size Factor',
            'min_value': '5',
            'max_value': '60',
            'default': '25',
            'step': 5,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Increase this when swapping faces zoomed out of the frame.'
        },
        'RestoreMouthFeatherBlendSlider': {
            'level': 2,
            'label': 'Mouth Feather Blend',
            'min_value': '1',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the border of Mouth blending. Increase this to show more of the original Mouth. Decrease this to show more of the swapped Mouth.'
        },
        'RestoreXMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X Mouth Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreYMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y Mouth Radius Factor',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'These parameters determine the shape of the mask. If both are equal to 1.0, the mask will be circular. If either one is greater or less than 1.0, the mask will become oval, stretching or shrinking along the corresponding direction.'
        },
        'RestoreXMouthOffsetSlider': {
            'level': 2,
            'label': 'X Mouth Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Mouth Mask on the X Axis.'
        },
        'RestoreYMouthOffsetSlider': {
            'level': 2,
            'label': 'Y Mouth Offset',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Moves the Mouth Mask on the Y Axis.'
        },
        'RestoreEyesMouthBlurSlider': {
            'level': 1,
            'label': 'Eyes/Mouth Blur',
            'min_value': '0',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle, RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blur of mask border.'
        },
    },

    'Face Compare':{
        'ViewFaceMaskEnableToggle':{
            'level': 1,
            'label': 'View Face Mask',
            'default': False,
            'help': 'Show Face Mask'
        },
        'ViewFaceCompareEnableToggle':{
            'level': 1,
            'label': 'View Face Compare',
            'default': False,
            'help': 'Show Face Compare'
        },
    },
    
    'Face Color Correction':{
        'AutoColorEnableToggle': {
            'level': 1,
            'label': 'AutoColor Transfer',
            'default': False,
            'help': 'Enable AutoColor Transfer: 1. Hans Test without mask, 2. Hans Test with mask, 3. DFL Method without mask, 4. DFL Original Method.'
        },
        'AutoColorTransferTypeSelection':{
            'level': 2,
            'label': 'Transfer Type',
            'options': ['Test', 'Test_Mask', 'DFL_Test', 'DFL_Orig'],
            'default': 'Test',
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the AutoColor transfer method type. Hans Method could have some artefacts sometimes.'
        },
        'AutoColorBlendAmountSlider': {
            'level': 1,
            'label': 'Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '80',
            'step': 5,
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the blend value.'
        },
        'ColorEnableToggle': {
            'level': 1,
            'label': 'Color Adjustments',
            'default': False,
            'help': 'Fine-tune the RGB color values of the swap.'
        },
        'ColorRedSlider': {
            'level': 1,
            'label': 'Red',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments'
        },
        'ColorGreenSlider': {
            'level': 1,
            'label': 'Green',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments'
        },
        'ColorBlueSlider': {
            'level': 1,
            'label': 'Blue',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments'
        },
        'ColorBrightnessDecimalSlider': {
            'level': 1,
            'label': 'Brightness',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Brightness.'
        },
        'ColorContrastDecimalSlider': {
            'level': 1,
            'label': 'Contrast',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Contrast.'
        },
        'ColorSaturationDecimalSlider': {
            'level': 1,
            'label': 'Saturation',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Saturation.'
        },
        'ColorSharpnessDecimalSlider': {
            'level': 1,
            'label': 'Sharpness',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Sharpness.'
        },
        'ColorHueDecimalSlider': {
            'level': 1,
            'label': 'Hue',
            'min_value': '-0.50',
            'max_value': '0.50',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Hue.'
        },
        'ColorGammaDecimalSlider': {
            'level': 1,
            'label': 'Gamma',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the Gamma.'
        },
        'ColorNoiseDecimalSlider': {
            'level': 1,
            'label': 'Noise',
            'min_value': '0.0',
            'max_value': '20.0',
            'default': '0.0',
            'step': 0.5,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Add noise to swapped face.'
        },
    },
    'Blend Adjustments':{
        'FinalBlendAdjEnableToggle': {
            'level': 1,
            'label': 'Final Blend',
            'default': False,
            'help': 'Blend at the end of pipeline.'
        },
        'FinalBlendAmountSlider': {
            'level': 2,
            'label': 'Final Blend Amount',
            'min_value': '1',
            'max_value': '50',
            'default': '1',
            'step': 1,
            'parentToggle': 'FinalBlendAdjEnableToggle',
            'requiredToggleValue': True,
            'help': 'Adjust the final blend value.'
        },
        'OverallMaskBlendAmountSlider': {
            'level': 1,
            'label': 'Overall Mask Blend Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': 'Combined masks blending distance. It is not applied to the border masks.'
        },        
    },
}