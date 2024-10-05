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
            'step': 1,
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
            'step': 1,
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
    },
    'Face Similarity': {
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
            'step': 1,
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
            'label': 'Background',
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