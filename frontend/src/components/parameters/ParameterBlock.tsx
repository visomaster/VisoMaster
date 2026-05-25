import { useAppStore } from '@/store/appStore'
import { useEvents } from '@/hooks/useEvents'

// Inline parameter definitions for each block
// In production these would come from GET /api/schema/parameters/*
// For now we hardcode the most common ones to get the UI running

interface Param {
  name: string
  label: string
  type: 'toggle' | 'slider' | 'decimal_slider' | 'selection'
  default: unknown
  options?: string[]
  min?: number
  max?: number
  step?: number
  decimals?: number
  parent?: string
  parentValue?: unknown
}

const BLOCK_PARAMS: Record<string, Param[]> = {
  'Face Restorer': [
    { name: 'FaceRestorerEnableToggle', label: 'Enable Restorer 1', type: 'toggle', default: false },
    { name: 'FaceRestorerTypeSelection', label: 'Type', type: 'selection', default: 'GFPGAN-v1.4', options: ['GFPGAN-v1.4', 'CodeFormer', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048', 'RestoreFormer++', 'VQFR-v2'], parent: 'FaceRestorerEnableToggle', parentValue: true },
    { name: 'FaceRestorerDetTypeSelection', label: 'Alignment', type: 'selection', default: 'Original', options: ['Original', 'Blend', 'Reference'], parent: 'FaceRestorerEnableToggle', parentValue: true },
    { name: 'FaceFidelityWeightDecimalSlider', label: 'Fidelity', type: 'decimal_slider', default: 0.9, min: 0, max: 1, step: 0.1, decimals: 1, parent: 'FaceRestorerEnableToggle', parentValue: true },
    { name: 'FaceRestorerBlendSlider', label: 'Blend', type: 'slider', default: 100, min: 0, max: 100, step: 1, parent: 'FaceRestorerEnableToggle', parentValue: true },
    { name: 'FaceRestorerEnable2Toggle', label: 'Enable Restorer 2', type: 'toggle', default: false },
    { name: 'FaceRestorerType2Selection', label: 'Type 2', type: 'selection', default: 'GFPGAN-v1.4', options: ['GFPGAN-v1.4', 'CodeFormer', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048', 'RestoreFormer++', 'VQFR-v2'], parent: 'FaceRestorerEnable2Toggle', parentValue: true },
    { name: 'FaceRestorerBlend2Slider', label: 'Blend 2', type: 'slider', default: 100, min: 0, max: 100, step: 1, parent: 'FaceRestorerEnable2Toggle', parentValue: true },
  ],
  'Face Similarity': [
    { name: 'SimilarityThresholdSlider', label: 'Threshold', type: 'slider', default: 60, min: 1, max: 100, step: 1 },
    { name: 'StrengthEnableToggle', label: 'Strength', type: 'toggle', default: false },
    { name: 'StrengthAmountSlider', label: 'Amount', type: 'slider', default: 100, min: 0, max: 500, step: 25, parent: 'StrengthEnableToggle', parentValue: true },
    { name: 'FaceLikenessEnableToggle', label: 'Face Likeness', type: 'toggle', default: false },
    { name: 'FaceLikenessFactorDecimalSlider', label: 'Likeness Amount', type: 'decimal_slider', default: 0, min: -1, max: 1, step: 0.05, decimals: 2, parent: 'FaceLikenessEnableToggle', parentValue: true },
    { name: 'DifferencingEnableToggle', label: 'Differencing', type: 'toggle', default: false },
    { name: 'DifferencingAmountSlider', label: 'Amount', type: 'slider', default: 4, min: 0, max: 100, step: 1, parent: 'DifferencingEnableToggle', parentValue: true },
  ],
  'Face Mask': [
    { name: 'BorderTopSlider', label: 'Top Border', type: 'slider', default: 10, min: 0, max: 100, step: 1 },
    { name: 'BorderBottomSlider', label: 'Bottom Border', type: 'slider', default: 10, min: 0, max: 100, step: 1 },
    { name: 'BorderLeftSlider', label: 'Left Border', type: 'slider', default: 10, min: 0, max: 100, step: 1 },
    { name: 'BorderRightSlider', label: 'Right Border', type: 'slider', default: 10, min: 0, max: 100, step: 1 },
    { name: 'BorderBlurSlider', label: 'Border Blur', type: 'slider', default: 10, min: 0, max: 100, step: 1 },
    { name: 'OccluderEnableToggle', label: 'Occlusion Mask', type: 'toggle', default: false },
    { name: 'OccluderSizeSlider', label: 'Occluder Size', type: 'slider', default: 0, min: -100, max: 100, step: 1, parent: 'OccluderEnableToggle', parentValue: true },
    { name: 'DFLXSegEnableToggle', label: 'DFL XSeg Mask', type: 'toggle', default: false },
    { name: 'ClipEnableToggle', label: 'Text Masking', type: 'toggle', default: false },
    { name: 'FaceParserEnableToggle', label: 'Face Parser Mask', type: 'toggle', default: false },
    { name: 'RestoreEyesEnableToggle', label: 'Restore Eyes', type: 'toggle', default: false },
    { name: 'RestoreMouthEnableToggle', label: 'Restore Mouth', type: 'toggle', default: false },
  ],
  'Frame Enhancer': [
    { name: 'FrameEnhancerEnableToggle', label: 'Enable', type: 'toggle', default: false },
    { name: 'FrameEnhancerTypeSelection', label: 'Type', type: 'selection', default: 'RealEsrgan-x2-Plus', options: ['RealEsrgan-x2-Plus', 'RealEsrgan-x4-Plus', 'RealEsr-General-x4v3', 'BSRGan-x2', 'BSRGan-x4', 'UltraSharp-x4', 'UltraMix-x4', 'DDColor-Artistic', 'DDColor', 'DeOldify-Artistic', 'DeOldify-Stable', 'DeOldify-Video'], parent: 'FrameEnhancerEnableToggle', parentValue: true },
    { name: 'FrameEnhancerBlendSlider', label: 'Blend', type: 'slider', default: 100, min: 0, max: 100, step: 1, parent: 'FrameEnhancerEnableToggle', parentValue: true },
  ],
  'Detection': [
    { name: 'DetectorModelSelection', label: 'Model', type: 'selection', default: 'RetinaFace', options: ['RetinaFace', 'Yolov8', 'SCRFD', 'Yunet'] },
    { name: 'DetectorScoreSlider', label: 'Score', type: 'slider', default: 50, min: 1, max: 100, step: 1 },
    { name: 'MaxFacesToDetectSlider', label: 'Max Faces', type: 'slider', default: 20, min: 1, max: 50, step: 1 },
    { name: 'AutoRotationToggle', label: 'Auto Rotation', type: 'toggle', default: false },
    { name: 'LandmarkDetectToggle', label: 'Landmark Detect', type: 'toggle', default: false },
    { name: 'ShowAllDetectedFacesBBoxToggle', label: 'Show Bounding Boxes', type: 'toggle', default: false },
    { name: 'ShowLandmarksEnableToggle', label: 'Show Landmarks', type: 'toggle', default: false },
  ],
  'Swapper': [
    { name: 'SwapModelSelection', label: 'Model', type: 'selection', default: 'Inswapper128', options: ['Inswapper128', 'InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C', 'DeepFaceLive (DFM)', 'SimSwap512', 'GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'] },
    { name: 'SwapperResSelection', label: 'Resolution', type: 'selection', default: '128', options: ['128', '256', '384', '512'] },
  ],
  'Color Correction': [
    { name: 'ColorEnableToggle', label: 'Enable', type: 'toggle', default: false },
    { name: 'ColorGammaDecimalSlider', label: 'Gamma', type: 'decimal_slider', default: 1.0, min: 0.1, max: 3.0, step: 0.1, decimals: 1, parent: 'ColorEnableToggle', parentValue: true },
    { name: 'ColorBrightnessDecimalSlider', label: 'Brightness', type: 'decimal_slider', default: 1.0, min: 0.1, max: 3.0, step: 0.1, decimals: 1, parent: 'ColorEnableToggle', parentValue: true },
    { name: 'ColorContrastDecimalSlider', label: 'Contrast', type: 'decimal_slider', default: 1.0, min: 0.1, max: 3.0, step: 0.1, decimals: 1, parent: 'ColorEnableToggle', parentValue: true },
    { name: 'ColorSaturationDecimalSlider', label: 'Saturation', type: 'decimal_slider', default: 1.0, min: 0.1, max: 3.0, step: 0.1, decimals: 1, parent: 'ColorEnableToggle', parentValue: true },
  ],
  'Expression Restorer': [
    { name: 'FaceExpressionEnableToggle', label: 'Enable', type: 'toggle', default: false },
    { name: 'FaceExpressionFriendlyFactorDecimalSlider', label: 'Friendly Factor', type: 'decimal_slider', default: 1.0, min: 0, max: 1, step: 0.1, decimals: 1, parent: 'FaceExpressionEnableToggle', parentValue: true },
    { name: 'FaceExpressionAnimationRegionSelection', label: 'Region', type: 'selection', default: 'all', options: ['all', 'eyes', 'lips'], parent: 'FaceExpressionEnableToggle', parentValue: true },
  ],
  'Face Editor': [
    { name: 'FaceEditorEnableToggle', label: 'Enable', type: 'toggle', default: false },
    { name: 'HeadPitchSlider', label: 'Head Pitch', type: 'slider', default: 0, min: -15, max: 15, step: 1, parent: 'FaceEditorEnableToggle', parentValue: true },
    { name: 'HeadYawSlider', label: 'Head Yaw', type: 'slider', default: 0, min: -15, max: 15, step: 1, parent: 'FaceEditorEnableToggle', parentValue: true },
    { name: 'HeadRollSlider', label: 'Head Roll', type: 'slider', default: 0, min: -15, max: 15, step: 1, parent: 'FaceEditorEnableToggle', parentValue: true },
    { name: 'EyesOpenRatioDecimalSlider', label: 'Eyes Open', type: 'decimal_slider', default: 0, min: -0.8, max: 0.8, step: 0.01, decimals: 2, parent: 'FaceEditorEnableToggle', parentValue: true },
    { name: 'LipsOpenRatioDecimalSlider', label: 'Lips Open', type: 'decimal_slider', default: 0, min: -0.8, max: 0.8, step: 0.01, decimals: 2, parent: 'FaceEditorEnableToggle', parentValue: true },
    { name: 'MouthSmileDecimalSlider', label: 'Smile', type: 'decimal_slider', default: 0, min: -0.3, max: 1.3, step: 0.01, decimals: 2, parent: 'FaceEditorEnableToggle', parentValue: true },
  ],
  'Landmarks Correction': [
    { name: 'FaceAdjEnableToggle', label: 'Face Adjustments', type: 'toggle', default: false },
    { name: 'KpsXSlider', label: 'Keypoints X', type: 'slider', default: 0, min: -100, max: 100, step: 1, parent: 'FaceAdjEnableToggle', parentValue: true },
    { name: 'KpsYSlider', label: 'Keypoints Y', type: 'slider', default: 0, min: -100, max: 100, step: 1, parent: 'FaceAdjEnableToggle', parentValue: true },
    { name: 'KpsScaleSlider', label: 'Keypoints Scale', type: 'slider', default: 0, min: -100, max: 100, step: 1, parent: 'FaceAdjEnableToggle', parentValue: true },
    { name: 'FaceScaleAmountSlider', label: 'Face Scale', type: 'slider', default: 0, min: -20, max: 20, step: 1, parent: 'FaceAdjEnableToggle', parentValue: true },
  ],
}

interface WidgetProps {
  param: Param
  value: unknown
  onChange: (v: unknown) => void
}

function Widget({ param, value, onChange }: WidgetProps) {
  const v = value ?? param.default

  if (param.type === 'toggle') {
    return (
      <div className="flex items-center justify-between py-1">
        <span className="text-xs text-zinc-400">{param.label}</span>
        <button
          onClick={() => onChange(!v)}
          className={`relative w-8 h-4 rounded-full transition-colors ${v ? 'bg-sky-500' : 'bg-zinc-600'}`}
        >
          <span className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${v ? 'translate-x-4' : 'translate-x-0.5'}`} />
        </button>
      </div>
    )
  }

  if (param.type === 'selection') {
    return (
      <div className="flex items-center gap-2 py-1">
        <span className="text-xs text-zinc-400 w-24 shrink-0">{param.label}</span>
        <select
          value={v as string}
          onChange={e => onChange(e.target.value)}
          className="flex-1 text-xs bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-zinc-300 focus:outline-none focus:border-sky-500"
        >
          {param.options?.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      </div>
    )
  }

  if (param.type === 'slider' || param.type === 'decimal_slider') {
    const numVal = Number(v)
    return (
      <div className="py-1">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-zinc-400">{param.label}</span>
          <span className="text-xs text-zinc-500 tabular-nums">
            {param.decimals ? numVal.toFixed(param.decimals) : numVal}
          </span>
        </div>
        <input
          type="range"
          min={param.min}
          max={param.max}
          step={param.step}
          value={numVal}
          onChange={e => onChange(param.decimals ? parseFloat(e.target.value) : parseInt(e.target.value))}
          className="w-full h-1 accent-sky-500 cursor-pointer"
        />
      </div>
    )
  }

  return null
}

interface Props { blockName: string }

export function ParameterBlock({ blockName }: Props) {
  const { selectedFaceId, parameters, updateFaceParameter } = useAppStore()
  const { send } = useEvents()

  const params = BLOCK_PARAMS[blockName] ?? []
  const faceParams = selectedFaceId ? (parameters[selectedFaceId] ?? {}) : {}

  const handleChange = (name: string, value: unknown) => {
    if (!selectedFaceId) return
    updateFaceParameter(selectedFaceId, name, value)
    send('set_parameter', { face_id: selectedFaceId, name, value })
  }

  return (
    <div className="flex flex-col divide-y divide-zinc-700/30">
      {params.map(p => {
        // Visibility: hide if parent toggle is off
        if (p.parent) {
          const parentVal = faceParams[p.parent] ?? BLOCK_PARAMS[blockName]?.find(x => x.name === p.parent)?.default
          if (parentVal !== p.parentValue) return null
        }
        return (
          <Widget
            key={p.name}
            param={p}
            value={faceParams[p.name]}
            onChange={v => handleChange(p.name, v)}
          />
        )
      })}
      {params.length === 0 && (
        <p className="text-xs text-zinc-600 py-2">No parameters</p>
      )}
    </div>
  )
}
