import { useState } from 'react'
import { Plus, Zap, ZapOff, UserSearch } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'
import { useEvents } from '@/hooks/useEvents'
import { FacePairRow } from './FacePairRow'
import { EmbeddingsSection } from './EmbeddingsSection'
import { cn } from '@/lib/utils'

const SWAPPER_MODELS = ['Inswapper128', 'InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C', 'DeepFaceLive (DFM)', 'SimSwap512', 'GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS']
const DETECTORS = ['RetinaFace', 'Yolov8', 'SCRFD', 'Yunet']
const ARCFACE_MODELS = ['Inswapper128ArcFace', 'SimSwapArcFace', 'GhostArcFace', 'CSCSArcFace']
const RESOLUTIONS = ['128', '256', '384', '512']

export function FaceSwapPanel() {
  const { facePairs, setFacePairs, playback, control, setControl } = useAppStore()
  const { send } = useEvents()
  const [loading, setLoading] = useState(false)

  const swapActive = playback.swap_enabled
  const editActive = playback.edit_enabled

  const handleActivate = async () => {
    setLoading(true)
    try {
      if (swapActive) {
        await api.post('/playback/swap/disable')
        send('swap_disable')
      } else {
        await api.post('/playback/swap/enable')
        send('swap_enable')
      }
    } finally { setLoading(false) }
  }

  const handleFindFaces = async () => {
    try {
      const res = await api.findFaces() as { found: number; faces: { face_id: string; thumbnail_url: string }[] }
      if (res.found > 0) {
        // Add new pairs for found faces
        const newPairs = res.faces.map(f => ({
          id: f.face_id,
          sourceFaceId: null,
          targetFaceId: f.face_id,
        }))
        setFacePairs([...facePairs, ...newPairs])
      }
    } catch (e) { alert(String(e)) }
  }

  const addPair = () => {
    setFacePairs([...facePairs, { id: crypto.randomUUID(), sourceFaceId: null, targetFaceId: null }])
  }

  const removePair = (id: string) => {
    setFacePairs(facePairs.filter(p => p.id !== id))
  }

  const setCtrl = (name: string, value: unknown) => {
    setControl({ [name]: value })
    send('set_control', { name, value })
  }

  return (
    <div className="flex flex-col h-full bg-zinc-900 border-r border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 shrink-0">
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Face Swapping</span>
        <div className="flex items-center gap-1">
          <button
            onClick={handleFindFaces}
            title="Find faces in current frame"
            className="flex items-center gap-1 px-2 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded transition-colors"
          >
            <UserSearch size={12} /> Find
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Activate button */}
        <div className="p-3 border-b border-zinc-800">
          <button
            onClick={handleActivate}
            disabled={loading}
            className={cn(
              'w-full flex items-center justify-center gap-2 py-2 rounded text-sm font-medium transition-all',
              swapActive
                ? 'bg-green-500/20 text-green-400 border border-green-500/40 hover:bg-green-500/30'
                : 'bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700',
              loading && 'opacity-60 cursor-wait',
            )}
          >
            {loading ? (
              <span className="animate-spin">⟳</span>
            ) : swapActive ? (
              <><Zap size={14} fill="currentColor" /> Swap Active</>
            ) : (
              <><ZapOff size={14} /> Activate Swap</>
            )}
          </button>
        </div>

        {/* Model settings */}
        <div className="p-3 border-b border-zinc-800 flex flex-col gap-2">
          <p className="text-xs text-zinc-500 font-medium">Model Settings</p>
          {[
            { label: 'Detector', key: 'DetectorModelSelection', opts: DETECTORS },
            { label: 'Swapper', key: 'SwapModelSelection', opts: SWAPPER_MODELS },
            { label: 'Resolution', key: 'SwapperResSelection', opts: RESOLUTIONS },
            { label: 'ArcFace', key: 'RecognitionModelSelection', opts: ARCFACE_MODELS },
          ].map(({ label, key, opts }) => (
            <div key={key} className="flex items-center gap-2">
              <span className="text-xs text-zinc-500 w-20 shrink-0">{label}</span>
              <select
                value={(control[key] as string) ?? opts[0]}
                onChange={e => setCtrl(key, e.target.value)}
                className="flex-1 text-xs bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-zinc-300 focus:outline-none focus:border-sky-500"
              >
                {opts.map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            </div>
          ))}
        </div>

        {/* Face pairs */}
        <div className="p-3 flex flex-col gap-2 border-b border-zinc-800">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 font-medium">Face Pairs</p>
            <button onClick={addPair} className="flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300 transition-colors">
              <Plus size={12} /> Add Pair
            </button>
          </div>
          {facePairs.length === 0 && (
            <div className="text-center py-6 text-xs text-zinc-600">
              Click "Find" to detect faces, or add a pair manually
            </div>
          )}
          {facePairs.map(pair => (
            <FacePairRow key={pair.id} pair={pair} onRemove={() => removePair(pair.id)} />
          ))}
        </div>

        {/* Embeddings */}
        <EmbeddingsSection />
      </div>
    </div>
  )
}
