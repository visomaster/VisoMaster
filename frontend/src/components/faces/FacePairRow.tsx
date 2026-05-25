import { useState } from 'react'
import { Trash2, UserCircle2 } from 'lucide-react'
import { useAppStore, type FacePair } from '@/store/appStore'
import { api } from '@/api/client'
import { TargetFaceDialog } from './TargetFaceDialog'

interface Props {
  pair: FacePair
  onRemove: () => void
}

export function FacePairRow({ pair, onRemove }: Props) {
  const { inputFaces, facePairs, setFacePairs, setSelectedFaceId } = useAppStore()
  const [showTargetDialog, setShowTargetDialog] = useState(false)

  const sourceFace = inputFaces.find(f => f.face_id === pair.sourceFaceId)

  const handleSelectTarget = (faceId: string) => {
    setFacePairs(facePairs.map(p => p.id === pair.id ? { ...p, targetFaceId: faceId } : p))
    setSelectedFaceId(faceId)
    if (pair.sourceFaceId) {
      api.assignInput(faceId, pair.sourceFaceId).catch(() => {})
    }
    setShowTargetDialog(false)
  }

  const handleBrowseSource = async () => {
    const path = window.prompt('Enter folder path for source faces:')
    if (!path) return
    try { await api.scanInputFolder(path) } catch (e) { alert(String(e)) }
  }

  return (
    <>
      <div className="bg-zinc-800/60 border border-zinc-700 rounded-lg p-2 flex flex-col gap-2">
        <div className="flex items-stretch gap-2">
          {/* Source face */}
          <div className="flex-1 flex flex-col gap-1">
            <span className="text-[10px] text-zinc-500">Source</span>
            <button
              onClick={handleBrowseSource}
              className="aspect-square w-full max-w-[72px] rounded border border-zinc-600 hover:border-sky-500 bg-zinc-900 flex items-center justify-center overflow-hidden transition-colors"
            >
              {sourceFace ? (
                <img src={`/api/input-faces/${sourceFace.face_id}/thumbnail`} alt="" className="w-full h-full object-cover" />
              ) : (
                <UserCircle2 size={24} className="text-zinc-600" />
              )}
            </button>
          </div>

          <div className="flex items-center text-zinc-600 text-lg">→</div>

          {/* Target face */}
          <div className="flex-1 flex flex-col gap-1">
            <span className="text-[10px] text-zinc-500">Target</span>
            <button
              onClick={() => setShowTargetDialog(true)}
              className="aspect-square w-full max-w-[72px] rounded border border-zinc-600 hover:border-sky-500 bg-zinc-900 flex items-center justify-center overflow-hidden transition-colors"
            >
              {pair.targetFaceId ? (
                <img src={`/api/target-faces/${pair.targetFaceId}/thumbnail`} alt="" className="w-full h-full object-cover" />
              ) : (
                <UserCircle2 size={24} className="text-zinc-600" />
              )}
            </button>
          </div>
        </div>

        {/* Remove */}
        <button
          onClick={onRemove}
          className="flex items-center gap-1 text-[10px] text-zinc-600 hover:text-red-400 transition-colors self-end"
        >
          <Trash2 size={10} /> Remove pair
        </button>
      </div>

      {showTargetDialog && (
        <TargetFaceDialog
          onSelect={handleSelectTarget}
          onClose={() => setShowTargetDialog(false)}
        />
      )}
    </>
  )
}
