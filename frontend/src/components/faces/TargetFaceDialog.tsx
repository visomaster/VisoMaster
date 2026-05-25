import { useState } from 'react'
import { X, FolderOpen, Search, UserSearch } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'
import { cn } from '@/lib/utils'

interface Props {
  onSelect: (faceId: string) => void
  onClose: () => void
}

export function TargetFaceDialog({ onSelect, onClose }: Props) {
  const { targetFaces, setTargetFaces } = useAppStore()
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)

  const filtered = targetFaces.filter(f =>
    f.face_id.toLowerCase().includes(search.toLowerCase())
  )

  const handleFind = async () => {
    setLoading(true)
    try {
      const res = await api.findFaces() as { found: number; faces: typeof targetFaces }
      if (res.found > 0) setTargetFaces([...targetFaces, ...res.faces])
    } catch (e) { alert(String(e)) }
    finally { setLoading(false) }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-zinc-700 rounded-xl w-[480px] max-h-[70vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
          <h2 className="text-sm font-semibold text-zinc-200">Choose Target Face</h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-800">
          <div className="flex-1 relative">
            <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search faces..."
              className="w-full pl-7 pr-3 py-1.5 text-xs bg-zinc-800 border border-zinc-700 rounded text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-sky-500"
            />
          </div>
          <button
            onClick={handleFind}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-sky-500 hover:bg-sky-400 text-white rounded transition-colors disabled:opacity-60"
          >
            <UserSearch size={12} />
            {loading ? 'Finding...' : 'Find in Frame'}
          </button>
        </div>

        {/* Face grid */}
        <div className="flex-1 overflow-y-auto p-3">
          {filtered.length === 0 ? (
            <div className="text-center py-12 text-xs text-zinc-600">
              No target faces yet. Click "Find in Frame" to detect faces.
            </div>
          ) : (
            <div className="grid grid-cols-5 gap-2">
              {filtered.map(f => (
                <button
                  key={f.face_id}
                  onClick={() => onSelect(f.face_id)}
                  className="aspect-square rounded-lg overflow-hidden border border-zinc-700 hover:border-sky-500 transition-all hover:scale-105"
                >
                  <img
                    src={`/api/target-faces/${f.face_id}/thumbnail`}
                    alt=""
                    className="w-full h-full object-cover"
                  />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-zinc-800 flex justify-end">
          <button onClick={onClose} className="px-4 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors">
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
