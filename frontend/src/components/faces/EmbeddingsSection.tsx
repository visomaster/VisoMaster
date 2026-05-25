import { useState } from 'react'
import { Plus, Download, Upload, Trash2 } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'

export function EmbeddingsSection() {
  const { embeddings, setEmbeddings } = useAppStore()
  const [merging, setMerging] = useState(false)

  const handleMerge = async () => {
    const name = window.prompt('Embedding name:')
    if (!name) return
    setMerging(true)
    try {
      const res = await api.mergeEmbeddings(name, []) as { embedding_id: string; name: string }
      setEmbeddings([...embeddings, res])
    } catch (e) { alert(String(e)) }
    finally { setMerging(false) }
  }

  const handleDelete = async (id: string) => {
    try {
      await api.deleteEmbedding(id)
      setEmbeddings(embeddings.filter(e => e.embedding_id !== id))
    } catch { /* ignore */ }
  }

  return (
    <div className="p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-zinc-500 font-medium">Embeddings</p>
        <div className="flex items-center gap-1">
          <button onClick={handleMerge} disabled={merging} title="Merge" className="p-1 text-zinc-500 hover:text-sky-400 transition-colors">
            <Plus size={13} />
          </button>
          <a href="/api/embeddings/export" download="embeddings.json" title="Export" className="p-1 text-zinc-500 hover:text-sky-400 transition-colors">
            <Download size={13} />
          </a>
        </div>
      </div>

      {embeddings.length === 0 ? (
        <p className="text-xs text-zinc-600 py-2">No embeddings yet</p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {embeddings.map(e => (
            <div key={e.embedding_id} className="flex items-center gap-1 px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-xs text-zinc-300 group">
              <span className="truncate max-w-[100px]">{e.name}</span>
              <button onClick={() => handleDelete(e.embedding_id)} className="opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-red-400 transition-all">
                <Trash2 size={10} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
