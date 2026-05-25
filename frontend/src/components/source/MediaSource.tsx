import { useState, useRef } from 'react'
import { FolderOpen, Search, Play, Square, SkipBack, SkipForward, Circle, ChevronLeft, ChevronRight, Bookmark, BookmarkX } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'
import { useEvents } from '@/hooks/useEvents'
import { cn } from '@/lib/utils'

export function MediaSource() {
  const { mediaList, setMediaList, selectedMediaId, setSelectedMediaId, playback, markers } = useAppStore()
  const { send } = useEvents()
  const [search, setSearch] = useState('')
  const [showImages, setShowImages] = useState(true)
  const [showVideos, setShowVideos] = useState(true)
  const frameRef = useRef<HTMLInputElement>(null)

  const filtered = mediaList.filter(m => {
    if (!showImages && m.file_type === 'image') return false
    if (!showVideos && m.file_type === 'video') return false
    return m.media_path.toLowerCase().includes(search.toLowerCase())
  })

  const handleBrowse = async () => {
    const path = window.prompt('Enter folder path:')
    if (!path) return
    try {
      const res = await api.scanFolder(path) as { items: unknown[] }
      const items = res.items as typeof mediaList
      setMediaList(items)
    } catch (e) { alert(String(e)) }
  }

  const handleSelect = async (id: string) => {
    try {
      await api.selectMedia(id)
      setSelectedMediaId(id)
    } catch (e) { alert(String(e)) }
  }

  const pct = playback.max_frame > 0
    ? (playback.current_frame / playback.max_frame) * 100
    : 0

  return (
    <div className="flex flex-col gap-0">
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 p-2 border-b border-zinc-800">
        <button onClick={handleBrowse} className="flex items-center gap-1 px-2 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 rounded text-zinc-300 transition-colors">
          <FolderOpen size={12} /> Browse
        </button>
        <div className="flex-1 relative">
          <Search size={11} className="absolute left-2 top-1/2 -translate-y-1/2 text-zinc-500" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search..."
            className="w-full pl-6 pr-2 py-1 text-xs bg-zinc-800 border border-zinc-700 rounded text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-sky-500"
          />
        </div>
        <button onClick={() => setShowImages(v => !v)} title="Images" className={cn('px-1.5 py-1 text-xs rounded', showImages ? 'text-sky-400' : 'text-zinc-600')}>🖼</button>
        <button onClick={() => setShowVideos(v => !v)} title="Videos" className={cn('px-1.5 py-1 text-xs rounded', showVideos ? 'text-sky-400' : 'text-zinc-600')}>▶</button>
      </div>

      {/* Media grid */}
      <div className="grid grid-cols-3 gap-1.5 p-2">
        {filtered.map(m => (
          <button
            key={m.media_id}
            onClick={() => handleSelect(m.media_id)}
            className={cn(
              'relative rounded overflow-hidden border transition-all text-left',
              selectedMediaId === m.media_id
                ? 'border-sky-500 ring-1 ring-sky-500/50'
                : 'border-zinc-700 hover:border-zinc-500',
            )}
          >
            <img
              src={`/api/target-media/${m.media_id}/thumbnail`}
              alt=""
              className="w-full aspect-video object-cover bg-zinc-800"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5">
              <p className="text-[10px] text-zinc-300 truncate">
                {m.media_path.split(/[\\/]/).pop()}
              </p>
            </div>
          </button>
        ))}
        {filtered.length === 0 && (
          <div className="col-span-3 py-8 text-center text-xs text-zinc-600">
            Browse a folder to load media
          </div>
        )}
      </div>

      {/* Seek bar — only when media selected */}
      {selectedMediaId && (
        <div className="border-t border-zinc-800 p-2 flex flex-col gap-2">
          {/* Seek slider */}
          <div className="relative">
            <input
              type="range"
              min={0}
              max={playback.max_frame || 1}
              value={playback.current_frame}
              onChange={e => send('seek', { frame: Number(e.target.value) })}
              className="w-full h-1.5 accent-sky-500 cursor-pointer"
            />
            {/* Marker ticks */}
            {markers.map(m => (
              <div
                key={m}
                className="absolute top-0 w-0.5 h-1.5 bg-amber-400 pointer-events-none"
                style={{ left: `${(m / (playback.max_frame || 1)) * 100}%` }}
              />
            ))}
          </div>

          {/* Frame counter */}
          <div className="flex items-center justify-between text-[10px] text-zinc-500">
            <span>0</span>
            <input
              ref={frameRef}
              defaultValue={playback.current_frame}
              onKeyDown={e => {
                if (e.key === 'Enter') send('seek', { frame: Number((e.target as HTMLInputElement).value) })
              }}
              className="w-16 text-center bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-zinc-300 focus:outline-none focus:border-sky-500"
            />
            <span>{playback.max_frame}</span>
          </div>

          {/* Controls */}
          <div className="flex items-center justify-center gap-1">
            <button onClick={() => send('step', { n: -30 })} className="p-1.5 text-zinc-400 hover:text-zinc-200 rounded hover:bg-zinc-800">
              <SkipBack size={14} />
            </button>
            <button onClick={() => send('step', { n: -1 })} className="p-1.5 text-zinc-400 hover:text-zinc-200 rounded hover:bg-zinc-800">
              <ChevronLeft size={14} />
            </button>
            <button
              onClick={() => send(playback.is_recording ? 'stop' : 'play')}
              className={cn('p-1.5 rounded transition-colors', playback.is_recording ? 'text-red-400 hover:bg-red-500/10' : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800')}
            >
              <Circle size={14} fill={playback.is_recording ? 'currentColor' : 'none'} />
            </button>
            <button
              onClick={() => send(playback.is_playing ? 'stop' : 'play')}
              className={cn('p-2 rounded-full transition-colors', playback.is_playing ? 'bg-sky-500 text-white' : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600')}
            >
              {playback.is_playing ? <Square size={12} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
            </button>
            <button onClick={() => send('step', { n: 1 })} className="p-1.5 text-zinc-400 hover:text-zinc-200 rounded hover:bg-zinc-800">
              <ChevronRight size={14} />
            </button>
            <button onClick={() => send('step', { n: 30 })} className="p-1.5 text-zinc-400 hover:text-zinc-200 rounded hover:bg-zinc-800">
              <SkipForward size={14} />
            </button>
            <div className="w-px h-4 bg-zinc-700 mx-1" />
            <button onClick={() => api.addMarker()} title="Add marker" className="p-1.5 text-zinc-400 hover:text-amber-400 rounded hover:bg-zinc-800">
              <Bookmark size={13} />
            </button>
            <button onClick={() => api.deleteMarker(playback.current_frame)} title="Remove marker" className="p-1.5 text-zinc-400 hover:text-red-400 rounded hover:bg-zinc-800">
              <BookmarkX size={13} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
