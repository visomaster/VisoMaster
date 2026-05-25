import { useState } from 'react'
import { Monitor, Camera, Radio } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { MediaSource } from './MediaSource'
import { WebcamSource } from './WebcamSource'
import { StreamingSource } from './StreamingSource'
import { usePreviewStream } from '@/hooks/usePreviewStream'
import { cn } from '@/lib/utils'

const TABS = [
  { id: 'media',     label: 'Media',    Icon: Monitor },
  { id: 'webcam',    label: 'Webcam',   Icon: Camera },
  { id: 'streaming', label: 'Stream',   Icon: Radio },
] as const

export function SourcePanel() {
  const { sourceType, setSourceType, webrtcFps, playback } = useAppStore()
  const previewSrc = usePreviewStream(20)

  return (
    <div className="flex flex-col h-full bg-zinc-900 border-r border-zinc-800">
      {/* Source preview */}
      <div className="relative bg-zinc-950 shrink-0">
        {previewSrc ? (
          <img src={previewSrc} alt="source preview" className="w-full aspect-video object-contain" />
        ) : (
          <div className="w-full aspect-video flex items-center justify-center text-zinc-600 text-xs">
            No source
          </div>
        )}
        {/* FPS badge */}
        {(sourceType === 'webcam' || sourceType === 'streaming') && webrtcFps > 0 && (
          <span className="absolute top-1.5 right-1.5 bg-black/60 text-xs text-green-400 px-1.5 py-0.5 rounded">
            {webrtcFps.toFixed(1)} fps
          </span>
        )}
        {/* Playing badge */}
        {playback.is_playing && (
          <span className="absolute top-1.5 left-1.5 bg-sky-500/80 text-xs text-white px-1.5 py-0.5 rounded">
            ▶ Live
          </span>
        )}
      </div>

      {/* Tab strip */}
      <div className="flex border-b border-zinc-800 shrink-0">
        {TABS.map(({ id, label, Icon }) => (
          <button
            key={id}
            onClick={() => setSourceType(id)}
            className={cn(
              'flex-1 flex items-center justify-center gap-1.5 py-2 text-xs transition-colors',
              sourceType === id
                ? 'text-sky-400 border-b-2 border-sky-500 -mb-px'
                : 'text-zinc-500 hover:text-zinc-300',
            )}
          >
            <Icon size={12} />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {sourceType === 'media'     && <MediaSource />}
        {sourceType === 'webcam'    && <WebcamSource />}
        {sourceType === 'streaming' && <StreamingSource />}
      </div>
    </div>
  )
}
