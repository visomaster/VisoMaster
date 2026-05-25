import { useEffect, useState } from 'react'
import { RotateCcw, RotateCw, FlipHorizontal, FlipVertical } from 'lucide-react'
import { api } from '@/api/client'
import { cn } from '@/lib/utils'

interface Webcam { index: number; label: string }

export function WebcamSource() {
  const [webcams, setWebcams] = useState<Webcam[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [rotation, setRotation] = useState(0)
  const [flipH, setFlipH] = useState(false)
  const [flipV, setFlipV] = useState(false)

  useEffect(() => {
    api.getWebcams().then((r: unknown) => {
      setWebcams((r as { webcams: Webcam[] }).webcams)
    }).catch(() => {})
  }, [])

  const handleSelect = async (index: number) => {
    try {
      await api.selectWebcam(index)
      setSelected(index)
    } catch (e) { alert(String(e)) }
  }

  const applyTransform = async (r: number, h: boolean, v: boolean) => {
    try { await api.setTransform(r, h, v) } catch { /* ignore */ }
  }

  const rotateCCW = () => { const r = (rotation - 90 + 360) % 360; setRotation(r); applyTransform(r, flipH, flipV) }
  const rotateCW  = () => { const r = (rotation + 90) % 360;       setRotation(r); applyTransform(r, flipH, flipV) }
  const toggleH   = () => { const h = !flipH; setFlipH(h); applyTransform(rotation, h, flipV) }
  const toggleV   = () => { const v = !flipV; setFlipV(v); applyTransform(rotation, flipH, v) }

  return (
    <div className="p-3 flex flex-col gap-3">
      {/* Webcam cards */}
      <div className="grid grid-cols-2 gap-2">
        {webcams.map(w => (
          <button
            key={w.index}
            onClick={() => handleSelect(w.index)}
            className={cn(
              'flex flex-col items-center gap-1 p-3 rounded border text-xs transition-all',
              selected === w.index
                ? 'border-sky-500 bg-sky-500/10 text-sky-400'
                : 'border-zinc-700 hover:border-zinc-500 text-zinc-400',
            )}
          >
            <span className="text-2xl">📷</span>
            <span>{w.label}</span>
          </button>
        ))}
        {webcams.length === 0 && (
          <div className="col-span-2 text-center text-xs text-zinc-600 py-6">
            No webcams found
          </div>
        )}
      </div>

      {/* Transform */}
      <div className="border-t border-zinc-800 pt-3">
        <p className="text-xs text-zinc-500 mb-2">Transform</p>
        <div className="flex items-center gap-2">
          <button onClick={rotateCCW} title="Rotate CCW" className="p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors">
            <RotateCcw size={13} />
          </button>
          <button onClick={rotateCW} title="Rotate CW" className="p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors">
            <RotateCw size={13} />
          </button>
          <span className="text-xs text-zinc-500 w-8 text-center">{rotation}°</span>
          <button onClick={toggleH} title="Flip H" className={cn('p-1.5 rounded transition-colors', flipH ? 'bg-sky-500/20 text-sky-400' : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-400')}>
            <FlipHorizontal size={13} />
          </button>
          <button onClick={toggleV} title="Flip V" className={cn('p-1.5 rounded transition-colors', flipV ? 'bg-sky-500/20 text-sky-400' : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-400')}>
            <FlipVertical size={13} />
          </button>
        </div>
      </div>
    </div>
  )
}
