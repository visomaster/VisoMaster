import { useEffect } from 'react'
import { Trash2 } from 'lucide-react'
import { useAppStore, type Provider } from '@/store/appStore'
import { ResourceBar } from '@/components/shared/ResourceBar'
import { api } from '@/api/client'
import { cn } from '@/lib/utils'

const PROVIDERS: Provider[] = ['CUDA', 'TensorRT', 'TensorRT-Engine']

export function TopBar() {
  const { gpuMemory, setGpuMemory, cpuPercent, gpuPercent, provider, setProvider } = useAppStore()

  // Poll GPU memory every 3s
  useEffect(() => {
    const poll = async () => {
      try {
        const m = await api.getGpuMemory()
        setGpuMemory(m)
      } catch { /* server not ready */ }
    }
    poll()
    const id = setInterval(poll, 3000)
    return () => clearInterval(id)
  }, [])

  const vramPct = gpuMemory.total_mb > 0
    ? (gpuMemory.used_mb / gpuMemory.total_mb) * 100
    : 0
  const vramDetail = gpuMemory.total_mb > 0
    ? `${(gpuMemory.used_mb / 1024).toFixed(1)} / ${(gpuMemory.total_mb / 1024).toFixed(0)} GB`
    : '— GB'

  const handleProvider = async (p: Provider) => {
    setProvider(p)
    try { await api.setProvider(p) } catch { /* toast */ }
  }

  const handleClearVram = async () => {
    try { await api.clearMemory() } catch { /* toast */ }
  }

  return (
    <header className="h-11 bg-zinc-900 border-b border-zinc-800 flex items-center px-4 gap-6 shrink-0 z-50">
      {/* Logo */}
      <div className="flex items-center gap-2 shrink-0">
        <div className="w-6 h-6 bg-sky-500 rounded flex items-center justify-center text-xs font-bold text-white">VM</div>
        <span className="text-sm font-semibold text-zinc-200 hidden sm:block">VisoMaster</span>
      </div>

      {/* Resource bars */}
      <div className="flex items-center gap-4 flex-1 min-w-0">
        <ResourceBar label="CPU" value={cpuPercent} />
        <ResourceBar label="GPU" value={gpuPercent} />
        <ResourceBar label="VRAM" value={vramPct} detail={vramDetail} />
      </div>

      {/* Provider selector */}
      <div className="flex items-center gap-1 bg-zinc-800 rounded-md p-0.5 shrink-0">
        {PROVIDERS.map((p) => (
          <button
            key={p}
            onClick={() => handleProvider(p)}
            className={cn(
              'px-2.5 py-1 text-xs rounded transition-colors',
              provider === p
                ? 'bg-sky-500 text-white font-medium'
                : 'text-zinc-400 hover:text-zinc-200',
            )}
          >
            {p === 'TensorRT-Engine' ? 'TRT-Engine' : p}
          </button>
        ))}
      </div>

      {/* Clear VRAM */}
      <button
        onClick={handleClearVram}
        title="Clear VRAM"
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-zinc-400 hover:text-red-400 hover:bg-zinc-800 rounded transition-colors shrink-0"
      >
        <Trash2 size={13} />
        <span className="hidden md:block">Clear VRAM</span>
      </button>
    </header>
  )
}
