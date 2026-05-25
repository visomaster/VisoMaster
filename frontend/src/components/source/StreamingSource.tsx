import { useState } from 'react'
import { Play, Square, Settings, RotateCcw, RotateCw, FlipHorizontal, FlipVertical } from 'lucide-react'
import { QRCodeSVG } from 'qrcode.react'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'
import { cn } from '@/lib/utils'

export function StreamingSource() {
  const { webrtcRunning, setWebrtcRunning, webrtcUrls, setWebrtcUrls, webrtcFps } = useAppStore()
  const [showSettings, setShowSettings] = useState(false)
  const [httpPort, setHttpPort] = useState('9091')
  const [httpsPort, setHttpsPort] = useState('9090')
  const [bindAddr, setBindAddr] = useState('0.0.0.0')
  const [rotation, setRotation] = useState(0)
  const [flipH, setFlipH] = useState(false)
  const [flipV, setFlipV] = useState(false)

  const handleStart = async () => {
    try {
      const res = await api.startWebrtc()
      setWebrtcUrls({ http_url: res.http_url, whip_url: res.whip_url })
      setWebrtcRunning(true)
    } catch (e) { alert(String(e)) }
  }

  const handleStop = async () => {
    try {
      await api.stopWebrtc()
      setWebrtcRunning(false)
      setWebrtcUrls(null)
    } catch { /* ignore */ }
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
      {/* Status */}
      <div className={cn(
        'flex items-center gap-2 px-3 py-2 rounded border text-xs',
        webrtcRunning && webrtcFps > 0
          ? 'border-green-500/40 bg-green-500/10 text-green-400'
          : webrtcRunning
          ? 'border-amber-500/40 bg-amber-500/10 text-amber-400'
          : 'border-zinc-700 text-zinc-500',
      )}>
        <span className={cn('w-2 h-2 rounded-full', webrtcRunning && webrtcFps > 0 ? 'bg-green-400' : webrtcRunning ? 'bg-amber-400 animate-pulse' : 'bg-zinc-600')} />
        {webrtcRunning && webrtcFps > 0
          ? `Live · ${webrtcFps.toFixed(1)} fps`
          : webrtcRunning
          ? 'Waiting for connection...'
          : 'Server stopped'}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2">
        {!webrtcRunning ? (
          <button onClick={handleStart} className="flex items-center gap-1.5 px-3 py-1.5 bg-sky-500 hover:bg-sky-400 text-white text-xs rounded transition-colors">
            <Play size={12} /> Start Server
          </button>
        ) : (
          <button onClick={handleStop} className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-700 hover:bg-zinc-600 text-zinc-300 text-xs rounded transition-colors">
            <Square size={12} /> Stop
          </button>
        )}
        <button onClick={() => setShowSettings(v => !v)} className={cn('p-1.5 rounded transition-colors', showSettings ? 'bg-sky-500/20 text-sky-400' : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-400')}>
          <Settings size={13} />
        </button>
      </div>

      {/* Settings popover */}
      {showSettings && (
        <div className="bg-zinc-800 border border-zinc-700 rounded p-3 flex flex-col gap-2 text-xs">
          <p className="text-zinc-400 font-medium">Port Settings</p>
          {[
            { label: 'HTTP Port', value: httpPort, set: setHttpPort },
            { label: 'HTTPS Port', value: httpsPort, set: setHttpsPort },
            { label: 'Bind Address', value: bindAddr, set: setBindAddr },
          ].map(({ label, value, set }) => (
            <div key={label} className="flex items-center gap-2">
              <span className="text-zinc-500 w-24 shrink-0">{label}</span>
              <input
                value={value}
                onChange={e => set(e.target.value)}
                className="flex-1 px-2 py-1 bg-zinc-900 border border-zinc-700 rounded text-zinc-300 focus:outline-none focus:border-sky-500"
              />
            </div>
          ))}
          <button
            onClick={() => {
              api.patchControl({ WebRTCHttpPortText: httpPort, WebRTCHttpsPortText: httpsPort, WebRTCBindAddressText: bindAddr })
              setShowSettings(false)
            }}
            className="mt-1 px-3 py-1.5 bg-sky-500 hover:bg-sky-400 text-white rounded transition-colors"
          >
            Apply
          </button>
        </div>
      )}

      {/* URLs + QR */}
      {webrtcUrls && (
        <div className="flex flex-col gap-2">
          <div className="text-xs space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 w-12 shrink-0">HTTP</span>
              <span className="text-zinc-300 truncate">{webrtcUrls.http_url}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 w-12 shrink-0">WHIP</span>
              <span className="text-zinc-300 truncate">{webrtcUrls.whip_url}</span>
            </div>
          </div>
          <div className="flex justify-center p-2 bg-white rounded">
            <QRCodeSVG value={webrtcUrls.http_url} size={120} />
          </div>
        </div>
      )}

      {/* Transform */}
      <div className="border-t border-zinc-800 pt-3">
        <p className="text-xs text-zinc-500 mb-2">Transform</p>
        <div className="flex items-center gap-2">
          <button onClick={rotateCCW} className="p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors"><RotateCcw size={13} /></button>
          <button onClick={rotateCW}  className="p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors"><RotateCw size={13} /></button>
          <span className="text-xs text-zinc-500 w-8 text-center">{rotation}°</span>
          <button onClick={toggleH} className={cn('p-1.5 rounded transition-colors', flipH ? 'bg-sky-500/20 text-sky-400' : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-400')}><FlipHorizontal size={13} /></button>
          <button onClick={toggleV} className={cn('p-1.5 rounded transition-colors', flipV ? 'bg-sky-500/20 text-sky-400' : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-400')}><FlipVertical size={13} /></button>
        </div>
      </div>
    </div>
  )
}
