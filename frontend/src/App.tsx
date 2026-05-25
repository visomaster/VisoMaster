import { useEffect } from 'react'
import { TopBar } from '@/components/layout/TopBar'
import { SourcePanel } from '@/components/source/SourcePanel'
import { FaceSwapPanel } from '@/components/faces/FaceSwapPanel'
import { FaceOptionsPanel } from '@/components/parameters/FaceOptionsPanel'
import { OutputPanel } from '@/components/output/OutputPanel'
import { useEvents } from '@/hooks/useEvents'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'

function AppInner() {
  // Boot the event listener
  useEvents()

  const { setPlayback, setMarkers, setTargetFaces, setInputFaces, setEmbeddings, setControl, setProvider } = useAppStore()

  // Load initial state from server
  useEffect(() => {
    const load = async () => {
      try {
        const [state, playback, markers] = await Promise.all([
          api.getState() as Promise<Record<string, unknown>>,
          api.get<Record<string, unknown>>('/playback'),
          api.get<{ markers: number[] }>('/playback/markers'),
        ])

        // Hydrate store
        if (state.control) setControl(state.control as Record<string, unknown>)
        if (state.target_faces) setTargetFaces(Object.values(state.target_faces as Record<string, unknown>) as never)
        if (state.input_faces) setInputFaces(Object.values(state.input_faces as Record<string, unknown>) as never)
        if (state.embeddings) setEmbeddings(Object.values(state.embeddings as Record<string, unknown>) as never)
        setPlayback(playback as never)
        setMarkers(markers.markers)

        // Sync provider
        const ctrl = state.control as Record<string, unknown>
        if (ctrl?.ProvidersPrioritySelection) {
          setProvider(ctrl.ProvidersPrioritySelection as never)
        }
      } catch {
        // Server not ready yet — will retry via WS reconnect
      }
    }
    load()
  }, [])

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-zinc-950">
      <TopBar />
      <div className="flex flex-1 overflow-hidden">
        {/* Col 1 — Input Source */}
        <div className="w-64 shrink-0 overflow-hidden">
          <SourcePanel />
        </div>

        {/* Col 2 — Face Swapping */}
        <div className="w-64 shrink-0 overflow-hidden">
          <FaceSwapPanel />
        </div>

        {/* Col 3 — Face Options */}
        <div className="w-72 shrink-0 overflow-hidden">
          <FaceOptionsPanel />
        </div>

        {/* Col 4 — Output */}
        <div className="flex-1 min-w-0 overflow-hidden border-l border-zinc-800">
          <OutputPanel />
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return <AppInner />
}
