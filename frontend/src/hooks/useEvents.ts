import { useEffect, useRef } from 'react'
import useWebSocket, { ReadyState } from 'react-use-websocket'
import { useAppStore } from '@/store/appStore'

const WS_URL = 'ws://localhost:8000/ws/events'

export function useEvents() {
  const store = useAppStore()

  const { lastMessage, sendMessage, readyState } = useWebSocket(WS_URL, {
    shouldReconnect: () => true,
    reconnectInterval: 2000,
  })

  useEffect(() => {
    if (!lastMessage?.data) return
    try {
      const { type, payload } = JSON.parse(lastMessage.data as string)
      switch (type) {
        case 'playback_state':
          store.setPlayback(payload)
          break
        case 'fps_update':
          store.setWebrtcFps(payload.fps ?? 0)
          break
        case 'state_updated':
          if (payload.section === 'control') {
            store.setControl({ [payload.name]: payload.value })
          } else if (payload.section === 'parameters' && payload.face_id) {
            store.updateFaceParameter(payload.face_id, payload.name, payload.value)
          }
          break
        case 'recording_finished':
          console.log('[recording] saved to', payload.output_path)
          break
      }
    } catch { /* ignore */ }
  }, [lastMessage])

  const send = (type: string, payload?: Record<string, unknown>) => {
    sendMessage(JSON.stringify({ type, payload: payload ?? {} }))
  }

  return { send, readyState, isConnected: readyState === ReadyState.OPEN }
}
