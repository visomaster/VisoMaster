import { useEffect, useRef, useState } from 'react'
import useWebSocket from 'react-use-websocket'

export function usePreviewStream(quality = 75) {
  const [src, setSrc] = useState<string>('')
  const prevSrc = useRef<string>('')

  const { lastMessage, sendMessage } = useWebSocket('ws://localhost:8000/ws/preview', {
    shouldReconnect: () => true,
    reconnectInterval: 2000,
  })

  // Send quality preference once connected
  useEffect(() => {
    sendMessage(JSON.stringify({ quality }))
  }, [quality])

  useEffect(() => {
    if (!lastMessage) return
    const blob = new Blob([lastMessage.data as ArrayBuffer], { type: 'image/jpeg' })
    const url = URL.createObjectURL(blob)
    setSrc(url)
    if (prevSrc.current) URL.revokeObjectURL(prevSrc.current)
    prevSrc.current = url
  }, [lastMessage])

  useEffect(() => () => { if (prevSrc.current) URL.revokeObjectURL(prevSrc.current) }, [])

  return src
}
