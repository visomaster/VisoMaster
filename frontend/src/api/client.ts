const BASE = '/api'

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Request failed')
  }
  return res.json()
}

export const api = {
  get:    <T>(path: string) => req<T>('GET', path),
  post:   <T>(path: string, body?: unknown) => req<T>('POST', path, body),
  put:    <T>(path: string, body?: unknown) => req<T>('PUT', path, body),
  delete: <T>(path: string) => req<T>('DELETE', path),

  // System
  getGpuMemory: () => api.get<{ used_mb: number; total_mb: number }>('/system/gpu-memory'),
  clearMemory:  () => api.post('/system/clear-memory'),
  setProvider:  (provider: string) => api.post('/system/providers', { provider }),

  // Media
  scanFolder:   (path: string, recursive = false) =>
    api.post<{ items: unknown[] }>('/target-media/scan-folder', { path, recursive }),
  selectMedia:  (id: string) => api.post(`/target-media/${id}/select`),
  deleteMedia:  (id: string) => api.delete(`/target-media/${id}`),

  // Faces
  findFaces:    () => api.post<{ found: number; faces: unknown[] }>('/target-faces/find'),
  clearFaces:   () => api.post('/target-faces/clear'),
  selectFace:   (id: string) => api.post(`/target-faces/${id}/select`),
  deleteFace:   (id: string) => api.delete(`/target-faces/${id}`),
  assignInput:  (faceId: string, inputId: string) =>
    api.post(`/target-faces/${faceId}/assign-input/${inputId}`),
  unassignInput:(faceId: string, inputId: string) =>
    api.delete(`/target-faces/${faceId}/assign-input/${inputId}`),
  assignEmbed:  (faceId: string, embedId: string) =>
    api.post(`/target-faces/${faceId}/assign-embedding/${embedId}`),

  // Input faces
  scanInputFolder: (path: string, recursive = false) =>
    api.post('/input-faces/scan-folder', { path, recursive }),
  clearInputFaces: () => api.post('/input-faces/clear'),

  // Embeddings
  mergeEmbeddings: (name: string, ids: string[]) =>
    api.post('/embeddings/merge', { name, input_face_ids: ids }),
  deleteEmbedding: (id: string) => api.delete(`/embeddings/${id}`),

  // Playback
  play:         () => api.post('/playback/play'),
  stop:         () => api.post('/playback/stop'),
  seek:         (frame: number) => api.post('/playback/seek', { frame }),
  step:         (n: number) => api.post('/playback/step', { n }),
  recordStart:  (folder?: string) => api.post('/playback/record/start', { output_folder: folder }),
  recordStop:   () => api.post<{ output_path: string }>('/playback/record/stop'),
  saveFrame:    () => api.post('/playback/save-frame'),
  addMarker:    () => api.post('/playback/markers'),
  deleteMarker: (frame: number) => api.delete(`/playback/markers/${frame}`),

  // Sources
  getWebcams:   () => api.get<{ webcams: unknown[] }>('/sources/webcams'),
  selectWebcam: (index: number) => api.post(`/sources/webcams/${index}/select`),
  startWebrtc:  () => api.post<{ http_url: string; https_url: string; whip_url: string; whip_https_url: string }>('/sources/webrtc/start'),
  stopWebrtc:   () => api.post('/sources/webrtc/stop'),
  setTransform: (rotation: number, flip_h: boolean, flip_v: boolean) =>
    api.put('/sources/transform', { rotation, flip_h, flip_v }),

  // State
  getState:     () => api.get('/state'),
  patchControl: (updates: Record<string, unknown>) =>
    api.put('/state/control', { updates }),
  patchParams:  (faceId: string, updates: Record<string, unknown>) =>
    api.put(`/state/parameters/${faceId}`, { updates }),
  copyParams:   (faceId: string) => api.post(`/state/copy/${faceId}`),
  pasteParams:  (faceId: string) => api.post(`/state/paste/${faceId}`),
  resetParams:  (faceId: string) => api.post(`/state/reset/${faceId}`),

  // Workspace
  saveWorkspace: (filename: string) => api.post('/workspace/save', { filename }),
  loadWorkspace: (filename: string) => api.post('/workspace/load', { filename }),
  resetWorkspace:() => api.post('/workspace/reset'),
}
