/* VisoMaster Streaming Client — High-Performance WebSocket */
'use strict';

// ── DOM refs ────────────────────────────────────────────────────────────────
const localVideo       = document.getElementById('localVideo');
const videoOverlay     = document.getElementById('videoOverlay');
const videoContainer   = document.getElementById('videoContainer');
const streamBtn        = document.getElementById('streamBtn');
const streamBtnIcon    = document.getElementById('streamBtnIcon');
const streamBtnText    = document.getElementById('streamBtnText');
const cameraSelect     = document.getElementById('cameraSelect');
const resolutionSelect = document.getElementById('resolutionSelect');
const statusDot        = document.getElementById('statusDot');
const panelToggle      = document.getElementById('panelToggle');
const panelBody        = document.getElementById('panelBody');

const statFps          = document.getElementById('statFps');
const statBitrate      = document.getElementById('statBitrate');
const statRss          = document.getElementById('statRss');
const statResolution   = document.getElementById('statResolution');

let localStream = null;
let isStreaming = false;
let ws = null;
let captureLoop = null;
let hiddenVideo = null;

// Encoding state
let encoder = null;       // VideoEncoder (WebCodecs) or null
let useWebCodecs = false;
let sendCanvas = null;
let sendCtx = null;
let pendingEncode = false;

// Mirror preview
let mirrorCanvas = null;
let mirrorCtx = null;

// Stats
let statsInterval = null;
let framesSent = 0;
let bytesSent = 0;
let lastStatsTime = 0;
let lastStatsFrames = 0;
let lastStatsBytes = 0;

// Target FPS
const TARGET_FPS = 30;

// ── Icons ────────────────────────────────────────────────────────────────────
const playIcon = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>`;
const stopIcon = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="3"/></svg>`;

panelToggle.addEventListener('click', () => {
  panelBody.classList.toggle('collapsed');
  panelToggle.classList.toggle('active');
});

function setStatus(state) { statusDot.className = 'status-dot ' + state; }

function updateStreamButton(streaming) {
  isStreaming = streaming;
  if (streaming) {
    streamBtn.classList.add('streaming');
    streamBtnIcon.innerHTML = stopIcon;
    streamBtnText.textContent = 'Stop Streaming';
  } else {
    streamBtn.classList.remove('streaming');
    streamBtnIcon.innerHTML = playIcon;
    streamBtnText.textContent = 'Start Streaming';
  }
}

// ── Stats ────────────────────────────────────────────────────────────────────
function startStats() {
  lastStatsTime = performance.now();
  lastStatsFrames = 0;
  lastStatsBytes = 0;
  statsInterval = setInterval(() => {
    const now = performance.now();
    const elapsed = (now - lastStatsTime) / 1000;
    if (elapsed < 0.5) return;
    const fps = Math.round((framesSent - lastStatsFrames) / elapsed);
    const bitrate = Math.round(((bytesSent - lastStatsBytes) * 8) / elapsed / 1000);
    statFps.textContent = fps;
    statBitrate.textContent = bitrate > 1000 ? (bitrate / 1000).toFixed(1) + ' Mbps' : bitrate + ' kbps';
    if (performance.memory) statRss.textContent = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(0) + ' MB';
    lastStatsTime = now;
    lastStatsFrames = framesSent;
    lastStatsBytes = bytesSent;
  }, 1000);
}

function stopStats() {
  if (statsInterval) { clearInterval(statsInterval); statsInterval = null; }
  statFps.textContent = '0';
  statBitrate.textContent = '0 kbps';
  statRss.textContent = '—';
  statResolution.textContent = '—';
}

// ── Cameras ──────────────────────────────────────────────────────────────────
async function enumerateCameras() {
  try { const t = await navigator.mediaDevices.getUserMedia({video:true}); t.getTracks().forEach(t=>t.stop()); } catch(_){}
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(d => d.kind === 'videoinput');
  cameraSelect.innerHTML = '';
  if (!cameras.length) {
    cameraSelect.innerHTML = '<option>No cameras found</option>';
    streamBtn.disabled = true;
    return;
  }
  cameras.forEach((cam, i) => {
    const opt = document.createElement('option');
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Camera ${i + 1}`;
    cameraSelect.appendChild(opt);
  });
}

function getConstraints() {
  const [w, h] = resolutionSelect.value.split('x').map(Number);
  return {
    video: {
      deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined,
      width: { ideal: Math.max(w, h) },
      height: { ideal: Math.min(w, h) },
      frameRate: { ideal: 30 },
      facingMode: 'user'
    },
    audio: false,
  };
}

// ── Stream toggle ────────────────────────────────────────────────────────────
streamBtn.addEventListener('click', () => {
  if (isStreaming) cleanUp(); else startStreaming();
});

// ── WebCodecs detection ──────────────────────────────────────────────────────
// H.264 works on iPhone/Safari but may fail on some Windows browsers.
// Auto-detects and falls back to JPEG if encoding fails.
function webCodecsAvailable() {
  return typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined';
}

// ── AVCC to Annex B conversion ───────────────────────────────────────────────
// WebCodecs outputs H.264 in AVCC format (length-prefixed NALUs).
// PyAV/FFmpeg expects Annex B format (start code prefixed: 00 00 00 01).
function avccToAnnexB(avccData) {
  // AVCC extradata format:
  // [0] version, [1] profile, [2] compat, [3] level, [4] lengthSizeMinusOne
  // [5] numSPS, then SPS entries, then numPPS, then PPS entries
  const view = new DataView(avccData.buffer, avccData.byteOffset, avccData.byteLength);
  const startCode = new Uint8Array([0, 0, 0, 1]);
  const parts = [];
  
  try {
    let offset = 5;
    const numSPS = avccData[offset] & 0x1f;
    offset++;
    
    for (let i = 0; i < numSPS; i++) {
      const spsLen = view.getUint16(offset);
      offset += 2;
      parts.push(startCode);
      parts.push(avccData.slice(offset, offset + spsLen));
      offset += spsLen;
    }
    
    const numPPS = avccData[offset];
    offset++;
    
    for (let i = 0; i < numPPS; i++) {
      const ppsLen = view.getUint16(offset);
      offset += 2;
      parts.push(startCode);
      parts.push(avccData.slice(offset, offset + ppsLen));
      offset += ppsLen;
    }
  } catch (e) {
    // If parsing fails, just return empty — frame data will still have start code
    return new Uint8Array(0);
  }
  
  // Concatenate all parts
  const totalLen = parts.reduce((sum, p) => sum + p.length, 0);
  const result = new Uint8Array(totalLen);
  let pos = 0;
  for (const part of parts) {
    result.set(part, pos);
    pos += part.length;
  }
  return result;
}

// ── Start streaming ──────────────────────────────────────────────────────────
async function startStreaming() {
  streamBtn.disabled = true;
  setStatus('connecting');

  if (ws) { cleanUp(); await new Promise(r => setTimeout(r, 200)); }

  try {
    localStream = await navigator.mediaDevices.getUserMedia(getConstraints());
  } catch (err) {
    console.error('[Stream] Camera error:', err);
    setStatus('error');
    streamBtn.disabled = false;
    return;
  }

  localVideo.classList.add('visible');
  videoOverlay.classList.add('hidden');

  const videoTrack = localStream.getVideoTracks()[0];
  const settings = videoTrack.getSettings();
  const vw = settings.width || 1280;
  const vh = settings.height || 720;

  // Determine send resolution — lower = faster encoding = higher FPS
  // 640px width gives ~3x faster encoding than 1280px
  const maxSendW = 640;
  const sendW = Math.min(vw, maxSendW);
  const sendH = Math.round(sendW * vh / vw);
  statResolution.textContent = sendW + '×' + sendH;

  // Mirror canvas for local preview
  mirrorCanvas = document.createElement('canvas');
  mirrorCanvas.width = vw;
  mirrorCanvas.height = vh;
  mirrorCtx = mirrorCanvas.getContext('2d');

  // Send canvas for encoding
  sendCanvas = document.createElement('canvas');
  sendCanvas.width = sendW;
  sendCanvas.height = sendH;
  sendCtx = sendCanvas.getContext('2d', { willReadFrequently: false });

  hiddenVideo = document.createElement('video');
  hiddenVideo.srcObject = localStream;
  hiddenVideo.muted = true;
  hiddenVideo.playsInline = true;
  await hiddenVideo.play();

  // Show mirrored preview
  const previewStream = mirrorCanvas.captureStream(30);
  localVideo.srcObject = previewStream;

  // Connect WebSocket
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = async () => {
    console.log('[Stream] WebSocket connected');
    setStatus('active');
    updateStreamButton(true);
    streamBtn.disabled = false;
    framesSent = 0;
    bytesSent = 0;
    startStats();

    // Try WebCodecs H.264 encoding (hardware accelerated)
    if (webCodecsAvailable()) {
      try {
        await initWebCodecsEncoder(sendW, sendH);
        useWebCodecs = true;
        console.log('[Stream] Using WebCodecs H.264 encoder (hardware accelerated)');
        ws.send(JSON.stringify({ type: 'codec', codec: 'h264', width: sendW, height: sendH }));
      } catch (e) {
        console.warn('[Stream] WebCodecs init failed, using JPEG fallback:', e.message);
        useWebCodecs = false;
        ws.send(JSON.stringify({ type: 'codec', codec: 'jpeg', width: sendW, height: sendH }));
      }
    } else {
      console.log('[Stream] WebCodecs not available, using JPEG');
      useWebCodecs = false;
      ws.send(JSON.stringify({ type: 'codec', codec: 'jpeg', width: sendW, height: sendH }));
    }

    startCaptureLoop();
  };

  ws.onmessage = (event) => {
    // Handle server messages (e.g., fallback request)
    if (typeof event.data === 'string') {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'fallback' && msg.codec === 'jpeg') {
          console.warn('[Stream] Server requested JPEG fallback');
          useWebCodecs = false;
          if (encoder) { try { encoder.close(); } catch(_){} encoder = null; }
        }
      } catch(_) {}
    }
  };

  ws.onclose = () => { if (isStreaming) { setStatus('error'); cleanUp(); } };
  ws.onerror = () => { setStatus('error'); cleanUp(); };
}

// ── WebCodecs H.264 Encoder ──────────────────────────────────────────────────
async function initWebCodecsEncoder(width, height) {
  const config = {
    codec: 'avc1.42001f',  // H.264 Baseline Level 3.1
    width: width,
    height: height,
    bitrate: 4_000_000,     // 4 Mbps
    framerate: TARGET_FPS,
    latencyMode: 'realtime',
    hardwareAcceleration: 'prefer-hardware',
    avc: { format: 'annexb' },  // Output Annex B format (start codes included)
  };

  // Check if config is supported
  const support = await VideoEncoder.isConfigSupported(config);
  if (!support.supported) {
    // Try without hardware preference
    config.hardwareAcceleration = 'no-preference';
    const support2 = await VideoEncoder.isConfigSupported(config);
    if (!support2.supported) throw new Error('H.264 encoding not supported');
  }

  encoder = new VideoEncoder({
    output: (chunk, metadata) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      
      // With annexb format, data already has start codes.
      // On keyframes, prepend SPS/PPS from decoderConfig description.
      if (chunk.type === 'key' && metadata && metadata.decoderConfig && metadata.decoderConfig.description) {
        const desc = new Uint8Array(metadata.decoderConfig.description);
        // Description in annexb format is already start-code prefixed SPS/PPS
        const sendBuf = new Uint8Array(desc.length + data.length);
        sendBuf.set(desc, 0);
        sendBuf.set(data, desc.length);
        ws.send(sendBuf.buffer);
        bytesSent += sendBuf.byteLength;
      } else {
        ws.send(data.buffer);
        bytesSent += data.byteLength;
      }
      framesSent++;
    },
    error: (e) => {
      console.error('[Encoder] Error:', e);
      useWebCodecs = false;
      if (encoder) { encoder.close(); encoder = null; }
    }
  });

  encoder.configure(config);
}

// ── Capture loop ─────────────────────────────────────────────────────────────
function startCaptureLoop() {
  let lastFrameTime = 0;
  const interval = 1000 / TARGET_FPS;

  // Use dual-canvas pipeline: while one blob is encoding, draw to the other
  let canvasA = sendCanvas;
  let ctxA = sendCtx;
  let canvasB = document.createElement('canvas');
  canvasB.width = sendCanvas.width;
  canvasB.height = sendCanvas.height;
  let ctxB = canvasB.getContext('2d', { willReadFrequently: false });
  let useA = true;
  let encoding = false;
  let frameIndex = 0;

  function capture(timestamp) {
    captureLoop = requestAnimationFrame(capture);
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!hiddenVideo || hiddenVideo.readyState < 2) return;
    if (timestamp - lastFrameTime < interval) return;
    lastFrameTime = timestamp;

    // Backpressure
    if (ws.bufferedAmount > 150 * 1024) return;

    // Draw mirrored preview
    mirrorCtx.save();
    mirrorCtx.translate(mirrorCanvas.width, 0);
    mirrorCtx.scale(-1, 1);
    mirrorCtx.drawImage(hiddenVideo, 0, 0, mirrorCanvas.width, mirrorCanvas.height);
    mirrorCtx.restore();

    // WebCodecs H.264 path
    if (useWebCodecs && encoder && encoder.state === 'configured') {
      // Draw to send canvas
      const canvas = useA ? canvasA : canvasB;
      const ctx = useA ? ctxA : ctxB;
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);
      ctx.restore();
      useA = !useA;

      const frame = new VideoFrame(canvas, { timestamp: frameIndex * interval * 1000 });
      const keyFrame = frameIndex % 60 === 0;
      encoder.encode(frame, { keyFrame });
      frame.close();
      frameIndex++;
      return;
    }

    // JPEG fallback path — skip if still encoding previous frame
    if (encoding) return;

    const canvas = useA ? canvasA : canvasB;
    const ctx = useA ? ctxA : ctxB;
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);
    ctx.restore();
    useA = !useA;

    encoding = true;
    canvas.toBlob((blob) => {
      encoding = false;
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
      const reader = new FileReader();
      reader.onload = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(reader.result);
          framesSent++;
          bytesSent += reader.result.byteLength;
        }
      };
      reader.readAsArrayBuffer(blob);
    }, 'image/jpeg', 0.70);
  }

  captureLoop = requestAnimationFrame(capture);
}

// ── Cleanup ──────────────────────────────────────────────────────────────────
function cleanUp() {
  stopStats();
  if (captureLoop) { cancelAnimationFrame(captureLoop); captureLoop = null; }
  pendingEncode = false;

  if (encoder) {
    try { encoder.close(); } catch(_) {}
    encoder = null;
  }
  useWebCodecs = false;

  if (ws) { ws.close(); ws = null; }

  mirrorCanvas = null; mirrorCtx = null;
  sendCanvas = null; sendCtx = null;
  hiddenVideo = null;

  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  localVideo.srcObject = null;
  localVideo.classList.remove('visible');
  videoOverlay.classList.remove('hidden');

  updateStreamButton(false);
  streamBtn.disabled = false;
  setStatus('idle');
}

// ── Init ─────────────────────────────────────────────────────────────────────
(async () => { await enumerateCameras(); setStatus('idle'); })();
