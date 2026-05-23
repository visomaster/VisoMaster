/* VisoMaster Streaming Client — app.js */
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

// Stats elements
const statFps          = document.getElementById('statFps');
const statBitrate      = document.getElementById('statBitrate');
const statRss          = document.getElementById('statRss');
const statResolution   = document.getElementById('statResolution');

let localStream  = null;
let isStreaming  = false;
let ws = null;
let captureLoop = null;
let mirrorCanvas = null;
let mirrorCtx = null;
let hiddenVideo = null;
let sendCanvas = null;
let sendCtx = null;

// Stats
let statsInterval = null;
let framesSent = 0;
let bytesSent = 0;
let lastStatsTime = 0;
let lastStatsFrames = 0;
let lastStatsBytes = 0;

// ── Icons ────────────────────────────────────────────────────────────────────
const playIcon = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>`;
const stopIcon = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="3"/></svg>`;

// ── Panel toggle ─────────────────────────────────────────────────────────────
panelToggle.addEventListener('click', () => {
  panelBody.classList.toggle('collapsed');
  panelToggle.classList.toggle('active');
});

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(state) {
  statusDot.className = 'status-dot ' + state;
}

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
    statBitrate.textContent = bitrate > 1000
      ? (bitrate / 1000).toFixed(1) + ' Mbps'
      : bitrate + ' kbps';

    if (performance.memory) {
      statRss.textContent = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(0) + ' MB';
    }

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

// ── Enumerate cameras ────────────────────────────────────────────────────────
async function enumerateCameras() {
  try {
    const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
    tmp.getTracks().forEach(t => t.stop());
  } catch (_) {}

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(d => d.kind === 'videoinput');
  cameraSelect.innerHTML = '';
  if (cameras.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'No cameras found';
    cameraSelect.appendChild(opt);
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

// ── Camera constraints ───────────────────────────────────────────────────────
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
  if (isStreaming) {
    cleanUp();
  } else {
    startStreaming();
  }
});

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

  // Show preview
  localVideo.classList.add('visible');
  videoOverlay.classList.add('hidden');

  const videoTrack = localStream.getVideoTracks()[0];
  const settings = videoTrack.getSettings();
  const vw = settings.width || 1280;
  const vh = settings.height || 720;

  // Mirror canvas (for local preview — flipped horizontally)
  mirrorCanvas = document.createElement('canvas');
  mirrorCanvas.width = vw;
  mirrorCanvas.height = vh;
  mirrorCtx = mirrorCanvas.getContext('2d', { willReadFrequently: false });

  // Send canvas (for encoding — also flipped to match preview)
  // Use a smaller resolution for the wire to boost FPS
  const sendW = Math.min(vw, 960);
  const sendH = Math.round(sendW * vh / vw);
  sendCanvas = document.createElement('canvas');
  sendCanvas.width = sendW;
  sendCanvas.height = sendH;
  sendCtx = sendCanvas.getContext('2d', { willReadFrequently: false });

  statResolution.textContent = sendW + '×' + sendH;

  hiddenVideo = document.createElement('video');
  hiddenVideo.srcObject = localStream;
  hiddenVideo.muted = true;
  hiddenVideo.playsInline = true;
  await hiddenVideo.play();

  // Show mirrored preview
  const mirroredStream = mirrorCanvas.captureStream(30);
  localVideo.srcObject = mirroredStream;

  // Connect WebSocket
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${proto}//${location.host}/ws`;
  console.log('[Stream] Connecting to', wsUrl);

  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    console.log('[Stream] Connected');
    setStatus('active');
    updateStreamButton(true);
    streamBtn.disabled = false;
    framesSent = 0;
    bytesSent = 0;
    startStats();
    startCaptureLoop();
  };

  ws.onclose = () => {
    console.log('[Stream] Disconnected');
    if (isStreaming) { setStatus('error'); cleanUp(); }
  };

  ws.onerror = () => {
    setStatus('error');
    cleanUp();
  };
}

// ── High-performance capture loop ───────────────────────────────────────────
// Uses requestAnimationFrame for smooth timing + toBlob with callback pipelining
// to maintain 30 FPS without blocking.

let pendingBlob = false;  // Prevents overlapping toBlob calls

function startCaptureLoop() {
  let lastFrameTime = 0;
  const targetInterval = 1000 / 30;  // 30 FPS target

  function captureFrame(timestamp) {
    captureLoop = requestAnimationFrame(captureFrame);

    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!hiddenVideo || hiddenVideo.readyState < 2) return;

    // Throttle to target FPS
    if (timestamp - lastFrameTime < targetInterval) return;
    lastFrameTime = timestamp;

    // Skip if previous frame is still encoding or buffer is full
    if (pendingBlob) return;
    if (ws.bufferedAmount > 200 * 1024) return;  // 200KB backpressure limit

    // Draw mirrored frame to preview canvas
    mirrorCtx.save();
    mirrorCtx.translate(mirrorCanvas.width, 0);
    mirrorCtx.scale(-1, 1);
    mirrorCtx.drawImage(hiddenVideo, 0, 0, mirrorCanvas.width, mirrorCanvas.height);
    mirrorCtx.restore();

    // Draw mirrored frame to send canvas (potentially downscaled)
    sendCtx.save();
    sendCtx.translate(sendCanvas.width, 0);
    sendCtx.scale(-1, 1);
    sendCtx.drawImage(hiddenVideo, 0, 0, sendCanvas.width, sendCanvas.height);
    sendCtx.restore();

    // Encode as JPEG and send — use callback to pipeline
    pendingBlob = true;
    sendCanvas.toBlob((blob) => {
      pendingBlob = false;
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
      blob.arrayBuffer().then((buf) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(buf);
          framesSent++;
          bytesSent += buf.byteLength;
        }
      });
    }, 'image/jpeg', 0.80);  // 80% quality — good balance of size vs quality
  }

  captureLoop = requestAnimationFrame(captureFrame);
}

// ── Stop streaming ───────────────────────────────────────────────────────────
function cleanUp() {
  stopStats();

  if (captureLoop) { cancelAnimationFrame(captureLoop); captureLoop = null; }
  pendingBlob = false;

  if (ws) { ws.close(); ws = null; }

  mirrorCanvas = null;
  mirrorCtx = null;
  sendCanvas = null;
  sendCtx = null;
  hiddenVideo = null;

  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }

  localVideo.srcObject = null;
  localVideo.classList.remove('visible');
  videoOverlay.classList.remove('hidden');

  updateStreamButton(false);
  streamBtn.disabled = false;
  setStatus('idle');
}

// ── Init ─────────────────────────────────────────────────────────────────────
(async () => {
  await enumerateCameras();
  setStatus('idle');
})();
