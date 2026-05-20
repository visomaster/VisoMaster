/* VisoMaster WebRTC Client — app.js (Mobile-First Redesign) */
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
const statsBar         = document.getElementById('statsBar');
const panelToggle      = document.getElementById('panelToggle');
const panelBody        = document.getElementById('panelBody');
const controlsPanel    = document.getElementById('controlsPanel');

// Stats elements
const statFps          = document.getElementById('statFps');
const statBitrate      = document.getElementById('statBitrate');
const statRss          = document.getElementById('statRss');
const statResolution   = document.getElementById('statResolution');

let pc           = null;   // RTCPeerConnection
let localStream  = null;   // MediaStream from getUserMedia
let isStreaming  = false;
let statsInterval = null;
let prevBytesSent = 0;
let prevTimestamp = 0;
let frameCount   = 0;
let fpsInterval  = null;
let lastFpsTime  = 0;
let lastFrameCount = 0;

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

// ── Stats collection ─────────────────────────────────────────────────────────
function startStatsCollection() {
  prevBytesSent = 0;
  prevTimestamp = 0;
  lastFpsTime = performance.now();
  lastFrameCount = 0;

  statsInterval = setInterval(async () => {
    if (!pc) return;

    try {
      const stats = await pc.getStats();
      stats.forEach(report => {
        if (report.type === 'outbound-rtp' && report.kind === 'video') {
          // Bitrate calculation
          const now = report.timestamp;
          const bytes = report.bytesSent || 0;
          if (prevTimestamp > 0) {
            const timeDiff = (now - prevTimestamp) / 1000;
            const byteDiff = bytes - prevBytesSent;
            const bitrate = Math.round((byteDiff * 8) / timeDiff / 1000);
            statBitrate.textContent = bitrate > 1000
              ? (bitrate / 1000).toFixed(1) + ' Mbps'
              : bitrate + ' kbps';
          }
          prevBytesSent = bytes;
          prevTimestamp = now;

          // FPS from frames sent
          if (report.framesPerSecond !== undefined) {
            statFps.textContent = Math.round(report.framesPerSecond);
          } else if (report.framesSent !== undefined) {
            const nowMs = performance.now();
            const elapsed = (nowMs - lastFpsTime) / 1000;
            if (elapsed >= 1) {
              const fps = Math.round((report.framesSent - lastFrameCount) / elapsed);
              statFps.textContent = fps;
              lastFpsTime = nowMs;
              lastFrameCount = report.framesSent;
            }
          }

          // Resolution
          if (report.frameWidth && report.frameHeight) {
            statResolution.textContent = report.frameWidth + '×' + report.frameHeight;
          }
        }
      });

      // RSS (memory) — use performance.memory if available (Chrome)
      if (performance.memory) {
        const rss = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(1);
        statRss.textContent = rss + ' MB';
      } else {
        statRss.textContent = '—';
      }
    } catch (e) {
      // Stats not available
    }
  }, 1000);
}

function stopStatsCollection() {
  if (statsInterval) {
    clearInterval(statsInterval);
    statsInterval = null;
  }
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
  } catch (_) { /* ignore */ }

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

// ── Camera constraints — use native resolution, let device handle rotation ──
function getConstraints() {
  const [w, h] = resolutionSelect.value.split('x').map(Number);

  // Always request the larger dimension as width — the camera sensor is
  // physically landscape. The device/browser applies rotation metadata
  // automatically so the stream appears portrait when the phone is upright.
  // We do NOT swap width/height; that forces the camera into a cropped or
  // incompatible mode on most mobile devices.
  return {
    video: {
      deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined,
      width: { ideal: Math.max(w, h) },
      height: { ideal: Math.min(w, h) },
      facingMode: 'user'
    },
    audio: false,
  };
}

// ── Stream toggle ────────────────────────────────────────────────────────────
streamBtn.addEventListener('click', () => {
  if (isStreaming) {
    _cleanUp();
  } else {
    _startStreaming();
  }
});

let mirrorCanvas = null;
let mirrorCtx = null;
let mirrorAnimId = null;

async function _startStreaming() {
  streamBtn.disabled = true;
  setStatus('connecting');

  const constraints = getConstraints();

  try {
    localStream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    setStatus('error');
    streamBtn.disabled = false;
    return;
  }

  // Show mirrored preview (will be set to canvas stream after flip setup)
  localVideo.classList.add('visible');
  videoOverlay.classList.add('hidden');

  setStatus('connecting');

  // ── Create a canvas to horizontally flip the video before sending ──────
  const videoTrack = localStream.getVideoTracks()[0];
  const settings = videoTrack.getSettings();
  const vw = settings.width || 1280;
  const vh = settings.height || 720;

  mirrorCanvas = document.createElement('canvas');
  mirrorCanvas.width = vw;
  mirrorCanvas.height = vh;
  mirrorCtx = mirrorCanvas.getContext('2d');

  // Create a hidden video element to draw from
  const hiddenVideo = document.createElement('video');
  hiddenVideo.srcObject = localStream;
  hiddenVideo.muted = true;
  hiddenVideo.playsInline = true;
  await hiddenVideo.play();

  // Draw loop: flip horizontally and render to canvas
  function drawMirrored() {
    if (!mirrorCanvas) return;
    // Update canvas size if track resolution changed
    const curSettings = videoTrack.getSettings();
    if (curSettings.width && curSettings.width !== mirrorCanvas.width) {
      mirrorCanvas.width = curSettings.width;
    }
    if (curSettings.height && curSettings.height !== mirrorCanvas.height) {
      mirrorCanvas.height = curSettings.height;
    }

    mirrorCtx.save();
    mirrorCtx.translate(mirrorCanvas.width, 0);
    mirrorCtx.scale(-1, 1);
    mirrorCtx.drawImage(hiddenVideo, 0, 0, mirrorCanvas.width, mirrorCanvas.height);
    mirrorCtx.restore();

    mirrorAnimId = requestAnimationFrame(drawMirrored);
  }
  drawMirrored();

  // Capture the flipped stream from the canvas
  const mirroredStream = mirrorCanvas.captureStream(30);
  const mirroredTrack = mirroredStream.getVideoTracks()[0];

  // Show the flipped stream as the local preview — what you see is what the server gets
  localVideo.srcObject = mirroredStream;

  // Build RTCPeerConnection — send the mirrored track, not the raw camera
  pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
  pc.addTrack(mirroredTrack, mirroredStream);

  pc.oniceconnectionstatechange = () => {
    const s = pc.iceConnectionState;
    if (s === 'connected' || s === 'completed') {
      setStatus('active');
      updateStreamButton(true);
      streamBtn.disabled = false;
      startStatsCollection();
    } else if (s === 'disconnected' || s === 'failed' || s === 'closed') {
      setStatus('error');
      _cleanUp();
    }
  };

  // Create offer
  const offer = await pc.createOffer({ offerToReceiveVideo: false, offerToReceiveAudio: false });
  await pc.setLocalDescription(offer);

  try {
    const resp = await fetch('/offer', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
    const answer = await resp.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
  } catch (err) {
    setStatus('error');
    _cleanUp();
  }
}

// ── Stop streaming ───────────────────────────────────────────────────────────
function _cleanUp() {
  stopStatsCollection();

  // Stop canvas mirror loop
  if (mirrorAnimId) {
    cancelAnimationFrame(mirrorAnimId);
    mirrorAnimId = null;
  }
  mirrorCanvas = null;
  mirrorCtx = null;

  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
  if (pc) {
    pc.close();
    pc = null;
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
