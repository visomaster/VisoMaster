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

// WebSocket fallback state
let ws = null;
let wsFrameLoop = null;
let useWebSocket = false;
let wsFps = 0;
let wsLastFpsTime = 0;
let wsFrameCount = 0;
let wsBytesSent = 0;
let wsLastBytesTime = 0;
let wsLastBytes = 0;

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

// ── Stats collection (WebRTC mode) ──────────────────────────────────────────
function startStatsCollection() {
  prevBytesSent = 0;
  prevTimestamp = 0;
  lastFpsTime = performance.now();
  lastFrameCount = 0;

  statsInterval = setInterval(async () => {
    if (useWebSocket) {
      // WebSocket stats
      const now = performance.now();
      const elapsed = (now - wsLastFpsTime) / 1000;
      if (elapsed >= 1) {
        statFps.textContent = Math.round((wsFrameCount - lastFrameCount) / elapsed);
        lastFrameCount = wsFrameCount;
        wsLastFpsTime = now;
      }
      // Bitrate
      const byteElapsed = (now - wsLastBytesTime) / 1000;
      if (byteElapsed >= 1) {
        const bitrate = Math.round(((wsBytesSent - wsLastBytes) * 8) / byteElapsed / 1000);
        statBitrate.textContent = bitrate > 1000
          ? (bitrate / 1000).toFixed(1) + ' Mbps'
          : bitrate + ' kbps';
        wsLastBytes = wsBytesSent;
        wsLastBytesTime = now;
      }
      return;
    }

    if (!pc) return;

    try {
      const stats = await pc.getStats();
      stats.forEach(report => {
        if (report.type === 'outbound-rtp' && report.kind === 'video') {
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

          if (report.frameWidth && report.frameHeight) {
            statResolution.textContent = report.frameWidth + '×' + report.frameHeight;
          }
        }
      });

      if (performance.memory) {
        const rss = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(1);
        statRss.textContent = rss + ' MB';
      } else {
        statRss.textContent = '—';
      }
    } catch (e) { /* Stats not available */ }
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

// ── Camera constraints ───────────────────────────────────────────────────────
function getConstraints() {
  const [w, h] = resolutionSelect.value.split('x').map(Number);
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
let hiddenVideo = null;

async function _startStreaming() {
  streamBtn.disabled = true;
  setStatus('connecting');

  // Prevent double-invocation
  if (pc || ws) {
    _cleanUp();
    await new Promise(r => setTimeout(r, 300));
  }

  const constraints = getConstraints();

  try {
    localStream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    console.error('[Stream] Camera access failed:', err);
    setStatus('error');
    streamBtn.disabled = false;
    return;
  }

  // Show mirrored preview
  localVideo.classList.add('visible');
  videoOverlay.classList.add('hidden');

  // ── Create a canvas to horizontally flip the video ─────────────────────
  const videoTrack = localStream.getVideoTracks()[0];
  const settings = videoTrack.getSettings();
  const vw = settings.width || 1280;
  const vh = settings.height || 720;

  mirrorCanvas = document.createElement('canvas');
  mirrorCanvas.width = vw;
  mirrorCanvas.height = vh;
  mirrorCtx = mirrorCanvas.getContext('2d');

  hiddenVideo = document.createElement('video');
  hiddenVideo.srcObject = localStream;
  hiddenVideo.muted = true;
  hiddenVideo.playsInline = true;
  await hiddenVideo.play();

  // Draw loop: flip horizontally
  function drawMirrored() {
    if (!mirrorCanvas) return;
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

  // Show the flipped stream as local preview
  const mirroredStream = mirrorCanvas.captureStream(30);
  localVideo.srcObject = mirroredStream;

  // Try WebRTC first, fall back to WebSocket
  const webrtcSuccess = await _tryWebRTC(mirroredStream);
  if (!webrtcSuccess) {
    console.log('[Stream] WebRTC failed, using WebSocket fallback');
    _startWebSocketStream();
  }
}

// ── WebRTC attempt (with timeout) ────────────────────────────────────────────
async function _tryWebRTC(mirroredStream) {
  return new Promise(async (resolve) => {
    // Fetch TURN credentials from server (configured via environment variables)
    let iceServers = [{ urls: 'stun:stun.l.google.com:19302' }];
    let useRelay = false;
    try {
      const turnResp = await fetch('/turn-credentials');
      if (turnResp.ok) {
        const turnConfig = await turnResp.json();
        if (turnConfig.iceServers && turnConfig.iceServers.length > 0) {
          iceServers = turnConfig.iceServers;
          useRelay = true;
          console.log('[WebRTC] TURN servers configured from server');
        }
      }
    } catch (e) {
      console.warn('[WebRTC] Could not fetch TURN credentials, using STUN only');
    }

    console.log('[WebRTC] Attempting connection...');
    pc = new RTCPeerConnection({
      iceServers,
      iceTransportPolicy: useRelay ? 'relay' : 'all'
    });

    const mirroredTrack = mirroredStream.getVideoTracks()[0];
    pc.addTrack(mirroredTrack, mirroredStream);

    let resolved = false;
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        console.warn('[WebRTC] Connection timeout (10s), falling back to WebSocket');
        if (pc) { pc.close(); pc = null; }
        resolve(false);
      }
    }, 10000);

    pc.oniceconnectionstatechange = () => {
      const s = pc.iceConnectionState;
      console.log('[WebRTC] ICE state:', s);
      if (s === 'connected' || s === 'completed') {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          console.log('[WebRTC] Connected successfully!');
          setStatus('active');
          updateStreamButton(true);
          streamBtn.disabled = false;
          useWebSocket = false;
          startStatsCollection();
          resolve(true);
        }
      } else if (s === 'failed') {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          console.warn('[WebRTC] ICE failed');
          if (pc) { pc.close(); pc = null; }
          resolve(false);
        }
      }
    };

    // Create offer and exchange with server
    try {
      const offer = await pc.createOffer({ offerToReceiveVideo: false, offerToReceiveAudio: false });
      await pc.setLocalDescription(offer);

      const resp = await fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
      });
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      const answer = await resp.json();
      console.log('[WebRTC] Got answer, relay candidates:', answer.sdp.includes('typ relay'));
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
    } catch (err) {
      console.error('[WebRTC] Signaling error:', err);
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        if (pc) { pc.close(); pc = null; }
        resolve(false);
      }
    }
  });
}

// ── WebSocket streaming ──────────────────────────────────────────────────────
function _startWebSocketStream() {
  useWebSocket = true;
  wsFrameCount = 0;
  wsBytesSent = 0;
  wsLastFpsTime = performance.now();
  wsLastBytesTime = performance.now();
  wsLastBytes = 0;
  lastFrameCount = 0;

  // Build WebSocket URL from current page location
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${proto}//${location.host}/ws`;
  console.log('[WebSocket] Connecting to', wsUrl);

  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    console.log('[WebSocket] Connected — streaming frames');
    setStatus('active');
    updateStreamButton(true);
    streamBtn.disabled = false;
    startStatsCollection();

    // Update resolution stat
    if (mirrorCanvas) {
      statResolution.textContent = mirrorCanvas.width + '×' + mirrorCanvas.height;
    }

    // Start frame capture loop — target ~20 FPS for good balance of quality/bandwidth
    const targetFps = 20;
    const frameInterval = 1000 / targetFps;

    wsFrameLoop = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN || !mirrorCanvas) return;

      // Check backpressure — skip frame if buffer is too full (> 1MB queued)
      if (ws.bufferedAmount > 1024 * 1024) return;

      // Encode canvas as JPEG and send
      mirrorCanvas.toBlob((blob) => {
        if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
        blob.arrayBuffer().then((buf) => {
          ws.send(buf);
          wsFrameCount++;
          wsBytesSent += buf.byteLength;
        });
      }, 'image/jpeg', 0.75);  // 75% JPEG quality
    }, frameInterval);
  };

  ws.onclose = () => {
    console.log('[WebSocket] Connection closed');
    if (isStreaming) {
      setStatus('error');
      _cleanUp();
    }
  };

  ws.onerror = (err) => {
    console.error('[WebSocket] Error:', err);
    setStatus('error');
    _cleanUp();
  };
}

// ── Stop streaming ───────────────────────────────────────────────────────────
function _cleanUp() {
  stopStatsCollection();

  // Stop WebSocket
  if (wsFrameLoop) {
    clearInterval(wsFrameLoop);
    wsFrameLoop = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  useWebSocket = false;

  // Stop canvas mirror loop
  if (mirrorAnimId) {
    cancelAnimationFrame(mirrorAnimId);
    mirrorAnimId = null;
  }
  mirrorCanvas = null;
  mirrorCtx = null;
  hiddenVideo = null;

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
