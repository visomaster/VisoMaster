/* WebStreamer — High-Performance WebSocket Streaming Client */
// @ts-check
'use strict';

// ── DOM refs ────────────────────────────────────────────────────────────────
const localVideo       = /** @type {HTMLVideoElement}  */ (document.getElementById('localVideo'));
const videoOverlay     = /** @type {HTMLElement}       */ (document.getElementById('videoOverlay'));
const videoContainer   = /** @type {HTMLElement}       */ (document.getElementById('videoContainer'));
const streamBtn        = /** @type {HTMLButtonElement} */ (document.getElementById('streamBtn'));
const streamBtnIcon    = /** @type {HTMLElement}       */ (document.getElementById('streamBtnIcon'));
const streamBtnText    = /** @type {HTMLElement}       */ (document.getElementById('streamBtnText'));
const cameraSelect     = /** @type {HTMLSelectElement} */ (document.getElementById('cameraSelect'));
const resolutionSelect = /** @type {HTMLSelectElement} */ (document.getElementById('resolutionSelect'));
const codecSelect      = /** @type {HTMLSelectElement} */ (document.getElementById('codecSelect'));
const qualityRange     = /** @type {HTMLInputElement}  */ (document.getElementById('qualityRange'));
const qualityLabel     = /** @type {HTMLElement}       */ (document.getElementById('qualityLabel'));
const statusDot        = /** @type {HTMLElement}       */ (document.getElementById('statusDot'));
const panelToggle      = /** @type {HTMLElement}       */ (document.getElementById('panelToggle'));
const panelBody        = /** @type {HTMLElement}       */ (document.getElementById('panelBody'));
const themeBtn         = /** @type {HTMLElement}       */ (document.getElementById('themeBtn'));

const statFps          = /** @type {HTMLElement} */ (document.getElementById('statFps'));
const statBitrate      = /** @type {HTMLElement} */ (document.getElementById('statBitrate'));
const statResolution   = /** @type {HTMLElement} */ (document.getElementById('statResolution'));
const statCodec        = /** @type {HTMLElement} */ (document.getElementById('statCodec'));

/** @type {MediaStream | null} */
let localStream = null;
/** @type {boolean} */
let isStreaming = false;
/** @type {WebSocket | null} */
let ws = null;
/** @type {number | null} */
let captureLoop = null;
/** @type {HTMLVideoElement | null} */
let hiddenVideo = null;

// Encoding state
/** @type {VideoEncoder | null} */
let encoder = null;
/** @type {boolean} */
let useWebCodecs = false;
/** @type {HTMLCanvasElement | null} */
let sendCanvas = null;
/** @type {CanvasRenderingContext2D | null} */
let sendCtx = null;
/** @type {boolean} */
let pendingEncode = false;

// Mirror preview
/** @type {HTMLCanvasElement | null} */
let mirrorCanvas = null;
/** @type {CanvasRenderingContext2D | null} */
let mirrorCtx = null;

// Stats
/** @type {ReturnType<typeof setInterval> | null} */
let statsInterval = null;
/** @type {number} */
let framesSent = 0;
/** @type {number} */
let bytesSent = 0;
/** @type {number} */
let lastStatsTime = 0;
/** @type {number} */
let lastStatsFrames = 0;
/** @type {number} */
let lastStatsBytes = 0;

/** @type {30} */
const TARGET_FPS = 30;

// ── Icons ────────────────────────────────────────────────────────────────────
const playIcon = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>`;
const stopIcon = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="3"/></svg>`;

// ── Panel collapse/expand ────────────────────────────────────────────────────
panelToggle.addEventListener('click', () => {
  const collapsed = panelBody.classList.toggle('collapsed');
  panelToggle.classList.toggle('collapsed', collapsed);
});

// ── Theme toggle ─────────────────────────────────────────────────────────────
function applyTheme(/** @type {string} */ theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('webstreamer_theme', theme);
}

themeBtn.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'light' ? 'dark' : 'light');
});

// Restore saved theme
applyTheme(localStorage.getItem('webstreamer_theme') || 'dark');

function setStatus(state) { statusDot.className = 'status-dot ' + state; }

/**
 * @param {'idle' | 'connecting' | 'streaming' | boolean} state
 */
function updateStreamButton(state) {
  // state: 'idle' | 'connecting' | 'streaming'
  if (state === 'streaming' || state === true) {
    isStreaming = true;
    streamBtn.classList.add('streaming');
    streamBtn.classList.remove('connecting');
    streamBtnIcon.innerHTML = stopIcon;
    streamBtnText.textContent = 'Stop Streaming';
    streamBtn.disabled = false;
  } else if (state === 'connecting') {
    isStreaming = false;
    streamBtn.classList.remove('streaming');
    streamBtn.classList.add('connecting');
    streamBtnIcon.innerHTML = playIcon;
    streamBtnText.textContent = 'Connecting...';
    streamBtn.disabled = true;
  } else {
    // idle / false
    isStreaming = false;
    streamBtn.classList.remove('streaming');
    streamBtn.classList.remove('connecting');
    streamBtnIcon.innerHTML = playIcon;
    streamBtnText.textContent = 'Start Streaming';
    streamBtn.disabled = false;
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
    statFps.textContent = String(fps);
    statBitrate.textContent = bitrate > 1000 ? (bitrate / 1000).toFixed(1) + ' Mbps' : bitrate + ' kbps';
    lastStatsTime = now;
    lastStatsFrames = framesSent;
    lastStatsBytes = bytesSent;
  }, 1000);
}

function stopStats() {
  if (statsInterval) { clearInterval(statsInterval); statsInterval = null; }
  statFps.textContent = '0';
  statBitrate.textContent = '0 kbps';
  statResolution.textContent = '—';
  statCodec.textContent = '—';
  statCodec.className = 'stat-value stat-codec';
}

// ── Cameras ──────────────────────────────────────────────────────────────────
/**
 * Clean up a raw camera label for display.
 * Strips USB IDs like "(30c9:0069)" and trims to 32 chars.
 * @param {string} label
 * @param {number} index
 * @returns {string}
 */
function formatCameraLabel(label, index) {
  if (!label) return `Camera ${index + 1}`;
  // Remove USB vendor:product IDs in parentheses e.g. "(30c9:0069)"
  let clean = label.replace(/\s*\([0-9a-f]{4}:[0-9a-f]{4}\)/gi, '').trim();
  // Truncate if still long
  if (clean.length > 32) clean = clean.slice(0, 30) + '…';
  return clean || `Camera ${index + 1}`;
}

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
    opt.textContent = formatCameraLabel(cam.label, i);
    cameraSelect.appendChild(opt);
  });
}

function getConstraints() {
  // Request the selected resolution as an ideal from the camera.
  // Using 'ideal' (not 'exact') means the browser will get as close as
  // possible without throwing an error if the camera can't deliver it.
  // We still downscale on the canvas if the camera overshoots.
  const [selW, selH] = resolutionSelect.value.split('x').map(Number);
  return {
    video: {
      deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined,
      frameRate: { ideal: 30 },
      width:  { ideal: selW },
      height: { ideal: selH },
    },
    audio: false,
  };
}

// ── Settings persistence ─────────────────────────────────────────────────────
const SETTINGS_KEY = 'webstreamer_settings';

/** Default quality per codec (0–1). */
const CODEC_DEFAULT_QUALITY = { auto: 0.85, h264: 0.85, jpeg: 0.85 };

/** @returns {number} quality as 0–1 float */
function getQuality() { return parseInt(qualityRange.value, 10) / 100; }

/** @param {number} q quality 0–1 */
function setQuality(q) {
  const pct = Math.round(q * 100);
  qualityRange.value = String(pct);
  qualityLabel.textContent = pct + '%';
}

// When codec changes, reset quality to the per-codec default
codecSelect.addEventListener('change', () => {
  const def = CODEC_DEFAULT_QUALITY[codecSelect.value] ?? 0.75;
  setQuality(def);
  // Show/hide quality row — not relevant for H.264 (bitrate-controlled by encoder)
  const isImageCodec = codecSelect.value !== 'h264';
  document.getElementById('qualityRow').style.display = isImageCodec ? '' : 'none';
  saveSettings();
});

qualityRange.addEventListener('input', () => {
  qualityLabel.textContent = qualityRange.value + '%';
});
qualityRange.addEventListener('change', saveSettings);

function saveSettings() {
  const settings = {
    cameraLabel:  cameraSelect.selectedOptions[0]?.text ?? '',
    cameraId:     cameraSelect.value,
    resolution:   resolutionSelect.value,
    codec:        codecSelect.value,
    quality:      qualityRange.value,
  };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

/** @param {HTMLSelectElement} select @param {string} value */
function restoreSelect(select, value) {
  const opt = Array.from(select.options).find(o => o.value === value);
  if (opt) select.value = value;
}

function restoreSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    if (s.resolution) restoreSelect(resolutionSelect, s.resolution);
    if (s.codec)      restoreSelect(codecSelect, s.codec);
    // Restore quality, or apply per-codec default if not saved
    const q = s.quality
      ? parseInt(s.quality, 10) / 100
      : (CODEC_DEFAULT_QUALITY[s.codec] ?? 0.75);
    setQuality(q);
    // Hide quality row for H.264
    const isImageCodec = codecSelect.value !== 'h264';
    document.getElementById('qualityRow').style.display = isImageCodec ? '' : 'none';
    // Camera is restored after enumeration in restoreCameraSelection()
  } catch (_) {}
}

/** Called after enumerateCameras() so options are populated. */
function restoreCameraSelection() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    // Prefer exact deviceId match, fall back to label match
    const byId    = s.cameraId    && Array.from(cameraSelect.options).find(o => o.value === s.cameraId);
    const byLabel = s.cameraLabel && Array.from(cameraSelect.options).find(o => o.text  === s.cameraLabel);
    const match   = byId || byLabel;
    if (match) cameraSelect.value = match.value;
  } catch (_) {}
}

// Persist on every change
cameraSelect.addEventListener('change',     saveSettings);
resolutionSelect.addEventListener('change', saveSettings);
codecSelect.addEventListener('change',      saveSettings);

// ── Stream toggle ────────────────────────────────────────────────────────────
streamBtn.addEventListener('click', () => {
  if (isStreaming) cleanUp(); else startStreaming();
});
// H.264 works on iPhone/Safari but may fail on some Windows browsers.
// Auto-detects and falls back to JPEG if encoding fails.
/** @returns {boolean} */
function webCodecsAvailable() {
  return typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined';
}

/**
 * Update the codec badge in the stats bar.
 * @param {'h264' | 'jpeg'} codec
 */
function setCodecStat(codec) {
  const labels = { h264: 'H.264', jpeg: 'JPEG' };
  statCodec.textContent = labels[codec] ?? codec.toUpperCase();
  statCodec.className   = 'stat-value stat-codec stat-codec--' + codec;
}
// ── AVCC to Annex B conversion ───────────────────────────────────────────────
// WebCodecs outputs H.264 in AVCC format (length-prefixed NALUs).
// PyAV/FFmpeg expects Annex B format (start code prefixed: 00 00 00 01).
/**
 * @param {Uint8Array} avccData
 * @returns {Uint8Array}
 */
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
  setStatus('connecting');
  updateStreamButton('connecting');

  if (ws) { cleanUp(); await new Promise(r => setTimeout(r, 200)); }

  try {
    localStream = await navigator.mediaDevices.getUserMedia(getConstraints());
  } catch (err) {
    console.error('[Stream] Camera error:', err);
    setStatus('error');
    updateStreamButton('idle');
    return;
  }

  localVideo.classList.add('visible');
  videoOverlay.classList.add('hidden');

  // Set up the hidden video element FIRST so we can read post-rotation
  // dimensions from videoWidth/videoHeight. Browsers (especially iOS Safari)
  // expose track settings as the sensor's native landscape resolution, but
  // <video>.videoWidth/Height reflect the user-visible orientation after the
  // browser applies any necessary rotation. Using those means we never need
  // to rotate the canvas manually.
  hiddenVideo = document.createElement('video');
  hiddenVideo.srcObject = localStream;
  hiddenVideo.muted = true;
  hiddenVideo.playsInline = true;
  await hiddenVideo.play();

  // Wait for metadata so videoWidth/videoHeight are populated
  if (!hiddenVideo.videoWidth || !hiddenVideo.videoHeight) {
    await new Promise((resolve) => {
      const onReady = () => { hiddenVideo.removeEventListener('loadedmetadata', onReady); resolve(); };
      hiddenVideo.addEventListener('loadedmetadata', onReady);
    });
  }

  const nativeW = hiddenVideo.videoWidth  || localStream.getVideoTracks()[0].getSettings().width  || 1280;
  const nativeH = hiddenVideo.videoHeight || localStream.getVideoTracks()[0].getSettings().height || 720;

  // The resolution dropdown is a max-dimension cap applied via canvas
  // downscaling. We never crop — the camera's full FOV is preserved by
  // scaling proportionally so the longer side fits within maxDim.
  const [selW, selH] = resolutionSelect.value.split('x').map(Number);
  const maxDim = Math.max(selW, selH);
  let sendW, sendH;
  if (Math.max(nativeW, nativeH) > maxDim) {
    const scale = maxDim / Math.max(nativeW, nativeH);
    sendW = Math.round(nativeW * scale);
    sendH = Math.round(nativeH * scale);
  } else {
    sendW = nativeW;
    sendH = nativeH;
  }
  // H.264 encoders require even dimensions
  sendW -= sendW % 2;
  sendH -= sendH % 2;
  statResolution.textContent = sendW + '×' + sendH;

  // Mirror canvas for local preview (matches device orientation)
  mirrorCanvas = document.createElement('canvas');
  mirrorCanvas.width = sendW;
  mirrorCanvas.height = sendH;
  mirrorCtx = mirrorCanvas.getContext('2d');

  // Send canvas for encoding (same orientation as preview)
  sendCanvas = document.createElement('canvas');
  sendCanvas.width = sendW;
  sendCanvas.height = sendH;
  sendCtx = sendCanvas.getContext('2d', { willReadFrequently: false });

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
    updateStreamButton('streaming');
    framesSent = 0;
    bytesSent = 0;
    startStats();

    // Codec selection: honour the dropdown, fall back gracefully.
    // 'auto'  → try H.264 WebCodecs, fall back to JPEG if unavailable
    // 'h264'  → force H.264; throw (and fall back to JPEG) if unsupported
    // 'jpeg'  → always JPEG, skip WebCodecs entirely
    const codecPref = codecSelect.value; // 'auto' | 'h264' | 'jpeg'
    const tryH264 = codecPref !== 'jpeg' && webCodecsAvailable();

    if (tryH264) {
      try {
        await initWebCodecsEncoder(sendW, sendH);
        useWebCodecs = true;
        console.log('[Stream] Using WebCodecs H.264 encoder');
        setCodecStat('h264');
        ws.send(JSON.stringify({ type: 'codec', codec: 'h264', width: sendW, height: sendH }));
      } catch (e) {
        if (codecPref === 'h264') {
          console.warn('[Stream] H.264 forced but init failed — falling back to JPEG:', /** @type {Error} */ (e).message);
        } else {
          console.warn('[Stream] WebCodecs init failed, using JPEG fallback:', /** @type {Error} */ (e).message);
        }
        useWebCodecs = false;
        setCodecStat('jpeg');
        ws.send(JSON.stringify({ type: 'codec', codec: 'jpeg', width: sendW, height: sendH }));
      }
    } else {
      const reason = codecPref === 'jpeg' ? 'JPEG selected' : 'WebCodecs not available';
      console.log(`[Stream] Using JPEG (${reason})`);
      useWebCodecs = false;
      setCodecStat('jpeg');
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
          setCodecStat('jpeg');
          if (encoder) { try { encoder.close(); } catch(_){} encoder = null; }
        }      } catch(_) {}
    }
  };

  ws.onclose = () => { if (isStreaming) { setStatus('error'); cleanUp(); } };
  ws.onerror = () => { setStatus('error'); cleanUp(); };
}

// ── WebCodecs H.264 Encoder ──────────────────────────────────────────────────
/**
 * @param {number} width
 * @param {number} height
 * @returns {Promise<void>}
 */
async function initWebCodecsEncoder(width, height) {
  /** @type {VideoEncoderConfig} */
  const config = {
    codec: 'avc1.42001f',  // H.264 Baseline Level 3.1
    width: width,
    height: height,
    bitrate: 6_000_000,     // 4 Mbps
    framerate: TARGET_FPS,
    latencyMode: 'quality',
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
        const desc = new Uint8Array(/** @type {ArrayBuffer} */ (metadata.decoderConfig.description));
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

// ── Draw video to canvas with selfie mirror ─────────────────────────────────
// The <video> element automatically reports the correctly oriented dimensions
// (videoWidth/videoHeight) regardless of device rotation, so drawImage with
// the canvas size matching those gives us the right aspect ratio without any
// manual rotation. We only mirror horizontally for the selfie preview.
/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLVideoElement} video
 */
function drawVideoToCanvas(ctx, canvas, video) {
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();
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
  let frameIndex = 0;

  function capture(timestamp) {
    captureLoop = requestAnimationFrame(capture);
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!hiddenVideo || hiddenVideo.readyState < 2) return;
    if (timestamp - lastFrameTime < interval) return;
    lastFrameTime = timestamp;

    // Backpressure — JPEG frames at 0.85 quality can be ~80–150 KB each,
    // so allow a larger buffer before dropping frames.
    if (ws.bufferedAmount > 512 * 1024) return;

    // Draw mirrored preview (with rotation if needed)
    drawVideoToCanvas(mirrorCtx, mirrorCanvas, hiddenVideo);

    // WebCodecs H.264 path
    if (useWebCodecs && encoder && encoder.state === 'configured') {
      // Draw to send canvas
      const canvas = useA ? canvasA : canvasB;
      const ctx = useA ? ctxA : ctxB;
      drawVideoToCanvas(ctx, canvas, hiddenVideo);
      useA = !useA;

      const frame = new VideoFrame(canvas, { timestamp: frameIndex * interval * 1000 });
      const keyFrame = frameIndex % 60 === 0;
      encoder.encode(frame, { keyFrame });
      frame.close();
      frameIndex++;
      return;
    }

    // Image (JPEG/WebP) path — fire-and-forget per frame; backpressure check above
    // prevents flooding the socket. No encoding flag needed since toBlob
    // callbacks are independent and the dual-canvas pipeline keeps draws safe.
    const canvas = useA ? canvasA : canvasB;
    const ctx = useA ? ctxA : ctxB;
    drawVideoToCanvas(ctx, canvas, hiddenVideo);
    useA = !useA;

    const mimeType = 'image/jpeg';
    canvas.toBlob((blob) => {
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
      blob.arrayBuffer().then((buf) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(buf);
        framesSent++;
        bytesSent += buf.byteLength;
      });
    }, mimeType, getQuality());
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

  updateStreamButton('idle');
  setStatus('idle');
}

// ── Init ─────────────────────────────────────────────────────────────────────
(async () => {
  restoreSettings();          // restore resolution + codec before paint
  await enumerateCameras();
  restoreCameraSelection();   // restore camera after options are populated
  setStatus('idle');
})();

// ── Live reload (development) ────────────────────────────────────────────────
(function () {
  /** @param {string} msg */
  function showReloadToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'reload-toast';
    toast.textContent = msg;
    document.body.appendChild(toast);
    // Trigger transition on next frame
    requestAnimationFrame(() => {
      requestAnimationFrame(() => toast.classList.add('visible'));
    });
    setTimeout(() => {
      toast.classList.remove('visible');
      toast.addEventListener('transitionend', () => toast.remove(), { once: true });
    }, 2000);
  }

  const es = new EventSource('/livereload');
  es.onmessage = (e) => {
    if (e.data === 'reload') {
      showReloadToast('↻  Reloading…');
      setTimeout(() => location.reload(), 300);
    }
  };
  es.onerror = () => {
    // SSE connection lost — silently retry (browser handles reconnect automatically)
  };
})();
