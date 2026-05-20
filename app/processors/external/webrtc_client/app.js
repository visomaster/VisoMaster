/* VisoMaster WebRTC Client — app.js */
'use strict';

// ── DOM refs ────────────────────────────────────────────────────────────────
const localVideo       = document.getElementById('localVideo');
const videoOverlay     = document.getElementById('videoOverlay');
const startBtn         = document.getElementById('startBtn');
const stopBtn          = document.getElementById('stopBtn');
const cameraSelect     = document.getElementById('cameraSelect');
const resolutionSelect = document.getElementById('resolutionSelect');
const statusBar        = document.getElementById('statusBar');
const statusText       = document.getElementById('statusText');

let pc           = null;   // RTCPeerConnection
let localStream  = null;   // MediaStream from getUserMedia

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(state, msg) {
  statusBar.className = 'status ' + state;
  statusText.textContent = msg;
}

// ── Enumerate cameras ────────────────────────────────────────────────────────
async function enumerateCameras() {
  try {
    // A brief getUserMedia is needed on some browsers before labels are revealed
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
    startBtn.disabled = true;
    return;
  }
  cameras.forEach((cam, i) => {
    const opt = document.createElement('option');
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Camera ${i + 1}`;
    cameraSelect.appendChild(opt);
  });
}

// ── Start streaming ──────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  setStatus('connecting', 'Requesting camera access…');

  const [w, h] = resolutionSelect.value.split('x').map(Number);
  const constraints = {
    video: { deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined,
             width: { ideal: w }, height: { ideal: h } },
    audio: false,
  };

  try {
    localStream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    setStatus('error', `Camera error: ${err.message}`);
    startBtn.disabled = false;
    return;
  }

  // Show preview
  localVideo.srcObject = localStream;
  localVideo.classList.add('visible');
  videoOverlay.classList.add('hidden');

  setStatus('connecting', 'Connecting to VisoMaster…');

  // Build RTCPeerConnection
  pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
  localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

  pc.oniceconnectionstatechange = () => {
    const s = pc.iceConnectionState;
    if (s === 'connected' || s === 'completed') {
      setStatus('active', 'Streaming to VisoMaster ✓');
      stopBtn.disabled  = false;
    } else if (s === 'disconnected' || s === 'failed' || s === 'closed') {
      setStatus('error', 'Connection lost — ' + s);
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
    setStatus('error', `Offer failed: ${err.message}`);
    _cleanUp();
  }
});

// ── Stop streaming ───────────────────────────────────────────────────────────
stopBtn.addEventListener('click', _cleanUp);

function _cleanUp() {
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

  startBtn.disabled = false;
  stopBtn.disabled  = true;
  setStatus('idle', 'Stopped — press Start to stream again');
}

// ── Init ─────────────────────────────────────────────────────────────────────
(async () => {
  await enumerateCameras();
  setStatus('idle', 'Idle — select a camera and press Start');
})();
