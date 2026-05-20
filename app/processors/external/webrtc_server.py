"""
WebRTC Server for VisoMaster
Serves a browser-facing WebRTC page over HTTPS (port 9090) and HTTP (port 9091).
Each incoming video track frame is written into a named shared memory block
so that the main VisoMaster process can poll it without requiring IPC queues.

Shared memory layout (fixed at 1920×1080×3 = 6 220 800 bytes):
  Bytes 0-3   : frame counter (uint32, little-endian) — incremented every write
  Bytes 4-7   : frame width  (uint32, little-endian)
  Bytes 8-11  : frame height (uint32, little-endian)
  Bytes 12-N  : raw BGR frame data (width * height * 3 bytes)

This module is designed to be launched as a separate subprocess via
multiprocessing.Process so it does not block the Qt event loop.
"""

import asyncio
import json
import logging
import os
import ssl
import fractions
from pathlib import Path

import numpy as np

# aiortc / aiohttp
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaBlackhole
from av import VideoFrame

# multiprocessing shared memory
from multiprocessing.shared_memory import SharedMemory
import struct

# Suppress noisy ICE binding errors for link-local addresses
logging.getLogger("aioice.ice").setLevel(logging.WARNING)
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

# ── Constants ───────────────────────────────────────────────────────────────

SHM_NAME         = "visomaster_webrtc_frame"
SHM_MAX_WIDTH    = 1920
SHM_MAX_HEIGHT   = 1080
SHM_FRAME_BYTES  = SHM_MAX_WIDTH * SHM_MAX_HEIGHT * 3          # BGR
SHM_HEADER_BYTES = 12                                           # counter + width + height
SHM_TOTAL_BYTES  = SHM_HEADER_BYTES + SHM_FRAME_BYTES

# Folder containing index.html, app.js, style.css
_THIS_DIR    = Path(__file__).parent
CLIENT_DIR   = _THIS_DIR / "webrtc_client"

# ── Shared-memory helpers ────────────────────────────────────────────────────

def _create_shm() -> SharedMemory:
    """Create (or re-attach to) the named shared memory block."""
    try:
        shm = SharedMemory(name=SHM_NAME, create=True, size=SHM_TOTAL_BYTES)
    except FileExistsError:
        shm = SharedMemory(name=SHM_NAME, create=False, size=SHM_TOTAL_BYTES)
    # Zero the header so consumers can detect "no frame yet"
    struct.pack_into("<III", shm.buf, 0, 0, 0, 0)
    return shm


def _write_frame(shm: SharedMemory, frame_bgr: np.ndarray):
    """Write one BGR frame into shared memory and increment the counter."""
    h, w = frame_bgr.shape[:2]

    # Clamp to max dimensions
    if h > SHM_MAX_HEIGHT or w > SHM_MAX_WIDTH:
        scale = min(SHM_MAX_WIDTH / w, SHM_MAX_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        import cv2
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h))
        h, w = new_h, new_w

    # Read current counter, increment
    counter = struct.unpack_from("<I", shm.buf, 0)[0]
    counter = (counter + 1) & 0xFFFFFFFF

    # Write header
    struct.pack_into("<III", shm.buf, 0, counter, w, h)

    # Write pixel data
    pixel_bytes = frame_bgr.tobytes()
    shm.buf[SHM_HEADER_BYTES: SHM_HEADER_BYTES + len(pixel_bytes)] = pixel_bytes


# ── WebRTC track handler ─────────────────────────────────────────────────────

class VideoStreamTrack:
    """Consumes an incoming RemoteStreamTrack and writes frames to shared memory."""

    def __init__(self, track, shm: SharedMemory):
        self._track = track
        self._shm   = shm
        self._task  = None

    def start(self):
        self._task = asyncio.ensure_future(self._run())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def _run(self):
        import cv2
        while True:
            try:
                frame: VideoFrame = await self._track.recv()
            except Exception:
                break
            # Convert to numpy BGR
            img = frame.to_ndarray(format="bgr24")
            _write_frame(self._shm, img)


# ── TURN server configuration ─────────────────────────────────────────────────
# Set these environment variables to enable TURN relay (required for RunPod/proxy setups):
#   TURN_URL      - e.g. "turn:your-turn-server.com:3478" or "turns:your-turn-server.com:5349"
#   TURN_USERNAME - TURN username
#   TURN_PASSWORD - TURN credential/password
#
# You can also use multiple TURN servers by separating URLs with commas:
#   TURN_URL="turn:server1.com:3478,turns:server1.com:5349"

def _get_ice_servers() -> list:
    """Build ICE server list. Uses Metered.ca TURN relay for proxy/NAT traversal."""
    # Default: use environment variables for TURN relay
    turn_url = os.environ.get("TURN_URL", "").strip()
    turn_username = os.environ.get("TURN_USERNAME", "").strip()
    turn_password = os.environ.get("TURN_PASSWORD", "").strip()
    
    if turn_url and turn_username and turn_password:
        # Filter out stun: URLs — they don't need credentials
        turn_urls = [u.strip() for u in turn_url.split(",") if u.strip() and u.strip().startswith("turn")]
        stun_urls = [u.strip() for u in turn_url.split(",") if u.strip() and u.strip().startswith("stun")]
        
        servers = []
        if stun_urls:
            servers.append({"urls": stun_urls})
        if turn_urls:
            servers.append({
                "urls": turn_urls,
                "username": turn_username,
                "credential": turn_password,
            })
        print(f"[WebRTC] TURN server configured from env: {turn_urls}")
        return servers
    
    print("[WebRTC] WARNING: No TURN server configured. WebRTC may fail behind proxies/NAT.")
    print("[WebRTC] Set TURN_URL, TURN_USERNAME, TURN_PASSWORD environment variables.")
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


async def _turn_credentials(request: web.Request):
    """Return TURN credentials to the browser client."""
    turn_url = os.environ.get("TURN_URL", "").strip()
    turn_username = os.environ.get("TURN_USERNAME", "").strip()
    turn_password = os.environ.get("TURN_PASSWORD", "").strip()
    
    ice_servers = []
    if turn_url and turn_username and turn_password:
        urls = [u.strip() for u in turn_url.split(",") if u.strip()]
        ice_servers.append({
            "urls": urls,
            "username": turn_username,
            "credential": turn_password,
        })
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({"iceServers": ice_servers}),
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ── aiohttp request handlers ─────────────────────────────────────────────────

async def _index(request: web.Request):
    content = (CLIENT_DIR / "index.html").read_text(encoding="utf-8")
    # Inject live-reload script before </body>
    livereload_script = """
<script>
(function() {
  var es = new EventSource('/livereload');
  es.onmessage = function(e) {
    if (e.data === 'reload') {
      console.log('[LiveReload] File changed, reloading...');
      window.location.reload();
    }
  };
  es.onerror = function() {
    console.log('[LiveReload] Connection lost, retrying...');
    setTimeout(function() { es.close(); es = new EventSource('/livereload'); }, 3000);
  };
})();
</script>
"""
    content = content.replace("</body>", livereload_script + "</body>")
    return web.Response(content_type="text/html", text=content)


async def _javascript(request: web.Request):
    content = (CLIENT_DIR / "app.js").read_text(encoding="utf-8")
    return web.Response(content_type="application/javascript", text=content)


async def _css(request: web.Request):
    content = (CLIENT_DIR / "style.css").read_text(encoding="utf-8")
    return web.Response(content_type="text/css", text=content)


def _set_codec_preferences(pc: RTCPeerConnection):
    """Set codec preferences on existing video transceivers to prioritize H.264.
    
    Must be called AFTER setRemoteDescription (so transceivers exist) but
    BEFORE createAnswer (so the answer reflects our preferences).
    """
    from aiortc import RTCRtpReceiver
    
    caps = RTCRtpReceiver.getCapabilities("video")
    if not caps or not caps.codecs:
        return
    
    # Put H.264 first, then VP8, then everything else
    h264_codecs = [c for c in caps.codecs if 'H264' in c.mimeType.upper()]
    vp8_codecs = [c for c in caps.codecs if 'VP8' in c.mimeType.upper()]
    other_codecs = [c for c in caps.codecs if 'H264' not in c.mimeType.upper() and 'VP8' not in c.mimeType.upper()]
    preferred = h264_codecs + vp8_codecs + other_codecs
    
    if not preferred:
        return
    
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            try:
                transceiver.setCodecPreferences(preferred)
            except Exception as e:
                print(f"[WebRTC] Could not set codec preferences: {e}")
    
    codec_names = [c.mimeType for c in preferred[:3]]
    print(f"[WebRTC] Codec preferences: {codec_names}")


async def _offer(request: web.Request):
    """Handle WebRTC offer from the browser."""
    params    = await request.json()
    offer     = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    shm: SharedMemory = request.app["shm"]
    
    # Use TURN relay — required for RunPod where inbound UDP is blocked.
    # Both sides need relay candidates for the connection to work.
    # Credentials come from environment variables.
    turn_url = os.environ.get("TURN_URL", "").strip()
    turn_username = os.environ.get("TURN_USERNAME", "").strip()
    turn_password = os.environ.get("TURN_PASSWORD", "").strip()
    
    if turn_url and turn_username and turn_password:
        turn_urls = [u.strip() for u in turn_url.split(",") if u.strip() and u.strip().startswith("turn")]
        turn_servers = [
            RTCIceServer(urls=turn_urls, username=turn_username, credential=turn_password),
        ]
        config = RTCConfiguration(iceServers=turn_servers)
    else:
        config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    
    pc = RTCPeerConnection(configuration=config)
    pcs: set  = request.app["pcs"]

    # Schedule closing old connections in background (don't block the new one)
    old_pcs = list(pcs)
    pcs.clear()
    pcs.add(pc)
    if old_pcs:
        asyncio.ensure_future(_close_old_connections(old_pcs))

    video_handler = None

    @pc.on("track")
    def on_track(track):
        nonlocal video_handler
        if track.kind == "video":
            video_handler = VideoStreamTrack(track, shm)
            video_handler.start()
            print("[WebRTC] Video track received from browser client")
        else:
            pc.addTrack(MediaBlackhole())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[WebRTC] Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            if video_handler:
                video_handler.stop()
            await pc.close()
            pcs.discard(pc)

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange():
        print(f"[WebRTC] ICE gathering state: {pc.iceGatheringState}")

    @pc.on("iceconnectionstatechange")
    def on_iceconnectionstatechange():
        print(f"[WebRTC] ICE connection state: {pc.iceConnectionState}")

    await pc.setRemoteDescription(offer)
    _set_codec_preferences(pc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Log the candidates in our answer for debugging
    relay_candidates = [line for line in pc.localDescription.sdp.split('\r\n') if 'candidate' in line and 'relay' in line]
    if relay_candidates:
        print(f"[WebRTC] Answer includes relay candidate(s): {len(relay_candidates)}")
    else:
        print("[WebRTC] WARNING: No relay candidates in answer — TURN may not be working")

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


# ── WHIP endpoint ─────────────────────────────────────────────────────────────

# Store active WHIP sessions for teardown via DELETE
_whip_sessions: dict = {}  # session_id -> RTCPeerConnection


async def _close_old_connections(old_pcs: list):
    """Close old peer connections in the background without blocking new ones."""
    await asyncio.sleep(5)  # Give new connection time to complete ICE negotiation via TURN
    for pc in old_pcs:
        try:
            if pc.connectionState not in ("connected", "completed"):
                await pc.close()
        except Exception:
            pass


async def _whip(request: web.Request):
    """WHIP-compliant endpoint for WebRTC ingestion from apps like Larix."""
    # WHIP expects raw SDP in the body with Content-Type: application/sdp
    content_type = request.content_type
    body = await request.text()

    if 'application/sdp' in content_type:
        sdp_offer = body
    else:
        # Fallback: try to parse as JSON (for flexibility)
        try:
            params = json.loads(body)
            sdp_offer = params.get("sdp", body)
        except (json.JSONDecodeError, ValueError):
            sdp_offer = body

    offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
    shm: SharedMemory = request.app["shm"]
    
    # WHIP also uses TURN relay from environment variables
    turn_url = os.environ.get("TURN_URL", "").strip()
    turn_username = os.environ.get("TURN_USERNAME", "").strip()
    turn_password = os.environ.get("TURN_PASSWORD", "").strip()
    
    if turn_url and turn_username and turn_password:
        turn_urls = [u.strip() for u in turn_url.split(",") if u.strip() and u.strip().startswith("turn")]
        turn_servers = [
            RTCIceServer(urls=turn_urls, username=turn_username, credential=turn_password),
        ]
        config = RTCConfiguration(iceServers=turn_servers)
    else:
        config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    
    pc = RTCPeerConnection(configuration=config)
    pcs: set = request.app["pcs"]

    # Schedule closing old connections in background (don't block the new one)
    old_pcs = list(pcs)
    pcs.clear()
    _whip_sessions.clear()
    pcs.add(pc)
    if old_pcs:
        asyncio.ensure_future(_close_old_connections(old_pcs))

    # Generate a session ID for this WHIP resource
    import uuid
    session_id = str(uuid.uuid4())
    _whip_sessions[session_id] = pc

    video_handler = None

    @pc.on("track")
    def on_track(track):
        nonlocal video_handler
        if track.kind == "video":
            video_handler = VideoStreamTrack(track, shm)
            video_handler.start()
            print(f"[WHIP] Video track received from session {session_id}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[WHIP] Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            if video_handler:
                video_handler.stop()
            await pc.close()
            pcs.discard(pc)
            _whip_sessions.pop(session_id, None)

    await pc.setRemoteDescription(offer)
    _set_codec_preferences(pc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Return 201 Created with SDP answer and Location header per WHIP spec
    resource_url = f"/whip/resource/{session_id}"
    return web.Response(
        status=201,
        content_type="application/sdp",
        text=pc.localDescription.sdp,
        headers={
            "Location": resource_url,
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Location",
        },
    )


async def _whip_resource_delete(request: web.Request):
    """Handle WHIP resource teardown via DELETE."""
    session_id = request.match_info["session_id"]
    pc = _whip_sessions.pop(session_id, None)
    if pc:
        await pc.close()
        request.app["pcs"].discard(pc)
        return web.Response(status=200, text="Session terminated")
    return web.Response(status=404, text="Session not found")


async def _whip_options(request: web.Request):
    """Handle CORS preflight for WHIP endpoint."""
    return web.Response(
        status=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )


# ── Live-reload: file watcher + SSE endpoint ─────────────────────────────────

# Tracks modification times of client files
_file_mtimes: dict = {}
_reload_clients: list = []  # list of asyncio.Queue for SSE subscribers


def _scan_client_files() -> dict:
    """Return {filepath: mtime} for all files in CLIENT_DIR."""
    mtimes = {}
    if CLIENT_DIR.is_dir():
        for f in CLIENT_DIR.iterdir():
            if f.is_file():
                mtimes[str(f)] = f.stat().st_mtime
    return mtimes


async def _file_watcher_task():
    """Background task that polls client files for changes and notifies SSE clients."""
    global _file_mtimes
    _file_mtimes = _scan_client_files()

    while True:
        await asyncio.sleep(1)  # Poll every second
        current = _scan_client_files()
        changed = False

        for path, mtime in current.items():
            if path not in _file_mtimes or _file_mtimes[path] != mtime:
                changed = True
                print(f"[WebRTC] File changed: {Path(path).name}")
                break

        # Also detect new or deleted files
        if not changed and set(current.keys()) != set(_file_mtimes.keys()):
            changed = True

        if changed:
            _file_mtimes = current
            # Notify all connected SSE clients
            for queue in _reload_clients:
                await queue.put("reload")


# ── WebSocket fallback for frame streaming ────────────────────────────────────
# When WebRTC ICE fails (e.g., RunPod blocking UDP), the browser falls back to
# sending JPEG-encoded frames over WebSocket. Higher latency than WebRTC but
# works reliably through any HTTP proxy.

async def _ws_stream(request: web.Request):
    """WebSocket endpoint that receives JPEG frames from the browser."""
    import cv2
    
    ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)  # 10MB max frame
    await ws.prepare(request)
    
    shm: SharedMemory = request.app["shm"]
    print("[WebSocket] Client connected for frame streaming")
    
    frame_count = 0
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                # Decode JPEG to BGR numpy array
                jpg_array = np.frombuffer(msg.data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    _write_frame(shm, frame_bgr)
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"[WebSocket] Received {frame_count} frames")
            elif msg.type == web.WSMsgType.TEXT:
                # Could be a control message
                if msg.data == 'ping':
                    await ws.send_str('pong')
            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        print(f"[WebSocket] Client disconnected after {frame_count} frames")
    
    return ws


# ── Live-reload: file watcher + SSE endpoint ─────────────────────────────────

# Tracks modification times of client files
_file_mtimes: dict = {}
_reload_clients: list = []  # list of asyncio.Queue for SSE subscribers


async def _livereload_sse(request: web.Request):
    """SSE endpoint that pushes 'reload' events when client files change."""
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await response.prepare(request)

    queue = asyncio.Queue()
    _reload_clients.append(queue)

    try:
        # Send initial heartbeat
        await response.write(b": heartbeat\n\n")

        while True:
            msg = await queue.get()
            await response.write(f"data: {msg}\n\n".encode())
    except (asyncio.CancelledError, ConnectionResetError, ConnectionError):
        pass
    finally:
        _reload_clients.remove(queue)

    return response


async def _on_startup(app: web.Application):
    """Start the file watcher background task."""
    app["file_watcher"] = asyncio.ensure_future(_file_watcher_task())


async def _on_shutdown(app: web.Application):
    # Stop file watcher
    watcher = app.get("file_watcher")
    if watcher:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass

    pcs: set = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    shm: SharedMemory = app["shm"]
    shm.close()


# ── Server entry-point ───────────────────────────────────────────────────────

def generate_self_signed_cert(cert_path: Path, key_path: Path):
    """Generate a self-signed certificate and private key using the cryptography package."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import datetime
    import ipaddress
    import socket

    # Make parent dirs
    cert_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[WebRTC] Generating self-signed SSL certificate...")
    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Get local hostname and IPs
    hostname = socket.gethostname()
    local_ips = ["127.0.0.1", "0.0.0.0"]
    try:
        # Try to resolve our main LAN IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        local_ips.append(lan_ip)
        s.close()
    except Exception:
        pass

    # Build subject and issuer
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VisoMaster"),
        x509.NameAttribute(NameOID.COMMON_NAME, "visomaster.local"),
    ])

    # Build SAN (Subject Alternative Name)
    san_entries = [x509.DNSName("localhost"), x509.DNSName(hostname)]
    for ip in local_ips:
        try:
            san_entries.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
        except ValueError:
            pass

    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow() - datetime.timedelta(days=1)
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=3650)  # 10 years
    ).add_extension(
        x509.SubjectAlternativeName(san_entries),
        critical=False,
    ).sign(key, hashes.SHA256())

    # Save files
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"[WebRTC] SSL certificate successfully generated: {cert_path}")


def _kill_process_on_port(port: int):
    """Kill any process using the specified port."""
    import psutil
    import signal
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    print(f"[WebRTC] Killing process {proc.pid} ({proc.name()}) on port {port}")
                    try:
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    except Exception as e:
                        print(f"[WebRTC] Error killing process: {e}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


def run_server(https_port: int = 9090, http_port: int = 9091,
               cert_file: str = "", key_file: str = "",
               host: str = "0.0.0.0"):
    """
    Start the WebRTC HTTP and HTTPS dual servers simultaneously.
    This function blocks; call it from a multiprocessing.Process.
    """
    # Kill any processes using these ports
    _kill_process_on_port(http_port)
    _kill_process_on_port(https_port)
    
    shm = _create_shm()

    app = web.Application()
    app["pcs"] = set()
    app["shm"] = shm
    app.on_shutdown.append(_on_shutdown)

    app.router.add_get("/",             _index)
    app.router.add_get("/app.js",       _javascript)
    app.router.add_get("/style.css",    _css)
    app.router.add_post("/offer",       _offer)
    app.router.add_get("/turn-credentials", _turn_credentials)
    app.router.add_get("/ws",           _ws_stream)
    app.router.add_post("/whip",        _whip)
    app.router.add_options("/whip",     _whip_options)
    app.router.add_delete("/whip/resource/{session_id}", _whip_resource_delete)
    app.router.add_get("/livereload",   _livereload_sse)

    app.on_startup.append(_on_startup)

    # ── Auto-generate certificates if not provided or missing ─────────────────
    ssl_ctx = None
    if not cert_file or not key_file:
        cert_file = str(CLIENT_DIR.parent / "certificates" / "cert.pem")
        key_file = str(CLIENT_DIR.parent / "certificates" / "key.pem")

    cert_path = Path(cert_file)
    key_path = Path(key_file)

    # Always regenerate the certificate on startup so it includes current IPs
    try:
        generate_self_signed_cert(cert_path, key_path)
    except Exception as e:
        print(f"[WebRTC] Error generating self-signed certificate: {e}")

    if cert_path.is_file() and key_path.is_file():
        try:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(str(cert_path), str(key_path))
        except Exception as e:
            print(f"[WebRTC] SSL Context error: {e}. Falling back to HTTP-only.")

    # ── Start Dual Servers ────────────────────────────────────────────────────
    async def start_dual_servers():
        runner = web.AppRunner(app)
        await runner.setup()

        # Start HTTP server
        http_site = web.TCPSite(runner, host, http_port)
        await http_site.start()
        print(f"[WebRTC] HTTP server running on {host}:{http_port} (http://{host}:{http_port})")

        # Start HTTPS server
        if ssl_ctx:
            https_site = web.TCPSite(runner, host, https_port, ssl_context=ssl_ctx)
            await https_site.start()
            print(f"[WebRTC] HTTPS server running on {host}:{https_port} (https://{host}:{https_port})")
        else:
            print("[WebRTC] HTTPS server NOT started due to missing/invalid certificates.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_dual_servers())

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
        try:
            shm.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    run_server()
