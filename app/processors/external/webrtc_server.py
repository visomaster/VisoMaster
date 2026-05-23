"""
Streaming Server for VisoMaster
Serves a browser-facing camera streaming page over HTTP (port 9091) and HTTPS (port 9090).
Frames are received via WebSocket and written into a named shared memory block
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
from pathlib import Path

import numpy as np
import cv2

from aiohttp import web

# multiprocessing shared memory
from multiprocessing.shared_memory import SharedMemory
import struct

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


# ── aiohttp request handlers ─────────────────────────────────────────────────

async def _index(request: web.Request):
    content = (CLIENT_DIR / "index.html").read_text(encoding="utf-8")
    return web.Response(content_type="text/html", text=content)


async def _javascript(request: web.Request):
    content = (CLIENT_DIR / "app.js").read_text(encoding="utf-8")
    return web.Response(content_type="application/javascript", text=content)


async def _css(request: web.Request):
    content = (CLIENT_DIR / "style.css").read_text(encoding="utf-8")
    return web.Response(content_type="text/css", text=content)


# ── WebSocket frame streaming (primary transport) ─────────────────────────────

async def _ws_stream(request: web.Request):
    """WebSocket endpoint that receives JPEG frames from the browser.
    
    Protocol:
      - Binary messages: JPEG-encoded frame data
      - Text 'ping': keepalive, responds with 'pong'
      - Text 'config:WxH': client reports its capture resolution
    """
    ws = web.WebSocketResponse(max_msg_size=5 * 1024 * 1024)  # 5MB max
    await ws.prepare(request)
    
    shm: SharedMemory = request.app["shm"]
    print("[Stream] Client connected")
    
    frame_count = 0
    import time
    start_time = time.time()
    last_log_time = start_time
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                # Decode JPEG to BGR — cv2.imdecode is fast for JPEG
                frame_bgr = cv2.imdecode(
                    np.frombuffer(msg.data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                if frame_bgr is not None:
                    _write_frame(shm, frame_bgr)
                    frame_count += 1
                    
                    # Log FPS every 5 seconds
                    now = time.time()
                    if now - last_log_time >= 5.0:
                        elapsed = now - last_log_time
                        fps = frame_count / (now - start_time)
                        recent_fps = (frame_count - int((last_log_time - start_time) * fps)) / elapsed if elapsed > 0 else 0
                        print(f"[Stream] {frame_count} frames, ~{fps:.1f} avg FPS, last 5s: ~{recent_fps:.1f} FPS")
                        last_log_time = now
                        
            elif msg.type == web.WSMsgType.TEXT:
                if msg.data == 'ping':
                    await ws.send_str('pong')
            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break
    except Exception as e:
        print(f"[Stream] Error: {e}")
    finally:
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"[Stream] Disconnected — {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS avg)")
    
    return ws


# ── Live-reload: file watcher + SSE endpoint ─────────────────────────────────

_file_mtimes: dict = {}
_reload_clients: list = []


def _scan_client_files() -> dict:
    mtimes = {}
    if CLIENT_DIR.is_dir():
        for f in CLIENT_DIR.iterdir():
            if f.is_file():
                mtimes[str(f)] = f.stat().st_mtime
    return mtimes


async def _file_watcher_task():
    global _file_mtimes
    _file_mtimes = _scan_client_files()
    while True:
        await asyncio.sleep(1)
        current = _scan_client_files()
        changed = False
        for path, mtime in current.items():
            if path not in _file_mtimes or _file_mtimes[path] != mtime:
                changed = True
                break
        if not changed and set(current.keys()) != set(_file_mtimes.keys()):
            changed = True
        if changed:
            _file_mtimes = current
            for queue in _reload_clients:
                await queue.put("reload")


async def _livereload_sse(request: web.Request):
    response = web.StreamResponse(
        status=200, reason='OK',
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
    app["file_watcher"] = asyncio.ensure_future(_file_watcher_task())


async def _on_shutdown(app: web.Application):
    watcher = app.get("file_watcher")
    if watcher:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
    shm: SharedMemory = app["shm"]
    shm.close()


# ── Server entry-point ───────────────────────────────────────────────────────

def generate_self_signed_cert(cert_path: Path, key_path: Path):
    """Generate a self-signed certificate for HTTPS."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import datetime
    import ipaddress
    import socket

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    print("[Server] Generating self-signed SSL certificate...")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    hostname = socket.gethostname()
    local_ips = ["127.0.0.1", "0.0.0.0"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VisoMaster"),
        x509.NameAttribute(NameOID.COMMON_NAME, "visomaster.local"),
    ])
    san_entries = [x509.DNSName("localhost"), x509.DNSName(hostname)]
    for ip in local_ips:
        try:
            san_entries.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
        except ValueError:
            pass

    cert = x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(
        key.public_key()
    ).serial_number(x509.random_serial_number()).not_valid_before(
        datetime.datetime.utcnow() - datetime.timedelta(days=1)
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=3650)
    ).add_extension(
        x509.SubjectAlternativeName(san_entries), critical=False,
    ).sign(key, hashes.SHA256())

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"[Server] SSL certificate generated: {cert_path}")


def _kill_process_on_port(port: int):
    """Kill any process using the specified port."""
    import psutil
    import signal
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    print(f"[Server] Killing process {proc.pid} ({proc.name()}) on port {port}")
                    try:
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()
        except (Exception):
            pass


def run_server(https_port: int = 9090, http_port: int = 9091,
               cert_file: str = "", key_file: str = "",
               host: str = "0.0.0.0"):
    """
    Start the streaming server (HTTP + HTTPS).
    This function blocks; call it from a multiprocessing.Process.
    """
    _kill_process_on_port(http_port)
    _kill_process_on_port(https_port)
    
    shm = _create_shm()

    app = web.Application()
    app["shm"] = shm
    app.on_shutdown.append(_on_shutdown)
    app.on_startup.append(_on_startup)

    app.router.add_get("/",           _index)
    app.router.add_get("/app.js",     _javascript)
    app.router.add_get("/style.css",  _css)
    app.router.add_get("/ws",         _ws_stream)
    app.router.add_get("/livereload", _livereload_sse)

    # ── Certificates ──────────────────────────────────────────────────────────
    ssl_ctx = None
    if not cert_file or not key_file:
        cert_file = str(CLIENT_DIR.parent / "certificates" / "cert.pem")
        key_file = str(CLIENT_DIR.parent / "certificates" / "key.pem")

    cert_path = Path(cert_file)
    key_path = Path(key_file)
    try:
        generate_self_signed_cert(cert_path, key_path)
    except Exception as e:
        print(f"[Server] Certificate generation error: {e}")

    if cert_path.is_file() and key_path.is_file():
        try:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(str(cert_path), str(key_path))
        except Exception as e:
            print(f"[Server] SSL error: {e}")

    # ── Start servers ─────────────────────────────────────────────────────────
    async def start_servers():
        runner = web.AppRunner(app)
        await runner.setup()

        http_site = web.TCPSite(runner, host, http_port)
        await http_site.start()
        print(f"[Server] HTTP on {host}:{http_port}")

        if ssl_ctx:
            https_site = web.TCPSite(runner, host, https_port, ssl_context=ssl_ctx)
            await https_site.start()
            print(f"[Server] HTTPS on {host}:{https_port}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_servers())

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
