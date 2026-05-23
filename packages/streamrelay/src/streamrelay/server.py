"""HTTP/HTTPS + WebSocket server that ingests camera frames from a browser.

Spawn this in a subprocess so it never blocks your AI app's event loop:

    import multiprocessing
    from streamrelay import StreamServer

    def _run():
        StreamServer(shm_name="visomaster_frames").run()

    if __name__ == "__main__":
        multiprocessing.Process(target=_run, daemon=True).start()

The server:
* serves the bundled web UI at ``GET /``
* accepts JPEG or H.264 binary frames over ``GET /ws`` (WebSocket)
* writes decoded BGR pixels into a named shared-memory block
* hot-reloads the web UI for development via SSE on ``/livereload``
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from aiohttp import web

from . import protocol

logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


# ── Default client assets bundled with the package ───────────────────────────
_PKG_DIR        = Path(__file__).parent
DEFAULT_CLIENT_DIR = _PKG_DIR / "client"


# ── Shared-memory helpers ────────────────────────────────────────────────────
def _create_shm(name: str) -> SharedMemory:
    try:
        shm = SharedMemory(name=name, create=True, size=protocol.SHM_TOTAL_BYTES)
    except FileExistsError:
        shm = SharedMemory(name=name, create=False, size=protocol.SHM_TOTAL_BYTES)
    protocol.pack_header(shm.buf, 0, 0, 0)
    return shm


def _write_frame(shm: SharedMemory, frame_bgr: np.ndarray) -> None:
    h, w = frame_bgr.shape[:2]
    if h > protocol.SHM_MAX_HEIGHT or w > protocol.SHM_MAX_WIDTH:
        scale = min(protocol.SHM_MAX_WIDTH / w, protocol.SHM_MAX_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h))
        h, w = new_h, new_w
    counter, _, _ = protocol.unpack_header(shm.buf)
    counter = (counter + 1) & 0xFFFFFFFF
    protocol.pack_header(shm.buf, counter, w, h)
    pixel_bytes = frame_bgr.tobytes()
    shm.buf[protocol.SHM_HEADER_BYTES:
            protocol.SHM_HEADER_BYTES + len(pixel_bytes)] = pixel_bytes


def _kill_process_on_port(port: int) -> None:
    """Best-effort: kill any process holding the given port."""
    try:
        import psutil
        import signal
    except ImportError:
        return
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # net_connections is the new API; connections() is deprecated.
            connections = (
                proc.net_connections() if hasattr(proc, "net_connections")
                else proc.connections()
            )
            for conn in connections:
                if conn.laddr and conn.laddr.port == port and conn.status == 'LISTEN':
                    print(f"[streamrelay] Releasing port {port} from PID {proc.pid}")
                    try:
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()
        except Exception:
            pass


# ── TLS certificate helper ───────────────────────────────────────────────────
def generate_self_signed_cert(
    cert_path: Path,
    key_path: Path,
    common_name: str = "streamrelay.local",
    organization: str = "streamrelay",
) -> None:
    """Generate a self-signed certificate so phones can use getUserMedia
    (which requires HTTPS for non-localhost origins)."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime
    import ipaddress
    import socket

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    print("[streamrelay] Generating self-signed SSL certificate…")

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
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])
    san_entries = [x509.DNSName("localhost"), x509.DNSName(hostname)]
    for ip in local_ips:
        try:
            san_entries.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
        except ValueError:
            pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"[streamrelay] SSL certificate written to {cert_path}")


# ── The server itself ────────────────────────────────────────────────────────
FrameCallback = Callable[[np.ndarray], None]


class StreamServer:
    """A bundle of HTTP + HTTPS + WebSocket endpoints that turns a phone
    or browser camera into BGR frames in a shared-memory block.

    Parameters
    ----------
    shm_name:
        Name of the shared-memory block to create. Must be unique per host.
    http_port, https_port:
        Ports to bind. Set ``https_port=0`` to disable TLS.
    host:
        Bind address. Defaults to all interfaces.
    cert_file, key_file:
        Paths to an existing TLS cert/key. If empty, a self-signed pair
        is generated next to the bundled client folder.
    client_dir:
        Override the static-asset folder. Defaults to the bundled UI.
    on_frame:
        Optional callback ``fn(frame_bgr) -> None`` invoked for every
        decoded frame **in addition to** the shared-memory write. Useful
        for in-process consumers that don't need the shm dance.
    """

    def __init__(
        self,
        shm_name: str = protocol.DEFAULT_SHM_NAME,
        http_port: int = 9091,
        https_port: int = 9090,
        host: str = "0.0.0.0",
        cert_file: str = "",
        key_file: str = "",
        client_dir: Optional[Path] = None,
        on_frame: Optional[FrameCallback] = None,
    ):
        self.shm_name = shm_name
        self.http_port = http_port
        self.https_port = https_port
        self.host = host
        self.cert_file = cert_file
        self.key_file = key_file
        self.client_dir = Path(client_dir) if client_dir else DEFAULT_CLIENT_DIR
        self.on_frame = on_frame

        self._shm: Optional[SharedMemory] = None
        self._file_mtimes: dict = {}
        self._reload_clients: list = []

    # ── Static handlers ──────────────────────────────────────────────────────
    async def _index(self, request: web.Request) -> web.Response:
        path = self.client_dir / "index.html"
        return web.Response(content_type="text/html",
                            text=path.read_text(encoding="utf-8"))

    async def _javascript(self, request: web.Request) -> web.Response:
        path = self.client_dir / "app.js"
        return web.Response(content_type="application/javascript",
                            text=path.read_text(encoding="utf-8"))

    async def _css(self, request: web.Request) -> web.Response:
        path = self.client_dir / "style.css"
        return web.Response(content_type="text/css",
                            text=path.read_text(encoding="utf-8"))

    # ── Frame ingestion (WebSocket) ──────────────────────────────────────────
    async def _ws_stream(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=5 * 1024 * 1024)
        await ws.prepare(request)
        print("[streamrelay] Client connected")

        frame_count = 0
        codec = "jpeg"
        h264_decoder = None
        start_time = time.time()
        last_log = start_time
        h264_errors = 0

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    if msg.data == "ping":
                        await ws.send_str("pong")
                        continue
                    try:
                        config = json.loads(msg.data)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if config.get("type") == "codec":
                        codec = config.get("codec", "jpeg")
                        w = config.get("width", 1280)
                        h = config.get("height", 720)
                        print(f"[streamrelay] Codec: {codec}, "
                              f"resolution: {w}x{h}")
                        if codec == "h264":
                            try:
                                import av  # type: ignore
                                h264_decoder = av.CodecContext.create("h264", "r")
                                h264_decoder.extradata = None
                                print("[streamrelay] H.264 decoder ready (PyAV)")
                            except ImportError:
                                print("[streamrelay] PyAV missing; "
                                      "falling back to JPEG")
                                codec = "jpeg"
                                await ws.send_str(json.dumps(
                                    {"type": "fallback", "codec": "jpeg"}))
                elif msg.type == web.WSMsgType.BINARY:
                    if codec == "h264" and h264_decoder is not None:
                        try:
                            import av  # type: ignore
                            packet = av.Packet(msg.data)
                            for frame in h264_decoder.decode(packet):
                                img = frame.to_ndarray(format="bgr24")
                                self._dispatch_frame(img)
                                frame_count += 1
                        except Exception:
                            h264_errors += 1
                            if h264_errors > 10:
                                print("[streamrelay] Too many H.264 errors; "
                                      "switching to JPEG")
                                codec = "jpeg"
                                h264_decoder = None
                                await ws.send_str(json.dumps(
                                    {"type": "fallback", "codec": "jpeg"}))
                    else:
                        frame_bgr = cv2.imdecode(
                            np.frombuffer(msg.data, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if frame_bgr is not None:
                            self._dispatch_frame(frame_bgr)
                            frame_count += 1

                    now = time.time()
                    if now - last_log >= 5.0:
                        elapsed = now - start_time
                        fps = frame_count / elapsed if elapsed else 0
                        print(f"[streamrelay] {frame_count} frames, "
                              f"{fps:.1f} FPS avg ({codec})")
                        last_log = now
                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[streamrelay] WS error: {e}")
        finally:
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed else 0
            print(f"[streamrelay] Disconnected — {frame_count} frames in "
                  f"{elapsed:.1f}s ({avg_fps:.1f} FPS)")
        return ws

    def _dispatch_frame(self, frame_bgr: np.ndarray) -> None:
        if self._shm is not None:
            _write_frame(self._shm, frame_bgr)
        if self.on_frame is not None:
            try:
                self.on_frame(frame_bgr)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"[streamrelay] on_frame callback raised: {e}")

    # ── Live-reload (development convenience) ────────────────────────────────
    def _scan_client_files(self) -> dict:
        out: dict = {}
        if self.client_dir.is_dir():
            for f in self.client_dir.iterdir():
                if f.is_file():
                    out[str(f)] = f.stat().st_mtime
        return out

    async def _file_watcher_task(self) -> None:
        self._file_mtimes = self._scan_client_files()
        while True:
            await asyncio.sleep(1)
            current = self._scan_client_files()
            if current != self._file_mtimes:
                self._file_mtimes = current
                for q in self._reload_clients:
                    await q.put("reload")

    async def _livereload_sse(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200, reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)
        queue: asyncio.Queue = asyncio.Queue()
        self._reload_clients.append(queue)
        try:
            await response.write(b": heartbeat\n\n")
            while True:
                msg = await queue.get()
                await response.write(f"data: {msg}\n\n".encode())
        except (asyncio.CancelledError, ConnectionResetError, ConnectionError):
            pass
        finally:
            self._reload_clients.remove(queue)
        return response

    # ── App lifecycle ────────────────────────────────────────────────────────
    async def _on_startup(self, app: web.Application) -> None:
        app["file_watcher"] = asyncio.ensure_future(self._file_watcher_task())

    async def _on_shutdown(self, app: web.Application) -> None:
        watcher = app.get("file_watcher")
        if watcher:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        if self._shm is not None:
            self._shm.close()

    # ── Entry points ─────────────────────────────────────────────────────────
    def run(self) -> None:
        """Block the current process and serve until interrupted."""
        if self.http_port:
            _kill_process_on_port(self.http_port)
        if self.https_port:
            _kill_process_on_port(self.https_port)

        self._shm = _create_shm(self.shm_name)

        app = web.Application()
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)

        app.router.add_get("/", self._index)
        app.router.add_get("/app.js", self._javascript)
        app.router.add_get("/style.css", self._css)
        app.router.add_get("/ws", self._ws_stream)
        app.router.add_get("/livereload", self._livereload_sse)

        ssl_ctx = self._build_ssl_context()

        async def start_servers():
            runner = web.AppRunner(app)
            await runner.setup()
            if self.http_port:
                http_site = web.TCPSite(runner, self.host, self.http_port)
                await http_site.start()
                print(f"[streamrelay] HTTP  on {self.host}:{self.http_port}")
            if ssl_ctx and self.https_port:
                https_site = web.TCPSite(
                    runner, self.host, self.https_port, ssl_context=ssl_ctx
                )
                await https_site.start()
                print(f"[streamrelay] HTTPS on {self.host}:{self.https_port}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_servers())
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if self._shm is not None:
                self._shm.close()
                try:
                    self._shm.unlink()
                except Exception:
                    pass

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _build_ssl_context(self) -> Optional[ssl.SSLContext]:
        if self.https_port == 0:
            return None
        cert_file = self.cert_file
        key_file = self.key_file
        if not cert_file or not key_file:
            # Default: write certs into the current working directory under
            # ./streamrelay-certs/. Avoids polluting the package install dir
            # (which may be read-only on system Python installs).
            base = Path.cwd() / "streamrelay-certs"
            cert_file = str(base / "cert.pem")
            key_file = str(base / "key.pem")
        cert_path = Path(cert_file)
        key_path = Path(key_file)
        if not (cert_path.is_file() and key_path.is_file()):
            try:
                generate_self_signed_cert(cert_path, key_path)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"[streamrelay] Cert generation skipped: {e}")
        if cert_path.is_file() and key_path.is_file():
            try:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(str(cert_path), str(key_path))
                return ctx
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"[streamrelay] SSL load error: {e}")
        return None


# ── Backwards-compatible function form ───────────────────────────────────────
def run_server(
    https_port: int = 9090,
    http_port: int = 9091,
    cert_file: str = "",
    key_file: str = "",
    host: str = "0.0.0.0",
    shm_name: str = protocol.DEFAULT_SHM_NAME,
) -> None:
    """Functional wrapper kept for legacy callers (e.g. VisoMaster)."""
    StreamServer(
        shm_name=shm_name,
        http_port=http_port,
        https_port=https_port,
        host=host,
        cert_file=cert_file,
        key_file=key_file,
    ).run()


if __name__ == "__main__":
    StreamServer().run()


def _cli() -> None:
    """Console-script entry point: ``streamrelay-server`` after pip install."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="streamrelay-server",
        description="Run a streamrelay WebSocket video ingestion server.",
    )
    parser.add_argument("--shm-name", default=protocol.DEFAULT_SHM_NAME,
                        help="Name of the shared-memory block to expose")
    parser.add_argument("--http-port", type=int, default=9091)
    parser.add_argument("--https-port", type=int, default=9090,
                        help="Set to 0 to disable HTTPS")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--cert-file", default="")
    parser.add_argument("--key-file", default="")
    args = parser.parse_args()

    StreamServer(
        shm_name=args.shm_name,
        http_port=args.http_port,
        https_port=args.https_port,
        host=args.host,
        cert_file=args.cert_file,
        key_file=args.key_file,
    ).run()
