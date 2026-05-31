# caption_server.py
# Local WebSocket server that pushes Kira's spoken-line captions (with Azure
# word-boundary timing) to the OBS browser-source overlay in caption_overlay/.
#
# Design intent:
#   * STRICTLY additive. Failure to start the server, or to send a frame,
#     MUST NOT block or slow down TTS. Every send is fire-and-forget,
#     guarded, and silent on no-connection.
#   * Captions are LIVE-ONLY: if no OBS source is connected when a line
#     speaks, the frame is dropped (we don't buffer past audio).
#   * One global server instance, started from bot.py once the asyncio loop
#     is running. All ai_core.py calls go through `enqueue_caption()`.
#
# Wire protocol (server → page, JSON):
#     { "type": "caption",
#       "text": "Hello chat how are you",
#       "words": [{"word":"Hello","offset_ms":0}, ...],
#       "clear_after_ms": 1500 }
#     { "type": "clear" }
#     { "type": "hello" }
#
# The overlay is purely client-side timed off these per-word offsets — see
# caption_overlay/caption.js. We do not stream word-by-word frames; one frame
# per spoken chunk keeps the network chatter minimal and avoids jitter.

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore
    WebSocketServerProtocol = Any  # type: ignore
    WEBSOCKETS_AVAILABLE = False

from config import (
    ENABLE_CAPTIONS,
    CAPTION_SERVER_PORT,
    CAPTION_CLEAR_DELAY_MS,
)


class CaptionServer:
    """Tiny broadcast WS server. Holds the set of connected overlay pages and
    forwards caption frames to all of them. No persistence, no buffering —
    captions are live only.

    Usage:
        server = CaptionServer()
        await server.start()                       # idempotent, fail-graceful
        await server.send_caption(text, words)     # called by ai_core
        await server.send_clear()                  # on interrupt
        await server.stop()                        # on shutdown
    """

    def __init__(self) -> None:
        self.enabled = ENABLE_CAPTIONS
        self.port = CAPTION_SERVER_PORT
        self.clear_delay_ms = CAPTION_CLEAR_DELAY_MS
        self._clients: set = set()
        self._server = None
        self._lock = asyncio.Lock()
        self._started = False

        if not self.enabled:
            print("   [Captions] Disabled (ENABLE_CAPTIONS=false). No server will start.")
        elif not WEBSOCKETS_AVAILABLE:
            print("   [Captions] websockets library missing \u2014 captions disabled. Run: pip install websockets")
            self.enabled = False

    # ── lifecycle ─────────────────────────────────────────────────────────
    async def start(self) -> bool:
        """Bind the WS server. Retries a few times on EADDRINUSE in case a
        previous bot instance is mid-shutdown and the socket is still
        lingering. On final failure logs a loud, actionable message and
        disables captions for the session."""
        if not self.enabled or self._started:
            return self._started

        # Errno 10048 (WSAEADDRINUSE) on Windows / 98 (EADDRINUSE) on POSIX.
        port_in_use_codes = {10048, 98}
        max_attempts = 4
        retry_delay = 0.75  # seconds; total worst-case wait ~3s

        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                # Bind to both v4 and v6 loopback so OBS-CEF (which sometimes
                # resolves "localhost" to ::1) and Chrome (127.0.0.1) can both
                # connect. Do NOT include "localhost" in the host list — it
                # resolves to BOTH 127.0.0.1 and ::1 and causes a self-collision
                # (the second/third bind hits a port already held by the first
                # bind in this same process, raising EADDRINUSE on every start).
                #
                # NOTE: We do NOT set SO_REUSEADDR here. On Windows SO_REUSEADDR
                # allows a second process to silently STEAL the port from a
                # running one (the opposite of POSIX behavior) — a worse
                # failure mode than the brief startup retry below. The retry
                # loop handles the only legitimate case: a previous instance
                # mid-shutdown still releasing the socket.
                self._server = await websockets.serve(
                    self._handle_client,
                    host=["127.0.0.1", "::1"],
                    port=self.port,
                    ping_interval=20,
                    ping_timeout=20,
                )
                self._started = True
                print(f"   [Captions] WebSocket server listening on ws://localhost:{self.port} (127.0.0.1 + ::1)")
                overlay_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caption_overlay", "index.html")
                # OBS browser source expects forward slashes in the file:// URL.
                print(f"   [Captions] OBS browser source URL: file:///{overlay_path.replace(os.sep, '/')}")
                return True
            except OSError as e:
                last_err = e
                errno_val = getattr(e, "errno", None) or getattr(e, "winerror", None)
                if errno_val in port_in_use_codes and attempt < max_attempts:
                    print(
                        f"   [Captions] Port {self.port} busy (attempt {attempt}/{max_attempts}) "
                        f"\u2014 retrying in {retry_delay:.1f}s (previous instance may be releasing it)..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                # Final failure on port-in-use \u2014 log loud and actionable.
                if errno_val in port_in_use_codes:
                    print("")
                    print("   " + "=" * 72)
                    print(f"   [Captions] !!! PORT {self.port} IN USE \u2014 CAPTIONS DISABLED THIS SESSION !!!")
                    print(f"   [Captions]     A previous bot instance is still running and holding the port.")
                    print(f"   [Captions]     FIX: kill the orphan python.exe, or run:")
                    print(f"   [Captions]         Get-NetTCPConnection -LocalPort {self.port} | "
                          f"Select-Object -Expand OwningProcess | ForEach-Object {{ Stop-Process -Id $_ -Force }}")
                    print(f"   [Captions]     Then restart the bot.")
                    print("   " + "=" * 72)
                    print("")
                else:
                    print(f"   [Captions] Failed to start server (OSError {errno_val}): {e} \u2014 captions disabled for this session.")
                self.enabled = False
                return False
            except Exception as e:
                last_err = e
                print(f"   [Captions] Failed to start server: {e} \u2014 captions disabled for this session.")
                self.enabled = False
                return False
        # Shouldn't reach here, but be defensive.
        if last_err:
            print(f"   [Captions] Start failed after {max_attempts} attempts: {last_err}")
        self.enabled = False
        return False

    async def stop(self) -> None:
        """Close the server and forcibly disconnect every client so the OS
        actually releases the listening socket. Without the explicit client
        close, lingering WS connections can keep the listener half-alive
        for a few seconds and block the next start."""
        # 1. Close every open client first so wait_closed() doesn't hang on
        #    a stuck async-for in _handle_client.
        clients_snapshot = list(self._clients)
        for ws in clients_snapshot:
            try:
                await ws.close(code=1001, reason="server shutdown")
            except Exception:
                pass
        self._clients.clear()

        # 2. Close the listening sockets and wait for the server task to
        #    actually exit so the port is fully released.
        if self._server is not None:
            try:
                self._server.close()
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                print("   [Captions] stop(): wait_closed timed out; socket may linger briefly.")
            except Exception as e:
                print(f"   [Captions] stop(): error during close: {e}")
            self._server = None
        self._started = False
        print("   [Captions] Server stopped, port released.")

    # ── client handling ───────────────────────────────────────────────────
    async def _handle_client(self, ws: WebSocketServerProtocol) -> None:
        """One coroutine per connected overlay page. We send a greeting then
        idle \u2014 the overlay is receive-only.""" 
        peer = getattr(ws, "remote_address", ("?", 0))
        async with self._lock:
            self._clients.add(ws)
        print(f"   [Captions] Overlay connected: {peer[0]}:{peer[1]} ({len(self._clients)} total)")
        try:
            await ws.send(json.dumps({"type": "hello"}))
            # Just hold the connection open; we don't expect inbound messages.
            async for _ in ws:
                pass
        except Exception:
            pass
        finally:
            async with self._lock:
                self._clients.discard(ws)
            print(f"   [Captions] Overlay disconnected ({len(self._clients)} remaining)")

    # ── public send API ───────────────────────────────────────────────────
    async def send_caption(self, text: str, words: list[dict]) -> None:
        """Push one caption frame. ``words`` is a list of dicts shaped
        ``{"word": str, "offset_ms": int}`` where offset_ms is the milliseconds
        from synthesis start (== now, on localhost) at which to reveal that word.

        No-op if disabled, no clients connected, or the frame is empty.
        Never raises \u2014 failures are logged once at debug-volume.
        """
        if not self.enabled or not self._started:
            print(f"   [Captions] Drop frame '{text[:30]}': server not started (enabled={self.enabled}, started={self._started}).")
            return
        if not text or not words:
            print(f"   [Captions] Drop frame: empty text or words list (text_len={len(text)}, words={len(words)}).")
            return
        if not self._clients:
            print(f"   [Captions] Drop frame '{text[:30]}': no overlay clients connected.")
            return
        frame = {
            "type": "caption",
            "text": text,
            "words": words,
            "clear_after_ms": self.clear_delay_ms,
        }
        await self._broadcast(frame)

    async def send_clear(self) -> None:
        """Immediately clear the active caption. Use on user interrupt."""
        if not self.enabled or not self._started or not self._clients:
            return
        await self._broadcast({"type": "clear"})

    async def send_cookie(self, shared: int, milestone: bool = False) -> None:
        """Push a cookie-jar update to the overlay. Two shapes:
            {"type":"cookie","shared":N,"cap":35,"milestone":false}  (normal)
            {"type":"cookie","milestone":true}                        (fire)
        Fire-and-forget; drops silently if no clients connected."""
        if not self.enabled or not self._started or not self._clients:
            return
        if milestone:
            await self._broadcast({"type": "cookie", "milestone": True})
        else:
            await self._broadcast({"type": "cookie", "shared": int(shared), "cap": 35, "milestone": False})

    async def send_chaos(self, active: bool, remaining: int = 0) -> None:
        """Push Chaos Mode state to the overlay. Shape:
            {"type":"chaos","active":bool,"remaining":int_seconds}
        Fire-and-forget; drops silently if no clients connected."""
        if not self.enabled or not self._started or not self._clients:
            return
        await self._broadcast({"type": "chaos", "active": bool(active), "remaining": int(remaining)})

    async def _broadcast(self, frame: dict) -> None:
        payload = json.dumps(frame, ensure_ascii=False)
        dead = []
        # Snapshot the client set so we don't mutate during iteration.
        clients = list(self._clients)
        sent = 0
        for ws in clients:
            try:
                await ws.send(payload)
                sent += 1
            except Exception as e:
                dead.append(ws)
                print(f"   [Captions] Client send failed ({type(e).__name__}: {e}) — dropping client.")
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)
            print(f"   [Captions] Removed {len(dead)} dead client(s) ({len(self._clients)} remaining).")
        if frame.get("type") == "caption":
            print(f"   [Captions] Broadcast OK to {sent}/{len(clients)} client(s).")


# Module-level singleton so ai_core can import and call without holding a ref.
# bot.py owns the lifecycle (start/stop).
caption_server = CaptionServer()


def enqueue_caption(loop: Optional[asyncio.AbstractEventLoop], text: str, words: list[dict]) -> None:
    """Thread-safe entry point for ai_core to push a caption frame.

    Caption sends originate from the async TTS path which already runs on the
    main loop, but Azure's word-boundary callbacks fire on the SDK's worker
    thread \u2014 hence the explicit ``loop`` arg so we can hop back. If ``loop``
    is None we fall back to ``asyncio.get_event_loop()`` and log on failure.

    Never raises; never blocks.
    """
    if not caption_server.enabled:
        return
    try:
        target_loop = loop or asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(
            caption_server.send_caption(text, words),
            target_loop,
        )
    except Exception as e:
        print(f"   [Captions] enqueue failed (suppressed): {e}")


def enqueue_clear(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    if not caption_server.enabled:
        return
    try:
        target_loop = loop or asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(
            caption_server.send_clear(),
            target_loop,
        )
    except Exception:
        pass
