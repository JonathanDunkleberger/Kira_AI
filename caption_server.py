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
        if not self.enabled or self._started:
            return self._started
        try:
            # Bind to ALL local interfaces so both Chrome (localhost / 127.0.0.1)
            # and OBS's embedded CEF browser can connect. OBS-CEF sometimes
            # resolves "localhost" to ::1 while a 127.0.0.1-only bind silently
            # refuses it \u2014 binding to both v4+v6 wildcards avoids the quirk.
            # We pass a list of hosts so websockets binds multiple sockets.
            self._server = await websockets.serve(
                self._handle_client,
                host=["127.0.0.1", "::1", "localhost"],
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
        except Exception as e:
            print(f"   [Captions] Failed to start server: {e} \u2014 captions disabled for this session.")
            self.enabled = False
            return False

    async def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.close()
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None
        self._started = False

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
        if not self.enabled or not self._started or not text or not words:
            return
        if not self._clients:
            return  # No overlay open in OBS \u2014 silently drop.
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

    async def _broadcast(self, frame: dict) -> None:
        payload = json.dumps(frame, ensure_ascii=False)
        dead = []
        # Snapshot the client set so we don't mutate during iteration.
        clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


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
