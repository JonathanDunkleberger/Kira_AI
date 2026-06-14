"""
control_server.py — Kira local web-dashboard backend
FastAPI + uvicorn running INSIDE the existing bot.event_loop.
Port: CONTROL_SERVER_PORT (default 8766), 127.0.0.1 only.

Architecture:
  GET  /state            — full JSON snapshot (one-shot, initial page load)
  WS   /ws               — 500ms state push stream
  GET  /vision/thumbnail — current frame as JPEG (poll at ~2s)
  POST /cmd/{action}     — all dashboard commands

This file is PURELY additive. It does not modify any agent logic.
caption_server.py on port 8765 is untouched.
F8/F9 global hotkeys registered by dashboard.py are untouched.
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
import traceback
from typing import TYPE_CHECKING, Any

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_DASHBOARD_HTML = Path(__file__).parent.parent.parent / 'web_dashboard' / 'index.html'

if TYPE_CHECKING:
    from kira.bot import VTubeBot

from kira.config import CONTROL_SERVER_PORT
from kira.senses.audio_agent import AUDIO_MODE_OFF, AUDIO_MODE_MEDIA, AUDIO_MODE_MUSIC
from kira.tools.music_tools import skip_song, clear_queue
from kira.persona.persona import EmotionalState
from kira.brain.game_mode_controller import ACTIVITY_VN, ACTIVITY_GAME, ACTIVITY_MEDIA, ACTIVITY_GENERAL

# ── Emotion → hex color (mirrors dashboard.py EMOTION_COLORS) ─────────────────
import kira.dashboard.theme as T
_EMOTION_COLORS: dict[str, str] = {
    "HAPPY":       T.EMOTION_HAPPY,
    "SASSY":       T.EMOTION_SASSY,
    "MOODY":       T.EMOTION_MOODY,
    "EMOTIONAL":   T.EMOTION_EMOTIONAL,
    "HYPERACTIVE": T.EMOTION_HYPERACTIVE,
}

# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPT DISPLAY HELPER
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

def _strip_user_wrapper(text: str) -> str:
    """Strip the prompt-framing wrapper stored in user turns of conversation_history.

    Stored formats observed in the wild:
      Voice path (main):
        [JONNY — your creator and the person you talk with, speaking to you]
        Jonny says: "ACTUAL MESSAGE"

      Vision bypass path (no bracket prefix):
        Jonny says: "ACTUAL MESSAGE"

    Strategy:
      - Match optional bracketed label (any content), then the literal
        `Jonny says: "` sentinel, then capture everything to the LAST `"`.
        Greedy `.+` naturally handles messages that themselves contain quotes.
      - If pattern does NOT match (assistant turns, legacy formats, anything
        unexpected), return the original text unchanged — never blank it out.
    """
    m = _re.search(
        r'(?:\[[^\]]*\]\s*)?Jonny says:\s*"(.+)"',
        text,
        _re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return text  # pass-through: never blank an unrecognised format


# ─────────────────────────────────────────────────────────────────────────────
# STATE SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def state_snapshot(bot: "VTubeBot") -> dict:
    """
    Read the same attributes the Tkinter _update_loop reads and return a
    flat JSON-serializable dict. Every field is individually guarded so one
    broken attribute never crashes the whole snapshot.
    """
    def _get(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    now = time.time()

    # ── Speaking + chat recency ───────────────────────────────────────────────
    is_speaking = _get(lambda: bool(bot.ai_core.is_speaking), False)
    _ts_list    = _get(lambda: bot.chat_msg_timestamps, [])
    since_last_chat_msg = _get(
        lambda: int(now - max(_ts_list)) if _ts_list else None, None
    )

    # ── Emotion ───────────────────────────────────────────────────────────────
    emotion_name = _get(lambda: bot.current_emotion.name, "HAPPY")
    emotion_color = _EMOTION_COLORS.get(emotion_name, T.TEXT_PRIMARY)

    # ── Chat rate / vibe meter ────────────────────────────────────────────────
    chat_rate = _get(lambda: round(bot.get_chat_rate_per_min(), 2), 0.0)
    last_spoke_ts = _get(lambda: bot.ai_core.last_speech_finish_time, 0)
    since_kira_spoke = int(now - last_spoke_ts) if last_spoke_ts and last_spoke_ts > 0 else None
    session_chatters = _get(lambda: len(bot.session_chatters_seen), 0)

    # ── Vision ────────────────────────────────────────────────────────────────
    vision_on = _get(lambda: bot.game_mode_controller.is_active, False)
    va = _get(lambda: bot.vision_agent, None)
    vision_summary = _get(
        lambda: (va.scene_summary or va.last_description or "").strip()
        if va else "", ""
    )
    vis_ts = _get(lambda: va.last_capture_time if va else 0, 0) or 0
    vision_age_s = int(now - vis_ts) if vis_ts > 0 else None

    # ── Audio / Hearing ───────────────────────────────────────────────────────
    aa = _get(lambda: bot.audio_agent, None)
    audio_on = _get(lambda: aa.is_active() if aa else False, False)
    audio_summary = _get(lambda: (aa.audio_summary or "").strip() if aa else "", "")
    audio_ts = _get(lambda: aa.last_capture_time if aa else 0, 0) or 0
    audio_age_s = int(now - audio_ts) if audio_ts > 0 else None
    audio_capture_count = _get(lambda: aa.capture_count if aa else 0, 0)

    # ── Loopback STT ──────────────────────────────────────────────────────────
    lt = _get(lambda: bot.loopback_transcriber, None)
    loopback_on = _get(lambda: lt.is_running() if lt else False, False)
    loopback_status = _get(lambda: lt.get_status_summary() if lt else "disabled", "disabled")

    def _loopback_feed():
        if not lt or not lt.is_running():
            return []
        import time as _t
        now = _t.time()
        out = []
        for seg in (lt.get_segments() or [])[-6:]:
            age = int(now - seg.get("ts", now))
            out.append({"age": age, "text": (seg.get("text", "") or "").strip()})
        return out

    loopback_feed = _get(_loopback_feed, [])
    loopback_summary = _get(lambda: (lt.get_dialogue_summary() or "") if lt else "", "")

    # ── Activity + mode flags ─────────────────────────────────────────────────
    activity = _get(lambda: bot.current_activity or "", "")
    mode = _get(lambda: bot.mode, "companion")
    carry_mode = _get(lambda: bot.carry_mode, False)
    immersive = _get(lambda: bot.immersive, False)
    presence_level = _get(lambda: bot.presence_level, "normal")

    # ── Effective (post-reconcile) state — the single truth both UIs render ───
    effective = _get(lambda: bot._compute_effective_state(), {}) or {}

    # ── Autopilot ─────────────────────────────────────────────────────────────
    ap = _get(lambda: bot.vn_autopilot, None)
    autopilot = {
        "running": _get(lambda: ap.is_running if ap else False, False),
        "paused":  _get(lambda: ap.is_paused if ap else False, False),
        "reason":  _get(lambda: ap.pause_reason if ap else None, None),
    }

    # ── Media Watch ───────────────────────────────────────────────────────────
    mw = _get(lambda: bot.media_watch, None)

    def _mw_latest():
        """Latest MW analysis entry as {text, age, tag} so the EYES panel can show
        Kira's real-time visual sync during Media Watch instead of the stale parked
        heartbeat description."""
        if not mw or not getattr(mw, "is_running", False):
            return None
        log = getattr(mw, "episode_log", None)
        if not log:
            return None
        e = log[-1]
        tag = "UNCERTAIN" if e.get("uncertain") else ("STATIC" if e.get("static") else "")
        return {
            "text": (e.get("summary", "") or "").strip(),
            "age": int(now - e.get("ts", now)),
            "tag": tag,
        }

    media_watch_state = {
        "running": _get(lambda: mw.is_running if mw else False, False),
        "status":  _get(lambda: mw.get_status_str() if mw else "OFF", "OFF"),
        "reactions": _get(lambda: getattr(mw, "reactions_enabled", True) if mw else False, False),
        "calls": _get(lambda: getattr(mw, "_calls_count", 0) if mw else 0, 0),
        "cost_usd": _get(lambda: round(getattr(mw, "_calls_cost_usd", 0.0), 3) if mw else 0.0, 0.0),
        "latest": _get(_mw_latest, None),
        "window": _get(lambda: (getattr(mw, "window_title", "") or "").strip() if mw else "", ""),
    }

    # ── Chess Mode ────────────────────────────────────────────────────────────
    ca = _get(lambda: bot.chess_agent, None)
    chess_state = {
        "running":               _get(lambda: ca.is_running if ca else False, False),
        "accepting_challenges":  _get(lambda: ca.accepting_challenges if ca else False, False),
        "status":                _get(lambda: ca.get_status_str() if ca else "\u265f CLOSED", "\u265f CLOSED"),
        "spectate_url":          _get(lambda: ca.get_spectate_url() if ca else "", ""),
        "score":                 _get(lambda: ca.get_score_data() if ca else {}, {}),
        "kira_elo":              _get(lambda: ca.kira_elo if ca else 1400, 1400),
    }

    # ── Mute / Pause ─────────────────────────────────────────────────────────
    muted = _get(lambda: bot.is_muted(), False)
    mute_remaining = _get(
        lambda: max(0, int(bot.mute_until - now))
        if bot.mute_until > now else 0,
        0
    )
    model_paused = _get(lambda: bot.is_paused, False)

    # ── Status bar ────────────────────────────────────────────────────────────
    llm_ready = _get(lambda: bot.ai_core.is_initialized, False)
    tts_backend = _get(lambda: bot.ai_core.tts_backend, "azure")
    from kira.config import ENABLE_TWITCH_CHAT
    twitch_on = _get(lambda: ENABLE_TWITCH_CHAT, False)
    fish_voice_id = _get(lambda: bot.ai_core.fish_voice_id or "", "")

    # VN agent on = vision active AND activity type is VN
    vn_agent_on = _get(
        lambda: (bot.game_mode_controller.is_active
                 and bot.game_mode_controller.activity_type == ACTIVITY_VN),
        False
    )

    # VRAM — whole-card via NVML (used/total), so headroom is visible during
    # AAA sessions. torch's allocator reads ~0 because the game isn't on torch.
    vram_used_gb = None
    vram_total_gb = None
    try:
        from kira.bot import read_gpu_memory_gb
        u, t = read_gpu_memory_gb()
        if u is not None and t is not None:
            vram_used_gb = round(u, 2)
            vram_total_gb = round(t, 1)
    except Exception:
        pass

    # ── YouTube ───────────────────────────────────────────────────────────────
    yt = _get(lambda: bot.youtube_bot, None)
    if yt is None:
        youtube_status = "disabled"
    elif _get(lambda: yt.running, False):
        vid = _get(lambda: yt.video_id or "", "")
        youtube_status = f"live({vid})" if vid else "live"
    else:
        youtube_status = "idle"
    yt_auto_status = _get(lambda: bot._yt_auto_search_status, "idle")

    # ── Chat queue instrumentation ────────────────────────────────────────────
    _age_log = _get(lambda: list(bot._chat_age_log[-20:]), [])
    if len(_age_log) >= 3:
        import statistics as _stat
        chat_median_age_s = round(_stat.median(_age_log), 1)
    else:
        chat_median_age_s = None

    # ── Music ─────────────────────────────────────────────────────────────────
    from kira.tools.music_tools import get_now_playing
    now_playing = _get(lambda: get_now_playing(), "Nothing")

    # ── Discord diary (Phase 1, review mode) ──────────────────────────────────
    discord_diary = {
        "text":   _get(lambda: getattr(bot, "pending_discord_summary", "") or "", ""),
        "path":   _get(lambda: getattr(bot, "pending_discord_summary_path", "") or "", ""),
        "posted": _get(lambda: bool(getattr(bot, "pending_discord_summary_posted", False)), False),
    }

    # ── Transcript (last 8 turns) ─────────────────────────────────────────────
    history = _get(lambda: bot.conversation_history, [])
    transcript = []
    for turn in history[-8:]:
        try:
            role = turn.get("role", "")
            raw = (turn.get("content") or "")
            text = (_strip_user_wrapper(raw) if role == "user" else raw)[:200]
            transcript.append({"role": role, "text": text})
        except Exception:
            pass

    return {
        # Emotion
        "emotion": emotion_name,
        "emotion_color": emotion_color,
        # Vibe meter
        "chat_rate": chat_rate,
        "since_kira_spoke": since_kira_spoke,
        "session_chatters": session_chatters,
        # Vision
        "vision_on": vision_on,
        "vision_summary": vision_summary,
        "vision_last_capture_age": vision_age_s,
        # Audio / hearing
        "audio_on": audio_on,
        "audio_summary": audio_summary,
        "audio_last_heard_age": audio_age_s,
        "audio_capture_count": audio_capture_count,
        # Loopback STT
        "loopback_on": loopback_on,
        "loopback_status": loopback_status,
        "loopback_feed": loopback_feed,
        "loopback_summary": loopback_summary,
        # Activity / mode
        "activity": activity,
        "mode": mode,
        "carry_mode": carry_mode,
        "immersive": immersive,
        "presence_level": presence_level,
        # Effective state (post-reconcile) — strip + three-state toggles render
        # from THIS, never from the raw toggle booleans above.
        "effective": effective,
        # Subsystem states
        "autopilot": autopilot,
        "media_watch": media_watch_state,
        "chess": chess_state,
        # Mute / pause
        "muted": muted,
        "mute_seconds_remaining": mute_remaining,
        "model_paused": model_paused,
        # Status bar
        "llm_ready": llm_ready,
        "tts_backend": tts_backend,
        "twitch_on": twitch_on,
        "vn_agent_on": vn_agent_on,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "youtube_status": youtube_status,
        "yt_auto_status": yt_auto_status,
        "chat_median_age_s": chat_median_age_s,
        "overlay_vis": dict(_overlay_vis),
        # TTS
        "fish_voice_id": fish_voice_id,
        # Music
        "now_playing": now_playing,
        # Discord diary (Phase 1 review mode)
        "discord_diary": discord_diary,
        # Transcript
        "transcript": transcript,
        # LLM cost telemetry
        "session_cost_usd": _get(lambda: (
            __import__("kira.brain.cost_tracker", fromlist=["cost_tracker"]).cost_tracker.session_cost_usd()
        ), 0.0),
        # Speaking state + chat recency (used by kira_cam CRT display)
        "is_speaking":       is_speaking,
        "since_last_chat_msg": since_last_chat_msg,
        # Ambient audio state
        "ambience":          dict(_ambience_state),
        # Cookie jar leaderboard + IOUs
        "cookie_top3": _get(lambda: (
            __import__("kira.memory.cookie_jar", fromlist=["cookie_jar"]).cookie_jar.get_session_top3()
        ), []),
        "ious_open": _get(lambda: (
            __import__("kira.memory.cookie_jar", fromlist=["cookie_jar"]).cookie_jar.open_iou_count()
        ), 0),
        # Server timestamp for the client
        "ts": round(now, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# APP + WEBSOCKET MANAGER
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Kira Control Server", version="1.0")

# ── Static overlay routes ────────────────────────────────────────────────────
# Serve the OBS browser-source overlays (chat / scoreboard / cookie jar) and
# their assets over HTTP so they can be pointed at
#   http://127.0.0.1:8766/web_dashboard/score_overlay.html
#   http://127.0.0.1:8766/web_dashboard/chat_overlay.html
#   http://127.0.0.1:8766/cookie_jar_overlay/cookie_jar_overlay.html
# instead of file://. Mounted at the directory names so every relative asset
# path resolves unchanged — including the cookie jar's ../web_dashboard/...
# reference. These are sub-path mounts, so they never shadow /state, /ws, etc.
# Server is 127.0.0.1-bound and these dirs contain only front-end assets.
_REPO_ROOT = Path(__file__).parent.parent.parent

# Short-URL aliases so OBS can use compact paths instead of full web_dashboard paths.
# FileResponse is used (not RedirectResponse) so OBS CEF never needs to follow a 302.
@app.get("/wheel")
async def _wheel_handler():
    return FileResponse(str(_REPO_ROOT / "web_dashboard" / "wheel_overlay.html"))

@app.get("/card_overlay")
async def _card_overlay_handler():
    # Redirect (not FileResponse) so relative asset paths (assets/cards/*.png)
    # resolve correctly against /web_dashboard/ in the browser.
    return RedirectResponse(url="/web_dashboard/card_overlay.html")

for _name in ("web_dashboard", "cookie_jar_overlay"):
    _dir = _REPO_ROOT / _name
    if _dir.is_dir():
        app.mount(f"/{_name}", StaticFiles(directory=str(_dir)), name=_name)

class _WSManager:
    """Tracks connected /ws clients and broadcasts state pushes."""
    def __init__(self):
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)

    async def broadcast(self, data: dict):
        if not self._clients:
            return
        import json
        payload = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


_ws_manager = _WSManager()

# Stream-screen overlay WS manager + in-memory override store.
# Keys: "starting" | "brb" | "ending"  →  {line1: str, line2: str}
_screen_ws_manager = _WSManager()
_screen_overrides: dict[str, dict] = {}

# Unified chat relay for chat_overlay browser source.
_chat_ws_manager = _WSManager()

# Overlay events relay — card_show/card_hide/banner_show/banner_hide/overlay_vis
_overlay_ws_manager = _WSManager()

# Ambient audio state (server-side authoritative copy).
# file: basename of file in web_dashboard/screens/ambience/, or "" for off.
_ambience_state: dict = {"file": "", "volume": 0.06}

# Overlay visibility state (server-side authoritative copy).
# False = hidden (fade); True = visible. Banner/spectate default AUTO (True = event-driven).
_overlay_vis: dict = {
    "chat":        True,
    "cards":       True,
    "banner":      True,
    "scoreboard":  True,
    "spectate":    True,
    "wheel":       True,
    "cookies_jar": True,
}


async def push_overlay_event(event: dict) -> None:
    """Push a raw overlay event to all /ws/overlays clients."""
    await _overlay_ws_manager.broadcast(event)


async def push_wheel_spin(slice_id: str, duration_ms: int = 3500,
                          tipper: str = "", label: str = "") -> None:
    """Broadcast a wheel_spin event to /ws/overlays clients."""
    await _overlay_ws_manager.broadcast({
        "type":        "wheel_spin",
        "result":      slice_id,
        "duration_ms": duration_ms,
        "tipper":      tipper,
        "label":       label,
    })


async def push_wheel_veto() -> None:
    """Broadcast a wheel_veto event — hides wheel overlay immediately."""
    await _overlay_ws_manager.broadcast({"type": "wheel_veto"})


async def push_card_show(chatter: str, message: str, platform: str) -> None:
    """Show a response-card for *chatter* on the card overlay.
    No-ops if overlay_vis['cards'] is False."""
    if not _overlay_vis.get("cards", True):
        return
    import time as _t
    n = len(_overlay_ws_manager._clients)
    print(f"   [CardOverlay] card_show → {n} WS client(s) connected to /ws/overlays")
    await push_overlay_event({
        "type":     "card_show",
        "chatter":  chatter,
        "message":  (message or "")[:200],
        "platform": platform,
        "ts":       round(_t.time(), 2),
    })


async def push_card_hide() -> None:
    """Signal TTS complete — overlay will hide the card once min_display has elapsed."""
    await push_overlay_event({"type": "card_hide"})


async def push_banner_show(text: str, duration_s: int = 8) -> None:
    """Display an event banner for *duration_s* seconds.
    No-ops if overlay_vis['banner'] is False."""
    if not _overlay_vis.get("banner", True):
        return
    await push_overlay_event({
        "type":       "banner_show",
        "text":       (text or "")[:160],
        "duration_s": duration_s,
    })


async def push_banner_hide() -> None:
    await push_overlay_event({"type": "banner_hide"})


async def push_cookie_drop(chatter: str = "", gold: bool = False) -> None:
    """Broadcast a cookie_drop to /ws/overlays so the card overlay can flash a
    '+1 🍪' badge attributed to *chatter*. The card overlay only flashes when
    *chatter* matches the chatter currently shown on its card — keeping the
    badge on the right person. Empty chatter means 'no attribution' (skipped)."""
    await push_overlay_event({
        "type":    "cookie_drop",
        "chatter": (chatter or ""),
        "gold":    bool(gold),
    })


async def push_score_update(
    session_wins:    int, session_losses:  int, session_draws:  int,
    lifetime_wins:   int, lifetime_losses: int, lifetime_draws: int,
    cookies: int = 0, cookies_max: int = 50,
    ious_open: int = 0,
) -> None:
    """Notify the score overlay of a change (chess game end or cookie update)."""
    await push_overlay_event({
        "type":           "score_update",
        "session_wins":    session_wins,
        "session_losses":  session_losses,
        "session_draws":   session_draws,
        "lifetime_wins":   lifetime_wins,
        "lifetime_losses": lifetime_losses,
        "lifetime_draws":  lifetime_draws,
        "cookies":         cookies,
        "cookies_max":     cookies_max,
        "ious_open":       ious_open,
    })


async def push_spectate_show(url: str, opponent: str) -> None:
    """Show the spectate embed for a viewer game."""
    if not _overlay_vis.get("spectate", True):
        return
    await push_overlay_event({
        "type":     "spectate_show",
        "url":      url,
        "opponent": opponent,
    })


async def push_spectate_hide() -> None:
    """Hide the spectate embed."""
    await push_overlay_event({"type": "spectate_hide"})


async def push_chat_message(platform: str, username: str, text: str) -> None:
    """Push a validated chat message to all connected /ws/chat clients.

    Called from brain_worker after the message has been parsed and filtered.
    Safe to call when no overlay is connected — no-ops cleanly.
    """
    import json as _json
    import time as _t
    await _chat_ws_manager.broadcast({
        "type":     "chat",
        "platform": platform,
        "username": username,
        "text":     text,
        "ts":       round(_t.time(), 2),
    })


def _bot() -> "VTubeBot":
    if _bot_ref is None:
        raise RuntimeError("control_server: bot not yet injected")
    return _bot_ref


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD UI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    """Serve the single-page web dashboard.

    no-store headers: the dashboard is a single inline HTML/JS/CSS document.
    Browsers were caching it, so UI fixes appeared not to 'stick' across
    sessions until a hard-refresh. Force a fresh fetch every load."""
    no_cache = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    try:
        return HTMLResponse(
            content=_DASHBOARD_HTML.read_text(encoding='utf-8'),
            headers=no_cache,
        )
    except FileNotFoundError:
        return HTMLResponse(
            content='<h1>Dashboard not found</h1><p>web_dashboard/index.html missing</p>',
            status_code=404,
        )


# ─────────────────────────────────────────────────────────────────────────────
# STATE ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/state")
async def get_state():
    """Full snapshot, one-shot (for initial page load)."""
    return JSONResponse(content=state_snapshot(_bot()))


@app.get("/screens/ambience/list")
async def list_ambience_files():
    """List audio files available in web_dashboard/screens/ambience/.
    Returns {files: ["name.mp3", ...]} sorted alphabetically."""
    import pathlib
    amb_dir = pathlib.Path("web_dashboard/screens/ambience")
    exts    = {".mp3", ".ogg", ".wav", ".flac"}
    if not amb_dir.is_dir():
        return JSONResponse(content={"files": []})
    files = sorted(
        p.name for p in amb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    return JSONResponse(content={"files": files})


@app.websocket("/ws")
async def ws_state(ws: WebSocket):
    """Push state_snapshot every 500ms. Handles disconnects gracefully."""
    await _ws_manager.connect(ws)
    try:
        while True:
            snap = state_snapshot(_bot())
            import json
            await ws.send_text(json.dumps(snap))
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass  # task cancelled during shutdown — not an error
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_manager.disconnect(ws)


@app.websocket("/ws/screens")
async def ws_screens(ws: WebSocket):
    """Stream-screen overlay connection.

    On connect: sends all current overrides so reconnecting screens restore
    their override state immediately. Then keeps the connection alive.
    """
    import json as _json
    await _screen_ws_manager.connect(ws)
    try:
        await ws.send_text(_json.dumps({
            "type": "overrides",
            "data": dict(_screen_overrides),
        }))
        while True:
            await ws.receive_text()  # keep alive
    except asyncio.CancelledError:
        pass  # task cancelled during shutdown — not an error
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _screen_ws_manager.disconnect(ws)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """Unified chat relay for the chat_overlay browser source.

    Messages are pushed by push_chat_message(); this endpoint just keeps the
    client connected and handles clean disconnect.
    """
    await _chat_ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep alive; client sends nothing
    except asyncio.CancelledError:
        pass  # task cancelled during shutdown — not an error
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _chat_ws_manager.disconnect(ws)


@app.websocket("/ws/overlays")
async def ws_overlays(ws: WebSocket):
    """Card / banner / visibility overlay relay.

    On connect: sends current overlay_vis + current chess score so reconnecting
    overlays restore their state immediately.  Events (card_show, card_hide,
    banner_show, banner_hide, spectate_show, spectate_hide, score_update,
    overlay_vis) are pushed by the push_* helpers above.
    """
    import json as _j
    await _overlay_ws_manager.connect(ws)
    print(f"   [Overlay] connected → /ws/overlays now has "
          f"{len(_overlay_ws_manager._clients)} client(s)")
    try:
        await ws.send_text(_j.dumps({"type": "overlay_vis", **_overlay_vis}))
        # Send current chess score state so the score overlay is correct on reconnect
        try:
            _bot_ref = _bot()
            _ca = getattr(_bot_ref, "chess_agent", None)
            if _ca:
                _sd = _ca.get_score_data()
                _cj = getattr(_bot_ref, "cookie_jar", None)
                _cookies     = int(_cj.get_shared()    if _cj else 0)
                _cookies_max = int(__import__("kira.memory.cookie_jar",
                                              fromlist=["MILESTONE_CAP"]).MILESTONE_CAP)
                await ws.send_text(_j.dumps({
                    "type": "score_update",
                    **_sd,
                    "cookies":     _cookies,
                    "cookies_max": _cookies_max,
                }))
        except Exception:
            pass
        while True:
            await ws.receive_text()  # keep alive; client sends nothing
    except asyncio.CancelledError:
        pass  # task cancelled during shutdown — not an error
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _overlay_ws_manager.disconnect(ws)
async def vision_thumbnail():
    """
    Returns the latest vision frame as a JPEG image (Content-Type: image/jpeg).
    Returns 204 No Content if no frame is available.
    Polled by the browser at ~2s — matches the existing _vision_loop cadence.
    NOT included in the 500ms state push.
    """
    try:
        va = _bot().vision_agent
        frame = getattr(va, "last_frame", None)
        if frame is None:
            return Response(status_code=204)
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=70)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/jpeg")
    except Exception:
        return Response(status_code=204)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class _CmdBody(BaseModel):
    """Generic command body — all fields optional. Each endpoint uses only what it needs."""
    class Config:
        extra = "allow"

    # Activity
    name: str | None = None
    slug: str | None = None
    # Audio
    mode: str | None = None          # hearing mode label
    label: str | None = None         # audio device label
    # Autopilot / pacing
    enabled: bool | None = None
    title: str | None = None
    key: str | None = None
    base: float | None = None
    max: float | None = None
    seconds: float | None = None
    # TTS / voice
    voice_id: str | None = None
    # Emotion
    emotion: str | None = None
    # YouTube
    url: str | None = None
    # Chess
    level: int | None = None
    # Codenames
    words: list[str] | str | None = None   # 25-word grid (list or delimited string)
    word: str | None = None                # a single board word
    identity: str | None = None            # team | opponent | neutral | assassin
    role: str | None = None                # guesser | spymaster
    team: str | None = None                # cosmetic team label (red/blue)
    clue: str | None = None                # clue word
    number: int | None = None              # clue number
    by: str | None = None                  # "me" | "opponent"
    targets: list[str] | str | None = None # intended clue targets (safety check)
    # Storytime / Puppet Show
    theme: str | None = None               # one-line story theme
    beats: int | None = None               # how many scene beats to write
    idx: int | None = None                 # beat index (regenerate one scene)
    preset: str | None = None              # structure/tone preset key
    note: str | None = None                # operator note for a single-beat rewrite
    narration: str | None = None           # hand-edited narration for one beat
    show_id: str | None = None             # saved-show folder id (load from library)

    # ── Stream screen fields (for screen_text command)
    screen: str | None = None
    line1:  str | None = None
    line2:  str | None = None

    # ── Generic text payload (test_banner, etc.)
    text:   str | None = None

    # ── Presence dial level (sleepy | normal | chatty)
    presence: str | None = None


def _ok(**kwargs) -> dict:
    return {"ok": True, **kwargs}

def _err(msg: str, **kwargs) -> dict:
    return {"ok": False, "error": msg, **kwargs}


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# Actions that change which subsystem owns perception/agenda → re-run the
# cross-mode reconciler afterward.
_MODE_ACTIONS = frozenset({
    "activity_go", "exit_game_mode", "vision_toggle", "loopback_toggle",
    "passive_watching_toggle", "carry_mode_toggle", "autopilot_toggle",
    "media_watch_toggle", "media_watch_react_toggle", "chess_toggle",
    "chess_accept_toggle",
})


@app.post("/cmd/{action}")
async def cmd(action: str, body: _CmdBody = _CmdBody()):
    """
    Central command dispatcher. Each action maps 1:1 to the same bot call
    the Tkinter handler made. Returns {ok: true} or {ok: false, error: "..."}.
    """
    bot = _bot()
    try:
        result = await _dispatch(action, body, bot)
        # Re-assert cross-mode invariants after any toggle that changes which
        # subsystem owns perception / agenda. _reconcile_modes() is idempotent
        # and order-independent, so calling it here makes the web dashboard
        # converge to the same state as the desktop dashboard regardless of the
        # order toggles are flipped in.
        if action in _MODE_ACTIONS and hasattr(bot, "_reconcile_modes"):
            try:
                bot._reconcile_modes(trigger=action)
            except Exception as _re:
                print(f"   [Reconcile] error after {action}: {_re}")
        return result
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content=_err(f"Internal error in cmd/{action}: {exc}"),
        )


async def _dispatch(action: str, body: _CmdBody, bot: "VTubeBot") -> dict:  # noqa: C901
    # ── Mode ──────────────────────────────────────────────────────────────────
    if action == "mode_toggle":
        if bot.mode == "companion":
            bot.mode = "streamer"
        else:
            bot.mode = "companion"
        return _ok(mode=bot.mode)

    # ── Activity ──────────────────────────────────────────────────────────────
    if action == "activity_go":
        name = (body.name or "").strip()
        slug = (body.slug or "").strip()
        if name:
            new_type = bot.activate_game_mode(name, known_slug=slug)
        else:
            bot.current_activity = ""
            new_type = ACTIVITY_GENERAL
        return _ok(activity=bot.current_activity, activity_type=new_type)

    if action == "exit_game_mode":
        await bot.deactivate_game_mode_async()
        return _ok()

    # ── Vision ────────────────────────────────────────────────────────────────
    if action == "vision_toggle":
        if bot.game_mode_controller.is_active:
            bot.game_mode_controller.deactivate()
        else:
            bot.game_mode_controller.activate(bot.game_mode_controller.activity_type)
        return _ok(vision_on=bot.game_mode_controller.is_active)

    # ── Audio / Hearing ───────────────────────────────────────────────────────
    if action == "audio_mode":
        if not bot.audio_agent:
            return _err("audio_agent disabled in config")
        label_to_mode = {
            "Off": AUDIO_MODE_OFF,
            "Media (game/anime)": AUDIO_MODE_MEDIA,
            "Music (singing/guitar)": AUDIO_MODE_MUSIC,
        }
        choice = body.mode or "Off"
        mode_val = label_to_mode.get(choice, AUDIO_MODE_OFF)
        bot.audio_agent.set_mode(mode_val)
        return _ok(hearing=choice)

    if action == "audio_device":
        if not bot.audio_agent:
            return _err("audio_agent disabled in config")
        label = body.label or ""
        if not label or label == "Auto-detect":
            bot.audio_agent.preferred_loopback_name = None
        else:
            cleaned = label.replace("⚠ ", "").replace(" (virtual)", "").rstrip(".")
            bot.audio_agent.preferred_loopback_name = cleaned
        return _ok(device=label)

    if action == "audio_devices_refresh":
        if not bot.audio_agent:
            return _err("audio_agent disabled in config")
        devices = bot.audio_agent.list_available_loopback_devices()
        labels = ["Auto-detect"]
        for dname, is_virtual in (devices or []):
            short = dname[:40] + ("..." if len(dname) > 40 else "")
            if is_virtual:
                short = f"⚠ {short} (virtual)"
            labels.append(short)
        return _ok(devices=labels)

    # ── Loopback STT ──────────────────────────────────────────────────────────
    if action == "loopback_toggle":
        lt = bot.loopback_transcriber
        if lt is None:
            return _err("Loopback STT disabled in config (ENABLE_LOOPBACK_TRANSCRIBER=false)")
        if lt.is_running():
            # Stop on a thread so we don't block the event loop during model unload
            await asyncio.to_thread(lt.stop)
            return _ok(loopback_on=False)
        else:
            if not bot.audio_agent or not bot.audio_agent.is_active():
                return _err("Enable Audio Hearing (Media mode) first")
            ai_core_ref = bot.ai_core
            speaking_fn = lambda: bool(getattr(ai_core_ref, "is_speaking", False))
            ok = await asyncio.to_thread(lt.start, bot.audio_agent, speaking_fn)
            return _ok(loopback_on=ok) if ok else _err("Loopback STT failed to start — check logs")

    # ── Passive Watching ──────────────────────────────────────────────────────
    if action == "passive_watching_toggle":
        bot.immersive = not bot.immersive
        return _ok(immersive=bot.immersive)

    # ── Carry Mode ────────────────────────────────────────────────────────────
    if action == "carry_mode_toggle":
        bot.carry_mode = not bot.carry_mode
        return _ok(carry_mode=bot.carry_mode)

    # ── Presence dial ─────────────────────────────────────────────────────────
    if action == "set_presence":
        level = str(getattr(body, "presence", None) or "").strip().lower()
        if level not in ("sleepy", "normal", "chatty"):
            return _err("presence must be one of: sleepy, normal, chatty")
        bot.presence_level = level
        return _ok(presence_level=bot.presence_level)

    # ── Discord diary (Phase 1, REVIEW MODE) ──────────────────────────────────
    # Manual post of the pending diary entry. Fires the webhook ONLY here, on a
    # deliberate dashboard click — never automatically at session end.
    if action == "post_discord_summary":
        text = (getattr(body, "text", None) or getattr(bot, "pending_discord_summary", "") or "").strip()
        if not text:
            return _err("no diary entry to post — generate one at session end first")
        from kira.streaming.discord_poster import post_discord_message
        ok, detail = await post_discord_message(text)
        if ok:
            bot.pending_discord_summary_posted = True
        return (_ok(detail=detail) if ok else _err(detail))

    # ── VN Autopilot ──────────────────────────────────────────────────────────
    if action == "autopilot_toggle":
        ap = bot.vn_autopilot
        if ap is None:
            return _err("vn_autopilot not initialized yet")
        enabled = body.enabled if body.enabled is not None else (not ap.enabled)
        title = (body.title or "").strip()
        if title:
            ap.vn_window_title = title
        ap.enabled = enabled
        if enabled:
            bot.autopilot_paused_for_input = False
            bot.event_loop.call_soon_threadsafe(ap.start)
        else:
            bot.event_loop.call_soon_threadsafe(ap.stop)
        return _ok(autopilot_enabled=enabled)

    if action == "vn_window":
        ap = bot.vn_autopilot
        if ap is None:
            return _err("vn_autopilot not initialized yet")
        title = (body.title or "").strip()
        ap.vn_window_title = title
        return _ok(vn_window_title=title)

    if action == "vn_redetect":
        ap = bot.vn_autopilot
        if ap is None:
            return _err("vn_autopilot not initialized yet")
        detected = await ap._autodetect_vn_window()
        if detected and ap:
            ap.vn_window_title = detected
        return _ok(detected_title=detected)

    if action == "advance_key":
        ap = bot.vn_autopilot
        if ap is None:
            return _err("vn_autopilot not initialized yet")
        key_map = {"Space": "space", "Enter": "enter", "Left Click": "click"}
        new_key = key_map.get(body.key or "Enter", "enter")
        ap.input_controller.set_advance_key(new_key)
        ap._working_advance_method = None
        try:
            ap._recent_advance_hashes.clear()
        except Exception:
            pass
        return _ok(advance_key=new_key)

    if action == "autopilot_pacing":
        ap = bot.vn_autopilot
        if ap is None:
            return _err("vn_autopilot not initialized yet")
        if body.base is not None:
            ap.pacing_base = float(body.base)
        if body.max is not None:
            ap.pacing_max = float(body.max)
        return _ok(pacing_base=ap.pacing_base, pacing_max=ap.pacing_max)

    if action == "autopilot_resume":
        ap = bot.vn_autopilot
        if ap is None:
            return _err("vn_autopilot not initialized yet")
        if not ap.is_paused:
            return _err("autopilot is not paused")
        bot.autopilot_paused_for_input = False
        bot.event_loop.call_soon_threadsafe(ap.resume_after_failsafe)
        return _ok()

    # ── Media Watch ───────────────────────────────────────────────────────────
    if action == "media_watch_toggle":
        mw = bot.media_watch
        if mw is None:
            return _err("media_watch not initialized yet")
        enabled = body.enabled if body.enabled is not None else (not mw.enabled)
        title = (body.title or "").strip() or getattr(mw, "window_title", "")
        auto_targeted = False
        if enabled and not title:
            # C1: empty field → auto-target the current foreground window so the
            # text field is an OVERRIDE, not a requirement. Write the chosen
            # title back so the dashboard shows what was picked.
            fg = mw.get_foreground_window_title()
            if fg:
                title = fg
                auto_targeted = True
                print(f"   [MediaWatch] Auto-targeted foreground window: '{title}'")
            else:
                mw.enabled = False
                return _err("No window title set and no foreground window detected")
        mw.window_title = title
        mw.enabled = enabled
        if enabled:
            bot.event_loop.call_soon_threadsafe(mw.start)
        else:
            bot.event_loop.call_soon_threadsafe(mw.stop)
        return _ok(media_watch_enabled=enabled, window_title=title, auto_targeted=auto_targeted)

    if action == "media_watch_window":
        mw = bot.media_watch
        if mw is None:
            return _err("media_watch not initialized yet")
        mw.window_title = (body.title or "").strip()
        return _ok(window_title=mw.window_title)

    if action == "media_watch_interval":
        mw = bot.media_watch
        if mw is None:
            return _err("media_watch not initialized yet")
        if body.seconds is not None:
            mw.analysis_interval_s = float(body.seconds)
        return _ok(interval_s=mw.analysis_interval_s)

    if action == "media_watch_react_toggle":
        mw = bot.media_watch
        if mw is None:
            return _err("media_watch not initialized yet")
        # State-explicit: the React-to-scenes switch is mw.reactions_enabled (a
        # real backing bool), NOT the presence/absence of the on_react handler.
        # on_react stays wired for the whole session; reactions_enabled gates it.
        if body.enabled is not None:
            mw.reactions_enabled = bool(body.enabled)
        else:
            mw.reactions_enabled = not getattr(mw, "reactions_enabled", True)
        return _ok(reactions_on=mw.reactions_enabled)

    # ── Chess Mode ─────────────────────────────────────────────────────────────
    if action == "chess_toggle":
        ca = bot.chess_agent
        if ca is None:
            return _err("chess_agent not initialized yet")
        enabled = body.enabled if body.enabled is not None else (not ca.enabled)
        if enabled:
            # Mutually exclusive with Media Watch and VN autopilot.
            mw = bot.media_watch
            if mw is not None and mw.is_running:
                return _err("Media Watch is running — stop it before arming Chess Mode")
            ap = bot.vn_autopilot
            if ap is not None and ap.is_running:
                return _err("VN autopilot is running — stop it before arming Chess Mode")
            ca.enabled = True
            bot.event_loop.call_soon_threadsafe(ca.start)
        else:
            bot.event_loop.call_soon_threadsafe(ca.stop)
        return _ok(chess_enabled=enabled)

    if action == "chess_accept_toggle":
        # Master switch for challenge acceptance. Default OFF each boot.
        # Toggling ON = "we're doing chess now"; OFF = polite declines resume.
        ca = bot.chess_agent
        if ca is None:
            return _err("chess_agent not initialized yet")
        if not ca.is_running:
            return _err("Chess Mode is not armed — enable Chess Mode first")
        on = body.enabled if body.enabled is not None else (not ca.accepting_challenges)
        ca.accepting_challenges = bool(on)
        state = "OPEN" if on else "CLOSED"
        print(f"   [Chess] Challenge acceptance: {state}")
        return _ok(accepting_challenges=on)

    if action == "chess_set_elo":
        ca = bot.chess_agent
        if ca is None:
            return _err("chess_agent not initialized yet")
        new_elo = int(body.level) if body.level is not None else 1400
        bot.event_loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(ca.update_elo(new_elo))
        )
        return _ok(kira_elo=new_elo)

    if action == "chess_challenge_ai":
        ca = bot.chess_agent
        if ca is None:
            return _err("chess_agent not initialized yet")
        if not ca.is_running:
            return _err("Chess Mode is not armed")
        level = int(body.level) if body.level is not None else 3
        # challenge_ai bypasses the accepting_challenges gate intentionally —
        # Jonny clicking this button is explicit consent. Same event path as a
        # human challenge: gameStart arrives on the stream, voice-line fires,
        # chip updates. Test path == real path.
        bot.event_loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(ca.challenge_ai(level))
        )
        return _ok(challenged_level=level)

    # ── Codenames (structured board tracker) ──────────────────────────────────
    # Kira reasons over this persistent in-memory board instead of re-reading a
    # single (often stale) vision frame each turn. These actions drive the model.
    if action.startswith("codenames_"):
        cn = getattr(bot, "codenames", None)
        if cn is None:
            return _err("codenames tracker not initialized")

        def _as_list(val) -> list[str]:
            if val is None:
                return []
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            # Accept comma- or whitespace-delimited strings from the dashboard.
            return [p.strip() for p in str(val).replace(",", " ").split() if p.strip()]

        if action == "codenames_start":
            cn.start(_as_list(body.words),
                     role=(body.role or "guesser"),
                     my_team_label=(body.team or ""))
            return _ok(**cn.snapshot())

        if action == "codenames_reset":
            cn.reset()
            return _ok(active=False)

        if action == "codenames_set_role":
            cn.set_role(body.role or "guesser")
            return _ok(role=cn.role)

        if action == "codenames_set_grid":
            cn.set_grid(_as_list(body.words))
            return _ok(**cn.snapshot())

        if action == "codenames_reveal":
            if not body.word or not body.identity:
                return _err("reveal needs 'word' and 'identity'")
            ok = cn.reveal(body.word, body.identity)
            return _ok(**cn.snapshot()) if ok else _err("invalid identity")

        if action == "codenames_clue":
            if not body.clue:
                return _err("clue needs 'clue'")
            cn.record_clue(body.clue, body.number or 0, by=(body.by or "me"))
            return _ok(**cn.snapshot())

        if action == "codenames_guess":
            if not body.word:
                return _err("guess needs 'word'")
            res = cn.record_guess(body.word, body.identity)
            return _ok(result=res, **cn.snapshot())

        if action == "codenames_check_clue":
            if not body.clue:
                return _err("check_clue needs 'clue'")
            return _ok(**cn.check_clue(body.clue, _as_list(body.targets)))

        if action == "codenames_state":
            return _ok(**cn.snapshot())

        return _err(f"Unknown codenames action '{action}'")

    # ── Storytime / Puppet Show ───────────────────────────────────────────────
    # GENERATE-THEN-PERFORM, review-gated. prepare writes a script + pre-gens all
    # scene images; the dashboard reviews/re-rolls; perform plays it live. Long
    # async work is scheduled onto the bot loop; the dashboard polls *_state.
    if action.startswith("storytime_"):
        st = getattr(bot, "storytime", None)
        # LOUD: every storytime click prints here so the terminal proves the
        # request reached the backend at all (silence = click never arrived).
        print(f"   [Storytime] ▶ action='{action}' theme={body.theme!r} "
              f"beats={body.beats} preset={body.preset!r} idx={body.idx} "
              f"(st={'ok' if st is not None else 'MISSING'})")
        if st is None:
            print("   [Storytime] ✖ bot.storytime is None — not initialized")
            return _err("storytime not initialized")

        if action == "storytime_state":
            return _ok(**st.snapshot())

        if action == "storytime_presets":
            from kira.storytime.script_writer import list_presets
            return _ok(presets=list_presets())

        if action == "storytime_library":
            return _ok(shows=st.list_library())

        if action == "storytime_load":
            if not body.show_id:
                return _err("load needs 'show_id'")
            if st.snapshot().get("busy"):
                return _err("Storytime is busy — wait for the current step to finish")
            ok = st.load_show(body.show_id)
            if not ok:
                return _err("Could not load that saved show (missing or no scenes)")
            return _ok(**st.snapshot())

        if action == "storytime_prepare":
            if st.snapshot().get("busy"):
                print("   [Storytime] ✖ prepare rejected — already busy")
                return _err("Storytime is busy — wait for the current step to finish")
            theme = (body.theme or body.text or body.title or "").strip()
            n_beats = int(body.beats) if body.beats is not None else 16
            preset = (body.preset or "").strip()
            print(f"   [Storytime] ✎ Generate Show scheduled — theme={theme!r} "
                  f"beats={n_beats} preset={preset!r}")
            bot.event_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(st.prepare(theme, n_beats, preset))
            )
            return _ok(status="scripting")

        if action == "storytime_regenerate":
            if body.idx is None:
                return _err("regenerate needs 'idx'")
            if st.snapshot().get("busy"):
                return _err("Storytime is busy — wait for the current step to finish")
            bot.event_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(st.regenerate_beat(int(body.idx)))
            )
            return _ok(status="generating")

        if action == "storytime_edit_beat":
            if body.idx is None:
                return _err("edit_beat needs 'idx'")
            ok = st.edit_beat_narration(int(body.idx), body.narration or "")
            if not ok:
                return _err("beat index out of range")
            return _ok(status="edited")

        if action == "storytime_rewrite_beat":
            if body.idx is None:
                return _err("rewrite_beat needs 'idx'")
            if st.snapshot().get("busy"):
                return _err("Storytime is busy — wait for the current step to finish")
            bot.event_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    st.rewrite_beat_script(int(body.idx), (body.note or "").strip())
                )
            )
            return _ok(status="generating")

        if action == "storytime_perform":
            snap = st.snapshot()
            if snap.get("status") not in ("ready", "done"):
                print(f"   [Storytime] ✖ perform rejected — status={snap.get('status')!r} "
                      f"progress={snap.get('progress')} (need a generated show first)")
                return _err("Nothing to perform — prepare a show first")
            speak = bot.ai_core.speak_text
            print(f"   [Storytime] ▶ Perform Live starting — \"{snap.get('title')}\" "
                  f"({snap.get('progress')} beats ready)")

            # Logged push wrapper: prints how many /ws/overlays clients each scene
            # is broadcast to, so a "nothing in OBS" can be told apart (0 clients =
            # overlay not connected; >0 = it's receiving and the issue is rendering).
            async def _logged_push(event: dict) -> None:
                n = len(_overlay_ws_manager._clients)
                etype = event.get("type", "?")
                if etype == "scene_show":
                    print(f"   [Storytime] scene_show beat {event.get('idx')} "
                          f"→ {n} overlay client(s): {event.get('src')}")
                elif etype == "scene_hide":
                    print(f"   [Storytime] scene_hide → {n} overlay client(s)")
                await push_overlay_event(event)

            bot.event_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(st.perform(speak, _logged_push))
            )
            return _ok(status="performing")

        if action == "storytime_stop":
            st.stop()
            return _ok(status="stopping")

        if action == "storytime_reset":
            st.reset()
            return _ok(status="idle")

        return _err(f"Unknown storytime action '{action}'")

    # ── Interrupt / Mute / Pause ──────────────────────────────────────────────
    # NOTE: F8/F9 global hotkeys registered in dashboard.py are UNTOUCHED.
    # These are SECONDARY triggers that call the exact same bot methods.
    if action == "interrupt":
        bot.interrupt()
        return _ok()

    if action == "mute_toggle":
        if bot.is_paused:
            return _err("Pause Model is active — use pause_toggle to release")
        if bot.is_muted():
            bot.unmute()
            return _ok(muted=False)
        else:
            bot.mute_for(60)
            return _ok(muted=True, mute_seconds=60)

    if action == "pause_toggle":
        if bot.is_paused:
            bot.resume_model()
        else:
            bot.pause_model()
        return _ok(model_paused=bot.is_paused)

    # ── Stream opener / closer ────────────────────────────────────────────────
    if action == "stream_start":
        await bot.run_stream_opener()
        return _ok()

    if action == "stream_end":
        await bot.run_stream_closer()
        return _ok()

    # ── Invite / Thoughts ─────────────────────────────────────────────────────
    if action == "invite_kira":
        await bot.request_thoughts()
        return _ok()

    # ── TTS ───────────────────────────────────────────────────────────────────
    if action == "tts_toggle":
        current = getattr(bot.ai_core, "tts_backend", "azure")
        bot.ai_core.tts_backend = "fish" if current == "azure" else "azure"
        return _ok(tts_backend=bot.ai_core.tts_backend)

    if action == "fish_voice_apply":
        vid = (body.voice_id or "").strip()
        if not vid:
            return _err("voice_id is required")
        bot.ai_core.fish_voice_id = vid
        return _ok(fish_voice_id=vid)

    if action == "reload_personality":
        bot.ai_core.reload_personality()
        return _ok()

    # ── Emotion ───────────────────────────────────────────────────────────────
    if action == "emotion_set":
        name = (body.emotion or "").upper().strip()
        try:
            new_state = EmotionalState[name]
        except KeyError:
            valid = [e.name for e in EmotionalState]
            return _err(f"Unknown emotion '{name}'. Valid: {valid}")
        bot.current_emotion = new_state
        try:
            bot.vts_expressions.fire_and_forget(new_state, loop=bot.event_loop)
        except Exception:
            pass
        return _ok(emotion=new_state.name)

    # ── Music ─────────────────────────────────────────────────────────────────
    if action == "skip_song":
        await asyncio.to_thread(skip_song)
        return _ok()

    if action == "clear_queue":
        await asyncio.to_thread(clear_queue)
        return _ok()

    # ── YouTube ───────────────────────────────────────────────────────────────
    if action == "youtube_connect":
        yt = bot.youtube_bot
        if yt is None:
            return _err("youtube_bot not initialized (ENABLE_YOUTUBE_CHAT=false?)")
        url = (body.url or "").strip()
        if not url:
            return _err("url is required")
        ok = yt.start(url)
        vid = yt.video_id if ok else None
        if ok:
            return _ok(video_id=vid, youtube_status=f"live({vid})")
        else:
            return _err("YouTube connect failed — check URL/video ID", youtube_status="failed")

    if action == "youtube_disconnect":
        yt = bot.youtube_bot
        if yt is None:
            return _err("youtube_bot not initialized")
        yt.stop()
        return _ok(youtube_status="idle")

    # ── Stream screen text ────────────────────────────────────────────────────
    _VALID_SCREENS = {"starting", "brb", "ending"}

    if action == "screen_text":
        screen = (body.screen or "").strip()
        if screen not in _VALID_SCREENS:
            return _err(f"Unknown screen '{screen}'. Valid: {sorted(_VALID_SCREENS)}")
        line1 = (body.line1 or "").strip()
        line2 = (body.line2 or "").strip()
        _screen_overrides[screen] = {"line1": line1, "line2": line2}
        await _screen_ws_manager.broadcast({
            "type":   "screen_text",
            "screen": screen,
            "line1":  line1,
            "line2":  line2,
        })
        return _ok(screen=screen)

    if action == "screen_text_clear":
        screen = (body.screen or "").strip()
        if screen not in _VALID_SCREENS:
            return _err(f"Unknown screen '{screen}'. Valid: {sorted(_VALID_SCREENS)}")
        _screen_overrides.pop(screen, None)
        await _screen_ws_manager.broadcast({
            "type":   "screen_text_clear",
            "screen": screen,
        })
        return _ok(screen=screen)

    # ── Overlay visibility ────────────────────────────────────────────────────
    if action == "overlay_vis":
        import json as _j
        key   = (body.key   or "").strip()
        value = body.enabled
        if key not in _overlay_vis:
            return _err(f"Unknown overlay key '{key}'. Valid: {sorted(_overlay_vis)}")
        if value is None:
            return _err("'enabled' bool required")
        _overlay_vis[key] = bool(value)
        asyncio.ensure_future(_overlay_ws_manager.broadcast(
            {"type": "overlay_vis", **_overlay_vis}
        ))
        return _ok(overlay_vis=dict(_overlay_vis))

    # ── Wheel test spin (dashboard / debug) ────────────────────────────────────
    if action == "wheel_spin_test":
        from kira.memory.wheel_slices import spin as _spin_wheel
        chosen = _spin_wheel()
        asyncio.ensure_future(push_overlay_event({
            "type":        "wheel_spin",
            "result":      chosen["id"],
            "label":       chosen["label"],
            "tipper":      "(test)",
            "duration_ms": 3500,
        }))
        return _ok(slice=chosen["id"], label=chosen["label"])

    # ── Wheel veto ────────────────────────────────────────────────────────────
    if action == "wheel_veto":
        # Broadcast wheel_veto to overlays
        asyncio.ensure_future(push_wheel_veto())
        # Also set _wheel_vetoed on the running bot instance if accessible
        try:
            import kira.bot as _bot_mod
            bot_ref = getattr(_bot_mod, "_GLOBAL_BOT_REF", None)
            if bot_ref is not None:
                bot_ref._wheel_vetoed = True
        except Exception:
            pass
        return _ok(message="wheel vetoed")

    # ── IOU redeem ────────────────────────────────────────────────────────────
    if action == "redeem_iou":
        idx = getattr(body, "idx", None)
        if idx is None:
            return _err("'idx' required")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return _err("'idx' must be an integer")
        try:
            from kira.memory.cookie_jar import cookie_jar as _cj
            ok = _cj.redeem_iou(idx)
            return _ok(redeemed=ok) if ok else _err(f"IOU {idx} not found or not open")
        except Exception as e:
            return _err(str(e))

    # ── Test overlay events (card / banner — debug / OBS setup verification) ───
    if action == "test_card":
        asyncio.ensure_future(push_card_show(
            chatter="Test",
            message="This is a test response card. If you can see this, card_overlay is working!",
            platform="twitch",
        ))
        # Auto-hide after MIN_DISPLAY_MS
        async def _auto_hide_card():
            import asyncio as _aio
            await _aio.sleep(5)
            await push_card_hide()
        asyncio.ensure_future(_auto_hide_card())
        return _ok(message="test card fired")

    if action == "test_banner":
        text = (getattr(body, 'text', None) or "🎉 Test Banner — Event banner is working!")[:160]
        asyncio.ensure_future(push_banner_show(text, duration_s=6))
        return _ok(message="test banner fired")

    # ── Ambience control ──────────────────────────────────────────────────────
    if action == "set_ambience":
        file   = str(getattr(body, "file",   "") or "").strip()
        volume = getattr(body, "volume", None)
        _ambience_state["file"] = file
        if volume is not None:
            try:
                _ambience_state["volume"] = max(0.0, min(1.0, float(volume)))
            except (TypeError, ValueError):
                pass
        return _ok(ambience=dict(_ambience_state))

    # ── Unknown action ────────────────────────────────────────────────────────
    return JSONResponse(
        status_code=404,
        content=_err(f"Unknown action '{action}'"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SERVER STARTUP  (called once from bot._main_loop)
# ─────────────────────────────────────────────────────────────────────────────

async def start_control_server(bot: "VTubeBot") -> None:
    """
    Start the FastAPI/uvicorn server inside the EXISTING asyncio event loop.
    Called by bot._main_loop via asyncio.ensure_future() so it runs as a
    background task alongside all other bot coroutines.
    Never raises — a startup failure is logged and the bot continues normally.
    """
    global _bot_ref
    _bot_ref = bot

    try:
        import uvicorn
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=CONTROL_SERVER_PORT,
            loop="none",       # reuse the already-running asyncio event loop
            log_level="warning",
            access_log=False,  # keep bot console clean during streams
        )
        server = uvicorn.Server(config)
        print(f"   [ControlServer] Listening on http://127.0.0.1:{CONTROL_SERVER_PORT}")
        print(f"   [ControlServer] Endpoints: /state  /ws  /vision/thumbnail  POST /cmd/{{action}}")
        await server.serve()
    except Exception as e:
        print(f"   [ControlServer] Failed to start: {e} — dashboard will still work via Tkinter")
