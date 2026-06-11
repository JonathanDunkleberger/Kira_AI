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
from fastapi.responses import JSONResponse, Response, HTMLResponse
from pydantic import BaseModel

_DASHBOARD_HTML = Path(__file__).parent / 'web_dashboard' / 'index.html'

if TYPE_CHECKING:
    from bot import VTubeBot

from config import CONTROL_SERVER_PORT
from audio_agent import AUDIO_MODE_OFF, AUDIO_MODE_MEDIA, AUDIO_MODE_MUSIC
from music_tools import skip_song, clear_queue
from persona import EmotionalState
from game_mode_controller import ACTIVITY_VN, ACTIVITY_GAME, ACTIVITY_MEDIA, ACTIVITY_GENERAL

# ── Emotion → hex color (mirrors dashboard.py EMOTION_COLORS) ─────────────────
import theme as T
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
    media_watch_state = {
        "running": _get(lambda: mw.is_running if mw else False, False),
        "status":  _get(lambda: mw.get_status_str() if mw else "OFF", "OFF"),
        "reactions": _get(lambda: getattr(mw, "reactions_enabled", True) if mw else False, False),
        "calls": _get(lambda: getattr(mw, "_calls_count", 0) if mw else 0, 0),
        "cost_usd": _get(lambda: round(getattr(mw, "_calls_cost_usd", 0.0), 3) if mw else 0.0, 0.0),
    }

    # ── Chess Mode ────────────────────────────────────────────────────────────
    ca = _get(lambda: bot.chess_agent, None)
    chess_state = {
        "running": _get(lambda: ca.is_running if ca else False, False),
        "status":  _get(lambda: ca.get_status_str() if ca else "OFF", "OFF"),
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
    from config import ENABLE_TWITCH_CHAT
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
        from bot import read_gpu_memory_gb
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

    # ── Music ─────────────────────────────────────────────────────────────────
    from music_tools import get_now_playing
    now_playing = _get(lambda: get_now_playing(), "Nothing")

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
        # TTS
        "fish_voice_id": fish_voice_id,
        # Music
        "now_playing": now_playing,
        # Transcript
        "transcript": transcript,
        # Server timestamp for the client
        "ts": round(now, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# APP + WEBSOCKET MANAGER
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Kira Control Server", version="1.0")

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

# bot reference is injected by start_control_server()
_bot_ref: "VTubeBot | None" = None


def _bot() -> "VTubeBot":
    if _bot_ref is None:
        raise RuntimeError("control_server: bot not yet injected")
    return _bot_ref


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD UI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    """Serve the single-page web dashboard."""
    try:
        return HTMLResponse(content=_DASHBOARD_HTML.read_text(encoding='utf-8'))
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
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_manager.disconnect(ws)


@app.get("/vision/thumbnail")
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
        mw.window_title = title
        mw.enabled = enabled
        if enabled:
            if not title:
                mw.enabled = False
                return _err("Provide a window_title in the body")
            bot.event_loop.call_soon_threadsafe(mw.start)
        else:
            bot.event_loop.call_soon_threadsafe(mw.stop)
        return _ok(media_watch_enabled=enabled)

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

    if action == "chess_challenge_ai":
        ca = bot.chess_agent
        if ca is None:
            return _err("chess_agent not initialized yet")
        if not ca.is_running:
            return _err("Chess Mode is not armed")
        level = int(body.level) if body.level is not None else 3
        bot.event_loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(ca.challenge_ai(level))
        )
        return _ok(challenged_level=level)

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
