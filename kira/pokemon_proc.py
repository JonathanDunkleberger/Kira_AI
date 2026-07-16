"""pokemon_proc.py - lifecycle for the standalone Pokemon session subprocess.

The Pokemon "hands" run in their OWN process+window (pokemon_agent/session.py, a
pygame app that boots from the post-pick savestate). This module is the SINGLE
owner of that process so the dashboard can Start/Stop/Status it instead of the dev
memorising a terminal command. Deliberately tiny and isolated from bot.py (the
8k-line monolith) - it only knows how to spawn/kill/poll one child process.

MVP stop = terminate(); session.py's `finally` cleans up pygame. No IPC back-channel
yet (the session talks to the bot over HTTP, not vice-versa) - a stop-flag poll is a
later nicety. Loud, never silent: every transition logs.
"""
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # kira/ -> repo root
_PKDIR = os.path.join(_REPO, "pokemon_agent")
_SESSION = os.path.join(_PKDIR, "session.py")
_PLAY_LIVE = os.path.join(_PKDIR, "play_live.py")
# CRASH-SAFE LAUNCH (2026-07-08): EVERY dashboard launch goes through the supervisor — never raw
# play_live — so a mid-run crash auto-resumes instead of ending the show un-recovered (the Viridian
# landmine). ONE protected launch path for GO / Resume / Stop.
_SUPERVISOR = os.path.join(_PKDIR, "supervisor.py")
_BOOT_STATE = os.path.join(_PKDIR, "states", "after_pick_bulbasaur.state")
# BATCH 6 PHASE 7 — the two TIMELINES (separate save files, clearly labelled):
#   SHERPA   = her real persistent campaign (states/campaign/), the everyday 90% button.
#   SHOWTIME = the canonical recordable spine (states/kira/), the stream-day timeline.
_CAMPAIGN_STATE = os.path.join(_PKDIR, "states", "campaign", "kira_campaign.state")
# TIMELINE-SCOPED HEALTH (2026-07-08): each timeline publishes its OWN health.json so the
# dashboard panel never shows the other timeline's stale snapshot.
_HEALTH_BY_TIMELINE = {
    "sherpa":   os.path.join(_PKDIR, "states", "campaign", "health.json"),
    "showtime": os.path.join(_PKDIR, "states", "kira", "health.json"),
}
_HEALTH_JSON = _HEALTH_BY_TIMELINE["sherpa"]   # back-compat default (sherpa is the everyday button)

_proc = {"p": None, "started": 0.0, "timeline": None}


def _read_health_file(path):
    """Load one timeline's health.json (or None). Never raises."""
    import json
    try:
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _log(m):
    print(f"   [pokemon_proc] {m}", flush=True)


def _boot_starter() -> str:
    """Starter label parsed from the canonical boot-state filename (cheap, honest -
    no second emulator). 'after_pick_bulbasaur.state' -> 'bulbasaur'."""
    base = os.path.splitext(os.path.basename(_BOOT_STATE))[0]
    for s in ("bulbasaur", "charmander", "squirtle"):
        if s in base:
            return s
    return "?"


def is_running() -> bool:
    p = _proc["p"]
    return p is not None and p.poll() is None


def start() -> dict:
    if is_running():
        _log(f"start ignored - already running (pid {_proc['p'].pid})")
        return {"running": True, "pid": _proc["p"].pid, "already": True,
                "starter": _boot_starter()}
    if not os.path.exists(_SESSION):
        _log(f"FAIL - session script missing: {_SESSION}")
        return {"running": False, "error": "session.py missing"}
    if not os.path.exists(_BOOT_STATE):
        _log(f"FAIL - boot state missing: {_BOOT_STATE}")
        return {"running": False, "error": "boot state missing"}
    # own console window so pygame has a display and the session's logs are visible
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    p = subprocess.Popen([sys.executable, _SESSION], cwd=_PKDIR, creationflags=flags)
    _proc["p"] = p
    _proc["started"] = time.time()
    _log(f"started session.py (pid {p.pid}) booting {os.path.basename(_BOOT_STATE)}")
    return {"running": True, "pid": p.pid, "starter": _boot_starter()}


def _audio_args() -> list:
    """Shared audio flags for a play_live launch: route game audio to Jonny's DESKTOP/headphones, NEVER
    the VTS cable. REGRESSION-PROOFING (2026-06-28): always pass an EXPLICIT --phones device so the
    launch can never fall back to play_live's empty default -> AudioPump's auto-pick (the path the live
    log showed routing game audio onto 'default', which on Jonny's rig IS the VB-Audio cable VTS lip-
    syncs from). POKEMON_PHONES overrides; the default is Jonny's desktop headphones ('Leviathan'). If
    that name isn't present at runtime, AudioPump's firewall still refuses any cable and substitutes a
    real output — so it degrades to a non-cable device, never the cable. Jonny needn't remember a flag."""
    args = []
    # GAME AUDIO DEFAULT ON (2026-07-09): the emulator audio OUTPUT (PortAudio WRITE) — the native
    # SIGSEGV at the Viridian fanfare — is now PROCESS-ISOLATED in a child (pokemon_audio.AudioPump +
    # audio_child.py). A native abort kills only the child; the emulator survives and the child
    # respawns. Audio is non-fatal, so it's the resting state again. Force OFF with POKEMON_GAME_AUDIO=0.
    if os.getenv("POKEMON_GAME_AUDIO", os.getenv("POKEMON_AUDIO", "1")) == "1":
        args.append("--audio")
        phones = os.getenv("POKEMON_PHONES", "Leviathan")   # NEVER empty -> never auto-picks 'default'
        args += ["--phones", phones]
    return args


def _spawn_supervised(timeline, fresh, label) -> dict:
    """Launch the timeline THROUGH THE SUPERVISOR (crash-safe: auto-resume on any death + crash-loop
    guard + faulthandler + hang-kill), in its own console window. The supervisor spawns play_live as
    ITS child and keeps it alive; killing the supervisor tree (stop()) tears both down. One owned
    supervisor at a time — this is the SINGLE protected launch path for GO / Resume / Stop."""
    if is_running():
        _log(f"{label} start ignored — already running (pid {_proc['p'].pid}, timeline {_proc['timeline']})")
        return {"running": True, "pid": _proc["p"].pid, "already": True, "timeline": _proc["timeline"]}
    if not os.path.exists(_SUPERVISOR):
        _log(f"FAIL - supervisor.py missing: {_SUPERVISOR}")
        return {"running": False, "error": "supervisor.py missing"}
    argv = [sys.executable, "-u", _SUPERVISOR, "--timeline", timeline]
    if fresh:
        argv.append("--fresh")                 # showtime: --fresh-kira on FIRST launch only, resume after
    url = os.getenv("POKEMON_URL", "")
    if url:
        argv += ["--url", url]
    argv += _audio_args()                      # empty by default (game audio off) — the crash workaround
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    p = subprocess.Popen(argv, cwd=_PKDIR, creationflags=flags)
    _proc.update(p=p, started=time.time(), timeline=timeline)
    _log(f"started {label} via SUPERVISOR: {' '.join(argv[1:])}  (pid {p.pid})")
    return {"running": True, "pid": p.pid, "timeline": timeline, "supervised": True}


def start_sherpa() -> dict:
    """SHERPA: RESUME — the everyday button. Resume her REAL persistent campaign and climb from where she
    actually is (states/campaign/), via the crash-safe supervisor (--resume --free-roam under the hood)."""
    seeded = os.path.exists(_CAMPAIGN_STATE)
    res = _spawn_supervised("sherpa", False, "SHERPA:RESUME")
    res["resumed_existing"] = seeded
    return res


def start_showtime(fresh=False) -> dict:
    """SHOWTIME: GO/RESUME — the stream-day timeline (the canonical recordable spine, states/kira/), via
    the crash-safe supervisor. fresh=True (GO) archives + starts new (--fresh-kira on the FIRST launch
    only; auto-resumes on any later crash — never re-wipes mid-stream). fresh=False RESUMES the kira run."""
    return _spawn_supervised("showtime", fresh, "SHOWTIME:GO" if fresh else "SHOWTIME:RESUME")


def health() -> dict:
    """PHASE 7 cockpit readout: the game-side health snapshot play_live publishes (progress/where/badges/
    time-since-badge/last-checkpoint), merged with whether a process is live. API spend is merged in by the
    control server from the bot's own cost-tracker. Returns {} game-side fields if no snapshot yet."""
    running = is_running()
    active_tl = _proc.get("timeline")
    out = {"running": running, "timeline": active_tl,
           "pid": _proc["p"].pid if running else None,
           "uptime_s": round(time.time() - _proc["started"], 1) if running else 0}

    # ── SAVE-SLOTS: read BOTH timelines' own health files (independent of what's running) so
    # the dashboard can render two clearly-separate, labelled save-slots (SHERPA + SHOWTIME).
    slots = {}
    for tl, path in _HEALTH_BY_TIMELINE.items():
        g = _read_health_file(path)
        try:
            mtime = os.path.getmtime(path) if os.path.exists(path) else None
        except Exception:
            mtime = None
        slots[tl] = {
            "game": g,
            "health_age_s": round(time.time() - g.get("ts", 0), 1) if g and g.get("ts") else None,
            "last_played_ts": mtime,
            "running": running and active_tl == tl,
        }
    out["slots"] = slots

    # ── ACTIVE GAME: the live panel shows the CURRENTLY-ACTIVE timeline's own file. If nothing
    # is running, fall back to the most-recently-played slot so the panel isn't blank.
    if active_tl in _HEALTH_BY_TIMELINE:
        _pick = active_tl
    else:
        _played = [(s.get("last_played_ts") or 0, tl) for tl, s in slots.items()]
        _pick = max(_played)[1] if any(ts for ts, _ in _played) else "sherpa"
    out["active_slot"] = _pick
    out["game"] = slots[_pick]["game"]
    out["health_age_s"] = slots[_pick]["health_age_s"]
    return out


def stop() -> dict:
    p = _proc["p"]
    if p is None or p.poll() is not None:
        _proc["p"] = None; _proc["timeline"] = None
        _log("stop ignored - not running")
        return {"running": False, "stopped": False, "note": "not running"}
    pid, tl = p.pid, _proc.get("timeline")
    # The owned child is now the SUPERVISOR, which has play_live as ITS child. terminate() alone would
    # kill the supervisor and ORPHAN play_live (still running + holding the game window). On Windows,
    # taskkill /T kills the whole tree (supervisor + play_live) so STOP means STOP. Deliberate STOP is
    # the ONLY intended way to end the show — window-close auto-relaunches (resilience).
    killed_tree = False
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=15)
            killed_tree = True
        except Exception as e:
            _log(f"taskkill tree failed ({e}); falling back to terminate()")
    if not killed_tree:
        try:
            p.terminate()
        except Exception:
            pass
    _proc["p"] = None; _proc["timeline"] = None
    _log(f"STOPPED supervisor tree (pid {pid}, timeline {tl}, tree_kill={killed_tree})")
    return {"running": False, "stopped": True, "pid": pid, "timeline": tl}


def status() -> dict:
    running = is_running()
    return {
        "running": running,
        "pid": _proc["p"].pid if running else None,
        "timeline": _proc.get("timeline"),
        "uptime_s": round(time.time() - _proc["started"], 1) if running else 0,
        "starter": _boot_starter(),
        "boot_state": os.path.basename(_BOOT_STATE),
        "campaign_exists": os.path.exists(_CAMPAIGN_STATE),
    }
