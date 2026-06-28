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
_BOOT_STATE = os.path.join(_PKDIR, "states", "after_pick_bulbasaur.state")
# BATCH 6 PHASE 7 — the two TIMELINES (separate save files, clearly labelled):
#   SHERPA   = her real persistent campaign (states/campaign/), the everyday 90% button.
#   SHOWTIME = the canonical recordable spine (states/kira/), the stream-day timeline.
_CAMPAIGN_STATE = os.path.join(_PKDIR, "states", "campaign", "kira_campaign.state")
_HEALTH_JSON = os.path.join(_PKDIR, "states", "campaign", "health.json")

_proc = {"p": None, "started": 0.0, "timeline": None}


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
    if os.getenv("POKEMON_AUDIO", "1") == "1":
        args.append("--audio")
        phones = os.getenv("POKEMON_PHONES", "Leviathan")   # NEVER empty -> never auto-picks 'default'
        args += ["--phones", phones]
    return args


def _spawn_play_live(extra_args, timeline, label) -> dict:
    """Spawn play_live.py with timeline args in its own console window. One owned child at a time — this
    is what kills the 'ask Claude for the launch command' dependency for Jonny's real playthrough."""
    if is_running():
        _log(f"{label} start ignored — already running (pid {_proc['p'].pid}, timeline {_proc['timeline']})")
        return {"running": True, "pid": _proc["p"].pid, "already": True, "timeline": _proc["timeline"]}
    if not os.path.exists(_PLAY_LIVE):
        _log(f"FAIL - play_live.py missing: {_PLAY_LIVE}")
        return {"running": False, "error": "play_live.py missing"}
    url = os.getenv("POKEMON_URL", "")
    argv = [sys.executable, _PLAY_LIVE] + extra_args + _audio_args()
    if url:
        argv += ["--url", url]
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    p = subprocess.Popen(argv, cwd=_PKDIR, creationflags=flags)
    _proc.update(p=p, started=time.time(), timeline=timeline)
    _log(f"started {label}: {' '.join(argv[1:])}  (pid {p.pid})")
    return {"running": True, "pid": p.pid, "timeline": timeline}


def start_sherpa() -> dict:
    """SHERPA: RESUME — the everyday button. Resume her REAL persistent campaign and climb from where she
    actually is (states/campaign/). play_live --resume --free-roam. SEEDS from --boot on the very first run."""
    seeded = os.path.exists(_CAMPAIGN_STATE)
    res = _spawn_play_live(["--resume", "--free-roam"], "sherpa", "SHERPA:RESUME")
    res["resumed_existing"] = seeded
    return res


def start_showtime(fresh=False) -> dict:
    """SHOWTIME: GO/RESUME — the stream-day timeline (the canonical recordable spine, states/kira/).
    play_live --show resumes a kira checkpoint if present; --fresh-kira archives it and starts a new run."""
    args = ["--show"]
    if fresh:
        args.append("--fresh-kira")
    return _spawn_play_live(args, "showtime", "SHOWTIME:GO" if fresh else "SHOWTIME:RESUME")


def health() -> dict:
    """PHASE 7 cockpit readout: the game-side health snapshot play_live publishes (progress/where/badges/
    time-since-badge/last-checkpoint), merged with whether a process is live. API spend is merged in by the
    control server from the bot's own cost-tracker. Returns {} game-side fields if no snapshot yet."""
    import json
    out = {"running": is_running(), "timeline": _proc.get("timeline"),
           "pid": _proc["p"].pid if is_running() else None,
           "uptime_s": round(time.time() - _proc["started"], 1) if is_running() else 0}
    try:
        if os.path.exists(_HEALTH_JSON):
            with open(_HEALTH_JSON, encoding="utf-8") as f:
                out["game"] = json.load(f)
            out["health_age_s"] = round(time.time() - out["game"].get("ts", 0), 1)
        else:
            out["game"] = None
    except Exception as e:
        out["game"] = None
        out["health_error"] = str(e)
    return out


def stop() -> dict:
    p = _proc["p"]
    if p is None or p.poll() is not None:
        _proc["p"] = None; _proc["timeline"] = None
        _log("stop ignored - not running")
        return {"running": False, "stopped": False, "note": "not running"}
    pid, tl = p.pid, _proc.get("timeline")
    p.terminate()
    _proc["p"] = None; _proc["timeline"] = None
    _log(f"terminated play session (pid {pid}, timeline {tl})")
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
