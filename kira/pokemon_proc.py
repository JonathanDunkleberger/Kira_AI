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
_BOOT_STATE = os.path.join(_PKDIR, "states", "after_pick_bulbasaur.state")

_proc = {"p": None, "started": 0.0}


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


def stop() -> dict:
    p = _proc["p"]
    if p is None or p.poll() is not None:
        _proc["p"] = None
        _log("stop ignored - not running")
        return {"running": False, "stopped": False, "note": "not running"}
    pid = p.pid
    p.terminate()
    _proc["p"] = None
    _log(f"terminated session.py (pid {pid})")
    return {"running": False, "stopped": True, "pid": pid}


def status() -> dict:
    running = is_running()
    return {
        "running": running,
        "pid": _proc["p"].pid if running else None,
        "uptime_s": round(time.time() - _proc["started"], 1) if running else 0,
        "starter": _boot_starter(),
        "boot_state": os.path.basename(_BOOT_STATE),
    }
