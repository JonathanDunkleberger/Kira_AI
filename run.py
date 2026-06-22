#!/usr/bin/env python
"""Kira launcher — the documented way to start the bot.

Resolves the repo root from this file's location and ``os.chdir()``s there
before launching, so every CWD-relative runtime path (``logs/``, ``clips/``,
``lore/``, ``playthroughs/``, ``persona/private/``) keeps working no matter
where you invoke it from.

Usage:
    python run.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── venv self-heal guard ───────────────────────────────────────────────────────
# The #1 latency trap (cost a whole watch session): launching `python run.py` under
# the WRONG interpreter — e.g. the global Python with a CPU-only torch — silently
# boots Whisper on CPU (12-19s STT). The GPU/cu128 torch lives in THIS repo's .venv.
# If we're not already on that venv's python, re-exec ourselves with it so EVERY
# launch (any shortcut / terminal / IDE) lands on the GPU interpreter. This runs
# before ANY kira import, so the correct torch is the one that actually gets loaded.
def _venv_python(root):
    # Derived from run.py's own location (survives a repo move) — not hardcoded.
    if os.name == "nt":
        return os.path.join(root, ".venv", "Scripts", "python.exe")
    return os.path.join(root, ".venv", "bin", "python")

def _same_path(a, b):
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except Exception:
        return False

_VENV_PY = _venv_python(_ROOT)
if not _same_path(sys.executable, _VENV_PY) and os.environ.get("KIRA_VENV_REEXEC") != "1":
    # Not on the venv python and haven't re-exec'd yet.
    if os.path.isfile(_VENV_PY):
        print(f"[Launch] Re-executing under .venv python (was: {sys.executable})", flush=True)
        print(f"[Launch]   -> {_VENV_PY}", flush=True)
        os.environ["KIRA_VENV_REEXEC"] = "1"  # sentinel: child won't re-exec again (loop guard)
        import subprocess
        try:
            _proc = subprocess.run([_VENV_PY, os.path.abspath(__file__)] + sys.argv[1:])
            sys.exit(_proc.returncode)
        except KeyboardInterrupt:
            sys.exit(130)
    else:
        # No-silent-failure: a missing/moved .venv is exactly the degraded boot that
        # has cost days. Warn LOUDLY and continue under the current interpreter.
        print(f"[Launch] WARNING - expected .venv python not found at {_VENV_PY}; "
              f"continuing under {sys.executable} (STT may fall back to CPU). "
              "Create the .venv or fix the path.")
elif os.environ.get("KIRA_VENV_REEXEC") == "1" and not _same_path(sys.executable, _VENV_PY):
    # We already re-exec'd but STILL aren't on the venv python (broken/mismatched
    # venv). Proceed rather than loop forever — and say so.
    print(f"[Launch] WARNING - re-exec did not land on the .venv python "
          f"(now: {sys.executable}). Proceeding to avoid a re-exec loop.")

os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Install full verbatim console capture BEFORE importing kira.bot, so even
# import-time output (the heavy pyaudio / torch / faster-whisper imports — a real
# crash site) is mirrored to logs/debug/latest.log. Fail-graceful: never blocks
# startup. See kira/debug_tee.py.
from kira.debug_tee import install_console_tee
install_console_tee()

from kira.bot import launch

if __name__ == "__main__":
    launch()
