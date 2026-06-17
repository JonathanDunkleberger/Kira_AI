# debug_tee.py — Full verbatim console capture for debugging.
#
# Mirrors EVERYTHING printed to stdout/stderr during a run to log files on disk
# while STILL printing to the real terminal (a tee). This is distinct from the
# curated logs/streams/ session log — it is the RAW, complete terminal output,
# including tracebacks and native crash dumps, so Claude Code can read exactly
# what the operator saw without anything being pasted or summarized.
#
# Files (under logs/debug/):
#   console_<YYYY-MM-DD_HH-MM-SS>.log   one per run (archive)
#   latest.log                          always contains the most recent run
#
# We write BOTH files live (no symlink — symlinks need privilege on Windows) so
# latest.log is crash-safe rather than a post-run copy that would lose the tail.
#
# Crash-safety: flush after every write so a hard crash leaves the last lines on
# disk. faulthandler is pointed at latest.log's fd so NATIVE crashes (e.g. a CUDA
# / Whisper segfault that produces no Python traceback) are captured in the file
# too — not just on the terminal.
#
# Robustness (it is ONLY a debug aid): a failing file handle is disabled and a
# one-time warning is printed to the real stream (never silent — Constraint #3).
# The tee NEVER raises into the app; a failure here leaves Kira running.

import os
import sys
import threading
import faulthandler
from datetime import datetime

_INSTALLED = False
_LOCK = threading.Lock()
_FILES = []            # open file handles mirrored on every write [run, latest]
_WARNED = set()        # handles already warned about (warn-once, no flood)
_LATEST_FILE = None    # kept alive globally for faulthandler's fd


def _disable_handle(fh, err):
    """Drop a file handle that failed mid-write and warn ONCE to the real stream.
    Never silent (Constraint #3); writes to __stderr__ so it can't recurse back
    through the tee."""
    try:
        _FILES.remove(fh)
    except ValueError:
        pass
    if fh not in _WARNED:
        _WARNED.add(fh)
        try:
            sys.__stderr__.write(
                f"   [DebugTee] WARNING — a debug-log handle failed; disabling it, "
                f"app continues: {err}\n"
            )
            sys.__stderr__.flush()
        except Exception:
            pass


class _Tee:
    """Stand-in for a text stream that mirrors writes to the original stream AND
    the debug files. Delegates everything else to the original so it is a
    faithful replacement for sys.stdout / sys.stderr."""

    def __init__(self, original):
        self._original = original

    def write(self, s):
        # Original stream first so the live terminal is never blocked by disk.
        # Best-effort: a broken console must not stop disk capture.
        try:
            self._original.write(s)
        except Exception:
            pass
        with _LOCK:
            for fh in list(_FILES):
                try:
                    fh.write(s)
                    fh.flush()  # crash-safety: never leave the tail in a buffer
                except Exception as e:
                    _disable_handle(fh, e)
        return len(s) if s else 0

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        with _LOCK:
            for fh in list(_FILES):
                try:
                    fh.flush()
                except Exception:
                    pass

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    # ── Faithful-stand-in delegation ────────────────────────────────────────
    def fileno(self):
        # Delegate to the real stream so anything needing a real fd still works.
        return self._original.fileno()

    def isatty(self):
        try:
            return self._original.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    def __getattr__(self, name):
        # Anything not overridden (buffer, errors, …) falls through to the real
        # stream. _original is set in __init__, so this never recurses on it.
        return getattr(self._original, name)


def install_console_tee(log_dir="logs/debug"):
    """Install the stdout/stderr tee + faulthandler native-crash file capture.

    Idempotent and safe to call twice: once from run.py BEFORE importing
    kira.bot (so import-time output — the heavy pyaudio/torch/faster-whisper
    imports, a real crash site — is captured) and once from kira.bot.launch()
    (so the `python -m kira.bot` entry path is covered too). The second call
    only RE-ASSERTS faulthandler's file target, because kira.bot's import-time
    `faulthandler.enable(file=sys.stderr)` would otherwise have re-pointed native
    dumps back at the terminal.

    Fail-graceful: any failure here leaves the app running with no capture."""
    global _INSTALLED, _LATEST_FILE
    try:
        if not _INSTALLED:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_path = os.path.join(log_dir, f"console_{ts}.log")
            latest_path = os.path.join(log_dir, "latest.log")
            # Line-buffered, utf-8, errors='replace' so an un-encodable glyph
            # (🎤 / ♪ / ⚠) can never crash the writer. latest.log opens in "w"
            # so it only ever holds the current/most-recent run.
            run_fh = open(run_path, "w", encoding="utf-8", errors="replace", buffering=1)
            latest_fh = open(latest_path, "w", encoding="utf-8", errors="replace", buffering=1)
            _FILES.extend([run_fh, latest_fh])
            _LATEST_FILE = latest_fh

            header = (f"=== Kira debug console capture — run {ts} ===\n"
                      f"=== verbatim stdout + stderr; full mirror of terminal ===\n")
            for fh in _FILES:
                try:
                    fh.write(header)
                    fh.flush()
                except Exception:
                    pass

            sys.stdout = _Tee(sys.stdout)
            sys.stderr = _Tee(sys.stderr)
            _INSTALLED = True
            try:
                sys.__stderr__.write(
                    f"   [DebugTee] Console capture active → {run_path} (+ latest.log)\n"
                )
                sys.__stderr__.flush()
            except Exception:
                pass

        # (Re)assert native-crash capture into the file on EVERY call, so a later
        # faulthandler.enable(file=sys.stderr) at kira.bot import time is overridden
        # back to the file. Pointed at latest.log so native dumps land in the file
        # the operator reads.
        if _LATEST_FILE is not None:
            try:
                faulthandler.enable(file=_LATEST_FILE, all_threads=True)
            except Exception as e:
                try:
                    sys.__stderr__.write(
                        f"   [DebugTee] faulthandler file-capture enable failed "
                        f"(non-fatal): {e}\n"
                    )
                except Exception:
                    pass
    except Exception as e:
        # The tee must NEVER prevent the app from starting.
        try:
            sys.__stderr__.write(
                f"   [DebugTee] install failed (app continues, no capture): {e}\n"
            )
            sys.__stderr__.flush()
        except Exception:
            pass
