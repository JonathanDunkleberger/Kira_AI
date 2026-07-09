#!/usr/bin/env python
"""
supervisor.py — the auto-restart wrapper that keeps the Sherpa show alive for a 30h
unattended stream.

WHY: play_live.py hosts mGBA in-process. A hard emulator fault, an uncaught roam-loop
exception, or an external kill takes the whole play_live process down — and the show
goes dark until a human notices. This thin supervisor wraps play_live in a
resume-on-crash loop: on any UNEXPECTED exit it relaunches with --resume (NEVER fresh —
so it picks up the banked canonical checkpoint states/campaign/kira_campaign.state),
tees the child's output to a timestamped log (so a crash traceback is never lost again),
and pings Jonny via the dead-man's-switch seam.

CONTRACT (matches CLAUDE.md rule 17 — resume-safe by default):
  * relaunch = --resume --free-roam ONLY. This script never starts a fresh timeline.
  * a CLEAN stop (Jonny closes the window / Ctrl-C the child → exit 0) ends the loop.
  * a CRASH (non-zero exit, kill, or a stale health heartbeat) → relaunch + alert.
  * rapid-crash backoff prevents a tight respawn loop from hammering the rig.

USAGE:
  .venv\\Scripts\\python.exe pokemon_agent\\supervisor.py [--audio] [--phones NAME]
                                                          [--headless] [--url URL]

  Ctrl-C the SUPERVISOR to stop the whole show (it tears the child down cleanly).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# Windows console defaults to cp1252 — the child (play_live) prints emoji/arrows and so do
# we; without this a single unicode glyph in a log line would crash the supervisor itself
# (the cp1252 crash class). Force UTF-8 with replacement so nothing we print can ever kill us.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(_PY):
    _PY = sys.executable  # fallback to whatever launched us

# a crash faster than this = "not healthy" → escalate backoff. A run that survives longer
# resets the streak (a genuine long run that later dies is a fresh incident, not a loop).
MIN_HEALTHY_UPTIME_S = 60
BACKOFF_STEP_S = 5
BACKOFF_MAX_S = 60
# if the child is ALIVE but states/campaign/health.json's ts goes older than this, treat it
# as a hung/wedged run: kill + relaunch. Generous so a slow LLM decision never trips it.
HANG_THRESHOLD_S = 300
EXIT_POLL_S = 2       # how fast we notice the process died (relaunch "within seconds")
HEALTH_POLL_S = 15    # how often we check the heartbeat for a hung-but-alive run
# CRASH-LOOP guard: a DETERMINISTIC crash (e.g. always at the same game beat) makes a naive
# resume-loop replay to the same spot and die forever. After this many consecutive rapid crashes
# we treat it as a loop: keep the show ALIVE (a slow retry can still get past an intermittent
# crash) but stop hammering + escalate LOUD so a human knows it needs eyes.
CRASH_LOOP_THRESHOLD = 5
CRASH_LOOP_BACKOFF_S = 300   # slow-retry cadence once a loop is detected (vs the fast ramp below)
# BOT-DOWN guard: play_live posts every voice/oracle beat to the Kira bot at --url. If the bot is
# down (Ctrl-C'd, crashed, not up yet), a spawned play_live just thrashes — every POST fails
# (WinError 10061) and a window-closed child relaunches into the same dead bot, replaying the opening.
# So we WAIT for the bot to answer before (re)launching. This is an environment wait, NOT a game
# crash — it must not count toward the crash-loop streak.
BOT_DOWN_POLL_S = 10


def _log(msg):
    print(f"[SUPERVISOR {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ping_jonny(url, message):
    """Best-effort operator ping via the existing dead-man's-switch seam (→ Discord + loud
    log). Never raises — a failed ping must not stop the supervisor."""
    try:
        req = urllib.request.Request(
            f"{url}/cmd/pokemon_alert",
            data=json.dumps({"name": message}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=6).read()
        _log(f"pinged Jonny: {message}")
    except Exception as e:
        _log(f"!! could not ping Jonny ({e}) — bot down? message was: {message}")


def _bot_reachable(url):
    """True if the Kira bot's control endpoint answers. An HTTP response of ANY code (even 404) means
    the server is up; only a connection-level failure (refused/timeout/DNS) means it's down. Never
    raises — a probe error is treated as 'down' so the caller simply waits."""
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except urllib.error.HTTPError:
        return True   # the server answered (with an error code) → it's up
    except Exception:
        return False  # connection refused / timeout / unreachable → bot is down


def _wait_for_bot(url):
    """Block until the bot at `url` answers, so we never spawn play_live into a dead bot (the
    hot-relaunch/intro-thrash). Returns True when reachable, False if interrupted (Ctrl-C)."""
    if _bot_reachable(url):
        return True
    _log(f"bot endpoint {url} unreachable — NOT launching play_live into a dead bot "
         f"(would thrash: failed POSTs + window-closed relaunch replaying the opening). "
         f"Waiting for the bot to come back…")
    waited = 0
    try:
        while not _bot_reachable(url):
            time.sleep(BOT_DOWN_POLL_S)
            waited += BOT_DOWN_POLL_S
            if waited % 60 == 0:
                _log(f"still waiting for bot at {url} ({waited}s)…")
    except KeyboardInterrupt:
        _log("Ctrl-C while waiting for the bot — supervisor stopping.")
        return False
    _log(f"bot back up after {waited}s — launching play_live.")
    return True


def _tee(proc, logfile):
    """Stream the child's merged stdout to our console AND a logfile, live."""
    with open(logfile, "w", encoding="utf-8", errors="replace") as fh:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()


def _health_path(timeline="sherpa"):
    # Each timeline publishes its own heartbeat: showtime → states/kira, sherpa → states/campaign
    # (POKEMON_CAMPAIGN_DIR overrides the sherpa dir for sandboxed test runs).
    if timeline == "showtime":
        return os.path.join(_ROOT, "states", "kira", "health.json")
    camp_dir = os.environ.get("POKEMON_CAMPAIGN_DIR") or os.path.join(_ROOT, "states", "campaign")
    return os.path.join(camp_dir, "health.json")


def _health_stale_for(hp):
    """Seconds since the campaign heartbeat last updated, or None if unreadable/absent."""
    try:
        with open(hp, "r", encoding="utf-8") as fh:
            ts = float(json.load(fh).get("ts", 0))
        return max(0.0, time.time() - ts) if ts else None
    except Exception:
        return None


def _build_cmd(args, first_launch=False):
    cmd = [_PY, "-u", os.path.join(_HERE, "play_live.py")]
    if args.timeline == "showtime":
        # SHOWTIME (the KIRA stream timeline): resume the existing kira checkpoint. --fresh-kira
        # (GO = archive current + start new) fires ONLY on the FIRST launch — an auto-relaunch after
        # a crash ALWAYS resumes (never re-wipes the run mid-stream). --show alone resumes states/kira.
        cmd += ["--show"]
        if getattr(args, "fresh", False) and first_launch:
            cmd += ["--fresh-kira"]
    else:
        # SHERPA (everyday working line): resume the persistent campaign. RESUME ONLY, never fresh.
        cmd += ["--resume", "--free-roam", "--roam-ticks", "100000000", "--roam-seconds", "86400"]
    cmd += ["--url", args.url]
    if args.headless:
        cmd.append("--headless")
    if args.audio:
        cmd += ["--audio", "--phones", args.phones]
    return cmd


def main():
    ap = argparse.ArgumentParser(description="Auto-restart supervisor for play_live (Sherpa or Showtime).")
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    ap.add_argument("--timeline", choices=["sherpa", "showtime"], default="sherpa",
                    help="which timeline to keep alive: sherpa (--resume --free-roam) or "
                         "showtime (--show, the KIRA stream). Both RESUME on crash, never fresh.")
    ap.add_argument("--fresh", action="store_true",
                    help="showtime only: --fresh-kira on the FIRST launch (archive current + start new); "
                         "auto-relaunches after a crash ALWAYS resume, never re-wipe.")
    ap.add_argument("--audio", action="store_true", help="pump game BGM to desktop (off the cable).")
    ap.add_argument("--phones", default="Leviathan")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-restarts", type=int, default=0, help="0 = unlimited (30h show default).")
    args = ap.parse_args()

    logdir = os.path.join(_ROOT, "logs", "debug", "supervisor")
    os.makedirs(logdir, exist_ok=True)

    _log(f"starting [{args.timeline}]. resume-command = {' '.join(_build_cmd(args))}")
    _log(f"heartbeat = {_health_path(args.timeline)}")

    fail_streak = 0
    restarts = 0
    hp = _health_path(args.timeline)

    # Child env: unless we asked for headless, make sure NOTHING suppresses play_live's game WINDOW.
    # An inherited SDL_VIDEODRIVER=dummy (e.g. left over from a prior headless/proof run in this shell)
    # would silently hide the window — strip it so a windowed supervised run is always visible.
    _child_env = os.environ.copy()
    if not args.headless:
        _child_env.pop("SDL_VIDEODRIVER", None)
        _log("windowed mode — a game window will open (SDL_VIDEODRIVER cleared so it can't be hidden).")

    while True:
        # BOT-DOWN GUARD: never (re)launch play_live into a dead bot. If the bot is unreachable we
        # wait here instead of hot-relaunching (which just replays the opening against a dead endpoint).
        # An interrupted wait (Ctrl-C) tears the show down cleanly.
        if not _wait_for_bot(args.url):
            return
        cmd = _build_cmd(args, first_launch=(restarts == 0))
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        logfile = os.path.join(logdir, f"playlive_{stamp}.log")
        _log(f"launching play_live (attempt #{restarts + 1}) -> tee to {logfile}")
        start = time.time()

        proc = subprocess.Popen(
            cmd, cwd=_HERE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=_child_env,
        )
        tee = threading.Thread(target=_tee, args=(proc, logfile), daemon=True)
        tee.start()

        killed_for_hang = False
        try:
            # poll for exit FAST (relaunch within seconds); check the heartbeat SLOW
            _since_health = 0.0
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                if _since_health >= HEALTH_POLL_S:
                    _since_health = 0.0
                    stale = _health_stale_for(hp)
                    # only trust the heartbeat once the run has had time to write one
                    if stale is not None and (time.time() - start) > HANG_THRESHOLD_S and stale > HANG_THRESHOLD_S:
                        _log(f"!! HUNG: heartbeat stale {int(stale)}s (> {HANG_THRESHOLD_S}s) but process alive — killing to relaunch.")
                        killed_for_hang = True
                        proc.terminate()
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                time.sleep(EXIT_POLL_S)
                _since_health += EXIT_POLL_S
        except KeyboardInterrupt:
            _log("Ctrl-C — stopping the show. Tearing down play_live cleanly.")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            return

        rc = proc.poll()
        uptime = time.time() - start
        tee.join(timeout=5)

        # COMPLETION: play_live exits 0 ONLY on genuine done (all segments / credits). rc=3
        # (window-closed) and rc=1 (crash/stuck) both fall through to relaunch — the show survives
        # an accidental window close; the ONLY deliberate stop is killing this supervisor (dashboard
        # STOP → taskkill tree, or Ctrl-C here).
        if rc == 0 and not killed_for_hang:
            _log(f"play_live COMPLETED (rc=0, uptime={int(uptime)}s) — credits/all-segments done. "
                 f"Supervisor stopping (the show finished).")
            _ping_jonny(args.url, "🎉 the run COMPLETED (credits/all segments) — supervisor stopping.")
            return

        # CRASH / kill / hang → relaunch with --resume.
        restarts += 1
        reason = "hung" if killed_for_hang else f"rc={rc}"
        if uptime < MIN_HEALTHY_UPTIME_S and not killed_for_hang:
            fail_streak += 1
        else:
            fail_streak = 0  # a healthy-length run resets the streak

        # CRASH-LOOP: consecutive rapid crashes = replaying into the same deterministic death.
        # Don't give up the show (an intermittent crash can clear on a later resume), but slow the
        # retry way down and escalate LOUD so it doesn't silently hammer forever.
        looping = fail_streak >= CRASH_LOOP_THRESHOLD
        if looping:
            backoff = CRASH_LOOP_BACKOFF_S
            _log(f"!! CRASH LOOP — {fail_streak} rapid crashes in a row (same spot?). play_live keeps "
                 f"dying right after resume. Slow-retrying every {backoff}s + escalating; a human should look. "
                 f"Check logs/debug/playlive_faulthandler.log for the native C-stack of the crash.")
            # re-alert on the first crossing and then every few loops (not every single retry)
            if fail_streak == CRASH_LOOP_THRESHOLD or (fail_streak % 5 == 0):
                _ping_jonny(args.url, f"CRASH LOOP — play_live has crashed {fail_streak}x in a row right after "
                                      f"resume ({reason}). Still slow-retrying every {backoff}s, but it needs eyes "
                                      f"(see playlive_faulthandler.log).")
        else:
            backoff = min(BACKOFF_MAX_S, BACKOFF_STEP_S * max(1, fail_streak))
            _ping_jonny(args.url, f"play_live went down ({reason}, uptime {int(uptime)}s) — auto-resuming in {backoff}s")

        _log(f"play_live DOWN ({reason}, uptime={int(uptime)}s, streak={fail_streak}). Relaunching --resume in {backoff}s.")

        if args.max_restarts and restarts >= args.max_restarts:
            _log(f"hit --max-restarts={args.max_restarts}; stopping.")
            _ping_jonny(args.url, f"supervisor hit max-restarts ({args.max_restarts}) — show is DOWN, needs eyes.")
            return

        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            _log("Ctrl-C during backoff — supervisor stopping.")
            return


if __name__ == "__main__":
    main()
