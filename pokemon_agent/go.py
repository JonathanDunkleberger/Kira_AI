"""go.py — THE GO BUTTON (Phase E): the ONE command for THE KIRA SHOW TIMELINE (timeline 2).

    python pokemon_agent/go.py              start (fresh bedroom) or RESUME the show run
    python pokemon_agent/go.py --fresh      archive the existing show run, start a brand-new one
    python pokemon_agent/go.py --throwaway  full end-to-end TEST in a disposable sandbox — the
                                            real states/kira + Sherpa canonical are never touched

WHAT IT DOES (in order, refusing loudly at any broken rung — never a soul-blind show):
  1. PREFLIGHT the live bot (:8766). Bot down -> refuse with the exact boot command. The show
     NEVER runs soul-blind (the watch.py law).
  2. SAFETY RAILS: the Sherpa canonical (states/campaign) is redirected to a per-run scratch dir
     via POKEMON_CAMPAIGN_DIR — physically unopenable for write by this run. The show lineage
     (states/kira) is only ever extended by play_live --show's own banking; --fresh archives
     first (archive-first law, never a clobber). --throwaway redirects BOTH lineages to a
     disposable sandbox (POKEMON_KIRA_DIR + POKEMON_CAMPAIGN_DIR).
  3. COLD-OPEN RECAP: on a resume, her journey (journey_core.json) is POSTed through the event
     seam as kind='recap' -> she opens the session with a "previously on…" in her own voice.
  4. LAUNCH the show UNDER supervisor.py (--timeline showtime): windowed, TRUE SPEED, game audio ->
     --phones (desktop headphones; the virtual cable carries ONLY her TTS via the bot — AudioPump
     refuses cables by design, the mouth-flap/loopback firewall). Streaming-ready: OBS captures the
     window + desktop audio. The supervisor keeps a 30h unattended stream ALIVE — a crash/hang/kill
     auto-resumes (never re-fresh mid-stream) instead of going dark. Single un-wrapped run for a quick
     test: POKEMON_GO_NO_SUPERVISOR=1.

AUDIO DOCTRINE (unchanged, enforced by existing layers): her voice -> cable -> VTS/OBS; game
music -> phones/speakers only, NEVER the cable.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
STATES_KIRA_REAL = os.path.join(_HERE, "states", "kira")
PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")


def log(m):
    print(f"[GO] {m}", flush=True)


def bot_up(url):
    try:
        with urllib.request.urlopen(f"{url}/state", timeout=4) as r:
            json.load(r)
        return True
    except Exception:
        return False


def kira_checkpoints(kira_dir):
    if not os.path.isdir(kira_dir):
        return []
    return sorted(f for f in os.listdir(kira_dir) if f.endswith(".state"))


def journey_summary():
    """Compose the cold-open recap text from the persisted journey (core-side copy first,
    then the show-lineage copy). Returns '' if no journey exists yet (a fresh run)."""
    for p in (os.path.join(_ROOT, "states", "kira", "journey_core.json"),
              os.path.join(STATES_KIRA_REAL, "journey_core.json")):
        try:
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    j = json.load(f)
                summary = (j.get("summary") or "").strip()
                beats = sorted((j.get("saga") or []), key=lambda b: -float(b.get("weight", 0)))
                lines = [b.get("text", "").strip() for b in beats[:3] if b.get("text")]
                if summary or lines:
                    return summary + (" Standout moments: " + " | ".join(lines) if lines else "")
        except Exception as e:
            log(f"journey read skipped ({p}): {e}")
    return ""


def post_recap(url, text):
    try:
        body = json.dumps({"name": text, "tier": 3, "kind": "recap"}).encode()
        req = urllib.request.Request(f"{url}/cmd/pokemon_event", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            res = json.load(r)
        log(f"cold-open recap fired (fired={res.get('fired')})")
    except Exception as e:
        log(f"!! cold-open recap POST failed (continuing — the run itself is fine): {e}")


def main():
    ap = argparse.ArgumentParser(description="THE GO BUTTON — start/resume the Kira show timeline.")
    ap.add_argument("--fresh", action="store_true",
                    help="archive the existing show run (archive-first, never a clobber) and "
                         "start a brand-new bedroom run")
    ap.add_argument("--throwaway", action="store_true",
                    help="END-TO-END TEST: run everything in a disposable sandbox; the real "
                         "states/kira and the Sherpa canonical are never touched")
    ap.add_argument("--url", default="http://127.0.0.1:8766", help="Kira control server")
    ap.add_argument("--phones", default=os.getenv("POKEMON_PHONES", "Leviathan"),
                    help="desktop output device for GAME audio (never the cable)")
    ap.add_argument("--no-audio", action="store_true", help="skip the game-audio pump")
    args = ap.parse_args()

    # ── 1. PREFLIGHT: the soul stack. A show without her voice is not a show. ────────────────
    if not bot_up(args.url):
        log("=" * 72)
        log("REFUSING TO START — Kira's bot is DOWN (no control server on " + args.url + ").")
        log("The show timeline NEVER runs soul-blind. Boot her first, in its own terminal:")
        log("    python run.py")
        log("wait for the control server, then press GO again:")
        log("    python pokemon_agent/go.py" + (" --throwaway" if args.throwaway else ""))
        log("=" * 72)
        return 2

    # ── 2. SAFETY RAILS ───────────────────────────────────────────────────────────────────────
    env = dict(os.environ)
    if args.throwaway:
        sandbox = os.path.join(tempfile.mkdtemp(prefix="kira_go_"), "throwaway")
        kira_dir = os.path.join(sandbox, "kira")
        camp_dir = os.path.join(sandbox, "campaign")
        os.makedirs(kira_dir, exist_ok=True)
        os.makedirs(camp_dir, exist_ok=True)
        env["POKEMON_KIRA_DIR"] = kira_dir
        env["POKEMON_CAMPAIGN_DIR"] = camp_dir
        log(f"THROWAWAY sandbox: {sandbox} (real states/kira + Sherpa canonical untouchable)")
    else:
        kira_dir = STATES_KIRA_REAL
        # The Sherpa canonical is NOT this run's to write: campaign-path writes (health/heartbeat/
        # anything) land in a per-run scratch dir. The show lineage banks via --show as designed.
        camp_dir = os.path.join(tempfile.mkdtemp(prefix="kira_go_"), "campaign_scratch")
        os.makedirs(camp_dir, exist_ok=True)
        env["POKEMON_CAMPAIGN_DIR"] = camp_dir
        log(f"Sherpa-canonical rail: campaign writes -> scratch {camp_dir} (canonical untouchable)")

    cps = kira_checkpoints(kira_dir)
    resuming = bool(cps) and not args.fresh
    log(f"show lineage: {kira_dir} — " +
        (f"RESUMING from {cps[-1]}" if resuming else
         ("FRESH RUN (archive-first)" if args.fresh and cps else "FRESH RUN (first ever)")))

    # ── 3. COLD-OPEN RECAP (resume only) ─────────────────────────────────────────────────────
    if resuming:
        recap = journey_summary()
        if recap:
            post_recap(args.url, recap)
            time.sleep(1.0)               # let the recap enter her queue before the run's intro
        else:
            log("no journey on disk yet — skipping the recap (her intro beat opens instead)")

    # ── 4. LAUNCH the show UNDER THE SUPERVISOR (auto-restart / resume-on-crash) ─────────────────
    # THE GO BUTTON runs the show under supervisor.py, NOT a bare play_live: on a bare subprocess.call a
    # hard emulator fault or uncaught roam exception takes the whole 30h stream DARK until a human
    # notices — the single biggest live-stability risk for an unattended stream. The supervisor's
    # `showtime` timeline is a drop-in superset: the SAME `play_live --show` launch PLUS resume-on-crash
    # (never re-fresh mid-stream — a crash resumes states/kira, never re-wipes the run), a stale-heartbeat
    # hang detector, rapid-crash backoff, and the dead-man's-switch ping. go.py still does the bot
    # preflight + fresh/throwaway sandbox setup above; only the LAUNCH is now crash-recovered.
    # Escape hatch for a quick single un-wrapped run (no restart loop, e.g. a short test): POKEMON_GO_NO_SUPERVISOR=1.
    supervised = os.getenv("POKEMON_GO_NO_SUPERVISOR") != "1"
    if supervised:
        cmd = [PY, os.path.join(_HERE, "supervisor.py"), "--timeline", "showtime",
               "--url", args.url, "--phones", args.phones]
        if args.fresh and cps:
            cmd.append("--fresh")     # supervisor: --fresh-kira on the FIRST launch ONLY; crash-resumes never re-wipe
    else:
        cmd = [PY, os.path.join(_HERE, "play_live.py"), "--show", "--url", args.url, "--phones", args.phones]
        if args.fresh and cps:
            cmd.append("--fresh-kira")
    # GAME AUDIO DEFAULT ON (2026-07-09): the PortAudio output SIGSEGV is process-isolated (audio_child.py)
    # — a native abort kills only the child, the game survives + the child respawns. Force OFF: POKEMON_GAME_AUDIO=0.
    if os.getenv("POKEMON_GAME_AUDIO", "1") == "1" and not args.no_audio:
        cmd.append("--audio")
    log(("launching UNDER SUPERVISOR (auto-restart, resume-on-crash): " if supervised
         else "launching (single run, NO supervisor): ") + " ".join(cmd))
    try:
        rc = subprocess.call(cmd, env=env, cwd=_ROOT)
    except KeyboardInterrupt:
        rc = 0
    if args.throwaway:
        log(f"throwaway run ended (rc={rc}) — sandbox left for inspection: {kira_dir}")
        log("discard it with: python pokemon_agent/go.py --clean-throwaways")
    return rc


if __name__ == "__main__":
    if "--clean-throwaways" in sys.argv:
        n = 0
        tmp = tempfile.gettempdir()
        for d in os.listdir(tmp):
            if d.startswith("kira_go_"):
                shutil.rmtree(os.path.join(tmp, d), ignore_errors=True)
                n += 1
        print(f"[GO] cleaned {n} throwaway sandbox(es)")
        sys.exit(0)
    sys.exit(main())
