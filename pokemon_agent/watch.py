"""watch.py — ONE command to watch Kira play, SOUL-ON, from ANY point of the Sherpa timeline.

THE COUCH-TEST RIG (post-credits Phase B). Jonny picks a spawn point (pre-E4, pre-Giovanni,
the summit, …), and this:

  1. PREFLIGHTS THE SOUL STACK. GETs http://127.0.0.1:8766/state (the live Kira bot's control
     server). If the bot is DOWN it REFUSES LOUDLY with the exact boot command — so a soul-BLIND
     watch (tonight's 10061 pop-in, every oracle POST failing silently) is impossible by accident.
  2. BUILDS A DISPOSABLE SANDBOX. Copies the chosen banked sanctity bundle (state + soul + strat +
     world + journey) into a fresh temp dir. CANONICAL IS NEVER TOUCHED — play_live is pointed at
     the sandbox via POKEMON_CAMPAIGN_DIR, so even a hard-kill mid-watch can't write canonical.
  3. LAUNCHES play_live --resume --free-roam --audio, WINDOWED, TRUE SPEED (paced), game music on.
     Her VOICE rides the live bot's TTS (that's what the preflight guarantees); the game audio rides
     play_live's --audio pump. Both reach Jonny's speakers.
  4. CLEAN STOP. Ctrl-C ends the run; the sandbox is disposable (pruned by --clean or the next run).

USAGE
  .venv\\Scripts\\python.exe pokemon_agent\\watch.py                 # interactive spawn-point picker
  .venv\\Scripts\\python.exe pokemon_agent\\watch.py --at pre-e4      # spawn straight from a named point
  .venv\\Scripts\\python.exe pokemon_agent\\watch.py --canonical      # the true summit (hall_of_fame), sandboxed
  .venv\\Scripts\\python.exe pokemon_agent\\watch.py --list           # list spawn points and exit
  .venv\\Scripts\\python.exe pokemon_agent\\watch.py --clean          # delete old watch sandboxes and exit

  --headless / --no-audio  strip the window / game audio (her voice still rides the bot).
  --url URL                point at a non-default control server.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# cp1252 Windows consoles choke on the emoji/arrows in the picker + refusal banner (a raise here
# would kill the launcher on its first print). Force utf-8 with replacement, exactly like play_live.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# The sanctity bundle: exactly what a --resume free-roam needs on disk. soul.json is the bundle
# filename; the live campaign reads soul from pokemon_soul.json, so we remap it into the sandbox.
BUNDLE_FILES = ("kira_campaign.state", "world_model.json", "strat_memory.json",
                "soul.json", "journey_core.json")
SOUL_SRC, SOUL_DST = "soul.json", "pokemon_soul.json"

_LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
_CANON = os.path.join(_HERE, "states", "campaign")
_SANDBOX_ROOT = os.path.join(os.environ.get("TEMP", _HERE), "kira_watch")

# Curated friendly names + one-line descriptions for the common spawn points. Any banked_* dir with
# a complete bundle is still selectable by its raw name even if it's not listed here — this is just
# the nice picker labelling, not a gate.
CURATED = [
    # alias,           banked dir,               human description
    ("postgame",      "banked_POSTGAME",        "🏆 THE VICTORY LAP — Champion, post-credits, in control (the SAFE summit spawn)"),
    # ⚠️ banked_CREDITS is the MID-CEREMONY Hall-of-Fame moment (2026-07-08 diagnosis): resuming
    # it can drain the ceremony -> CREDITS -> post-credits SoftReset -> title screen (the QW-4
    # void). Kept for the record; spawn 'postgame' for the safe Champion watch.
    ("summit",        "banked_CREDITS",         "🏆 THE SUMMIT (mid-ceremony — credits WILL roll; prefer 'postgame')"),
    # ⚠️ ALIAS FIX (2026-07-08 probe): the true PRE-E4-COMBAT save is banked_VICTORY (Indigo Plateau,
    # 8 badges, game_clear=FALSE, E4 genuinely ahead). banked_E4 is the POST-victory Hall-of-Fame
    # ceremony (Lance already beaten) — the old 'pre-e4' alias dropped Jonny into the credits, not a
    # fight. 'pre-e4' now = the real doorstep; 'hall-of-fame' = the post-win ceremony (record only).
    ("pre-e4",        "banked_LORELEI",         "🥊 INSIDE Lorelei's room — the E4 gauntlet starts on the next step (no cave, no walk; goal-pin it)"),
    ("e4-approach",   "banked_VICTORY",         "The dramatic approach — Indigo Plateau exterior, ~14 tiles to the League building then Lorelei (past Victory Road)"),
    ("hall-of-fame",  "banked_E4",              "Post-victory Hall of Fame ceremony (Lance beaten; NO fights left)"),
    # banked_BLAINE has badge 7 (Blaine already beaten); the real PRE-Blaine save is banked_CINNABAR
    # (badge 6, at Cinnabar). Post-win banks kept under a *-done alias for record.
    ("pre-blaine",    "banked_CINNABAR",        "🥊 Cinnabar, badge 6 — Blaine NOT yet beaten (real gym combat ahead — goal-pin it)"),
    ("blaine-done",   "banked_BLAINE",          "AFTER Blaine — badge 7 won (no fight left)"),
    ("pre-giovanni",  "banked_GIOVANNI",        "AFTER Giovanni — badge 8 won (post-win; no fight left)"),
    ("pre-sabrina",   "banked_SABRINA",         "AFTER Sabrina — badge 6 won (post-win; no fight left)"),
    ("silph",         "banked_SILPH",           "Silph Co. cleared — Lapras in hand, Master Ball won"),
    ("snorlax",       "banked_SNORLAX",         "Snorlax woken — the coastal road opens"),
    ("lapras",        "banked_LAPRAS",          "Lapras just received — the future Surf carrier"),
    ("safari",        "banked_SAFARI",          "Safari Zone done — HM Surf + Strength, Gold Teeth"),
    ("surf",          "banked_SURF_TAUGHT",     "Surf taught — water routes open"),
    ("flute",         "banked_FLUTE",           "Poké Flute in hand — Snorlax is next"),
    ("scope",         "banked_SCOPE",           "Silph Scope — Pokémon Tower ghosts revealed"),
    ("rock-tunnel",   "banked_ROCKTUNNEL",      "Rock Tunnel — Flash lights the dark"),
    ("hm05",          "banked_HM05",            "HM05 Flash learned — early-game"),
]


def _preflight(url):
    """Is the live Kira bot (soul stack) up and answering on the control server? Returns
    (ok, detail). ok=False means her voice/oracle would be dead — we must refuse."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/state", timeout=4) as r:
            st = json.load(r)
        return True, st
    except Exception as e:
        return False, str(e)


def _refuse_bot_down(url, detail):
    py = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        py = "python"
    print("\n" + "=" * 70)
    print("  ✋ WATCH REFUSED — the Kira soul stack is DOWN.")
    print(f"     Control server {url} did not answer ({detail}).")
    print("     Every oracle/voice POST would fail 10061 and she'd play SOUL-BLIND —")
    print("     silent grinding, no wants, no reactions, no narration. Not a watch.")
    print("")
    print("  START THE BOT FIRST (its own terminal — it owns TTS / VTS / mic):")
    print(f"     {py} {os.path.join(_ROOT, 'run.py')}")
    print("")
    print("  Wait for it to finish booting (control server on :8766), then re-run watch.py.")
    print("=" * 70 + "\n")


def _valid_bundle(d):
    return os.path.isdir(d) and all(os.path.exists(os.path.join(d, f)) for f in BUNDLE_FILES)


def _discover():
    """All selectable spawn points as (alias, dir, desc), curated first (in order), then any other
    complete banked_* bundle by raw name, then canonical."""
    out, seen = [], set()
    for alias, name, desc in CURATED:
        d = os.path.join(_LONGRUN, name)
        if _valid_bundle(d):
            out.append((alias, d, desc)); seen.add(name)
    if os.path.isdir(_LONGRUN):
        for name in sorted(os.listdir(_LONGRUN)):
            if name.startswith("banked_") and name not in seen:
                d = os.path.join(_LONGRUN, name)
                if _valid_bundle(d):
                    out.append((name.replace("banked_", "").lower(), d, "(uncurated bank)"))
    return out


def _resolve(spawn, points):
    """Map a --at value (alias, raw dir name, or path) to a bundle dir."""
    if spawn in ("canonical", "summit-canonical"):
        return _CANON if _valid_bundle(_CANON) else None
    for alias, d, _ in points:
        if spawn.lower() in (alias.lower(), os.path.basename(d).lower(),
                             os.path.basename(d).replace("banked_", "").lower()):
            return d
    if os.path.isdir(spawn) and _valid_bundle(spawn):
        return spawn
    return None


def _clean():
    if not os.path.isdir(_SANDBOX_ROOT):
        print("no watch sandboxes to clean."); return
    n = 0
    for name in os.listdir(_SANDBOX_ROOT):
        p = os.path.join(_SANDBOX_ROOT, name)
        try:
            shutil.rmtree(p); n += 1
        except Exception as e:
            print(f"  could not remove {p}: {e}")
    print(f"cleaned {n} watch sandbox(es) from {_SANDBOX_ROOT}")


def _build_sandbox(bundle_dir, label):
    """Copy the sanctity bundle into a fresh disposable sandbox. Returns the sandbox path.
    Canonical is only ever READ here — never written."""
    os.makedirs(_SANDBOX_ROOT, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    sandbox = os.path.join(_SANDBOX_ROOT, f"sandbox_{label}_{ts}")
    os.makedirs(sandbox, exist_ok=True)
    for f in BUNDLE_FILES:
        shutil.copy2(os.path.join(bundle_dir, f), os.path.join(sandbox, f))
    # Remap the bundle's soul.json -> the pokemon_soul.json filename the live campaign loads.
    shutil.copy2(os.path.join(bundle_dir, SOUL_SRC), os.path.join(sandbox, SOUL_DST))
    # Optional sidecars (e.g. dialogue_hints.json — her overheard-intel ledger): ride when present,
    # never required (old banks without them just start empty in the sandbox).
    try:
        import sanctity as _sanctity
        for f in getattr(_sanctity, "OPTIONAL_SIDECARS", ()):
            src = os.path.join(bundle_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(sandbox, f))
    except Exception:
        pass
    return sandbox


def main():
    ap = argparse.ArgumentParser(description="Watch Kira play soul-on from any Sherpa-timeline point.")
    ap.add_argument("--at", default=None, help="spawn point (alias / banked_* name / path). Omit for picker.")
    ap.add_argument("--canonical", action="store_true", help="watch the true canonical (hall_of_fame summit).")
    ap.add_argument("--list", action="store_true", help="list spawn points and exit.")
    ap.add_argument("--clean", action="store_true", help="delete old watch sandboxes and exit.")
    ap.add_argument("--url", default="http://127.0.0.1:8766", help="Kira control server.")
    ap.add_argument("--headless", action="store_true", help="no game window (her voice still rides the bot).")
    ap.add_argument("--no-audio", action="store_true", help="no game-audio pump (her voice still rides the bot).")
    ap.add_argument("--roam-seconds", type=int, default=86400, help="watch length cap (default 24h — Ctrl-C ends it).")
    ap.add_argument("--goal", default=None,
                    help="GOAL-PIN the spawn to an era-correct objective (overrides the post-game "
                         "victory-lap frame) — e.g. --goal \"beat Sabrina and win the Marsh Badge\" or "
                         "--goal \"fight through the Elite Four and become Champion\". This is how you "
                         "get first-time-flavored gym/E4 footage from a post-credits save.")
    args = ap.parse_args()

    if args.clean:
        _clean(); return 0

    points = _discover()

    if args.list:
        print(f"\nSpawn points (bundles in {_LONGRUN}):\n")
        if _valid_bundle(_CANON):
            print(f"  {'canonical':<14} {'← the true summit (hall_of_fame)'}")
        for alias, d, desc in points:
            print(f"  {alias:<14} {desc}")
        print("\nWatch one:  watch.py --at <alias>     |  the summit:  watch.py --canonical\n")
        return 0

    # ── choose spawn point ────────────────────────────────────────────────────
    if args.canonical:
        bundle_dir, label = (_CANON if _valid_bundle(_CANON) else None), "canonical"
    elif args.at:
        bundle_dir = _resolve(args.at, points)
        label = args.at.lower().replace("banked_", "")
    else:
        # interactive picker
        print("\n  Pick a spawn point to watch Kira from (canonical = the summit):\n")
        menu = ([("canonical", _CANON, "🏆 the true summit — hall_of_fame (Champion)")]
                if _valid_bundle(_CANON) else []) + points
        for i, (alias, d, desc) in enumerate(menu):
            print(f"   [{i:>2}] {alias:<14} {desc}")
        print("")
        try:
            sel = input("  number (or alias) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  cancelled."); return 1
        if sel.isdigit() and 0 <= int(sel) < len(menu):
            alias, bundle_dir, _ = menu[int(sel)]
            label = alias
        else:
            bundle_dir = _resolve(sel, points)
            label = sel.lower().replace("banked_", "")

    if not bundle_dir or not _valid_bundle(bundle_dir):
        print(f"\n  ✋ no complete bundle for that spawn point. Try:  watch.py --list\n")
        return 1

    # ── PREFLIGHT the soul stack (refuse loudly if the bot is down) ───────────
    ok, detail = _preflight(args.url)
    if not ok:
        _refuse_bot_down(args.url, detail)
        return 2
    is_speaking = detail.get("is_speaking") if isinstance(detail, dict) else "?"
    print(f"\n  ✅ soul stack UP — {args.url} answering (is_speaking={is_speaking}). Her voice is live.")

    # ── build disposable sandbox (canonical never written) ────────────────────
    sandbox = _build_sandbox(bundle_dir, label)
    print(f"  🧪 sandbox: {sandbox}")
    print(f"     (from {bundle_dir} — CANONICAL states/campaign UNTOUCHED)")

    # ── launch play_live, windowed, true speed, soul-on ───────────────────────
    py = sys.executable
    cmd = [py, "-u", os.path.join(_HERE, "play_live.py"),
           "--resume", "--free-roam",
           "--url", args.url,
           "--roam-ticks", "100000000",
           "--roam-seconds", str(args.roam_seconds)]
    if args.headless:
        cmd.append("--headless")
    # GAME AUDIO DEFAULT ON (2026-07-09): PortAudio output is now process-isolated (audio_child.py) —
    # a native abort kills only the child, the game keeps running. Force OFF with POKEMON_GAME_AUDIO=0.
    if os.getenv("POKEMON_GAME_AUDIO", "1") == "1" and not args.no_audio:
        cmd.append("--audio")

    env = dict(os.environ)
    env["POKEMON_CAMPAIGN_DIR"] = sandbox   # THE time-machine seam — she reads/writes only the sandbox
    if args.goal:                           # GOAL-PIN: era-correct objective overrides post-game frame
        env["POKEMON_WATCH_GOAL"] = args.goal
        # If the goal names a gym leader, auto-route her to that gym (so a post-credits/badge-8 save
        # still PATHS INTO the gym instead of stalling with next_gym=None). One flag does both.
        _leaders = ("brock", "misty", "surge", "erika", "koga", "sabrina", "blaine", "giovanni")
        _gl = next((L for L in _leaders if L in args.goal.lower()), None)
        if _gl:
            env["POKEMON_WATCH_NEXT_GYM"] = _gl
        print(f"  🎯 GOAL-PINNED: {args.goal!r}"
              + (f" (routing to {_gl.title()}'s gym)" if _gl else "") + " — not the victory lap.")

    print(f"  ▶  starting watch — TRUE SPEED, game audio {'off' if args.no_audio else 'on'}, "
          f"window {'off' if args.headless else 'on'}. Ctrl-C to stop.\n")
    try:
        rc = subprocess.run(cmd, env=env, cwd=_HERE).returncode
    except KeyboardInterrupt:
        rc = 0
    print(f"\n  ⏹  watch ended (play_live rc={rc}). Sandbox left at {sandbox} (disposable; watch.py --clean prunes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
