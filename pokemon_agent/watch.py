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
    ("summit",        "banked_CREDITS",         "🏆 THE SUMMIT — Champion, credits rolled (Venusaur L95)"),
    ("pre-e4",        "banked_E4",              "The Elite Four doorstep — badges 8, the gauntlet ahead"),
    ("victory-road",  "banked_VICTORY",         "Victory Road — the final climb to Indigo Plateau"),
    ("pre-blaine",    "banked_BLAINE",          "Cinnabar, before Blaine — badge 7 gym"),
    ("cinnabar",      "banked_CINNABAR",        "Just off Seafoam — arrived at Cinnabar Island"),
    ("pre-giovanni",  "banked_GIOVANNI",        "Before Giovanni — the final badge-8 Viridian showdown"),
    ("pre-sabrina",   "banked_SABRINA",         "Saffron, before Sabrina — badge 6, the psychic gym"),
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
    if not args.no_audio:
        cmd.append("--audio")

    env = dict(os.environ)
    env["POKEMON_CAMPAIGN_DIR"] = sandbox   # THE time-machine seam — she reads/writes only the sandbox

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
