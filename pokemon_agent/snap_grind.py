"""snap_grind.py - poll the look-ahead STAGE save while a headless grind runs and keep the best
PRE-CREDITS snapshot for a live final shot.

WHY (2026-08-15): recon_longrun plays until it rolls credits, so its only bank is POST-credits and
useless for "grind her up, then let her finish it LIVE on screen". The campaign rewrites
%TEMP%/longrun/stage/kira_campaign.state as it banks during the run, so polling that file captures
her mid-grind. We keep every snapshot whose ace level is >= MIN_ACE and whose map is NOT the
Champion's room / Hall of Fame, naming each by ace level + place so the best one is obvious.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\snap_grind.py [minutes] [min_ace_level]
"""
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge              # noqa: E402
import firered_ram as ram              # noqa: E402
import travel as tv                    # noqa: E402
import pokemon_state as st             # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STAGE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "stage")
STATE = os.path.join(STAGE, "kira_campaign.state")
OUT = os.path.join("G:\\", "temp", "opencode", "snaps")
# Maps to REFUSE: (1,80) Hall of Fame and (1,79) Champion's room — a snapshot there is either
# post-credits or one fight from it, which is not a watchable "final shot".
REFUSE_MAPS = {(1, 79), (1, 80)}
SIDECARS = ("world_model.json", "strat_memory.json", "journey_core.json", "soul.json",
            "dialogue_hints.json", "team_plan_state.json")


def read_state(path):
    """(ace_level, party, map, coords) from a savestate, or None if it won't load."""
    try:
        b = Bridge(ROM)
        with open(path, "rb") as f:
            b.load_state(f.read())
        for _ in range(30):
            b.run_frame()
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        party = []
        for i in range(min(cnt, 6)):
            base = ram.GPLAYER_PARTY + i * 100
            party.append((st.SPECIES_NAME.get(st.read_party_species(b, i), "?"),
                          b.rd8(base + 0x54), b.rd16(base + 0x56)))
        mid = tuple(tv.map_id(b))
        xy = tuple(tv.coords(b))
        ace = max((lv for _s, lv, _hp in party), default=0)
        del b
        return ace, party, mid, xy
    except Exception as e:
        print(f"   (unreadable: {e})", flush=True)
        return None


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    min_ace = int(sys.argv[2]) if len(sys.argv) > 2 else 85
    os.makedirs(OUT, exist_ok=True)
    deadline = time.time() + minutes * 60
    seen_mtime = None
    best = (0, None)
    n = 0
    print(f"[snap] watching {STATE}\n[snap] keeping pre-credits snaps with ace >= L{min_ace} "
          f"for {minutes:.0f} min -> {OUT}", flush=True)
    while time.time() < deadline:
        try:
            mt = os.path.getmtime(STATE) if os.path.exists(STATE) else None
        except Exception:
            mt = None
        if mt and mt != seen_mtime:
            seen_mtime = mt
            tmp = os.path.join(OUT, "_probe.state")
            try:
                shutil.copy2(STATE, tmp)
            except Exception:
                time.sleep(5)
                continue
            info = read_state(tmp)
            if info:
                ace, party, mid, xy = info
                tag = f"ace L{ace} map={mid}{xy} party={[(s, l) for s, l, _h in party]}"
                if mid in REFUSE_MAPS:
                    print(f"[snap] skip (champion/HoF map): {tag}", flush=True)
                elif ace < min_ace:
                    print(f"[snap] skip (ace too low):      {tag}", flush=True)
                else:
                    n += 1
                    name = f"snap{n:02d}_aceL{ace}_map{mid[0]}-{mid[1]}"
                    dst = os.path.join(OUT, name)
                    os.makedirs(dst, exist_ok=True)
                    shutil.copy2(tmp, os.path.join(dst, "kira_campaign.state"))
                    for s in SIDECARS:
                        src = os.path.join(STAGE, s)
                        if os.path.exists(src):
                            shutil.copy2(src, os.path.join(dst, s))
                    print(f"[snap] KEPT {name}: {tag}", flush=True)
                    if ace > best[0]:
                        best = (ace, dst)
        time.sleep(10)
    print(f"[snap] done. best = ace L{best[0]} -> {best[1]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
