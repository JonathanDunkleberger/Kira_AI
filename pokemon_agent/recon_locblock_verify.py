"""recon_locblock_verify.py — F-8 LOCATION BLOCK verify (headless, read-only, no bot).

Loads a spread of banked bundles spanning the three setting classes (overworld / dungeon /
building interior) and prints the exact location block each soul tick would lead with.
PASS = every block's indoor/outdoor claim matches the map group, no known place reads
"DON'T recognize", and every unknown place carries the curiosity-not-assertion instruction.

RUN:  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_locblock_verify.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                  # noqa: E402
from campaign import Campaign                              # noqa: E402
import travel as tv                                        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
LONGRUN = os.path.join(os.environ.get("TEMP", _HERE), "longrun")

BANKS = ["banked_POSTGAME", "banked_E4", "banked_VICTORY", "banked_SNORLAX",
         "banked_ROCKTUNNEL", "banked_SILPH", "banked_SAFARI", "banked_CINNABAR"]


def main():
    fails = 0
    for name in BANKS:
        p = os.path.join(LONGRUN, name, "kira_campaign.state")
        if not os.path.exists(p):
            print(f"-- {name}: (no bundle, skipped)")
            continue
        b = Bridge(ROM)
        with open(p, "rb") as f:
            b.load_state(f.read())
        for _ in range(20):
            b.run_frame()
        camp = Campaign(b, battle_runner=lambda: "skipped",
                        on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                        render=lambda: None)
        state = camp.read_live_state()
        blk = camp._location_block(state)
        mp = tuple(state["map"])
        known = mp in camp._PLACE_NAMES
        ok = True
        outdoor = mp[0] == 3 or (mp[0] == 1 and mp[1] in tv.G1_OUTDOOR)
        cave = mp[0] == 1 and mp[1] in tv.G1_CAVES
        if outdoor and "outdoors" not in blk:
            ok = False
        if cave and "underground in a real cave" not in blk:
            ok = False
        if not outdoor and not cave and "inside a building" not in blk:
            ok = False
        if known and "DON'T recognize" in blk:
            ok = False
        if not known and "guess" not in blk:
            ok = False
        fails += (not ok)
        print(f"-- {name}: map={mp}@{tv.coords(b)} known={known}")
        print(f"   {'PASS' if ok else '!! FAIL'}: {blk}")
    print(f"\n==== {'ALL PASS' if fails == 0 else f'{fails} FAIL(S)'} ====")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
