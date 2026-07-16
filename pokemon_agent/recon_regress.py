"""recon_regress.py - run the REAL BattleAgent (no overrides) on each suite state and
report the outcome. The regression gate: battle.state, wild_battle.state, forest_battle.state
must all WIN. Pass extra state basenames as args to add cases (e.g. forest_metapod.state).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge              # noqa: E402
import pokemon_state as st             # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import resolve_state     # noqa: E402  (find fixtures across the 3 lineages)

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = sys.argv[1:] or ["battle.state", "wild_battle.state", "forest_battle.state"]


def run_one(name):
    path = resolve_state(name)
    if not path:
        return name, "MISSING"
    b = Bridge(ROM)
    with open(path, "rb") as f:
        b.load_state(f.read())
    log = []
    ag = BattleAgent(b, on_event=lambda s, **k: log.append(s),
                     log=lambda m: log.append(m))
    out = ag.run(max_seconds=120)
    # surface the last enemy/our HP for context
    return name, out


def main():
    print("   [regress] === battle engine regression ===")
    results = []
    for name in STATES:
        n, out = run_one(name)
        mark = "OK " if out == "win" else "!! "
        print(f"   [regress] {mark}{n:28s} -> {out}")
        results.append(out)
    wins = sum(1 for r in results if r == "win")
    print(f"   [regress] {wins}/{len(results)} WIN")


if __name__ == "__main__":
    main()
