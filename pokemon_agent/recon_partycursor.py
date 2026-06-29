"""recon_partycursor.py — find the in-battle PARTY-MENU cursor RAM address (to fix the switch slot-select).
Boots a battle fixture, drives to the action menu, opens POKEMON (party screen), then snapshots EWRAM,
presses DOWN, snapshots again, and reports bytes that incremented 0->1 (the cursor/slotId candidates)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                  # noqa: E402
import pokemon_state as st                 # noqa: E402
import firered_ram as ram                  # noqa: E402
from battle_agent import BattleAgent       # noqa: E402
from campaign import resolve_state         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
EWRAM_BASE, EWRAM_SIZE = 0x02000000, 0x40000


def snap(b):
    return bytes(b.rd8(EWRAM_BASE + i) for i in range(EWRAM_SIZE))


def main():
    fixture = sys.argv[1] if len(sys.argv) > 1 else "forest_battle.state"
    b = Bridge(ROM)
    with open(resolve_state(fixture), "rb") as f:
        b.load_state(f.read())
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=print)
    # drive to a clean action menu
    import time
    t0 = time.time()
    while time.time() - t0 < 20 and not st.in_battle(b):
        b.run_frame()
    ag._settle_action_menu()
    if not ag._goto_pokemon():
        print("could not reach POKEMON action cursor"); return
    b.press("A", ag.hold, ag.hold, ag.render, owner=ag.owner)
    for _ in range(40):
        b.run_frame()
    print(f"party screen open? white_box={ag._white_box()} (want False)")
    s0 = snap(b)
    # press DOWN a few times, snapshot after each
    snaps = [s0]
    for _ in range(3):
        ag._tap("DOWN")
        for _ in range(20):
            b.run_frame()
        snaps.append(snap(b))
    # find bytes that go 0,1,2,3... (monotonic small increments) across the snaps
    cands = []
    for off in range(EWRAM_SIZE):
        vals = [snaps[k][off] for k in range(len(snaps))]
        if vals[0] == 0 and vals == sorted(vals) and vals[-1] in (1, 2, 3) and len(set(vals)) >= 2 \
                and all(v <= 5 for v in vals):
            cands.append((EWRAM_BASE + off, vals))
    print(f"\n{len(cands)} cursor-candidate addresses (0->N on DOWN):")
    for addr, vals in cands[:40]:
        print(f"  0x{addr:08X}: {vals}")


if __name__ == "__main__":
    main()
