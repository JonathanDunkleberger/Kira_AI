"""recon_repair_kit.py — night shift 6: build bill_done_kit.state, giving bill_done's ivysaur the
NEUTRAL-COVERAGE kit the fixed move-manager now KEEPS (Razor Leaf + Vine Whip + Tackle + Sleep Powder)
instead of the degraded grass-ONLY [Razor Leaf, Poison, Sleep, EMPTY] that PP-famines the Route-6
flying/bug gauntlet. Tackle (normal, neutral) is the lifeline vs Pidgey/Butterfree that resist grass.
Proves the S.S.Anne -> Gary -> Cut -> Surge downstream with a realistic managed team; the _value fix
makes a FRESH run reproduce this kit. Canonical save never touched (scratch workshop only).
RUN: .venv/Scripts/python.exe pokemon_agent/recon_repair_kit.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge
import pokemon_state as st
from campaign import Campaign, resolve_state

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OUT = os.path.join(_HERE, "states", "workshop", "bill_done_kit.state")

# Razor Leaf(75) STAB nuke + Vine Whip(22) STAB backup + Tackle(33) NEUTRAL coverage + Sleep Powder(79)
MOVES = [75, 22, 33, 79]
PPS = [25, 15, 35, 15]


def main():
    b = Bridge(ROM)
    with open(resolve_state("bill_done.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    print("BEFORE:", [st.MOVE_NAMES.get(m, m) for m in camp._lead_moves()], camp._lead_pps())
    camp._set_lead_moves(MOVES, PPS)
    for _ in range(6):
        b.run_frame()
    print("AFTER :", [st.MOVE_NAMES.get(m, m) for m in camp._lead_moves()], camp._lead_pps())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(b.save_state())
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
