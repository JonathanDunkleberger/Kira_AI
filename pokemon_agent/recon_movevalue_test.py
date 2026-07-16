"""recon_movevalue_test.py — verify the STAB-aware _ensure_move_room decision (night shift 15).
Boots a state, prints the lead moveset, calls _ensure_move_room(), prints what it dropped + the result.
RUN: python pokemon_agent/recon_movevalue_test.py <state>
Expect:
  mtmoon_endgame.state  [Tackle, VineWhip, Poison, Sleep]  -> DROP Tackle (weak non-STAB filler), keep VineWhip
  ss_ticket_razorleaf   [VineWhip, RazorLeaf, Poison, Sleep] -> DROP nothing (2 STABs + 2 status all precious)
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


def show(camp):
    return [st.MOVE_NAMES.get(m, f"#{m}") for m in camp._lead_moves()]


def main():
    b = Bridge(ROM)
    with open(resolve_state(sys.argv[1]), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    print("BEFORE:", show(camp))
    dropped = camp._ensure_move_room()
    print("DROPPED:", dropped)
    print("AFTER :", show(camp))


if __name__ == "__main__":
    main()
