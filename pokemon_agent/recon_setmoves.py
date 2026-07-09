"""recon_setmoves.py — build a TEST FIXTURE: give ss_ticket's ivysaur the moveset the move-learn
fix WOULD produce (Vine Whip + Razor Leaf + PoisonPowder + Sleep Powder), so a look-ahead can answer
"does a 2-grass-attacker ivysaur BEAT the rival Gary?" (night shift 15). Writes a scratch workshop
state; the canonical save is never touched.
RUN: python pokemon_agent/recon_setmoves.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge
import firered_ram as ram
import pokemon_state as st
from campaign import Campaign, resolve_state

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OUT = os.path.join(_HERE, "states", "workshop", "ss_ticket_razorleaf.state")

# the fixed set: Vine Whip(22) + Razor Leaf(75) two grass attackers, PoisonPowder(77) + Sleep Powder(79)
MOVES = [22, 75, 77, 79]
PPS = [25, 25, 35, 15]


def main():
    b = Bridge(ROM)
    with open(resolve_state("ss_ticket.state"), "rb") as f:
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
    lvl = b.rd8(ram.GPLAYER_PARTY + 0x54)
    print("ivysaur level:", lvl)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(b.save_state())
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
