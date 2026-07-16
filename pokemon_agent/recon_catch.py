"""recon_catch.py - map the in-battle BALL-THROW flow + capture-outcome RAM (catch arc).

PASS 1 (--setup): from seg_route3_start, walk grass until a wild battle starts, bank route3_wild.state
  + report the enemy species (so every later pass starts INSIDE a fresh wild battle, no re-walk).
PASS 2 (default): from route3_wild.state, open BAG and explore the bag menu RAM (action cursor, pocket,
  item cursor) by pressing L/R/U/D and diffing, to find how to select a Poke Ball; attempt a throw and
  watch party-count (catch = party N->N+1) + battle state. Iterate here.

Run:  ...python recon_catch.py --setup     then     ...python recon_catch.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge        # noqa: E402
import travel as tv             # noqa: E402
import pokemon_state as st      # noqa: E402
import firered_ram as ram       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
START = os.path.join(_HERE, "states", "seg_route3_start.state")
WILD = os.path.join(_HERE, "states", "route3_wild.state")
HOLD = 8


def party_count(b):
    return b.rd8(ram.GPLAYER_PARTY_CNT)


def setup():
    from campaign import Campaign
    b = Bridge(ROM)
    with open(START, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    print(f"   [catch-setup] start {tv.coords(b)} party={party_count(b)}", flush=True)
    got = {"done": False}

    def battle_runner():
        # traveler hands off here on a wild encounter (after _warmup_battle). Settle to the action
        # menu, bank the fixture, and return 'loss' to make travel STOP cleanly.
        for _ in range(150):
            b.run_frame()
            if b.rd8(ram.GBATTLE_MENU_UP) == 1:
                break
        rb = st.read_battle(b)
        foe = st.SPECIES_NAME.get(rb["enemy"]["species"], "?") if rb else "?"
        with open(WILD, "wb") as f:
            f.write(bytes(b.save_state()))
        got["done"] = True
        print(f"   [catch-setup] WILD vs {foe} L{rb['enemy']['level'] if rb else '?'} "
              f"menu_up={b.rd8(ram.GBATTLE_MENU_UP)} at {tv.coords(b)} -> banked route3_wild.state",
              flush=True)
        return "loss"

    camp = Campaign(b, battle_runner=battle_runner, on_event=lambda *a, **k: None, render=lambda: None)
    camp._suppress_heal = True
    camp.trav.travel(target_map=(3, 22), edge="north", max_steps=200, max_seconds=120)
    if not got["done"]:
        print("   [catch-setup] !! no wild encounter during the north walk", flush=True)


def explore():
    from battle_agent import BattleAgent
    b = Bridge(ROM)
    with open(WILD, "rb") as f:
        b.load_state(f.read())
    for _ in range(8):
        b.run_frame()
    b.set_input_owner("agent")
    AC = ram.GBATTLE_ACTION_CURSOR
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
    for _ in range(180):                      # settle the intro to the action menu
        if b.rd8(ram.GBATTLE_MENU_UP) == 1:
            break
        ag._tap("B")
    ag._settle()
    print(f"   [catch] action menu: menu_up={b.rd8(ram.GBATTLE_MENU_UP)} cursor={b.rd8(AC)} "
          f"party={party_count(b)}", flush=True)
    # goto BAG (eaten-press tolerant, like _goto_run): FIGHT->RIGHT, RUN->UP, POKEMON->UP
    for _ in range(10):
        c = b.rd8(AC)
        if c == ram.ACT_BAG:
            break
        ag._tap("RIGHT" if c == ram.ACT_FIGHT else "UP")
        ag._wait(3)
    print(f"   [catch] cursor now {b.rd8(AC)} (want {ram.ACT_BAG}=BAG)", flush=True)
    ag._tap("A"); ag._wait(40)                # open the bag
    print(f"   [catch] bag opened: menu_up={b.rd8(ram.GBATTLE_MENU_UP)}", flush=True)
    # map the bag-menu state RAM: diff a WIDE region across each nav press
    LO, HI = 0x02023000, 0x02024400

    def snap():
        return bytes(b.read_bytes(LO, HI - LO))

    def diff(tag, prev):
        cur = snap()
        ds = [(LO + k, prev[k], cur[k]) for k in range(len(prev)) if prev[k] != cur[k]]
        print(f"   [catch] {tag}: {len(ds)} diffs -> " +
              ", ".join(f"{hex(a)}:{x}->{y}" for a, x, y in ds[:8]), flush=True)
        return cur
    s = snap()
    for press in ["RIGHT", "RIGHT", "DOWN", "DOWN", "UP", "LEFT"]:
        ag._tap(press); ag._wait(8)
        s = diff(f"after {press}", s)


if __name__ == "__main__":
    (setup if "--setup" in sys.argv else explore)()
