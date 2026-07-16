"""recon_fuchsia_mart.py — SHIFT-10: identify the Fuchsia City Poké Mart door + BUY-list row order, so
she can autonomously stock Super/Hyper Potions before Koga (the proven potion-stall win). Drives the
fuchsia_potions fixture to Fuchsia (patching beat_gym to snapshot at the gym-city instead of fighting),
then enters each candidate building door and tests the buy clerk: the interior where the BUY list opens
(MART_CURSOR responds) is the Mart. Dumps the interior item rows via a live buy-verify probe.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_FIELD_MOVES", "1")
os.environ.setdefault("POKEMON_ITEM_PICKUP", "1")

from bridge import Bridge                                          # noqa: E402
import travel as tv                                               # noqa: E402
import firered_ram as ram                                          # noqa: E402
import pokemon_state as st                                         # noqa: E402
from battle_agent import BattleAgent                               # noqa: E402
from campaign import (Campaign, resolve_state, MART_CLERK_FRONT,   # noqa: E402
                      MART_CURSOR, MART_SCROLL, FUCHSIA)

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OB, SZ = 0x02036E38, 0x24
OFF_ACTIVE, OFF_GFX, OFF_X, OFF_Y, OFF_FACING = 0x00, 0x05, 0x10, 0x12, 0x18
# Fuchsia (3,7) building doors learned live (world_model), minus gym (9,32) + Center (25,31):
DOORS = [(24, 5), (11, 15), (28, 16), (14, 31), (38, 31), (39, 28), (19, 31)]
CLERK_TILE = (2, 3)


def objs(b):
    out = []
    for i in range(1, 16):
        o = OB + i * SZ
        if not (b.rd8(o + OFF_ACTIVE) & 1):
            continue
        out.append((b.rd8(o + OFF_GFX), b.rds16(o + OFF_X) - 7, b.rds16(o + OFF_Y) - 7,
                    b.rd8(o + OFF_FACING) & 0x0F))
    return out


class _AtCity(Exception):
    pass


def main():
    b = Bridge(ROM)
    with open(resolve_state("fuchsia_potions.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    # IN-BATTLE ITEM chooser (mirror recon_longrun): let her heal-stall through fights on the drive.
    def chooser(kind, options, ctx):
        if kind == "battle_item":
            return ("use_potion" if "use_potion" in options else next(iter(options.keys())))
        return None

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: None, choose=chooser).run(max_seconds=180)

    camp = Campaign(b, battle_runner=runner, render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    # Drive to Fuchsia: patch beat_gym so the moment head_to_gym reaches the gym-city, we STOP with her
    # standing on the Fuchsia overworld (a clean probe origin) instead of entering the fight.
    def _stop(*a, **k):
        raise _AtCity()
    camp.beat_gym = _stop
    try:
        camp.free_roam(max_ticks=60, max_seconds=600)
    except _AtCity:
        pass
    except Exception as e:
        print(f"drive aborted: {e!r}")

    m = tv.map_id(b)
    print(f"\nafter drive: map={m} coords={tv.coords(b)}", flush=True)
    if tuple(m) != tuple(FUCHSIA):
        # fall back: try to force onto Fuchsia by exiting to overworld
        print("!! not on Fuchsia overworld — probing may fail")
    snap = b.save_state()

    for door in DOORS:
        b.load_state(snap)
        for _ in range(10):
            b.run_frame()
        try:
            r = camp.enter_warp(pick=door)
        except Exception as e:
            print(f"door {door}: enter_warp err {e}"); continue
        inside = tv.map_id(b)
        if inside[0] == 3:
            print(f"door {door}: did not enter (still overworld {inside})"); continue
        ob = objs(b)
        clerk = [o for o in ob if (o[1], o[2]) == CLERK_TILE]
        is_mart = False
        try:
            camp._step_to(MART_CLERK_FRONT)
            for _ in range(20):
                b.run_frame()
            is_mart = bool(camp._mart_enter_buylist())
        except Exception as e:
            print(f"door {door} -> {inside}: buylist probe err {e}")
        tag = ""
        if is_mart:
            rows = []
            try:
                # walk the buy list top->down reading the highlighted item id if exposed; fall back to
                # just reporting that the buy list opened (row order then control-verified by a buy).
                cur = b.rd16(MART_CURSOR); scr = b.rd16(MART_SCROLL)
                tag = f" *** MART (buylist OPENED; cursor={cur} scroll={scr}) ***"
            except Exception:
                tag = " *** MART (buylist OPENED) ***"
        elif clerk:
            tag = " (clerk at (2,3) — likely Mart, buylist probe failed)"
        print(f"door {door} -> interior {inside}: objs={ob}{tag}", flush=True)
        # B out
        for _ in range(8):
            b.press("B", 6, 12, lambda: None, owner="agent")
            for _ in range(10):
                b.run_frame()


if __name__ == "__main__":
    main()
