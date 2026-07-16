"""recon_mk_midvr_fixture.py — build a MID-VICTORY-ROAD party-6 fixture for the NS#17 cave-grind smoke.

The NS#16 frontier's #1 gap: "I have NO fixture inside an adequate Center-less cave" to smoke the party>=2
participation-switch cave-grind (Mt. Moon smoke was a solo-Ivysaur; the intended endgame case is a party-6
with an ace-protected bench). This boots giovanni_kit_g (badge 8, Venusaur L68 ace + L8-39 bench = the exact
underleveled-for-E4 shape) and runs victory_road.run_strike, but INTERCEPTS the map-keyed dispatch loop the
instant she first stands on a VR floor (VR1F/2F/3F) and BANKS the state (+ the giovanni_kit_g sidecars) as
`states/workshop/midvr_g.*`. That gives a party-6 fixture standing IN an adequate Center-less cave.

RUN: ../.venv/Scripts/python.exe -u recon_mk_midvr_fixture.py
"""
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import field_moves as fm             # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
import victory_road as vr            # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WS = os.path.join(_HERE, "states", "workshop")
DBG = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "midvr_fixture")
VR_FLOORS = {vr.VR1F, vr.VR2F, vr.VR3F}
SRC = os.environ.get("MIDVR_SRC", "giovanni_kit_g")
OUT = os.environ.get("MIDVR_OUT", "midvr_g")


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    src_state = os.path.join(WS, SRC + ".state")
    b = Bridge(ROM)
    with open(src_state, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    nb = [0]

    def fight():
        nb[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=420)

    camp = Campaign(b, battle_runner=fight, on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    for loader, side, fb in ((camp.world.load, SRC + ".world_model.json", C.WORLD_JSON),
                             (camp.strat.load, SRC + ".strat_memory.json", C.STRAT_JSON)):
        try:
            p = os.path.join(WS, side)
            loader(p if os.path.exists(p) else fb)
        except Exception:
            pass
    L(f"boot {SRC} map={tv.map_id(b)} coords={tv.coords(b)} badge8={int(fm.read_flag(b, vr.FLAG_BADGE_EARTH))} "
      f"party={camp._party_levels()}")

    def bank():
        os.makedirs(WS, exist_ok=True)
        with open(os.path.join(WS, OUT + ".state"), "wb") as f:
            f.write(b.save_state())
        for ext in ("world_model.json", "strat_memory.json", "journey_core.json", "soul.json"):
            s = os.path.join(WS, f"{SRC}.{ext}")
            if os.path.exists(s):
                shutil.copyfile(s, os.path.join(WS, f"{OUT}.{ext}"))
        L(f"BANKED {OUT} at {tv.map_id(b)}@{tv.coords(b)} party={camp._party_levels()}")

    vro = vr.VictoryRoad(camp, L, dbg_dir=DBG)
    _orig = vro.handle_interrupts

    def _hooked():
        r = _orig()
        if tuple(tv.map_id(b)) in VR_FLOORS:
            bank()
            L(f"REACHED a VR floor — fixture saved, stopping (battles {nb[0]})")
            raise SystemExit(0)
        return r

    vro.handle_interrupts = _hooked
    try:
        res = vro.run()
        L(f"strike returned {res} WITHOUT hitting a VR floor (at {tv.map_id(b)}) — NO fixture banked")
        return 1
    except SystemExit:
        return 0


if __name__ == "__main__":
    sys.exit(main())
