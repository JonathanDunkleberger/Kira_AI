"""recon_vr_grind_smoke.py — end-to-end smoke of the Indigo-anchored Victory-Road cave grind (NS#17).

Boots indigo_reach_g (at the Indigo Plateau, badge 8, underleveled bench: Venusaur L71 ace + Lapras L39 /
Kadabra L40 levelable + L9-14 chaff) and runs Campaign.prep_e4_in_victory_road — the pre-E4 team-depth
top-up. Watches the loop: descend Indigo -> R23-north -> VR2F, cave-grind the bench (boulder-avoided,
contained), heal at Indigo when the ace dips, re-dip, repeat. SUCCESS = a levelable bench mon RISES, she
draws step-encounters in VR2F, heals reach Indigo (not Viridian), and she returns 'ready' without wedging.

RUN: POKEMON_CAVE_GRIND=1 ../.venv/Scripts/python.exe -u recon_vr_grind_smoke.py
     GRIND_TARGET=<n> caps the target (default: a modest bench top-up so a stint completes in-window).
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ.setdefault("POKEMON_CAVE_GRIND", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WS = os.path.join(_HERE, "states", "workshop")
SRC = os.environ.get("INDIGO_STATE", "indigo_reach_g")


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(WS, SRC + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    nb = [0]

    def fight():
        nb[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=240)

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

    lv0 = camp._party_levels()
    st0 = camp.read_live_state()
    try:
        camp.team_planner.ensure_plan(st0["party"], st0["badge_count"])   # so _prep_e4_target resolves
    except Exception:
        pass
    L(f"boot {SRC} map={tv.map_id(b)}@{tv.coords(b)} party={lv0}")
    # a modest target so the smoke completes a couple of stints in-window (levelables are L39/L40 -> +3)
    tgt = int(os.environ.get("GRIND_TARGET", "43"))
    stints = int(os.environ.get("MAX_STINTS", "6"))
    L(f"prep_e4_in_victory_road(target={tgt}, max_stints={stints})  [E4-prep real target would be "
      f"{camp._prep_e4_target(camp.read_live_state(), camp.read_live_state()['party'])}]")
    r = camp.prep_e4_in_victory_road(target=tgt, max_stints=stints,
                                     budget_s=int(os.environ.get("VR_BUDGET_S", "600")))
    lvN = camp._party_levels()
    bench_up = sum(1 for a, c in zip(sorted(lv0), sorted(lvN)) if c > a)
    fin = tuple(tv.map_id(b))
    L(f"RESULT: {r!r} | party {lv0} -> {lvN} | bench slots rose={bench_up} | battles={nb[0]} | "
      f"final map={fin} ({'VIRIDIAN-ABANDON' if fin == (3, 1) else 'ok'})")
    ok = r == "ready" and nb[0] > 0 and bench_up >= 1 and fin != (3, 1)
    L(f"SMOKE {'PASS' if ok else 'FAIL'} (ready={r=='ready'} battles={nb[0]} bench_rose={bench_up} "
      f"no_viridian={fin != (3, 1)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
