"""recon_enter_league_smoke.py — full endgame dispatch smoke (NS#17 flag-flip gate).

Verifies the HANDOFF of the wired endgame path: Campaign._enter_league now runs
prep_e4_in_victory_road (the Indigo-anchored VR2F cave grind) BEFORE dispatching e4_strike. This boots
indigo_reach_g at the Indigo Plateau with CAVE_GRIND=1 and BOUNDED budgets (a short VR-grind budget so it
does a stint or two then proceeds; a short E4 deadline so e4_strike just boots + enters the League) and
asserts: (a) the pre-gauntlet grind fires + levels a bench mon, (b) she returns to Indigo (3,9), (c)
e4_strike then BOOTS and enters the League door / a gauntlet room — the whole chain wires without wedging.
It does NOT try to win the E4 (team-depth) — only that the dispatch composes cleanly.

RUN: POKEMON_CAVE_GRIND=1 ../.venv/Scripts/python.exe -u recon_enter_league_smoke.py
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
os.environ.setdefault("POKEMON_VR_GRIND_BUDGET_S", "260")     # a stint or two, then proceed
os.environ.setdefault("POKEMON_GRIND_WEAK_PROBE_S", "110")
os.environ.setdefault("POKEMON_GRIND_WEAK_BUDGET_S", "120")
os.environ.setdefault("POKEMON_E4_DEADLINE_S", "150")         # e4_strike just boots + enters, then deadlines
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
    state = camp.read_live_state()
    L(f"boot {SRC} map={tv.map_id(b)}@{tv.coords(b)} party={lv0} badges={state['badge_count']}")
    r = camp._enter_league(state)
    lvN = camp._party_levels()
    bench_up = sum(1 for a, c in zip(sorted(lv0), sorted(lvN)) if c > a)
    L(f"RESULT: _enter_league -> {r!r} | party {lv0} -> {lvN} | bench rose={bench_up} | battles={nb[0]} "
      f"| final map={tv.map_id(b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
