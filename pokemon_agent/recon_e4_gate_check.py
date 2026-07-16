"""recon_e4_gate_check.py — RUN-4 (2026-07-14) E4-READINESS GATE verify.

Two layers, both required to PASS before fresh_go_4 launches:

  (1) PREDICATE LOGIC (pure): _e4_entry_ready / _e4_prep_floor_target across synthetic parties — the
      qualifying-shape rule (full six, floor>=42, gap<=15) and the ace-capped floor target (ace-15).
      Also the DISPATCH gate returns None (ALLOW) on a synthetic GREEN party and non-None (BLOCK) on RED.

  (2) BEHAVIORAL BOOT (indigo_reach_g, a badge-8 AT-Indigo RED fixture: ace L71, bench floor L9): run the
      real _e4_readiness_gate a few stints with bounded budgets and assert the RUN-4 verify criteria:
        • NO league entry while RED   (gate returns non-None every stint — the strike never dispatches)
        • weak ones fielded + floor/band CLIMBING (a levelable bench mon gains levels)
        • ace FLAT   (the ace earns no XP while the gate is RED — the solo-ace root)

RUN: POKEMON_CAVE_GRIND=1 ../.venv/Scripts/python.exe -u recon_e4_gate_check.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ.setdefault("POKEMON_CAVE_GRIND", "1")
os.environ.setdefault("POKEMON_VR_GRIND_BUDGET_S", "200")     # a stint or two per gate call, then return
os.environ.setdefault("POKEMON_GRIND_WEAK_PROBE_S", "90")
os.environ.setdefault("POKEMON_GRIND_WEAK_BUDGET_S", "110")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge             # noqa: E402
import travel as tv                   # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign         # noqa: E402
import campaign as C                  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WS = os.path.join(_HERE, "states", "workshop")
SRC = os.environ.get("INDIGO_STATE", "indigo_reach_g")

results = []


def check(label, cond):
    results.append((label, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}", flush=True)


def _party(levels):
    return [{"level": l, "species": f"m{i}"} for i, l in enumerate(levels)]


def logic_checks(camp):
    print("\n── (1) PREDICATE LOGIC ──", flush=True)
    # _e4_entry_ready(party) -> (ready, floor, ceil, gap)
    r, fl, ce, gp = camp._e4_entry_ready(_party([42, 42, 42, 42, 42, 42]))
    check("all L42 -> READY (floor 42, gap 0)", r and fl == 42 and gp == 0)
    r, *_ = camp._e4_entry_ready(_party([42, 45, 50, 55, 57, 57]))
    check("floor42 ceil57 gap15 -> READY (boundary)", r)
    r, *_ = camp._e4_entry_ready(_party([42, 45, 50, 55, 57, 58]))
    check("gap 16 -> NOT ready", not r)
    r, *_ = camp._e4_entry_ready(_party([41, 50, 50, 50, 50, 50]))
    check("floor 41 (<42) -> NOT ready", not r)
    r, *_ = camp._e4_entry_ready(_party([55, 55, 60, 63, 70, 70]))
    check("fresh-run success shape [55..70] gap15 -> READY", r)
    r, *_ = camp._e4_entry_ready(_party([31, 32, 32, 31, 33, 63]))
    check("fresh_go_3 disqualify shape -> NOT ready", not r)
    r, *_ = camp._e4_entry_ready(_party([50, 50, 50, 50, 50]))
    check("only 5 mons -> NOT ready (need a full six)", not r)
    # _e4_prep_floor_target = max(42, ace-15)
    check("target([31..63]) == 48 (ace 63 - 15)",
          camp._e4_prep_floor_target(_party([31, 32, 32, 31, 33, 63])) == 48)
    check("target([44..44]) == 42 (floor min bar)",
          camp._e4_prep_floor_target(_party([44, 44, 44, 44, 44, 44])) == 42)
    # DISPATCH gate: GREEN -> None (allow the strike); RED synthetic is exercised behaviorally below
    g = camp._e4_readiness_gate({"party": _party([55, 55, 60, 63, 70, 70])}, "enter_league")
    check("gate GREEN party -> returns None (ALLOWS the strike)", g is None)


def behavioral(camp, b):
    print("\n── (2) BEHAVIORAL BOOT (indigo_reach_g, RED) ──", flush=True)
    lv0 = camp._party_levels()
    ace0 = max(lv0)
    floor0 = min(lv0)
    state = camp.read_live_state()
    ready0, fl0, ce0, gp0 = camp._e4_entry_ready(state.get("party") or [])
    tgt = camp._e4_prep_floor_target(state.get("party") or [])
    print(f"  boot {SRC} map={tv.map_id(b)}@{tv.coords(b)} party={lv0} badges={state['badge_count']} "
          f"ready={ready0} floor={fl0} ace={ce0} gap={gp0} target=L{tgt}", flush=True)
    check("RED at boot (ace L71 / bench floor L9 -> not qualifying shape)", not ready0)

    blocked_every_stint = True
    N = int(os.environ.get("GATE_STINTS", "3"))
    for i in range(N):
        st = camp.read_live_state()
        g = camp._e4_readiness_gate(st, "enter_league")
        lv = camp._party_levels()
        print(f"  [stint {i + 1}/{N}] gate -> {g!r} | party {lv} | ace L{max(lv)} floor L{min(lv)}", flush=True)
        if g is None:
            blocked_every_stint = False
            break
    lvN = camp._party_levels()
    aceN = max(lvN)
    # a levelable bench mon rose (floor OR any non-ace member gained a level)
    bench_rose = any(c > a for a, c in zip(sorted(lv0)[:-1], sorted(lvN)[:-1]))
    check("gate BLOCKED the strike every stint while RED (never returned None)", blocked_every_stint)
    check(f"a levelable bench mon CLIMBED ({sorted(lv0)} -> {sorted(lvN)})", bench_rose)
    check(f"ACE stayed FLAT while RED (L{ace0} -> L{aceN}, drift <= 1)", aceN - ace0 <= 1)


def main():
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

    camp = Campaign(b, battle_runner=fight, on_event=lambda *a, **k: None,
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

    logic_checks(camp)
    behavioral(camp, b)

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\nE4-GATE CHECK: {n_pass}/{len(results)} PASS | battles={nb[0]}", flush=True)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
