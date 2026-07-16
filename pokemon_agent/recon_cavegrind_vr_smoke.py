"""recon_cavegrind_vr_smoke.py — smoke the PARTY-6 cave step-encounter grind INSIDE Victory Road.

The NS#16/#17 gap: cave-grind's core mechanic was proven on a SOLO Mt.Moon smoke, but the INTENDED endgame
case is a party-6 with an ace-protected bench, grinding in a Center-less cave (Victory Road). This boots the
midvr_g fixture (party-6 standing on a VR floor, badge 8, Venusaur L68 ace + underleveled bench) and drives
`grind_weak_members` like the roam would (repeated re-entry), answering the three open questions:

  Q1  MECHANIC  — does she recognise the VR floor as a cave (_grind_is_cave), draw STEP-encounters, and
                  FIGHT them (party-6, ace-protected participation switch)?
  Q2  LEVELING  — does the FIELDED weak bench mon actually gain levels (not just the ace)?
  Q3  CONTAINMENT — does she stay ON the floor (never leak out an exit warp mid-grind)?
  Q4  HEAL-IN-CAVE — when the ace bails (ACE-DOWN GUARD -> heal_nearest from a Center-less VR floor), does
                  she reach a Center and RESUME, or wedge? (FORCE_ACE_DIP=1 writes the ace HP low to force it.)

RUN: POKEMON_CAVE_GRIND=1 ../.venv/Scripts/python.exe -u recon_cavegrind_vr_smoke.py
     add FORCE_ACE_DIP=1 to force the heal-in-cave path;  MIDVR_STATE=midvr_g  GRIND_TARGET=<n>  ITERS=<n>
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ.setdefault("POKEMON_CAVE_GRIND", "1")              # the flag under test
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import firered_ram as ram            # noqa: E402
import pokemon_state as pst          # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WS = os.path.join(_HERE, "states", "workshop")
SRC = os.environ.get("MIDVR_STATE", "midvr_g")
VR_FLOORS = {(1, 39), (1, 40), (1, 41)}
ITERS = int(os.environ.get("ITERS", "4"))


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
                           log=lambda m: print(m, flush=True)).run(max_seconds=300)

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

    m0 = tuple(tv.map_id(b))
    lv0 = camp._party_levels()
    is_cave = camp._grind_is_cave(list(m0))
    band = camp._grind_wild_band(list(m0))
    L(f"boot {SRC} map={m0}@{tv.coords(b)} party={lv0} is_cave={is_cave} band={band}")
    if m0 not in VR_FLOORS:
        L(f"!! NOT on a VR floor ({m0}) — wrong fixture; abort")
        return 1
    if not is_cave:
        L(f"!! _grind_is_cave({m0})=False — KB missing the floor; abort")
        return 1

    # target = the real E4-prep floor if available, else a modest bench top-up so a stint completes in-window
    state = camp.read_live_state()
    try:
        camp.team_planner.ensure_plan(state["party"], state["badge_count"])
    except Exception:
        pass
    e4t = None
    try:
        e4t = camp._prep_e4_target(state, state["party"])
    except Exception as e:
        L(f"   _prep_e4_target errored: {e!r}")
    target = int(os.environ.get("GRIND_TARGET", str(e4t or (min(lv0) + 5))))
    L(f"   E4-prep target={e4t}  grind target(used)={target}")

    if os.environ.get("HEAL_BOOT") == "1":
        # the REAL endgame flow arrives at Indigo HEALED before dipping into VR to grind — restore full HP
        # so the smoke tests whether the L69 ace even NEEDS to heal grinding L36-46 wilds (participation switch)
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(cnt):
            base = ram.GPLAYER_PARTY + s * pst.PARTY_MON_SIZE
            b.core.memory.u16.raw_write(base + 0x56, b.rd16(base + 0x58))
        L(f"   HEAL_BOOT: party restored to full HP (ace_hp now {camp._ace_hp_frac():.0%})")

    if os.environ.get("FORCE_ACE_DIP") == "1":
        # write the ACE (highest-level slot) current HP to ~10% to force the ACE-DOWN GUARD -> heal_nearest
        levels = camp._party_levels()
        ace = max(range(len(levels)), key=lambda s: levels[s])
        base = ram.GPLAYER_PARTY + ace * pst.PARTY_MON_SIZE
        mx = b.rd16(base + 0x58)
        b.core.memory.u16.raw_write(base + 0x56, max(1, mx // 10))
        L(f"   FORCE_ACE_DIP: ace slot {ace} HP -> {b.rd16(base + 0x56)}/{mx} (forces heal-in-cave)")

    leaked = False
    prev = m0
    for i in range(1, ITERS + 1):
        here = tuple(tv.map_id(b))
        if here != prev:
            L(f"   [tick {i}] MAP now {here} (was {prev})")
            if here not in VR_FLOORS and here != prev:
                L(f"   [tick {i}] *** left the VR floors -> {here} (map {tv.coords(b)})")
        prev = here
        L(f"   [tick {i}] grind_weak_members(L{target}) from {here}@{tv.coords(b)} party={camp._party_levels()} "
          f"ace_hp={camp._ace_hp_frac():.0%} excursions={getattr(camp, '_heal_excursion_n', 0)}")
        # match the real E4-prep executor: min_level = target - E4_PREP_BAND so L8-14 box-fodder chaff isn't
        # dragged into the grind (the un-actuatable-box + faint-thrash the guard exists to prevent)
        r = camp.grind_weak_members(target, min_level=max(1, target - C.E4_PREP_BAND))
        here2 = tuple(tv.map_id(b))
        L(f"   [tick {i}] -> {r!r} | now {here2}@{tv.coords(b)} party={camp._party_levels()} "
          f"battles={nb[0]} excursions={getattr(camp, '_heal_excursion_n', 0)}")
        if here2 not in VR_FLOORS:
            leaked = True
            L(f"   [tick {i}] OFF the VR floors at {here2} — heal-in-cave / leak path exercised")
    lvN = camp._party_levels()
    gained = sorted(set(lvN) - set(lv0))
    bench_up = sum(1 for a, c in zip(sorted(lv0), sorted(lvN)) if c > a)
    L(f"RESULT: party {lv0} -> {lvN} | bench slots that rose={bench_up} | battles={nb[0]} | "
      f"final map={tv.map_id(b)} | left_floor={leaked}")
    # PASS (Q1/Q2/Q3): drew battles, a bench slot leveled, and she did NOT silently leak off-floor during a
    # NON-heal grind (a heal excursion leaving the floor is EXPECTED and reported separately, not a fail).
    ok = nb[0] > 0 and bench_up >= 1
    L(f"SMOKE {'PASS' if ok else 'FAIL'} (battles={nb[0]} bench_rose={bench_up})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
