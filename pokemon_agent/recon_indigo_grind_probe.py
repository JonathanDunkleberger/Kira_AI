"""recon_indigo_grind_probe.py — is there a CLEAN Indigo-anchored endgame grind arena?

The NS#17 finding: cave-grinding INSIDE Victory Road is blocked by heal-to-Viridian (from a VR floor the
only reachable Center is a cross-region abandon to Viridian). The candidate fix: grind at an INDIGO-adjacent
spot (Indigo (3,9) HAS a League Center) where a heal excursion is SHORT. This probe boots indigo_reach_g
(at Indigo, badge 8, underleveled bench) and answers, from the FEET:
  (1) descend Indigo -> Route 23: which HALF does she land on, is there GRASS reachable (a clean grass
      arena needs NO cave-grind), and/or is a VR floor reachable to dip into?
  (2) heal_nearest from there: does it route to INDIGO (short) or abandon cross-region to Viridian?

RUN: ../.venv/Scripts/python.exe -u recon_indigo_grind_probe.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
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
INDIGO, R23 = (3, 9), (3, 42)


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(WS, SRC + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def fight():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=180)

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

    L(f"boot {SRC} map={tv.map_id(b)}@{tv.coords(b)} party={camp._party_levels()}")

    # (1) descend Indigo -> Route 23
    if tuple(tv.map_id(b)) == INDIGO:
        L("descending Indigo -> Route 23 (south)")
        camp.walk_to_map(R23, "south")
    here = tuple(tv.map_id(b))
    coords = tv.coords(b)
    grid = tv.Grid(b)
    grass = [(x - tv.MAP_OFFSET, y - tv.MAP_OFFSET) for (x, y) in grid.grass]
    reach_grass = [g for g in grass if tv.bfs(grid, coords, lambda q, gg=g: q == gg, walkable=grid.walkable)]
    warps = []
    try:
        warps = [(tuple(xy), tuple(d)) for xy, d, _w in tv.read_warps(b)]
    except Exception:
        pass
    L(f"AFTER DESCENT: map={here}@{coords} | grass tiles on map={len(grass)} reachable_from_feet={len(reach_grass)}")
    L(f"   reachable grass (first 8): {reach_grass[:8]}")
    L(f"   warps here: {warps}")
    cy = (coords or (0, 0))[1]
    L(f"   R23 half: cy={cy} ({'NORTH (Indigo side)' if cy <= 30 else 'SOUTH (VR side)'})")

    import firered_ram as ram
    import pokemon_state as pst

    def ding_ace(frac=0.15):
        levels = camp._party_levels()
        ace = max(range(len(levels)), key=lambda s: levels[s])
        base = ram.GPLAYER_PARTY + ace * pst.PARTY_MON_SIZE
        mx = b.rd16(base + 0x58)
        b.core.memory.u16.raw_write(base + 0x56, max(1, int(mx * frac)))
        return camp._ace_hp_frac()

    # (2) heal_nearest from R23-north with a DINGED ace — does it reach Indigo or abandon to Viridian?
    L(f"DING ace -> {ding_ace():.0%}; heal_nearest() from R23-north@{tv.coords(b)}")
    r = camp.heal_nearest()
    m_after = tuple(tv.map_id(b))
    L(f"HEAL(R23-north) RESULT: {r!r} | now {m_after}@{tv.coords(b)} ace_hp={camp._ace_hp_frac():.0%} "
      f"| {'INDIGO-clean' if m_after == INDIGO else 'VIRIDIAN-abandon' if m_after == (3, 1) else 'here=' + str(m_after)}")

    # (3) descend R23-north -> VR2F via the (18,28) warp, then ding + heal from VR2F
    if tuple(tv.map_id(b)) != R23:
        L("returning to R23 to test the VR2F dip")
        camp.walk_to_map(R23, "south")
    L("descending to VR2F: enter_warp(pick=(18,28)) on R23")
    try:
        camp.enter_warp(pick=(18, 28))
    except Exception as e:
        L(f"   descend errored: {e!r}")
    L(f"AFTER VR2F DIP: map={tv.map_id(b)}@{tv.coords(b)} is_cave={camp._grind_is_cave(list(tv.map_id(b)))}")
    if tuple(tv.map_id(b)) == (1, 40):
        L(f"DING ace -> {ding_ace():.0%}; heal_nearest() from VR2F@{tv.coords(b)}")
        r2 = camp.heal_nearest()
        m2 = tuple(tv.map_id(b))
        L(f"HEAL(VR2F) RESULT: {r2!r} | now {m2}@{tv.coords(b)} ace_hp={camp._ace_hp_frac():.0%} "
          f"| {'INDIGO-clean' if m2 == INDIGO else 'VIRIDIAN-abandon' if m2 == (3, 1) else 'here=' + str(m2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
