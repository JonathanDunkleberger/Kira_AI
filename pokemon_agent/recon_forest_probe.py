"""recon_forest_probe.py — NIGHT TRAIN shift 1: is Viridian Forest's NORTH exit reachable?

The fresh spine loops on ADVANCE_NORTH because advance_north warps INTO Viridian Forest (1,0) from
the south gate but can't reach the north exit gate. DECISIVE question: from the Forest's south entry,
is the NORTH exit warp LAND-REACHABLE (=> a small targeted fix: travel(arrive_coord)+enter_warp(pick))
or maze-sealed (=> a real maze build)? Drive in via the real gates, dump the Forest's warps + which
are reachable + which leads back to Route 2 north (the forward exit).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                # noqa: E402
import travel as tv                      # noqa: E402
from battle_agent import BattleAgent     # noqa: E402
from campaign import Campaign            # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
statef = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_POSTGAME", "kira_campaign.state")

b = Bridge(ROM)
with open(statef, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()
b.set_input_owner("agent")

_dirs = ["DOWN", "RIGHT", "LEFT", "UP"]
_di = 0
for i in range(80):
    if tuple(tv.map_id(b))[0] == 3:
        break
    before = (tuple(tv.map_id(b)), tuple(tv.coords(b) or (-1, -1)))
    b.press(_dirs[_di % 4], 8, 10, owner="agent")
    for _ in range(24):
        b.run_frame()
    after = (tuple(tv.map_id(b)), tuple(tv.coords(b) or (-1, -1)))
    if before[0] != after[0]:
        _di = 0
        continue
    if before[1] == after[1]:
        _di += 1
print(f"outside: map={tv.map_id(b)} coords={tv.coords(b)}")


def _flee(*a, **k):
    try:
        return BattleAgent(b, log=lambda *_: None).flee()
    except Exception:
        return "fled"


camp = Campaign(b, battle_runner=_flee)
camp.render = lambda: None
trav = camp.trav

# Pallet -> Route1 -> Viridian -> Route2
for tgt, edge in (((3, 19), "north"), ((3, 1), "north"), ((3, 20), "north")):
    r = trav.travel(target_map=tgt, edge=edge, max_steps=700, max_seconds=180)
    print(f"  travel -> {tgt} edge={edge}: {r} (now {tv.map_id(b)} {tv.coords(b)})")
    if tuple(tv.map_id(b)) != tgt:
        print(f"!! stuck reaching {tgt}"); sys.exit(1)


def dump_warps(label):
    g = tv.Grid(b)
    c0 = tuple(tv.coords(b))
    print(f"\n== {label}: map={tv.map_id(b)} stand={c0} dims sx{g.sx_lo}..{g.sx_hi} sy{g.sy_lo}..{g.sy_hi} ==")
    for (wxy, wdest, wid) in sorted(tv.read_warps(b), key=lambda w: w[0][1]):
        wx, wy = wxy
        appr = (wx, wy + 1)
        reach = bool(tv.bfs(g, c0, lambda t, a=appr: t == a, walkable=g.walkable)) or \
            bool(tv.bfs(g, c0, lambda t, w=(wx, wy): t == w, walkable=g.walkable))
        print(f"    warp {wxy} -> {wdest}  reachable={reach}")


# enter the Forest via the reachable south gate (6,51)->(15,0), then (7,1)->(1,0)
print("\n-- entering Route 2 -> Forest gate (15,x) --")
r = camp.enter_warp(pick=(6, 51))
print(f"   enter_warp (6,51) -> {r}  now {tv.map_id(b)} {tv.coords(b)}")
if tuple(tv.map_id(b))[0] == 15:
    dump_warps("Forest GATE building")
    # the gate's north door -> Viridian Forest
    r = camp.enter_warp(prefer="north")
    print(f"   gate enter_warp north -> {r}  now {tv.map_id(b)} {tv.coords(b)}")

if tuple(tv.map_id(b)) == (1, 0):
    dump_warps("VIRIDIAN FOREST (1,0)")
    # THE decisive test: from here, is any warp leading back to Route 2 (3,20) reachable, and is it
    # the NORTH one? classify each Forest warp by dest + reachability + y.
    g = tv.Grid(b)
    c0 = tuple(tv.coords(b))
    print("\n== Forest warp analysis (dest, y, reachable, approach) ==")
    for (wxy, wdest, wid) in sorted(tv.read_warps(b), key=lambda w: w[0][1]):
        wx, wy = wxy
        appr = (wx, wy + 1)
        p = tv.bfs(g, c0, lambda t, a=appr: t == a, walkable=g.walkable)
        p2 = tv.bfs(g, c0, lambda t, w=(wx, wy): t == w, walkable=g.walkable)
        reach = bool(p or p2)
        print(f"    warp {wxy} y={wy} -> dest {wdest}  reachable={reach} "
              f"pathlen={len(p or p2) if reach else '-'}")
    # how far north can she flood from the entry?
    seen = {c0}; frontier = [c0]; min_y = c0[1]
    while frontier:
        nf = []
        for (x, y) in frontier:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                t = (x + dx, y + dy)
                if t not in seen and g.walkable(*t):
                    seen.add(t); nf.append(t); min_y = min(min_y, t[1])
        frontier = nf
    print(f"\nForest land flood from {c0}: {len(seen)} tiles, shallowest y={min_y} (sy_lo={g.sy_lo})")

    # DECISIVE FIX TEST: with the FLEE runner (fleeing wilds, no heal-bounce), can she traverse the
    # Forest to the north gate and warp out to (15,3) -> Route 2 north? This is exactly what a
    # heal-suppressed + flee-wilds advance_north Forest leg would do.
    print("\n== FIX TEST: enter_warp(prefer='north') from the Forest with the flee runner ==")
    r = camp.enter_warp(prefer="north")
    print(f"   Forest enter_warp north -> {r}  now map={tv.map_id(b)} coords={tv.coords(b)}")
    if tuple(tv.map_id(b)) == (15, 3):
        print("   >>> reached the NORTH gate (15,3)! traversal WORKS — Route 2 north -> Pewter next")
        r2 = camp.enter_warp(prefer="north")
        print(f"   north gate enter_warp north -> {r2}  now map={tv.map_id(b)} coords={tv.coords(b)}")
else:
    print(f"!! not in Viridian Forest (at {tv.map_id(b)}) — could not complete the drive")
