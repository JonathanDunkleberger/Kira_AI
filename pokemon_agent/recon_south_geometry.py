"""recon_south_geometry.py — post-ticket Cerulean south-exit ground truth (read-only).

Boot the canonical save, walk her to Cerulean (3,3) via the harness bridge? NO — simpler: load the
banked stall state that's already ON Cerulean if present, else load canonical and just read the
CERULEAN map data by teleporting is not possible read-only... so: load canonical (Bill's cottage),
then this script only answers if she's on (3,3). Instead we load `stall_cerulean_L29b.state` (a
staged Cerulean state from the strike) — fallback: canonical.

ANSWERS: from the player's position on (3,3), post-ticket:
  - which SOUTH-border tiles (y = max) are BFS-reachable
  - which door tiles are reachable + where each warp leads (read_warps)
  - where the cut tree / field objects sit
  - the object events on the south fence line
RUN: python pokemon_agent/recon_south_geometry.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import field_moves as fm               # noqa: E402
from campaign import Campaign, resolve_state  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CAND = [os.path.join(_HERE, "states", "campaign", "stall_cerulean_L29b.state"),
        os.path.join(_HERE, "states", "campaign", "r24_for_bill.state"),
        os.path.join(_HERE, "states", "campaign", "kira_campaign.state")]


def main():
    b = Bridge(ROM)
    picked = None
    for p in CAND:
        if os.path.exists(p):
            with open(p, "rb") as f:
                b.load_state(f.read())
            for _ in range(30):
                b.run_frame()
            picked = p
            if tuple(tv.map_id(b)) == (3, 3):
                break
    print(f"state: {os.path.basename(picked)} map={tv.map_id(b)} pos={tv.coords(b)}")
    print(f"ticket flag: {fm.read_flag(b, 0x234)}")
    if tuple(tv.map_id(b)) != (3, 3):
        print("!! not on Cerulean — geometry read needs an on-map state; aborting")
        return
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    grid = tv.Grid(b)
    cur = tv.coords(b)
    # full reachable set
    seen = {tuple(cur)}
    frontier = [tuple(cur)]
    while frontier:
        x, y = frontier.pop()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            t = (x + dx, y + dy)
            if t not in seen and grid.walkable(t[0] + 7, t[1] + 7):
                seen.add(t)
                frontier.append(t)
    ys = [y for _, y in seen]
    y_max = max(ys)
    south_reach = sorted([t for t in seen if t[1] >= y_max - 1])
    print(f"reachable tiles: {len(seen)}; southmost reachable rows y>={y_max-1}: {south_reach}")
    # doors + warps
    doors = camp._door_tiles()
    warps = tv.read_warps(b)
    wmap = {tuple(w[0]): w[1] for w in warps}
    print(f"\ndoors on map: {len(doors)}; warp events: {len(warps)}")
    for d in doors:
        appr = (d[0], d[1] + 1)
        reach = appr in seen or tuple(d) in seen
        print(f"  door {d} -> {wmap.get(tuple(d), '?')}  approach-reachable={reach}")
    # field objects + fence-line NPCs
    objs = fm.scan_field_objects(b)
    print(f"\nfield objects (tree95/boulder97/ball92): {objs}")


if __name__ == "__main__":
    main()
