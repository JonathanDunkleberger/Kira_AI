"""recon_martdoor.py — diagnose why travel can't enter the Cerulean Mart door (30,11).
Exits to overworld, dumps ALL object events (x,y,gfx,facing) + the walkable/NPC map around the
door, probes travel to each tile adjacent to the door. Read-only (no canonical writes)."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge                                          # noqa: E402
import travel as tv                                                # noqa: E402
from campaign import Campaign, resolve_state                       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
DOOR = (30, 11)
OB, SZ = 0x02036E38, 0x24
OFF_ACTIVE, OFF_GFX, OFF_X, OFF_Y, OFF_FACING = 0x00, 0x05, 0x10, 0x12, 0x18


def dump_objects(b):
    out = []
    for i in range(1, 16):
        o = OB + i * SZ
        if not (b.rd8(o + OFF_ACTIVE) & 1):
            continue
        x = b.rds16(o + OFF_X) - 7
        y = b.rds16(o + OFF_Y) - 7
        out.append((i, b.rd8(o + OFF_GFX), x, y, b.rd8(o + OFF_FACING) & 0x0F))
    return out


def main():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._exit_to_overworld()
    for _ in range(30):
        b.run_frame()
    print(f"overworld map={tv.map_id(b)} player={tv.coords(b)}", flush=True)

    print("\n=== object events (idx gfx x y facing) ===")
    objs = dump_objects(b)
    for i, g, x, y, f in objs:
        near = "  <-- NEAR DOOR" if abs(x - DOOR[0]) <= 3 and abs(y - DOOR[1]) <= 3 else ""
        print(f"  obj{i:2} gfx={g:3} ({x},{y}) facing={f}{near}")
    occ = {(x, y) for _, _, x, y, _ in objs}

    print(f"\n=== tiles around door {DOOR} (W=walkable C=collide N=npc P=player D=door) ===")
    grid = tv.Grid(b)
    px, py = tv.coords(b)
    for yy in range(DOOR[1] - 1, DOOR[1] + 4):
        row = []
        for xx in range(DOOR[0] - 4, DOOR[0] + 5):
            t = (xx, yy)
            if t == DOOR:
                c = "D"
            elif t == (px, py):
                c = "P"
            elif t in occ:
                c = "N"
            else:
                c = "W" if grid.walkable(xx, yy) else "C"
            row.append(c)
        print(f"  y={yy:2}: " + " ".join(row) + f"   (x={DOOR[0]-4}..{DOOR[0]+4})")

    print("\n=== travel probes to door-adjacent tiles ===")
    snap = b.save_state()
    for appr in [(DOOR[0], DOOR[1] + 1), (DOOR[0], DOOR[1] - 1),
                 (DOOR[0] - 1, DOOR[1]), (DOOR[0] + 1, DOOR[1])]:
        b.load_state(snap)
        for _ in range(8):
            b.run_frame()
        reach = bool(tv.bfs(grid, tv.coords(b), lambda t: t == appr, walkable=grid.walkable))
        r = camp.trav.travel(target_map=None, arrive_coord=appr, max_steps=200, max_seconds=60)
        print(f"  approach {appr}: bfs_reachable={reach} travel={r} final={tv.coords(b)}")


if __name__ == "__main__":
    main()
