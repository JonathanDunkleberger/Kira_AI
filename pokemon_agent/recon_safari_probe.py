"""recon_safari_probe.py — ground truth at the (33-35,17) wedge: grid collision/elevation/
behavior + LIVE object-event positions vs template positions, from the stage_safari save
(she saved at Center arrival, same map as the wedge). Read-only.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_safari_probe.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STAGE = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "stage_safari")

OBJ_EVENTS = 0x02036E38              # gObjectEvents (matches field_moves/campaign _OB)
OBJ_SZ = 0x24
MAP_LAYOUT = 0x02036DFC              # gMapHeader base (travel.GMAPHEADER)


def main():
    b = Bridge(ROM)
    with open(os.path.join(STAGE, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()

    print(f"map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)
    g = tv.Grid(b)

    # grid dump around the wedge: x 28..40, y 13..21 (save coords)
    ml = b.rd32(tv.GMAPHEADER)
    mp = b.rd32(ml + 0x0C)
    w = b.rd32(ml + 0x00)
    print(f"grid {g.w}x{g.h} (sx_hi={g.sx_hi}, sy_hi={g.sy_hi})")
    print("     " + " ".join(f"{x:>4}" for x in range(5, 45)))
    for sy in range(14, 34):
        row = []
        for sx in range(5, 45):
            bx, by = sx + tv.MAP_OFFSET, sy + tv.MAP_OFFSET
            v = b.rd16(mp + (by * w + bx) * 2)
            col = (v >> 10) & 3
            elev = (v >> 12) & 0xF
            mark = "G" if (bx, by) in g.grass else ("W" if (bx, by) in g.water else " ")
            row.append(f"c{col}e{elev}{mark}")
        print(f"y={sy:>3} " + " ".join(row), flush=True)

    print("\nLIVE gObjectEvents (slot: active, gfx, current x,y):")
    for i in range(16):
        o = OBJ_EVENTS + i * OBJ_SZ
        flags = b.rd32(o)
        if not (flags & 1):
            continue
        gfx = b.rd8(o + 0x05)
        cx = b.rds16(o + 0x10) - tv.MAP_OFFSET
        cy = b.rds16(o + 0x12) - tv.MAP_OFFSET
        px = b.rds16(o + 0x0C) - tv.MAP_OFFSET
        py = b.rds16(o + 0x0E) - tv.MAP_OFFSET
        print(f"  slot {i:2}: gfx={gfx:3} current=({cx},{cy}) previous=({px},{py})",
              flush=True)

    print("\ntemplates (read_object_templates):")
    for xy, gfx, present in tv.read_object_templates(b):
        print(f"  tmpl @ {tuple(xy)} gfx={gfx} present={present}", flush=True)

    print("\nwarps (read_warps):")
    for xy, d, wid in tv.read_warps(b):
        print(f"  warp @ {tuple(xy)} -> {tuple(d)} (id {wid})", flush=True)

    # FLOOD from her tile with the strike's walkable; classify the frontier
    cur = tuple(tv.coords(b))
    wts = {tuple(w[0]) for w in tv.read_warps(b)}
    npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}

    def ok(sx, sy):
        return g.walkable(sx, sy) and (sx, sy) not in wts and (sx, sy) not in npcs

    seen = {cur}
    q = [cur]
    while q:
        x, y = q.pop()
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb not in seen and ok(*nb):
                seen.add(nb)
                q.append(nb)
    xs = [t[0] for t in seen]
    ys = [t[1] for t in seen]
    print(f"\nflood from {cur}: {len(seen)} tiles, x {min(xs)}..{max(xs)}, "
          f"y {min(ys)}..{max(ys)}")
    frontier = {}
    for (x, y) in seen:
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb in seen:
                continue
            bx, by = nb[0] + tv.MAP_OFFSET, nb[1] + tv.MAP_OFFSET
            why = ("warp" if nb in wts else
                   "npc" if nb in npcs else
                   "water" if (bx, by) in g.water else
                   "ledge" if (bx, by) in g.ledge else
                   "impass" if (bx, by) in g.impass else
                   f"col{g.col.get((bx, by), '?')}")
            frontier.setdefault(why, []).append(nb)
    for why, tiles in sorted(frontier.items()):
        show = sorted(set(tiles))
        print(f"  frontier[{why}] x{len(show)}: {show[:20]}{'...' if len(show) > 20 else ''}",
              flush=True)

    # the EXACT strike BFS (tv.bfs with edge_open/ledges/bound)
    p = tv.bfs(g, cur, lambda t: t == (9, 18),
               walkable=lambda sx, sy: g.walkable(sx, sy)
               and (sx, sy) not in wts and (sx, sy) not in npcs)
    print(f"\ntv.bfs (26,30)->(9,18): {'len ' + str(len(p)) if p else None}")
    # if None: flood via tv.bfs mechanics to find ITS true reach
    reach = {cur}
    q = [cur]
    while q:
        x, y = q.pop()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if g.ledge_dir(x + dx, y + dy) == (dx, dy):
                nx, ny = x + 2 * dx, y + 2 * dy
            else:
                nx, ny = x + dx, y + dy
                if not g.edge_open(x, y, dx, dy):
                    continue
            if not (g.sx_lo <= nx <= g.sx_hi and g.sy_lo <= ny <= g.sy_hi):
                continue
            nb = (nx, ny)
            if nb in reach or not ok(nx, ny):
                continue
            reach.add(nb)
            q.append(nb)
    xs2 = [t[0] for t in reach]
    ys2 = [t[1] for t in reach]
    print(f"tv.bfs-mechanics flood: {len(reach)} tiles, x {min(xs2)}..{max(xs2)}, "
          f"y {min(ys2)}..{max(ys2)}")
    only_plain = sorted(seen - reach)
    print(f"tiles plain-flood reaches but bfs-mechanics can't ({len(only_plain)}): "
          f"{only_plain[:30]}{'...' if len(only_plain) > 30 else ''}")
    # classify which impass behaviors fence the bfs reach
    fence = {}
    for (x, y) in reach:
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (x + dx, y + dy)
            if nb in reach or not ok(*nb):
                continue
            if not g.edge_open(x, y, dx, dy):
                a = g.impass.get((x + tv.MAP_OFFSET, y + tv.MAP_OFFSET))
                d = g.impass.get((nb[0] + tv.MAP_OFFSET, nb[1] + tv.MAP_OFFSET))
                fence.setdefault((a, d, (dx, dy)), []).append((x, y))
    for k, tiles in sorted(fence.items(), key=lambda kv: str(kv[0])):
        print(f"  edge_open fence src_bh={k[0]} dst_bh={k[1]} dir={k[2]}: "
              f"{sorted(set(tiles))[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
