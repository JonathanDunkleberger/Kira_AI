"""recon_choke_verdict.py — final verdict on the Cerulean south chokepoint.

Confirms: (1) the gap object (26,32) gfx == GFX_CUT_TREE (95); (2) the two flankers' gfx; (3) whether
the player HAS HM Cut + the Cascade badge (can she clear it at all?); (4) live Cerulean warps + whether
ANY south-edge tile or warp is reachable WITHOUT passing the tree (is there an alternate route, i.e. did
she route into a pocket, or is the tree the sole gate?). Read-only.
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

from bridge import Bridge          # noqa: E402
import travel as tv                # noqa: E402
from travel import Grid, bfs       # noqa: E402
import firered_ram as ram          # noqa: E402
import field_moves as fm           # noqa: E402
import pokemon_state as st         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OFF = tv.MAP_OFFSET
OB, SZ = 0x02036E38, 0x24
TREE = (26, 32)


def main():
    state = os.path.join(_HERE, "states", "campaign", "kira_campaign.state")
    b = Bridge(ROM)
    b.load_state(open(state, "rb").read())
    for _ in range(90):
        b.run_frame()
    cur = tv.coords(b)
    print(f"==== map={tv.map_id(b)} player={cur} (live campaign save) ====", flush=True)

    # (1)(2) gfx of the three choke objects
    print("\n-- choke objects --", flush=True)
    for i in range(1, 16):
        o = OB + i * SZ
        if not (b.rd8(o) & 1):
            continue
        c = (b.rds16(o + 0x10) - OFF, b.rds16(o + 0x12) - OFF)
        if c in ((26, 31), (26, 32), (27, 31)):
            g = b.rd8(o + 0x05)
            kind = {95: "CUT_TREE", 97: "BOULDER", 92: "ITEM_BALL"}.get(g, "NPC/other")
            print(f"   {c}: gfx={g} (0x{g:02X}) -> {kind}", flush=True)

    # (3) does she have Cut + Cascade badge?
    print("\n-- HM Cut capability --", flush=True)
    try:
        moves = []
        for s in range(ram.read_party_count(b)):
            moves.append(st.read_party_moves(b, s))
        has_cut = any(fm.MOVE_CUT in mv for mv in moves)
    except Exception as e:
        has_cut = f"read-failed: {e!r}"
        moves = []
    cascade = fm.read_flag(b, fm.FLAG_BADGE[2])     # Cut gate = Cascade badge (#2)
    print(f"   party moves: {moves}", flush=True)
    print(f"   knows CUT(15)? {has_cut}   Cascade badge(flag 0x821)? {cascade}", flush=True)
    try:
        print(f"   field_moves.can_use_hm('cut') = {fm.can_use_hm(b, 'cut')}", flush=True)
    except Exception as e:
        print(f"   can_use_hm err: {e!r}", flush=True)

    # (4) live warps + reachability without the tree
    print("\n-- Cerulean live warps --", flush=True)
    warps = tv.read_warps(b)
    for w in warps:
        print(f"   warp tile {w[0]} -> dest map {w[1]} (id {w[2]})", flush=True)

    grid = Grid(b)
    # flood from player, NOT crossing the tree tile
    seen = {cur}; stack = [cur]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n not in seen and n != TREE and 0 <= n[0] <= grid.sx_hi and 0 <= n[1] <= grid.sy_hi \
                    and grid.walkable(*n):
                seen.add(n); stack.append(n)
    south_cols = sorted(x for x in range(grid.sx_hi + 1) if (x, grid.sy_hi) in seen)
    reachable_warps = [w for w in warps if w[0] in seen]
    print(f"\n   reachable area size (no tree) = {len(seen)} tiles", flush=True)
    print(f"   south-edge cols reachable WITHOUT the tree: {south_cols}", flush=True)
    print(f"   warps reachable WITHOUT the tree: {[w[0] for w in reachable_warps]}", flush=True)
    # and WITH the tree removed (simulate Cut)
    seen2 = {cur}; stack = [cur]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n not in seen2 and 0 <= n[0] <= grid.sx_hi and 0 <= n[1] <= grid.sy_hi \
                    and grid.walkable(*n):
                seen2.add(n); stack.append(n)
    south_cols2 = sorted(x for x in range(grid.sx_hi + 1) if (x, grid.sy_hi) in seen2)
    print(f"   south-edge cols reachable IF tree cut: {south_cols2}", flush=True)
    print(f"\n   VERDICT: tree is {'THE SOLE south gate' if not south_cols and south_cols2 else 'NOT the sole gate'} "
          f"(no-tree south={south_cols}, cut-tree south={south_cols2})", flush=True)


if __name__ == "__main__":
    main()
