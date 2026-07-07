"""recon_silph_probe4.py — programmatic 5F route truth: the exact BFS path to the Card Key,
which NPC template tiles sit ON it, and whether a path survives masking them.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_silph_probe4.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SILPH = [(1, 47 + i) for i in range(11)]
BALL_FRONTS = [(21, 21), (23, 21)]


def main():
    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def fight():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=240)

    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    camp.enter_warp(pick=(33, 30))
    for _ in range(80):
        b.run_frame()
    for target in SILPH[1:5]:
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == target]
        camp.enter_warp(pick=cands[0])
        if st.in_battle(b):
            fight()
        for _ in range(60):
            b.run_frame()
    if tuple(tv.map_id(b)) != SILPH[4]:
        print(f"!! not on 5F (at {tv.map_id(b)})", flush=True)
        return 1

    cur = tuple(tv.coords(b) or (0, 0))
    g = tv.Grid(b)
    wts = {tuple(w[0]) for w in tv.read_warps(b)}
    npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
    print(f"cur={cur}", flush=True)

    def route(mask, tgt):
        return tv.bfs(g, cur, lambda t: t == tgt,
                      walkable=lambda sx, sy: g.walkable(sx, sy) and (sx, sy) not in mask)

    for tgt in BALL_FRONTS:
        p = route(wts, tgt)
        if not p:
            print(f"{tgt}: NO PATH even warp-masked-only", flush=True)
            continue
        on_path = [t for t in p if tuple(t) in npcs]
        print(f"{tgt}: warp-masked len={len(p)}; NPC tiles ON path: {on_path}", flush=True)
        p2 = route(wts | npcs, tgt)
        print(f"{tgt}: all-templates-masked -> "
              f"{'len ' + str(len(p2)) if p2 else 'NO PATH'}", flush=True)
        if not p2 and on_path:
            p3 = route(wts | set(map(tuple, on_path)), tgt)
            print(f"{tgt}: only-on-path-npcs-masked -> "
                  f"{'len ' + str(len(p3)) if p3 else 'NO PATH'}"
                  f"{' via ' + str([t for t in (p3 or []) if tuple(t) in npcs]) if p3 else ''}",
                  flush=True)
    # which template tiles sit in the SOUTH HALL region (y>=14) at all — the ball pocket
    print(f"templates in south hall: {[t for t in npcs if t[1] >= 14]}", flush=True)
    # reachable set with warps+templates masked: does ANY tile of row 21 get reached?
    seen = {cur}
    from collections import deque
    q = deque([cur])
    mask = wts | npcs
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (cx + dx, cy + dy)
            if nxt in seen or nxt in mask or not g.walkable(*nxt):
                continue
            if not (g.sx_lo <= nxt[0] <= g.sx_hi and g.sy_lo <= nxt[1] <= g.sy_hi):
                continue
            seen.add(nxt)
            q.append(nxt)
    r21 = sorted(t for t in seen if t[1] == 21)
    print(f"masked-reachable row-21 tiles: {r21}", flush=True)
    r20 = sorted(t for t in seen if t[1] == 20)
    print(f"masked-reachable row-20 tiles: {r20}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
