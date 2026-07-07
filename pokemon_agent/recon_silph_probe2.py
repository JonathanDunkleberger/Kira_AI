"""recon_silph_probe2.py — 5F ground truth: where is the Card Key ball REALLY, and what
seals it off? (strike5/6: BFS can't reach any neighbor of the billed (22,21).)
Climb 1F->5F, dump object templates (item balls = graphics id 0x5B usually; just dump all),
the ball tile's neighbors' walkability, BFS reachability, and a frame snap.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_silph_probe2.py
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
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
DBG = os.path.join(SCRATCH, "silph_probe")
SILPH = [(1, 47 + i) for i in range(11)]


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
                    on_event=lambda s, **k: print(f"[event] {s}", flush=True),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    os.makedirs(DBG, exist_ok=True)

    camp.enter_warp(pick=(33, 30))
    for _ in range(80):
        b.run_frame()
    for target in SILPH[1:5]:
        here = tuple(tv.map_id(b))
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == target]
        if not cands:
            print(f"!! no warp {here} -> {target}", flush=True)
            return 1
        camp.enter_warp(pick=cands[0])
        if st.in_battle(b):
            fight()
        for _ in range(60):
            b.run_frame()
        print(f"floor {here} -> {tuple(tv.map_id(b))} @ {tv.coords(b)}", flush=True)
    if tuple(tv.map_id(b)) != SILPH[4]:
        print(f"!! not on 5F (at {tv.map_id(b)}) — abort", flush=True)
        return 1

    print(f"5F objects [(x,y), gfx, present]: {tv.read_object_templates(b)}", flush=True)
    print(f"5F warps: {[(tuple(w[0]), tuple(w[1])) for w in tv.read_warps(b)]}", flush=True)
    cur = tuple(tv.coords(b) or (0, 0))
    wts = {tuple(w[0]) for w in tv.read_warps(b)}
    g = tv.Grid(b)

    def wk(t):
        return g.walkable(t[0] + tv.MAP_OFFSET, t[1] + tv.MAP_OFFSET)

    for ball in ((22, 21), (21, 21), (23, 21)):
        print(f"tile {ball}: walkable={wk(ball)} beh={camp._tile_behavior(*ball):#04x}",
              flush=True)
    for nb in ((21, 21), (23, 21), (22, 20), (22, 22)):
        p = tv.bfs(g, cur, lambda t, a=nb: t == a,
                   walkable=lambda sx, sy: g.walkable(sx, sy)
                   and (sx - tv.MAP_OFFSET, sy - tv.MAP_OFFSET) not in wts)
        print(f"bfs {cur} -> {nb}: {'len ' + str(len(p)) if p else 'NO PATH'}", flush=True)
    try:
        b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, "probe2_5f.png"))
        print(f"snap -> {os.path.join(DBG, 'probe2_5f.png')}", flush=True)
    except Exception as e:
        print(f"snap failed: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
