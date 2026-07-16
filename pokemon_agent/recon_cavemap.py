"""recon_cavemap.py - OBSERVED-landing mapper for Mt Moon (VISIBLE). Real spawns, not predicted.

The ROM's warpId->spawn isn't destMap.warp[warpId] (observed: (5,6) & (19,14) both spawn (25,4)).
So OBSERVE: enter each reachable warp once, record the REAL spawn; from each spawn take the FULL
reachable-warp set (not nearest - that bounced); backtrack via SAVESTATE jumps when a spawn's
frontier is exhausted (no wandering). Build the true observed graph, find the route to the FORWARD
region ((3,22) where the east edge / Cerulean is reachable), save it to mtmoon_plan.json, and print
the observed door->spawn map in plain terms for Jonny to confirm.

WATCH=1 (default): visible window so Jonny watches the doors get mapped. INVINCIBLE=1: nav-isolation.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_cavemap.py
"""
import os
import sys
import json
import ctypes

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

WATCH = os.getenv("WATCH", "1") == "1"
INVINCIBLE = os.getenv("INVINCIBLE", "1") == "1"
if not WATCH:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import firered_ram as ram             # noqa: E402
import travel as tv                   # noqa: E402
import pokemon_state as st            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign         # noqa: E402
from cave_nav import CaveNav          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
GH = 0x02036DFC
CAVE_GROUP = 1
MAX_STEPS = int(os.getenv("MAX_STEPS", "60"))


def warp_xys(b):
    ev = b.rd32(GH + 0x04); n = b.rd8(ev + 0x01); base = b.rd32(ev + 0x08)
    out = []
    for i in range(min(n, 32)):
        e = base + i * 8
        out.append((ctypes.c_int16(b.rd16(e)).value, ctypes.c_int16(b.rd16(e + 2)).value))
    return out


def east_reachable(b):
    g = tv.Grid(b)
    return bool(tv.bfs(g, tv.coords(b), (lambda t: t[0] == g.sx_hi), walkable=g.walkable))


def main():
    b = Bridge(ROM)
    with open(os.path.join(STATES, "mtmoon_interior.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    w16 = b.core.memory.u16.raw_write

    def heal():
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            w16(base + 0x56, b.rd16(base + 0x58))
    heal()

    render = (lambda: None)
    if WATCH:
        import pygame
        pygame.init()
        win = (b.width * 3, b.height * 3)
        screen = pygame.display.set_mode(win)
        pygame.display.set_caption("Kira - Mt Moon door-mapping (watch)")

        def render():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            if INVINCIBLE and st.in_battle(b):
                w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))
            surf = pygame.image.fromstring(b.frame_rgb().tobytes(), (b.width, b.height), "RGB")
            screen.blit(pygame.transform.scale(surf, win), (0, 0))
            pygame.display.flip()
    else:
        def render():
            if INVINCIBLE and st.in_battle(b):
                w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))

    def fr():
        BattleAgent(b, on_event=lambda *a, **k: None, render=render, log=lambda m: None).run(90)
        heal(); return "win"
    camp = Campaign(b, battle_runner=fr, on_event=lambda *a, **k: None, beat=lambda *a, **k: None,
                    render=render)
    camp._suppress_heal = True
    nav = CaveNav(b, camp, fog_path=None, on_event=lambda *a, **k: None, render=render, log=lambda m: None)

    def node():
        return (tv.map_id(b), tv.coords(b))
    observed = {}      # (map, warp_xy) -> (destMap, spawn_xy)
    parent = {}        # spawn_node -> (prev_node, warp_xy)
    saved = {}         # spawn_node -> state bytes
    reach_of = {}      # spawn_node -> full reachable warp set
    visited = set()
    start = node()
    saved[start] = bytes(b.save_state()); parent[start] = None
    cur = start
    forward = None
    log = []
    for step in range(MAX_STEPS):
        if cur not in visited:
            visited.add(cur)
            reach_of[cur] = [w for w in warp_xys(b) if nav._reachable(w)]
            log.append(f"spawn {cur[0]}@{cur[1]}: reachable warps {reach_of[cur]}")
            if cur[0][0] != CAVE_GROUP and east_reachable(b):
                forward = cur; log.append(f"  >>> FORWARD region (east reachable) at {cur} <<<"); break
        frontier = [w for w in reach_of[cur] if (cur[0], w) not in observed]
        if frontier:
            w = min(frontier, key=lambda t: abs(t[0] - cur[1][0]) + abs(t[1] - cur[1][1]))
            nm = nav._enter(w)
            if nm in (None, "BLACKOUT"):
                observed[(cur[0], w)] = ("FAIL", None); log.append(f"  warp {w}: enter {nm}")
                continue
            sp = node()
            observed[(cur[0], w)] = (sp[0], sp[1])
            log.append(f"  door {cur[0]}@{w} -> spawn {sp[0]}@{sp[1]}")
            if sp not in parent:
                parent[sp] = (cur, w)
            saved[sp] = bytes(b.save_state())
            cur = sp
        else:
            # backtrack: jump (load_state) to any saved spawn that still has an unobserved warp
            bt = next((nd for nd in saved if any((nd[0], w) not in observed
                       for w in reach_of.get(nd, []))), None)
            if bt is None:
                log.append("  all reachable doors mapped; no forward region found"); break
            b.load_state(saved[bt])
            for _ in range(6):
                b.run_frame()
            cur = bt
    # report
    print("\n".join(f"   [map] {ln}" for ln in log), flush=True)
    if not forward:
        print("   [map] !! NO forward region in the observed graph - precise wall", flush=True)
        return
    # reconstruct route
    seq = []
    n = forward
    while parent.get(n):
        pn, w = parent[n]
        seq.append((list(pn[0]), list(w)))
        n = pn
    seq = seq[::-1]
    json.dump(seq, open(os.path.join(STATES, "mtmoon_plan.json"), "w"))
    print(f"   [map] *** OBSERVED ROUTE ({len(seq)} doors) saved to mtmoon_plan.json ***", flush=True)
    for m, w in seq:
        print(f"   [map]   on {tuple(m)} take door {tuple(w)}", flush=True)


if __name__ == "__main__":
    main()
