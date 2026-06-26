"""recon_dfs.py - autonomous zone-aware route finder for Mt Moon (entrance -> forward exit).

The cave splits each interior map into disconnected ZONES; the route weaves (1,1)<->(1,2)<->(1,3)
through specific zones. Jonny proved a route EXISTS. This does a savestate-backtracking DFS over
(map, landing-coord) zone-states from the entrance, taking only _reachable warps, until it reaches a
(1,2) landing from which the forward exit (45,4)->(3,22) is reachable -- then takes it and confirms
emergence on map (3,22). Prints the resolved executable warp chain (to be hardcoded for the visible run).

Bounds: depth<=DEPTH, visited zone-states, per-_enter travel budget. Skips the backward Route-3 exit
(1,1)@(18,37). No claims beyond what each _enter/_reachable actually returns.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_dfs.py
"""
import os
import sys
import ctypes

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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
DEPTH = 10
CAVES = {(1, 1), (1, 2), (1, 3)}
BACKWARD = {((1, 1), (18, 37))}        # Route-3 entrance side; don't exit backward


def warps(b):
    ev = b.rd32(GH + 0x04); n = b.rd8(ev + 0x01); base = b.rd32(ev + 0x08)
    out = []
    for i in range(n):
        e = base + i * 8
        x = ctypes.c_int16(b.rd16(e)).value; y = ctypes.c_int16(b.rd16(e + 2)).value
        out.append(((x, y), (b.rd8(e + 7), b.rd8(e + 6))))
    return out


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
            w16(ram.GPLAYER_PARTY + s * 100 + 0x56, b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58))

    def render():
        if st.in_battle(b):
            w16(ram.GBATTLE_MONS + 0x28, b.rd16(ram.GBATTLE_MONS + 0x2C))

    def fr():
        BattleAgent(b, on_event=lambda *a, **k: None, render=render, log=lambda m: None).run(90); heal(); return "win"
    camp = Campaign(b, battle_runner=fr, on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=render)
    camp._suppress_heal = True
    nav = CaveNav(b, camp, fog_path=None, on_event=lambda *a, **k: None, render=render, log=lambda m: None)
    heal()

    visited = set()
    found = [None]

    def dfs(depth, trail):
        m = tv.map_id(b); c = tv.coords(b)
        key = (m, c)
        indent = "  " * (DEPTH - depth)
        # GOAL: on (1,2), ANY reachable warp whose dest LEAVES the cave group is a forward-exit
        # candidate (no hardcoded tile guess: the (45,4)/(39,4) human reads disagree). Try each;
        # FOUND when we actually emerge on a non-cave map.
        if m == (1, 2):
            snap0 = bytes(b.save_state())
            for wxy, dest in warps(b):
                if (m, wxy) in BACKWARD or dest in CAVES:
                    continue
                b.load_state(snap0); [b.run_frame() for _ in range(5)]; heal()
                try:
                    if not nav._reachable(wxy):
                        continue
                except Exception:
                    continue
                nav._enter(wxy)
                em = tv.map_id(b)
                print(f"   [dfs] {indent}GOAL-TRY: (1,2)@{c} --{wxy}(dest {dest})--> {em}@{tv.coords(b)}", flush=True)
                if em not in CAVES:
                    found[0] = trail + [(m, c, wxy, em, tv.coords(b))]
                    return True
            b.load_state(snap0); [b.run_frame() for _ in range(4)]
        if key in visited or depth <= 0:
            return False
        visited.add(key)
        snap = bytes(b.save_state())
        for wxy, dest in warps(b):
            if (m, wxy) in BACKWARD or dest not in CAVES:
                continue
            b.load_state(snap); [b.run_frame() for _ in range(5)]; heal()
            try:
                if not nav._reachable(wxy):
                    continue
            except Exception:
                continue
            res = nav._enter(wxy)
            nm = tv.map_id(b); nc = tv.coords(b)
            if res is None or res == "BLACKOUT" or nm == m:
                continue                      # no real floor change = enter failed (warp-avoid reliable)
            if (nm, nc) in visited:
                continue
            print(f"   [dfs] {indent}{m}@{c} --{wxy}--> {nm}@{nc}", flush=True)
            if dfs(depth - 1, trail + [(m, c, wxy, nm, nc)]):
                return True
            b.load_state(snap); [b.run_frame() for _ in range(4)]
        return False

    print(f"   [dfs] start {tv.map_id(b)}@{tv.coords(b)} depth={DEPTH}", flush=True)
    ok = dfs(DEPTH, [])
    if ok:
        print("   [dfs] ===== RESOLVED EXECUTABLE CHAIN (entrance -> forward exit) =====", flush=True)
        for hop in found[0]:
            print(f"   [dfs]   {hop[0]}@{hop[1]} --{hop[2]}--> {hop[3]}@{hop[4]}", flush=True)
    else:
        print("   [dfs] NO ROUTE FOUND within bounds. Zone-states explored:", flush=True)
        for v in sorted(visited):
            print(f"   [dfs]   visited {v}", flush=True)


if __name__ == "__main__":
    main()
