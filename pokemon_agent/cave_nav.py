"""cave_nav.py - GENERAL cave / multi-warp-map traversal (Mt Moon, Rock Tunnel, Victory Road).

Not turn-by-turn LLM nav (the 52-hour-stuck failure mode). Two facts make this tractable:
  1. The whole map's collision is already in RAM (travel.Grid reads the backup layout), so
     intra-floor pathfinding is solved by BFS - the model never steers tile-by-tile.
  2. Each warp's DESTINATION map is in the map's warp-events table - readable WITHOUT entering.
So we route over the FLOOR GRAPH by known destinations instead of blind tile exploration:

  - At each floor, read its real warps (events table) + their dest maps.
  - The warp nearest our spawn is the one we ARRIVED through (the back-door) - never re-enter it;
    its dest is a "back" map (toward the entrance), never an exit.
  - EXIT = a warp whose dest leaves the cave group AND is not a known back/entrance map (the far
    side, e.g. Mt Moon -> Route 4). Reachable exit -> take it, done.
  - Else descend/explore: take the nearest reachable warp to an UNVISITED cave floor (frontier).
  - ANTI-LOOP: visited-floor counts; if no frontier, take the least-visited cave neighbour; never
    step back toward the entrance. STUCK FALLBACK: nothing reachable -> LOUD abort.
  - FOG-OF-WAR JSON: visited floors + learned warp edges persist, so a resume never re-treads.
  - COMBAT GAUNTLET: the Traveler hands wild/trainer battles to the battle engine (real move-nav +
    forced faint-switch, both built) during every walk-to-warp, then pathfinding resumes.

Returns 'exited' / 'stuck' / 'timeout'.
"""
import ctypes
import json
import os
import time

import travel as tv
import pokemon_state as st

_STEP_ONTO = {"N": "UP", "S": "DOWN", "E": "RIGHT", "W": "LEFT"}
_GMAPHEADER = 0x02036DFC


class CaveNav:
    def __init__(self, bridge, campaign, fog_path=None, on_event=None, render=None, log=print):
        self.b = bridge
        self.camp = campaign
        self.on_event = on_event or (lambda *a, **k: None)
        self.render = render or (lambda: None)
        self.log = log
        self.fog_path = fog_path
        self.fog = {"visited": {}, "edges": {}, "back": []}
        if fog_path and os.path.exists(fog_path):
            try:
                self.fog = json.load(open(fog_path))
            except Exception as e:
                self.log(f"   [cave] fog load failed ({e}) - fresh")

    @staticmethod
    def _mk(m):
        return f"{m[0]},{m[1]}"

    def _save_fog(self):
        if self.fog_path:
            try:
                json.dump(self.fog, open(self.fog_path, "w"))
            except Exception as e:
                self.log(f"   [cave] fog save failed: {e}")

    # ── authoritative warps (events table): [(x,y), destMap] - dest known WITHOUT entering ──
    def _warps(self):
        events = self.b.rd32(_GMAPHEADER + 0x04)
        n = self.b.rd8(events + 0x01)
        base = self.b.rd32(events + 0x08)
        out = []
        for i in range(min(n, 32)):
            e = base + i * 8
            x = ctypes.c_int16(self.b.rd16(e + 0)).value
            y = ctypes.c_int16(self.b.rd16(e + 2)).value
            out.append(((x, y), (self.b.rd8(e + 7), self.b.rd8(e + 6))))
        return out

    # ── step onto a warp tile from whichever of the 4 sides is reachable ────────
    def _enter(self, warp):
        before = tv.map_id(self.b)
        # OTHER warp tiles on this floor: walking THROUGH one teleports us off the intended path
        # (the Mt Moon trap - pathing to a far warp crossed a nearer warp and fired it early). Avoid
        # them both in the reachability precheck and the actual walk; we only step onto OUR target.
        others = {w for w, _ in self._warps()} - {tuple(warp)}
        for (dx, dy), sdir in (((0, 1), "N"), ((0, -1), "S"), ((1, 0), "W"), ((-1, 0), "E")):
            grid = tv.Grid(self.b)
            here = tv.coords(self.b)
            appr = (warp[0] + dx, warp[1] + dy)
            if not grid.walkable(*appr):
                continue
            avoid_walk = (lambda g: lambda sx, sy: g.walkable(sx, sy) and (sx, sy) not in others)(grid)
            if here != appr and not tv.bfs(grid, here, (lambda a: lambda t: t == a)(appr),
                                           walkable=avoid_walk):
                continue
            if here != appr:
                r = self.camp.trav.travel(target_map=None, arrive_coord=appr, max_steps=800,
                                          max_seconds=600, avoid=others)
                if r == "battle_loss":
                    return "BLACKOUT"        # party blacked out mid-approach -> respawned at a
                                             # Center; NEVER mis-read that teleport as an exit
                if tv.map_id(self.b) != before:
                    return tv.map_id(self.b)
                if r != "arrived":
                    continue
            for _ in range(4):
                self.b.press(_STEP_ONTO[sdir], 8, 8, self.render, owner="agent")
                if tv.map_id(self.b) != before:
                    return tv.map_id(self.b)
                if st.in_battle(self.b):
                    self.camp.trav.battle_runner()
        return None

    def _reachable(self, warp):
        grid = tv.Grid(self.b); here = tv.coords(self.b)
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            appr = (warp[0] + dx, warp[1] + dy)
            if grid.walkable(*appr) and (here == appr or tv.bfs(
                    grid, here, (lambda a: lambda t: t == a)(appr), walkable=grid.walkable)):
                return True
        return False

    # ── WARP-LEVEL frontier explorer with LANDING-AWARE exit ─────────────────────
    # v2 (2026-06-26, after the phantom): the forward exit can share a map id with the entrance
    # (Mt Moon: both (3,22)), and the exit warp can live in a region of a floor reachable only via
    # a DEEPER loop (1F->B1F->B2F->B1F'->exit). So: (1) frontier is per-WARP, not per-floor, so a
    # multi-region floor gets fully explored; (2) "exit" is judged by LANDING, not map id - we only
    # count emergence as forward if the fwd_edge (toward the next area, e.g. Cerulean = east) is
    # BFS-reachable from where we land; the entrance plaza fails that test, so we re-enter and keep
    # exploring; (3) abort LOUD when the reachable warp graph is exhausted with no forward exit.
    def _fwd_reachable(self, fwd_edge):
        g = tv.Grid(self.b)
        line = {"east": g.sx_hi, "west": g.sx_lo, "north": g.sy_lo, "south": g.sy_hi}[fwd_edge]
        axis = 0 if fwd_edge in ("east", "west") else 1
        return bool(tv.bfs(g, tv.coords(self.b), (lambda t: t[axis] == line), walkable=g.walkable))

    def run_plan(self, plan, fwd_edge="east"):
        """Execute a PRE-COMPUTED warp route (region-graph planner output): a list of
        [expected_map, warp_xy]. On each map, navigate to the warp (ledge-aware BFS in _enter) and
        enter it, verifying we're on the expected map first. After the last warp we should be in the
        FORWARD region of a non-cave map -> cross the fwd edge to the far side (Cerulean) to confirm.
        Keeps the anti-phantom guard: only 'exited' if we actually cross out to a new non-cave map."""
        for i, (exp_map, wxy) in enumerate(plan):
            cur = tv.map_id(self.b)
            if tuple(cur) != tuple(exp_map):
                self.log(f"   [cave] PLAN MISMATCH step {i}: on {cur}, expected {exp_map} - ABORT LOUD")
                return "stuck"
            self.log(f"   [cave] plan step {i}: on {cur}@{tv.coords(self.b)} -> enter warp {tuple(wxy)}")
            nm = self._enter(tuple(wxy))
            if nm == "BLACKOUT":
                self.log("   [cave] BLACKED OUT executing plan - ABORT (survival, not nav)"); return "blackout"
            if nm is None:
                self.log(f"   [cave] plan step {i}: warp {tuple(wxy)} unenterable - ABORT LOUD"); return "stuck"
        cur = tv.map_id(self.b); here = tv.coords(self.b)
        if not self._fwd_reachable(fwd_edge):
            self.log(f"   [cave] plan done at {cur}@{here} but {fwd_edge} edge UNREACHABLE - ABORT LOUD")
            return "stuck"
        self.log(f"   [cave] FORWARD region reached {cur}@{here}; crossing {fwd_edge} to the far side")
        self.camp.trav.travel(target_map=(99, 99), edge=fwd_edge, max_steps=400, max_seconds=200)
        out = tv.map_id(self.b)
        if out != cur:
            self.log(f"   [cave] *** CLEARED: emerged + crossed {cur} -> {out} ***")
            self.on_event("made it all the way through to the other side"); return "exited"
        self.log(f"   [cave] tried to cross but still on {cur} - ABORT LOUD"); return "stuck"

    def clear_cave(self, entrance_map, fwd_edge="east", max_hops=80, max_seconds=1800):
        cave_group = tv.map_id(self.b)[0]
        entered = set()                  # (mapkey, warp_xy) we have GONE THROUGH (warp-level fog)
        t0 = time.time()
        stale = 0
        for hop in range(max_hops):
            if time.time() - t0 > max_seconds:
                self.log("   [cave] !! WALL-CLOCK budget blown - ABORT LOUD"); return "timeout"
            cur = tv.map_id(self.b); here = tv.coords(self.b); curk = self._mk(cur)
            # LANDING-AWARE exit test (only on non-cave maps)
            if cur[0] != cave_group:
                if self._fwd_reachable(fwd_edge):
                    self.log(f"   [cave] FORWARD emergence on {cur} at {here}: {fwd_edge} edge "
                             f"reachable -> crossing to confirm the far side")
                    self.camp.trav.travel(target_map=(99, 99), edge=fwd_edge,
                                          max_steps=400, max_seconds=150)
                    out = tv.map_id(self.b)
                    if out != cur and out[0] != cave_group:
                        self.log(f"   [cave] *** CLEARED: emerged + crossed {cur}->{out} ***")
                        self.on_event("made it all the way through the cave"); self._save_fog()
                        return "exited"
                    self.log(f"   [cave] crossed fwd; now {out} - treating as cleared")
                    self._save_fog(); return "exited"
                self.log(f"   [cave] non-cave {cur} at {here} but {fwd_edge} edge UNREACHABLE "
                         f"= back/entrance region -> re-enter + keep exploring")
            self.fog["visited"][curk] = self.fog["visited"].get(curk, 0) + 1
            warps = self._warps()
            reach = [(xy, d) for xy, d in warps if self._reachable(xy)]
            if not reach:
                self.log("   [cave] !! no reachable warp here - ABORT LOUD"); self._save_fog()
                return "stuck"

            def dist(xy):
                return abs(xy[0] - here[0]) + abs(xy[1] - here[1])
            frontier = [(xy, d) for xy, d in reach if (curk, xy) not in entered]   # unvisited warps
            if frontier:
                target = min(frontier, key=lambda c: dist(c[0])); kind = "frontier"; stale = 0
            else:                            # region exhausted -> re-enter a known warp to escape
                target = min(reach, key=lambda c: dist(c[0])); kind = "escape"; stale += 1
            if stale > 16:
                self.log("   [cave] !! reachable warp graph exhausted, NO forward exit found "
                         "- ABORT LOUD (not a phantom: never crossed a fwd edge)")
                self._save_fog(); return "stuck"
            self.log(f"   [cave] hop{hop} {cur}@{here} -> {kind} warp {target[0]}->{target[1]} "
                     f"(entered {len(entered)} warps)")
            entered.add((curk, target[0]))
            nm = self._enter(target[0])
            if nm == "BLACKOUT":
                self.log("   [cave] BLACKED OUT mid-traversal - ABORT (survival/roster, not nav)")
                self._save_fog(); return "blackout"
            if nm is None:
                self.log(f"   [cave] enter {target[0]} failed; counted as visited"); continue
            self.fog["edges"][f"{curk}@{target[0][0]},{target[0][1]}"] = self._mk(nm)
            self._save_fog()
        self.log("   [cave] hop budget exhausted - ABORT LOUD"); return "timeout"
