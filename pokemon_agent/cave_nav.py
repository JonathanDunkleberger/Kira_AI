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
        for (dx, dy), sdir in (((0, 1), "N"), ((0, -1), "S"), ((1, 0), "W"), ((-1, 0), "E")):
            grid = tv.Grid(self.b)
            here = tv.coords(self.b)
            appr = (warp[0] + dx, warp[1] + dy)
            if not grid.walkable(*appr):
                continue
            if here != appr and not tv.bfs(grid, here, (lambda a: lambda t: t == a)(appr),
                                           walkable=grid.walkable):
                continue
            if here != appr:
                r = self.camp.trav.travel(target_map=None, arrive_coord=appr, max_steps=400)
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

    # ── the floor-graph explorer ────────────────────────────────────────────────
    def clear_cave(self, entrance_map, max_hops=30, max_seconds=1500):
        cave_group = tv.map_id(self.b)[0]
        # BACK = the entrance side only (where we came from). Everything else non-cave is the EXIT.
        # (Do NOT auto-infer back from the nearest warp - on a hand-saved mid-cave state the nearest
        # warp can be the forward exit, which mislabeled Mt Moon's (3,22) Cerulean exit as "back".)
        back = {tuple(entrance_map)}
        t0 = time.time()
        for hop in range(max_hops):
            if time.time() - t0 > max_seconds:
                self.log("   [cave] !! WALL-CLOCK budget blown - ABORT LOUD"); return "timeout"
            cur = tv.map_id(self.b)
            # EXIT: left the cave group to a map that is NOT a known back/entrance map. A blackout
            # also leaves the cave (respawn at a Center), but that is caught at the source: _enter
            # returns "BLACKOUT" on travel's battle_loss, so we never reach here on a blackout.
            if cur[0] != cave_group and cur not in back:
                self.log(f"   [cave] EXITED -> map {cur} (far side)")
                self.on_event("made it through the cave"); self._save_fog(); return "exited"
            self.fog["visited"][self._mk(cur)] = self.fog["visited"].get(self._mk(cur), 0) + 1
            warps = self._warps()
            here = tv.coords(self.b)
            # candidates = reachable warps whose dest is NOT a back/entrance map (never walk back
            # out the way we came). A FORWARD EXIT = any reachable non-cave dest (e.g. (3,22)).
            cand = [(xy, dest) for xy, dest in warps if self._reachable(xy) and dest not in back]
            self.log(f"   [cave] hop{hop} map={cur} at={here} "
                     f"warps={[(xy, dest) for xy, dest in warps]} reachable={cand}")
            if not cand:
                self.log("   [cave] !! no reachable forward warp - ABORT LOUD")
                self._save_fog(); return "stuck"

            def dist(xy):
                return abs(xy[0] - here[0]) + abs(xy[1] - here[1])
            exits = [(xy, d) for xy, d in cand if d[0] != cave_group]          # non-cave, non-back
            frontier = [(xy, d) for xy, d in cand if d[0] == cave_group
                        and self._mk(d) not in self.fog["visited"]]            # unseen cave floor
            if exits:
                target = min(exits, key=lambda c: dist(c[0])); kind = "EXIT"
            elif frontier:
                target = min(frontier, key=lambda c: dist(c[0])); kind = "frontier"
            else:                                        # anti-loop: least-visited cave neighbour
                target = min(cand, key=lambda c: self.fog["visited"].get(self._mk(c[1]), 0))
                kind = "backtrack"
            self.log(f"   [cave] -> {kind} warp {target[0]} -> {target[1]}")
            newmap = self._enter(target[0])
            if newmap == "BLACKOUT":
                self.log("   [cave] party BLACKED OUT mid-traversal (lost a battle) -> ABORT LOUD; "
                         "survival is a roster/heal concern, not a nav failure")
                self._save_fog(); return "blackout"
            if newmap is None:
                self.log(f"   [cave] enter {target[0]} failed; marking unenterable")
                self.fog["edges"][f"{self._mk(cur)}@{target[0][0]},{target[0][1]}"] = "UNREACHABLE"
                continue
            self.fog["edges"][f"{self._mk(cur)}@{target[0][0]},{target[0][1]}"] = self._mk(newmap)
            self._save_fog()
        self.log("   [cave] hop budget exhausted"); return "timeout"
