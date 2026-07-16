"""pad_nav.py — THE TELEPORT-PAD ROUTER (general engine asset, extracted from
recon_sabrina.py after the Marsh Badge strike proved it end-to-end 2026-07-07;
ported into the engine 2026-07-08 night shift 5 — the descent pre-grade's SILPH WARN:
beat_gym travel-wedged x4 inside Saffron Gym and A-mashed Sabrina from 11 tiles away).

Ground truth (sabrina_run8 + the strike):
  A building interior can be WARP-PARTITIONED — walk-BFS reads no_route from the
  entrance pocket to ANYTHING (Saffron Gym (14,3) = 32 warp events, most of them
  teleport pads). A warp event whose DEST is the CURRENT map is a teleport pad; its
  warp_id indexes the LANDING warp's tile (travel.read_warps returns events in id
  order). The pad graph is computed AT RUNTIME — zero hardcoded room sequences; the
  class ports to any teleport maze (Sabrina's gym; Mansion-style layouts).

  REGIONS: flood-fill walk-regions with ALL warp tiles + NPC bodies masked, bounded
  to the playable rectangle, gated STRICT to the seed tile's elevation (0xF =
  multi-level pass). STRICTER than Grid.edge_open's per-edge law on purpose: that
  law allows 0-as-transition, and the void strip beside the rooms (col x29) reads
  collision-0 elevation-0 next to elevation-3 floor — a 0-lenient flood welds every
  room together (the strike-2 leak). Meta-BFS over regions, pad rides as edges.

  RIDING: approach a free pad-neighbor (nearest by real BFS), then ONE long-hold
  press onto the pad. Same-map pads never change map_id — arrival = coords JUMPED
  to a tile that is none of {start, approach, pad} (the teleport fade takes a beat
  after she lands on the pad; don't judge early).

  MOVEMENT INSIDE AN ARMED MAZE NEVER USES travel: travel's BFS is warp-blind and
  reads pads as plain floor, so its paths ride pads mid-route (the divergence class
  spin_nav documents for spinners). walk() here is the strike's masked static-BFS
  stepper verbatim.

Used by: campaign.beat_gym (router armed at gym entry when the interior has
same-map warps) — trainer engagement, leader approach, post-badge walk-out.
"""
from collections import deque

import travel as tv
import pokemon_state as st
from dialogue_drive import box_open as _box_open

_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def same_map_pads(b):
    """{pad_tile: landing_tile} for the current map — the arm-the-router probe.
    Empty dict = a normal interior (no teleport pads). Pure read."""
    here = tuple(tv.map_id(b))
    warps = tv.read_warps(b)
    pads = {}
    for xy, dest, wid in warps:
        if tuple(dest) == here and 0 <= wid < len(warps):
            pads[tuple(xy)] = tuple(warps[wid][0])
    return pads


class PadNav:
    """One instance per (bridge, campaign) pair, built at interior entry. All methods
    are same-map only: if the map changed since construction (whiteout / heal exit),
    they return False and the caller re-arms on re-entry."""

    def __init__(self, b, camp, log=lambda m: print(m, flush=True)):
        self.b, self.camp, self.L = b, camp, log
        self.map = tuple(tv.map_id(b))

    def _here(self):
        return tuple(tv.map_id(self.b)) == self.map

    def _elev(self, g, sx, sy):
        return g.elev.get((sx + tv.MAP_OFFSET, sy + tv.MAP_OFFSET), 0)

    def _region(self, g, seed, wtiles, npcs):
        """Walk-region flood from seed: warps + NPC bodies masked, playable-rect
        bounded, STRICT seed elevation (0xF pass), edge_open honored."""
        e0 = self._elev(g, *seed)
        seen, q = {seed}, [seed]
        while q:
            x, y = q.pop()
            for dx, dy in _DIRS:
                n = (x + dx, y + dy)
                if n in seen or n in wtiles or n in npcs:
                    continue
                if not (g.sx_lo <= n[0] <= g.sx_hi and g.sy_lo <= n[1] <= g.sy_hi):
                    continue
                if self._elev(g, *n) not in (e0, 0xF):
                    continue
                if not g.walkable(*n) or not g.edge_open(x, y, dx, dy):
                    continue
                seen.add(n)
                q.append(n)
        return frozenset(seen)

    def plan(self, goal_tiles, label="pad-plan"):
        """Pad-ride sequence from HERE until the current walk-region touches one of
        goal_tiles. [] = already in-region (walk the rest), [pad, ...] = rides in
        order, None = no route (LOUD)."""
        if not self._here():
            return None
        b = self.b
        g = tv.Grid(b)
        warps = tv.read_warps(b)
        npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
        wtiles = {tuple(w[0]) for w in warps}
        pads = same_map_pads(b)
        goal_tiles = set(goal_tiles)

        def pads_of(reg):
            return [p for p in pads
                    if any((p[0] + dx, p[1] + dy) in reg for dx, dy in _DIRS)]

        cur = tuple(tv.coords(b) or (0, 0))
        start = self._region(g, cur, wtiles, npcs)
        if goal_tiles & start:
            return []
        seen_regions = {start}
        q = deque([(start, [])])
        while q:
            reg, path = q.popleft()
            for p in pads_of(reg):
                land = pads[p]
                lr = None
                for dx, dy in _DIRS:
                    n = (land[0] + dx, land[1] + dy)
                    if n not in wtiles and n not in npcs and g.walkable(*n):
                        lr = self._region(g, n, wtiles, npcs)
                        break
                if lr is None or lr in seen_regions:
                    continue
                if goal_tiles & lr:
                    self.L(f"   [{label}] pad route: "
                           f"{' -> '.join(str(x) for x in path + [p])} then walk")
                    return path + [p]
                seen_regions.add(lr)
                q.append((lr, path + [p]))
        self.L(f"!! [{label}] NO pad route from {cur} toward {sorted(goal_tiles)[:4]} "
               f"({len(pads)} pads, {len(seen_regions)} regions swept)")
        return None

    def walk(self, tile, label="pad-walk", tries=6):
        """Deterministic same-map mover (strike-verbatim): static BFS with warps +
        NPC bodies masked + strict seed elevation, stepped tile-by-tile; battles run
        to completion and recompute; a step that fails outside battle after the
        spotting-wait is DEAD for this call."""
        b, camp = self.b, self.camp
        dead = set()
        for _ in range(tries):
            if not self._here():
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile:
                return True
            g = tv.Grid(b)
            e0 = self._elev(g, *cur)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
            npcs = ({tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
                    | dead) - {tile}
            p = tv.bfs(g, cur, lambda t: t == tile,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs
                       and self._elev(g, sx, sy) in (e0, 0xF))
            if not p:
                self.L(f"   [{label}] no masked static path {cur} -> {tile} "
                       f"(dead={sorted(dead)})")
                return False
            for t in p[1:]:
                ok = camp._step_to(tuple(t))
                if st.in_battle(b):
                    self.L(f"   [{label}] battle mid-path -> {camp.battle_runner()}")
                    camp._drain_overworld(label=label)
                    break
                if not ok:
                    for _ in range(120):
                        b.run_frame()
                    if _box_open(b):
                        camp._drain_overworld(label=label)
                    if st.in_battle(b):
                        self.L(f"   [{label}] step was a trainer spotting -> "
                               f"{camp.battle_runner()}")
                        camp._drain_overworld(label=label)
                        break
                    dead.add(tuple(t))
                    self.L(f"   [{label}] step into {tuple(t)} failed — dead-marked, "
                           f"recompute")
                    break
            if tuple(tv.coords(b) or ()) == tile:
                return True
        return tuple(tv.coords(b) or ()) == tile

    def ride(self, pad, label="pad-ride"):
        """Approach a free pad-neighbor (nearest by real BFS), then one LONG-HOLD
        press onto the pad (turn+walk in one continuous input). Arrival = coord
        JUMP (same-map pads never change map_id)."""
        b, camp = self.b, self.camp
        if not self._here():
            return False
        m0 = tuple(tv.map_id(b))
        c0 = tuple(tv.coords(b) or (0, 0))
        g = tv.Grid(b)
        e0 = self._elev(g, *c0)
        wts = {tuple(w[0]) for w in tv.read_warps(b)}
        npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
        cands = []
        for nb, kin in (((pad[0] - 1, pad[1]), "RIGHT"), ((pad[0] + 1, pad[1]), "LEFT"),
                        ((pad[0], pad[1] - 1), "DOWN"), ((pad[0], pad[1] + 1), "UP")):
            if nb in wts or not g.walkable(*nb) or self._elev(g, *nb) not in (e0, 0xF):
                continue
            p = (tv.bfs(g, c0, lambda t, a=nb: t == a,
                        walkable=lambda sx, sy: g.walkable(sx, sy)
                        and (sx, sy) not in wts and (sx, sy) not in npcs
                        and self._elev(g, sx, sy) in (e0, 0xF))
                 if c0 != nb else [c0])
            if p:
                cands.append((len(p), nb, kin))
        for _len, nb, kin in sorted(cands):
            if tuple(tv.coords(b) or ()) != nb and not self.walk(nb, f"{label}-approach"):
                continue
            for _try in range(3):
                b.press(kin, 26, 10, camp.render, owner="agent")
                for _ in range(180):
                    b.run_frame()
                    cnow = tuple(tv.coords(b) or (0, 0))
                    if tuple(tv.map_id(b)) != m0 or cnow not in (c0, nb, pad):
                        break
                cnow = tuple(tv.coords(b) or (0, 0))
                if tuple(tv.map_id(b)) != m0 or cnow not in (c0, nb, pad):
                    for _ in range(60):
                        b.run_frame()
                    self.L(f"   [{label}] rode pad {pad}: {c0} -> "
                           f"{tuple(tv.coords(b) or ())} (map {tuple(tv.map_id(b))})")
                    # a spotter may fire the instant she lands — resolve before returning
                    if _box_open(b):
                        camp._drain_overworld(label=label)
                    if st.in_battle(b):
                        self.L(f"   [{label}] battle at pad landing -> "
                               f"{camp.battle_runner()}")
                        camp._drain_overworld(label=label)
                    return True
            break
        self.L(f"!! [{label}] pad ride {pad} failed (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def goto_region(self, goal_tiles, label="pad", max_rides=8):
        """Ride pads until one of goal_tiles shares the current walk-region. True =
        the caller can WALK the rest; False = no route or a ride wedged (caller
        falls back LOUD — never silently)."""
        for _ in range(max_rides):
            if st.in_battle(self.b):
                self.L(f"   [{label}] battle before ride -> {self.camp.battle_runner()}")
                self.camp._drain_overworld(label=label)
            plan = self.plan(goal_tiles, label)
            if plan is None:
                return False
            if not plan:
                return True
            if not self.ride(plan[0], label):
                return False
        return False
