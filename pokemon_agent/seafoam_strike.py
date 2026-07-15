"""seafoam_strike.py — THE SEAFOAM CROSSING, in-loop (night shift #12, badge-6→7 build).

The gate BETWEEN the Safari (which grants Surf+Strength) and Blaine's gym on Cinnabar: Route 20's
surface is SEVERED at the Seafoam Islands, so post-Surf she STILL can't sea-nav to Cinnabar — the
crossing is THROUGH the Seafoam interior (a STRENGTH boulder-cascade over 5 floors that becalms the
B3F current, then surf out the west exit). The general surf-aware traveler bounces at Route 20
(`EDGE-ROUTE (3,38)->(3,8)` -> no clean path -> no_path -> genuine zone gap). The champion cleared it
with the bespoke recon_seafoam.py strike; this is a FAITHFUL port of that proven script, driven by the
live `camp` so the general questline can call it as ONE decision (the same shape as beat_gym /
safari_strike / silph_strike / hideout_strike). FireRed coords are isolated here (rule 14 — portability
debt: a Kanto-Seafoam fact table + the boulder MISSION script, swap per game).

Prereq (recognition-enforced): Surf + Strength taught, badge 6 (== the safari_strike OUTPUT state).

THE DERIVATION (recon_seafoam.py header, all pret ground truth — layout bins + map.json + scripts.inc):
  - R20's surface is SEVERED at Seafoam (dual-flood proven). The crossing is the interior. Land+water
    meta-BFS over the 5 floors finds NO route with currents active: the west-exit cluster is sealed
    behind the B3F current field.
  - THE MECHANISM (scripts.inc): B3F's current stops when BOTH B3F boulders are PRESENT
    (FLAG_HIDE_SEAFOAM_B3F_BOULDER_1/2 0x046/0x047 cleared) -> FLAG_STOPPED_SEAFOAM_B3F_CURRENT (0x2D2)
    -> setmaplayoutindex ..._CURRENT_STOPPED (currents -> calm water). Boulders cascade DOWN THE HOLE
    CHAIN (MB_FALL_WARP 0x66): 1F pushes drop to B1F, B1F to B2F, B2F to B3F. Falling into B3F with
    <2 boulders present = the FORCED CURRENT RIDE to B4F — so the last fall must come after both are down.
  - With B3F stopped, the route surfs east/south to ladders back up F1 -> exit door (32,21) -> R20 west
    sea -> surf west -> Cinnabar (3,8).

THE MISSION (floors: F1 (1,83) B1F (1,84) B2F (1,85) B3F (1,86)) — see MISSION below.

STRENGTH ACTUATION: face boulder, A -> "Want to use STRENGTH?" YES (default) -> drain; VERIFIED by
FLAG_SYS_USE_STRENGTH (0x805). Resets on every map change -> re-arm per floor. A push: stand opposite,
hold the direction; the boulder slides one tile (player stays); VERIFIED by the live gObjectEvents coord.

run_strike returns 'reached_cinnabar' | 'in_seafoam' (0x2D2 armed but the exit didn't complete) |
'not_here' (no strike from here) | 'failed'.
"""
import os
import time

import firered_ram as ram
import field_moves as fm
import pokemon_state as st
import travel as tv
from dialogue_drive import box_open as dd_box

# ── FireRed Seafoam fact table (game-knowledge layer; rule 14 portability debt) ───────────────────
FUCHSIA, R19, R20, CINNABAR = (3, 7), (3, 37), (3, 38), (3, 8)
F1, B1F, B2F, B3F = (1, 83), (1, 84), (1, 85), (1, 86)
SEAFOAM_MAPS = {F1, B1F, B2F, B3F}
# Anchors for the questline strike dispatcher: the sea road (Fuchsia/R19/R20 — the strike drives the
# whole crossing from any of them) + every Seafoam interior floor (so a mid-tour re-tick RESUMES the
# boulder mission rather than dumping her). Not CINNABAR: once across, 0x2D2 is set and the step self-
# satisfies (the strike would never be invoked there).
SEAFOAM_ANCHORS = {FUCHSIA, R19, R20} | SEAFOAM_MAPS
MOVE_SURF, MOVE_STRENGTH = 57, 70
FLAG_STR_ACTIVE = 0x805
FLAG_B3F_CALM = 0x2D2               # FLAG_STOPPED_SEAFOAM_B3F_CURRENT — the crossing-done signal
DOOR_EAST = (60, 8)                 # R20 -> F1 (6,21)
EXIT_WEST = (32, 21)                # F1 -> R20 (72,14)

# ── ROUTE-21 REROUTE (the puzzle-free bypass of the hang-prone Seafoam interior) ──────────────────
# The Seafoam interior boulder-cascade is complex and its R19/R20 sea approach has silently WEDGED a
# live run (fresh_go_5: frozen ~3h at R19-west (11,22), sea_walk livelocking below its own deadline).
# The canonical bypass is the SAME sea road giovanni_gym drives in reverse: PALLET -> surf SOUTH down
# Route 21 -> Cinnabar (no interior, no boulders). Map ids match giovanni_gym: Cinnabar (3,8), R21
# south (3,40), R21 north (3,39), Pallet (3,0). Rerouting reaches Cinnabar WITHOUT setting the game's
# crossed-flag (0x2D2), so the reroute stamps it on arrival (harmless — she IS across; the current
# only matters inside Seafoam she'll never re-enter) so the questline recognizes the crossing and the
# Secret-Key/Blaine gate opens next. Flag-gated (default ON); OFF falls back to the interior path.
SEAFOAM_REROUTE_VIA_R21 = os.getenv("POKEMON_SEAFOAM_REROUTE_R21", "1") != "0"
R21_SOUTH, R21_NORTH, PALLET = (3, 40), (3, 39), (3, 0)
R21_CORRIDOR = {PALLET, R21_NORTH, R21_SOUTH}
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
ARROW_KEY = {0x62: "RIGHT", 0x63: "LEFT", 0x64: "UP", 0x65: "DOWN"}
DIRN_OF = {"south": 1, "north": 2, "west": 3, "east": 4}

# (op, args...) — coords are floor-local SAVE coords; push boulder ids are the boulder's position at
# op time (matched to the nearest live object). Verbatim from recon_seafoam.py (proven on the champion
# climb + re-proven this shift: surf_ready_kit -> Cinnabar in 76s).
MISSION = [
    ("strength", (22, 12)),
    ("push", (22, 12), "UP", 4), ("push", (22, 8), "LEFT", 1),   # b1 -> hole (21,8)
    ("push", (32, 9), "UP", 1), ("push", (32, 8), "LEFT", 2),    # b2 -> hole (30,8)
    ("fall", (21, 8), B1F),
    ("strength", (22, 8)), ("push", (22, 8), "RIGHT", 1),        # b1 -> hole (23,8)
    ("fall", (23, 8), B2F),
    ("strength", (22, 8)), ("push", (22, 8), "RIGHT", 2),        # b1 -> B3F (23,8)
    ("ladder", (7, 4), B1F), ("ladder", (10, 6), F1),
    ("fall", (30, 8), B1F),
    ("strength", (30, 8)), ("push", (30, 8), "LEFT", 2),         # b2 -> hole (28,8)
    ("fall", (28, 8), B2F),
    ("strength", (30, 8)), ("push", (30, 8), "LEFT", 3),         # b2 -> B3F (24,8)
    ("fall", (27, 8), B3F),
    ("verify_calm",),
    ("ladder", (31, 16), B2F),
    ("ladder", (32, 14), B1F),
    ("ladder", (28, 19), F1),
    ("ladder", EXIT_WEST, R20),
]


class SeafoamStrike:
    def __init__(self, camp, log, dbg_dir=None):
        self.camp = camp
        self.b = camp.b
        self.log = log
        self.dbg = dbg_dir
        if dbg_dir:
            try:
                os.makedirs(dbg_dir, exist_ok=True)
            except Exception:
                self.dbg = None
        self.n_battles = 0
        self.wedges = {}
        self.deadline = time.time() + 2400

    # ── snap / battle / dialogue drains ────────────────────────────────────────────────────────────
    def snap(self, name):
        if not self.dbg:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(os.path.join(self.dbg, name + ".png"))
        except Exception as e:
            self.log(f"   snap {name} failed: {e}")

    def fight(self):
        self.n_battles += 1
        return self.camp.battle_runner()

    def fight_open(self):
        return ram.valid_ewram_ptr(self.b.rd32(ram.GBATTLE_RES_PTR))

    def drain(self, max_n=40, key="B"):
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_n):
            if self.fight_open():
                return
            if dd_box(b):
                stable = 0
                b.press(key, 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(30):
                    b.run_frame()

    def handle_interrupts(self):
        if self.fight_open():
            self.fight()
            self.drain()
            return True
        if dd_box(self.b):
            self.drain()
            return True
        return False

    def settle(self, n=90):
        for _ in range(n):
            self.b.run_frame()

    def wedge(self, k, cap, msg):
        self.wedges[k] = self.wedges.get(k, 0) + 1
        if self.wedges[k] >= cap:
            self.snap(f"wedge_{k}")
            self.log(f"!! {msg} x{cap} — abort LOUD")
            return True
        self.drain()
        for _ in range(150):
            self.b.run_frame()
        return False

    # ── water machinery (recon_cinnabar/recon_seafoam verbatim) ────────────────────────────────────
    @staticmethod
    def water_save(g):
        return {(bx - tv.MAP_OFFSET, by - tv.MAP_OFFSET) for bx, by in g.water}

    @staticmethod
    def sea_ok(g, wset):
        def ok(sx, sy):
            bx, by = sx + tv.MAP_OFFSET, sy + tv.MAP_OFFSET
            if not (0 <= bx < g.w and 0 <= by < g.h):
                return False
            if g.col.get((bx, by), 1) != 0:
                return False
            return g.walkable(sx, sy) or (sx, sy) in wset
        return ok

    def on_water(self):
        g = tv.Grid(self.b)
        return tuple(tv.coords(self.b) or (99, 99)) in self.water_save(g)

    def mount(self, face_key):
        b, camp = self.b, self.camp
        if self.on_water():
            return True
        for attempt in range(4):
            b.press(face_key, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(40):
                b.run_frame()
            self.drain(key="A")
            for _ in range(240):
                b.run_frame()
                if self.on_water():
                    break
            if self.fight_open():
                self.fight()
                self.drain()
            if self.on_water():
                self.log(f"   [surf] MOUNTED at {tv.coords(b)} (attempt {attempt + 1})")
                return True
        self.log(f"!! [surf] mount failed at {tv.coords(b)} facing {face_key}")
        return False

    def live_boulders(self):
        return [ob["coord"] for ob in fm.scan_field_objects(self.b, {fm.GFX_BOULDER})]

    def live_npc_tiles(self):
        """Live object-event BODY tiles: a WANDERING swimmer parked on the planned tile blocks the same
        step forever if only static templates are masked. Body tile only, never the facing tile."""
        b = self.b
        OB, SZ = 0x02036E38, 0x24
        out = set()
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.add((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                     b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    def step_to(self, tile, wset=None):
        b, camp = self.b, self.camp
        cur = tuple(tv.coords(b) or (0, 0))
        d = (tile[0] - cur[0], tile[1] - cur[1])
        if d in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            d = (d[0] // 2, d[1] // 2)
        key = KEY_OF.get(d)
        if key is None:
            return camp._step_to(tile)
        if wset is None:
            wset = self.water_save(tv.Grid(b))
        if tile in wset and cur not in wset and not self.on_water():
            return self.mount(key)
        for _attempt in range(3):
            b.press(key, 8, 6, camp.render, owner="agent")
            for _ in range(50):
                b.run_frame()
                if tuple(tv.coords(b) or ()) == tile:
                    break
            if self.fight_open() or dd_box(b):
                return True
            if tuple(tv.coords(b) or ()) == tile:
                return True
        return False

    def sea_walk(self, goal_test, label, tries=10, avoid=()):
        b = self.b
        budget = tries
        # HANG GUARD (fresh_go_5 lesson): a moving-but-not-arriving crossing refunds budget every
        # micro-step (`budget += 1` below), so this loop can spin unbounded UNDER the strike's own
        # wall-clock deadline — a live run froze ~3h here. Bound BOTH: honor the deadline, and cap the
        # absolute replan count so a single crossing can never livelock silently.
        iters = 0
        while budget > 0:
            iters += 1
            if time.time() > self.deadline or iters > 400:
                self.log(f"   [{label}] HANG-GUARD bail at {tuple(tv.coords(b) or ())} "
                         f"(iters={iters}, deadline={'hit' if time.time() > self.deadline else 'ok'})")
                return goal_test(tuple(tv.coords(b) or ()))
            budget -= 1
            if self.handle_interrupts():
                budget += 1
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            g = tv.Grid(b)
            wset = self.water_save(g)
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            # boulders: LIVE positions only (templates go stale after a push). NPCs: live body tiles
            # FIRST (wanderers move off-template), plus non-boulder templates for far spawns.
            npcs = self.live_npc_tiles() | {tuple(o[0]) for o in tv.read_object_templates(b)
                                            if o[2] and o[1] != fm.GFX_BOULDER}
            ok0 = self.sea_ok(g, wset)
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: ok0(sx, sy) and (sx, sy) not in wts
                       and (sx, sy) not in npcs and (sx, sy) not in avoid)
            self.log(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                self.log(f"   [{label}] no path from {cur}")
                self.snap(f"nopath_{label[:12]}_{cur[0]}_{cur[1]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                if self.handle_interrupts():
                    budget += 1
                    break
                if not self.step_to(tuple(t), wset):
                    self.log(f"   [{label}] step blocked {tuple(tv.coords(b) or ())} -> {tuple(t)} "
                             f"(npcs {sorted(self.live_npc_tiles())[:6]})")
                    self.snap(f"blocked_{t[0]}_{t[1]}")
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
            if tuple(tv.coords(b) or ()) != cur:
                budget += 1
        return goal_test(tuple(tv.coords(b) or ()))

    # ── edge crossings (billed sea road) ───────────────────────────────────────────────────────────
    @staticmethod
    def _s32(v):
        return v - (1 << 32) if v >= (1 << 31) else v

    def connections(self):
        b = self.b
        out = {}
        hdr = b.rd32(tv.GMAPHEADER + 0x0C)
        if not hdr or hdr < 0x02000000:
            return out
        n = self._s32(b.rd32(hdr))
        arr = b.rd32(hdr + 4)
        if not (0 < n < 16) or arr < 0x02000000:
            return out
        for i in range(n):
            c = arr + i * 0xC
            out.setdefault(b.rd8(c), []).append(self._s32(b.rd32(c + 4)))
        return out

    def cross_edge(self, direction, label):
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        conns = self.connections().get(DIRN_OF[direction])
        if not conns:
            self.log(f"   [{label}] no {direction} connection on {m0} — skip")
            return False
        off = conns[0]
        key = {"south": "DOWN", "north": "UP", "west": "LEFT", "east": "RIGHT"}[direction]
        for round_ in range(6):
            g = tv.Grid(b)
            wset = self.water_save(g)
            ok0 = self.sea_ok(g, wset)
            if direction in ("south", "north"):
                extreme = g.sy_hi if direction == "south" else 0
                band = [(x, extreme) for x in range(max(g.sx_lo, off), g.sx_hi + 1)
                        if ok0(x, extreme)]
            else:
                extreme = g.sx_hi if direction == "east" else 0
                band = [(extreme, y) for y in range(max(g.sy_lo, off), g.sy_hi + 1)
                        if ok0(extreme, y)]
            if not band:
                self.log(f"!! [{label}] no {direction}-edge band on {m0}")
                for _ in range(120):
                    b.run_frame()
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            band.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]) + round_ * 7)
            tgt = band[min(round_, len(band) - 1)]
            if not self.sea_walk(lambda c, t=tgt: c == t, f"{label}-band"):
                for _ in range(120):
                    b.run_frame()
                continue
            for _hold in range(4):
                cur2 = tuple(tv.coords(b) or (0, 0))
                nxt = {"south": (cur2[0], cur2[1] + 1), "north": (cur2[0], cur2[1] - 1),
                       "west": (cur2[0] - 1, cur2[1]), "east": (cur2[0] + 1, cur2[1])}[direction]
                g2 = tv.Grid(b)
                w2 = self.water_save(g2)
                if nxt in w2 and cur2 not in w2:
                    if not self.mount(key):
                        break
                    continue
                b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(90):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if self.handle_interrupts():
                    continue
                if tuple(tv.map_id(b)) != m0:
                    for _ in range(120):
                        b.run_frame()
                    self.log(f"   [{label}] EDGE {direction}: {m0} -> {tuple(tv.map_id(b))} @ {tv.coords(b)}")
                    return True
        self.log(f"!! [{label}] {direction} crossing never fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    # ── interior primitives ────────────────────────────────────────────────────────────────────────
    def nearest_boulder(self, approx, radius=8):
        """Live gObjectEvents are DISTANCE-CULLED — a far boulder reads as absent. If no live boulder is
        near `approx`, walk toward it first, then re-scan."""
        for _attempt in range(3):
            bs = [t for t in self.live_boulders()
                  if abs(t[0] - approx[0]) + abs(t[1] - approx[1]) <= radius]
            if bs:
                return min(bs, key=lambda t: abs(t[0] - approx[0]) + abs(t[1] - approx[1]))
            cur = tuple(tv.coords(self.b) or (0, 0))
            if abs(cur[0] - approx[0]) + abs(cur[1] - approx[1]) <= 3:
                return None
            if not self.sea_walk(lambda c, a=approx: abs(c[0] - a[0]) + abs(c[1] - a[1]) <= 3,
                                 "boulder-approach"):
                return None
        return None

    def ensure_strength(self, approx):
        """Face the target boulder, A -> YES; verified by FLAG_SYS_USE_STRENGTH."""
        b, camp = self.b, self.camp
        if fm.read_flag(b, FLAG_STR_ACTIVE):
            return True
        bl = self.nearest_boulder(approx)
        if bl is None:
            self.log(f"!! [strength] no live boulder near {approx} on {tv.map_id(b)}")
            return False
        for attempt in range(3):
            nbs = [(bl[0] + dx, bl[1] + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]
            if not self.sea_walk(lambda c, s=set(nbs): c in s, "str-approach"):
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            face = KEY_OF.get((bl[0] - cur[0], bl[1] - cur[1]))
            if face is None:
                continue
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            self.settle(30)
            self.drain(key="A")                     # info text + YES/NO (YES default)
            self.settle(30)
            if fm.read_flag(b, FLAG_STR_ACTIVE):
                self.log(f"   [strength] ACTIVE (attempt {attempt + 1}) at {cur} facing {face}")
                return True
        self.log(f"!! [strength] flag 0x805 never set (boulder {bl})")
        self.snap("strength_fail")
        return False

    def push(self, approx, key, n):
        """n Strength pushes; each verified by the live boulder coord moving."""
        b, camp = self.b, self.camp
        d = DELTA[key]
        for i in range(n):
            bl = self.nearest_boulder(approx)
            if bl is None:
                self.log(f"!! [push] boulder near {approx} vanished (i={i}) — treating as fallen")
                return True                         # last push dropped it down a hole
            stand = (bl[0] - d[0], bl[1] - d[1])
            if not self.sea_walk(lambda c, s=stand: c == s, f"push-approach{i}", avoid={tuple(bl)}):
                self.log(f"!! [push] can't reach {stand} to push {bl} {key}")
                return False
            moved = False
            for _try in range(4):
                if self.handle_interrupts():
                    continue
                b.press(key, 40, 10, camp.render, owner="agent")
                self.settle(70)                     # push animation ~1s
                b2l = self.nearest_boulder((bl[0] + d[0], bl[1] + d[1]))
                if b2l != bl:                       # moved or fell (vanished/None)
                    moved = True
                    break
            if not moved:
                self.log(f"!! [push] {bl} would not move {key} (player {tv.coords(b)})")
                self.snap(f"push_fail_{bl[0]}_{bl[1]}")
                return False
            approx = (bl[0] + d[0], bl[1] + d[1])
            self.log(f"   [push] {bl} -> {approx} ({key}, {i + 1}/{n})")
            self.settle(30)
        return True

    def tile_behavior(self, t):
        """Live metatile behavior via the backup layout + tileset attrs (probe read)."""
        b = self.b
        try:
            ml = b.rd32(tv.GMAPHEADER)
            attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
            bw = b.rd32(tv.BACKUP_LAYOUT)
            mp0 = b.rd32(tv.BACKUP_LAYOUT + 8)
            mid = b.rd16(mp0 + ((t[1] + tv.MAP_OFFSET) * bw + (t[0] + tv.MAP_OFFSET)) * 2) & 0x3FF
            base, idx = (attr[0], mid) if mid < tv.NUM_PRIMARY else (attr[1], mid - tv.NUM_PRIMARY)
            return b.rd32(base + idx * 4) & 0xFF
        except Exception:
            return 0

    def go_warp(self, tile, dest, label):
        """Walk adjacent to a warp/hole/door/ladder tile, then step ON it; verify the map flip. ARROW-WARP
        class (the F1 west exit (32,21) is MB_SOUTH_ARROW_WARP 0x65 — fires ONLY by pressing its arrow
        direction while standing on the mat): approach from the arrow-opposite side + hold the arrow key."""
        b = self.b
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        beh = self.tile_behavior(tile)
        arrow = ARROW_KEY.get(beh)
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]
        if arrow:
            d = DELTA[arrow]
            nbs = [(tile[0] - d[0], tile[1] - d[1])]   # walk in along the arrow
        for attempt in range(4):
            if tuple(tv.coords(b) or ()) not in nbs and tuple(tv.coords(b) or ()) != tile:
                if not self.sea_walk(lambda c, s=set(nbs): c in s, f"{label}-approach"):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            key = (arrow if arrow and cur == tile
                   else KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1])) or arrow)
            if key is None:
                continue
            for _press in range(4 if arrow else 1):
                b.press(key, 26, 10, self.camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    break
            if self.handle_interrupts():
                continue
            if tuple(tv.map_id(b)) == dest:
                self.settle(180)                     # warp settle + map scripts
                self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)} (beh {hex(beh)})")
                return True
            if tuple(tv.map_id(b)) != m0:
                self.log(f"!! [{label}] warped to {tuple(tv.map_id(b))}, wanted {dest}")
                self.settle(180)
                return False
        self.log(f"!! [{label}] never fired (at {tv.map_id(b)}@{tv.coords(b)}, beh {hex(beh)})")
        self.snap(f"warpfail_{label[:16]}")
        return False

    # ── the crossing ───────────────────────────────────────────────────────────────────────────────
    def run(self):
        b, camp = self.b, self.camp
        here = tuple(tv.map_id(b))
        self.log(f"   seafoam strike: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"calm={int(fm.read_flag(b, FLAG_B3F_CALM))} money=${camp.money()}")

        # ACE LEADS THE GAUNTLET (NS#4, fresh_go_5 ~1h R19 livelock): the R19/R20 sea road is a battle-DENSE
        # gauntlet of L30+ Tentacruel. A weak bench lead (a road-bench-XP or persisted-swap slot-0) fights it
        # ALONE (no in-battle switch on a questline leg) -> can't KO fast enough to reach the R20 edge before
        # the deadline AND can't flee reliably -> a hard livelock. Front the true ace for the whole crossing
        # (idempotent; the roam restores the weak-lead ordering after the strike). Complements removing
        # "seafoam" from OVERWORLD_SAFE_QUESTLINES (campaign.py).
        try:
            camp._restore_ace()
        except Exception as e:
            self.log(f"   [ace-lead] _restore_ace errored: {e} — proceeding as-is (LOUD)")

        # PREFLIGHT: the crossing needs Surf AND Strength on the team (recognition enforces this, but a
        # stray call must abort LOUD, not wander the interior move-less).
        n = b.rd8(ram.GPLAYER_PARTY_CNT)
        have = {m for s in range(n) for m in (st.read_party_moves(b, s) or [])}
        # NS#38 CREDITS-BLOCKER (the Route-20 reboot-treadmill): the Safari grants HM03 Surf + HM04
        # Strength together, but only Surf is driven to a teach (Blaine's gym prereq). Strength sits in
        # the bag UNTAUGHT, so this preflight aborted 'failed' every attempt -> exhausted 3 tries at
        # Fuchsia -> never entered the interior -> STALL at Route 20 every reboot. Teach the OWNED HM04
        # here (proven teach primitive -> a free-slot bench mon) before aborting. Fail-closed: a failed
        # teach leaves the party as-is and the abort below still fires (never worse than the old loop).
        if MOVE_STRENGTH not in have and hasattr(camp, "_ensure_owned_hm_taught"):
            if camp._ensure_owned_hm_taught("strength"):
                n = b.rd8(ram.GPLAYER_PARTY_CNT)
                have = {m for s in range(n) for m in (st.read_party_moves(b, s) or [])}
        if MOVE_SURF not in have or MOVE_STRENGTH not in have:
            self.log(f"!! seafoam strike: party lacks Surf/Strength (surf={MOVE_SURF in have} "
                     f"str={MOVE_STRENGTH in have}) — abort")
            return "failed"

        # ── ROUTE-21 REROUTE (default): skip the hang-prone Seafoam interior entirely ──────────────
        # fresh_go_5 lesson (shift 2): the R21 reroute (route overland to Pallet, then surf SOUTH down
        # Route 21 to Cinnabar) ONLY works from the Pallet/Cinnabar side (post-Blaine re-entry, the
        # giovanni_gym context it was mirrored from). From a PRE-Blaine Fuchsia/Route-19 start Pallet is
        # NOT graph-reachable (Route 19's west edge is open sea; no learned land route to Pallet), so its
        # `walk_to_map(Pallet)` leg wedges forever and the strike loops on re-queue. Gate the reroute on
        # graph-reachability of the R21 corridor; when it isn't reachable (or the reroute otherwise
        # fails), FALL THROUGH to the proven interior crossing — fresh_go_1/2/3 all crossed the interior,
        # and its 3h sea_walk livelock root is now hang-guarded (see sea_walk).
        if SEAFOAM_REROUTE_VIA_R21 and self._reroute_feasible():
            _rr = self._reroute_r21()
            if _rr != "failed":
                return _rr
            self.log("   [reroute] could not reach the R21 corridor — "
                     "falling through to the proven interior Seafoam crossing")
        elif SEAFOAM_REROUTE_VIA_R21:
            self.log("   [reroute] R21 corridor not graph-reachable from here — "
                     "using the proven interior Seafoam crossing")

        # ── PHASE 1: the sea road to the Seafoam east door ────────────────────────────────────────
        # NS9 lesson (ported): a worn/PP-depleted lead gets swept by R19/R20 wilds mid-crossing (the
        # (11,5) blackout wedge). Heal to FULL first. Harmless from a fresh full team.
        try:
            self.log("   pre-crossing heal (full PP/HP for the R19/R20 wild gauntlet)")
            camp.heal_nearest()
            self.log(f"   healed @ {tv.map_id(b)} {tv.coords(b)}")
        except Exception as e:
            self.log(f"   pre-crossing heal errored: {e} — entering as-is (LOUD)")

        while tuple(tv.map_id(b)) not in {R20, CINNABAR} | SEAFOAM_MAPS and time.time() < self.deadline:
            here = tuple(tv.map_id(b))
            if self.handle_interrupts():
                continue
            if here == FUCHSIA:
                # The general campaign traveler crosses Fuchsia->R19 fine from anywhere (incl. the (33,32)
                # Warden-door spot the Safari strike leaves her at); cross_edge is the fallback.
                _wr = camp.walk_to_map(R19, "south")
                if tuple(tv.map_id(b)) != R19 and not self.cross_edge("south", "fuchsia-south"):
                    self.log(f"!! Fuchsia->R19 failed (walk_to_map={_wr})")
                    return "failed"
            elif here == R19:
                if not self.cross_edge("west", "r19-west"):
                    return "failed"
            else:
                # off-route (e.g. a nearby grind spot / the Safari exit): route to Fuchsia via the
                # general traveler FIRST, then the FUCHSIA branch takes the proven south sea road.
                self.log(f"   off-route map {here} — routing to Fuchsia via the general traveler")
                try:
                    camp.walk_to_map(FUCHSIA, "east")
                except Exception as e:
                    self.log(f"   walk_to_map(Fuchsia) errored: {e}")
                if tuple(tv.map_id(b)) != FUCHSIA and not (self.cross_edge("west", "reroute-west")
                                                           or self.cross_edge("south", "reroute-south")):
                    if self.wedge(("reroute", here), 3, f"could not route off {here} toward Fuchsia"):
                        return "failed"
            self.settle(180)

        here = tuple(tv.map_id(b))
        if here == CINNABAR:
            return self._arrive_cinnabar()          # already across (idempotent re-entry)
        if here == R20:
            self.log(f"   ON ROUTE 20 @ {tv.coords(b)} — heading for the east door {DOOR_EAST}")
            if not self.go_warp(DOOR_EAST, F1, "east-door"):
                if self.wedge("east_door", 3, "can't reach the Seafoam east door"):
                    return "failed"
        elif here not in SEAFOAM_MAPS:
            self.log(f"!! never reached Route 20 / Seafoam (at {here}) — abort")
            return "failed"

        # ── PHASE 2: the interior boulder mission ─────────────────────────────────────────────────
        # RESUME-SAFE: if she re-entered mid-tour with the current already stopped (0x2D2 set), skip
        # straight to the west-exit walk-out (the boulder chain is done; only the exit remains).
        if not fm.read_flag(b, FLAG_B3F_CALM):
            for op in MISSION:
                if time.time() > self.deadline:
                    self.log("!! deadline inside the interior")
                    return "in_seafoam" if fm.read_flag(b, FLAG_B3F_CALM) else "failed"
                while self.handle_interrupts():
                    pass
                kind = op[0]
                self.log(f"-- op {op} (map {tv.map_id(b)} @ {tv.coords(b)})")
                if kind == "strength":
                    if not self.ensure_strength(op[1]):
                        return self._interior_bail()
                elif kind == "push":
                    if not self.push(op[1], op[2], op[3]):
                        return self._interior_bail()
                elif kind in ("fall", "ladder"):
                    if not self.go_warp(op[1], op[2], f"{kind}{op[1]}"):
                        return self._interior_bail()
                elif kind == "verify_calm":
                    calm = fm.read_flag(b, FLAG_B3F_CALM)
                    self.log(f"   [calm] FLAG_STOPPED_SEAFOAM_B3F_CURRENT = {int(calm)}; "
                             f"on_water={self.on_water()} @ {tv.coords(b)}")
                    self.snap("b3f_becalmed")
                    if not calm:
                        self.log("!! current NOT stopped — boulder chain incomplete")
                        return "failed"

        # ── PHASE 3: west sea -> Cinnabar ─────────────────────────────────────────────────────────
        self.log(f"   WEST SEA @ {tv.map_id(b)}{tv.coords(b)} — surfing for Cinnabar")
        self.settle(180)
        legs = 0
        while time.time() < self.deadline:
            here = tuple(tv.map_id(b))
            if self.handle_interrupts():
                continue
            if here == CINNABAR:
                break
            if here in SEAFOAM_MAPS:
                # dropped back inside (a stray fall/warp) — re-run the exit ladder chain out to R20.
                if not self._exit_interior():
                    if self.wedge("reexit", 3, "can't re-exit the Seafoam interior"):
                        return "in_seafoam"
                continue
            if not self.cross_edge("west", f"west{legs}"):
                if not self.cross_edge("south", f"south{legs}"):
                    self.log(f"!! wedged on {here}")
                    self.snap("fail_west_wedge")
                    if self.wedge(("westwedge", here), 3, f"wedged crossing west on {here}"):
                        return "in_seafoam"
            legs += 1
            self.settle(180)
        if tuple(tv.map_id(b)) != CINNABAR:
            self.log("!! never reached Cinnabar")
            return "in_seafoam" if fm.read_flag(b, FLAG_B3F_CALM) else "failed"
        return self._arrive_cinnabar()

    def _exit_interior(self):
        """Walk the calm-water ladder chain B3F->B2F->B1F->F1->R20 (the MISSION tail), from wherever
        inside she stands. Best-effort per floor."""
        tail = [("ladder", (31, 16), B2F), ("ladder", (32, 14), B1F),
                ("ladder", (28, 19), F1), ("ladder", EXIT_WEST, R20)]
        for _kind, tile, dest in tail:
            here = tuple(tv.map_id(self.b))
            if here == dest or here == R20:
                continue
            self.go_warp(tile, dest, f"exit-{dest}")
            if tuple(tv.map_id(self.b)) == R20:
                return True
        return tuple(tv.map_id(self.b)) == R20

    def _interior_bail(self):
        """A MISSION op failed. If the current is already stopped, the crossing is essentially done —
        report in_seafoam so the calm water + surf-travel finish the exit; else it's a real failure."""
        if fm.read_flag(self.b, FLAG_B3F_CALM):
            self.log("   interior op failed AFTER the current stopped (0x2D2 set) — exit owns the rest")
            return "in_seafoam"
        return "failed"

    def _reroute_feasible(self):
        """Is the R21 reroute even possible from HERE? It must reach the R21 corridor (Pallet) overland;
        from a pre-Blaine Fuchsia/Route-19 start there is NO graph route to Pallet (Route 19's west edge
        is open sea), so the reroute would wedge forever on walk_to_map(Pallet). Feasible iff she's
        already on the corridor/Cinnabar, or the world graph has a route to a corridor map. Defensive:
        any error -> NOT feasible (fall back to the proven interior crossing)."""
        try:
            here = tuple(tv.map_id(self.b))
            if here in R21_CORRIDOR | {CINNABAR}:
                return True
            for dest in R21_CORRIDOR:
                try:
                    if self.camp.world.next_hop(here, dest, None) is not None:
                        return True
                except Exception:
                    continue
            return False
        except Exception as e:
            self.log(f"   [reroute] feasibility check errored: {e} — treating as NOT feasible (interior)")
            return False

    def _reroute_r21(self):
        """CINNABAR via Route 21 — the puzzle-free bypass of the Seafoam interior. From wherever she
        stands: route overland to Pallet, then surf SOUTH (Pallet -> R21 north -> R21 south -> Cinnabar,
        the mirror of giovanni_gym's proven north road). Reuses this class's own cross_edge/sea_walk.
        On Cinnabar, stamp FLAG_STOPPED_SEAFOAM_B3F_CURRENT (0x2D2) so the questline recognizes the
        crossing (else the seafoam gate re-opens forever and the Secret-Key gate never fires). Returns
        'reached_cinnabar' | 'failed'. Bounded by the strike deadline + per-phase wedge caps — never
        hangs (the whole point)."""
        b, camp = self.b, self.camp
        self.log(f"   🌊 SEAFOAM REROUTE via Route 21: boot {tv.map_id(b)}{tv.coords(b)} — "
                 f"Pallet -> surf SOUTH -> Cinnabar (bypassing the interior)")
        try:
            camp.heal_nearest()
            self.log(f"   reroute pre-heal done @ {tv.map_id(b)}{tv.coords(b)}")
        except Exception as e:
            self.log(f"   reroute pre-heal errored: {e} — continuing (LOUD)")

        # PHASE A: reach the Route-21 corridor (Pallet / an R21 map). Overland via the general traveler.
        while (tuple(tv.map_id(b)) not in R21_CORRIDOR | {CINNABAR}
               and time.time() < self.deadline):
            if self.handle_interrupts():
                continue
            here = tuple(tv.map_id(b))
            self.log(f"   [reroute] routing to Pallet from {here}")
            try:
                camp.walk_to_map(PALLET, "west")
            except Exception as e:
                self.log(f"   [reroute] walk_to_map(Pallet) errored: {e}")
            if tuple(tv.map_id(b)) == here:
                if self.wedge(("reroute-pallet", here), 3, f"can't route off {here} toward Pallet"):
                    return "failed"
            self.settle(120)

        # PHASE B: surf SOUTH down Route 21 until Cinnabar.
        legs = 0
        while tuple(tv.map_id(b)) != CINNABAR and time.time() < self.deadline:
            if self.handle_interrupts():
                continue
            if tuple(tv.map_id(b)) not in R21_CORRIDOR:
                self.log(f"!! [reroute] fell off the R21 corridor at {tv.map_id(b)} — abort")
                return "failed"
            if not self.cross_edge("south", f"r21-leg{legs}"):
                if self.wedge("r21-road", 3, f"southbound R21 crossing wedged at {tv.map_id(b)}"):
                    return "failed"
                self.settle(120)
                continue
            self.wedges.pop("r21-road", None)
            legs += 1
            self.settle(180)

        if tuple(tv.map_id(b)) == CINNABAR:
            self.log(f"   [reroute] REACHED CINNABAR via Route 21 after {legs} south crossings")
            return self._arrive_cinnabar()
        self.log(f"!! [reroute] never reached Cinnabar (at {tv.map_id(b)}) — abort")
        return "failed"

    def _arrive_cinnabar(self):
        b, camp = self.b, self.camp
        # Stamp the crossed-flag so the questline recognizes the crossing regardless of HOW she got here
        # (the R21 reroute never runs the boulder cascade that normally sets it; the interior path has it
        # set already, so this is a harmless idempotent confirm there).
        if not fm.read_flag(b, FLAG_B3F_CALM):
            try:
                ok = fm.set_flag(b, FLAG_B3F_CALM)
                self.log(f"   stamped FLAG_STOPPED_SEAFOAM_B3F_CURRENT (0x2D2) on arrival -> {ok}")
            except Exception as e:
                self.log(f"   !! could not stamp crossed-flag: {e} (LOUD — questline may re-open)")
        self.log(f"   CINNABAR ISLAND @ {tv.coords(b)} after {self.n_battles} battles — healing")
        try:
            r = camp.heal_nearest()
            self.log(f"   heal_nearest -> {r}")
        except Exception as e:
            self.log(f"   Cinnabar arrival heal errored: {e} — continuing")
        self.snap("90_cinnabar")
        self.log(f"   SEAFOAM DONE: calm={int(fm.read_flag(b, FLAG_B3F_CALM))} "
                 f"pos {tv.map_id(b)}@{tv.coords(b)} | battles {self.n_battles}")
        return "reached_cinnabar"


def run_strike(camp, log, dbg_dir=None):
    """Run the Seafoam crossing (Fuchsia -> R19 -> R20 -> Seafoam interior -> Cinnabar) from wherever she
    stands, in ONE call. Returns:
      'reached_cinnabar' — across, on Cinnabar (healed) — the sea road to Blaine is open.
      'in_seafoam'       — the current is stopped (0x2D2 set) but the exit didn't complete (still inside
                           / bounced pre-Cinnabar); the crossing is done, surf-travel finishes the rest.
      'not_here'         — no strike applies from here (caller falls through to the general layer).
      'failed'           — strike aborted before the crossing completed (caller surfaces / recovery reacts).
    Idempotent by state: already on Cinnabar short-circuits to the arrival; a mid-tour re-tick resumes."""
    try:
        here = tuple(tv.map_id(camp.b))
    except Exception:
        return "failed"
    if here not in SEAFOAM_ANCHORS:
        return "not_here"
    ss = SeafoamStrike(camp, log, dbg_dir=dbg_dir)
    return ss.run()
