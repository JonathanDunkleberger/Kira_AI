"""victory_road.py — THE ROAD TO THE PLATEAU (post-badge-8 -> Indigo), in-loop (night shift).

A FAITHFUL port of the proven recon_victory.py vehicle into an in-loop module driven by the live `camp`
bridge, so the endgame push can call it as ONE decision (the same shape as giovanni_gym / blaine_gym /
mansion_strike / seafoam_strike). FireRed coords/puzzles are isolated here (rule 14 portability debt: the
Route-22/23 + Victory-Road boulder-puzzle fact tables, swap per game).

Ground truth (pret; the puzzle sequences are recon_victory's offline-solved, elevation-aware constants —
vr1f/2f/3f_probe*.py; proven on the champion climb, banked indigo_reach_g):
- Phase 1: Viridian west edge -> Route 22. GARY trigger (col 33, armed by badge 8) fires a forced scene ->
  his strongest pre-E4 team; handle_interrupts owns it. A loss whiteouts to Viridian -> the loop re-crosses.
- Phase 2: R22 -> the north-entrance gate (28,0) -> Route 23.
- Phase 3: Route 23 south leg northward through the six badge-guard lockall scenes (all msgbox drains — she
  holds all 8) -> the Victory Road door (5,28) -> VR 1F.
- Phase 4: every VR-floor barrier opens ONLY by pushing a boulder onto its STRENGTH switch. The three-floor
  boulder chain (1F/2F/3F, incl. the 3F row-19-boulder reveal via the (34,18) hole drop) is the hand-solved
  VRnF_PUZZLE constants below. NEVER push boulder (35,13) — it seals the (37,10) pocket.
- Phase 5: VR 2F east pocket -> R23 north -> Indigo Plateau exterior (3,9) -> heal at the League center.

WHITEOUT-TOLERANT: progress RATCHETS (Gary's var, the gauntlet scenes and every VR switch var persist in
the save), so the dispatch loop keys on the CURRENT map every iteration and skips already-open barriers —
a mid-VR whiteout costs a re-cross, never solved ground.

Resume-safe: already at Indigo -> 'reached_indigo'; anywhere on the road/floors -> the map dispatch picks
up from there. run_strike returns:
  'reached_indigo' — at the Indigo Plateau exterior, healed. (Success — the E4 vehicle takes it from here.)
  'battle_loss'    — a fight loss loop the caller's recovery should own (rare; the loop self-recovers most).
  'stuck'          — a wedge cap hit (puzzle/warp/edge) or the deadline. Surfaces LOUD.

EQ teach (Phase 0) is DEFAULT OFF (POKEMON_TEACH_EQ) — recon_victory NS12 proved it did net harm on a
non-EQ kit (flaky TM-case actuation + it could forget Razor Leaf, Venusaur's only Grass STAB). Razor Leaf
x2 carries VR's Water/Rock/Ground; enable only with a clean droppable slot + a fixed teacher.
"""
import os
import time

import travel as tv
import pokemon_state as st
import firered_ram as ram
import field_moves as fm
from dialogue_drive import box_open as dd_box

# ── FireRed Route-22/23 + Victory-Road fact table (game-knowledge layer; rule 14 portability debt) ──────
VIRIDIAN = (3, 1)
R22 = (3, 41)
R23 = (3, 42)
GATE = (28, 0)                       # Route22 North-Entrance gatehouse (group 28)
VR1F, VR2F, VR3F = (1, 39), (1, 40), (1, 41)
INDIGO = (3, 9)
FLAG_BADGE_EARTH = 0x827             # badge 8 — the prereq; also the strike's own preflight guard
FLAG_STR_ACTIVE = 0x805             # Strength armed for the session
TM26_ITEM, MOVE_EQ = 314, 89
TEACH_EQ = os.getenv("POKEMON_TEACH_EQ") == "1"    # default OFF — see module docstring

KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
ARROW_KEY = {0x62: "RIGHT", 0x63: "LEFT", 0x64: "UP", 0x65: "DOWN"}
DIRN_OF = {"south": 1, "north": 2, "west": 3, "east": 4}

# ── the hand-solved, elevation-aware boulder-push puzzles (recon_victory constants, verbatim) ───────────
VR1F_PUZZLE = [("strength", (7, 18)),
               ("push", (7, 18), "DOWN", 1),
               ("push", (7, 19), "RIGHT", 4),
               # stand (11,20) = the entrance arrow tile (0x65: warps on DOWN only; this push presses UP)
               ("push", (11, 19), "UP", 1, ((11, 20),)),
               ("push", (11, 18), "RIGHT", 1),
               ("push", (12, 18), "UP", 1),
               ("push", (12, 17), "RIGHT", 7),
               ("push", (19, 17), "UP", 2),
               ("push", (19, 15), "RIGHT", 1),
               ("push", (20, 15), "DOWN", 1)]      # lands (20,16) = the switch
VR2F_PUZZLE1 = [("strength", (6, 17)),
                ("push", (6, 17), "DOWN", 1),
                ("push", (6, 18), "LEFT", 2),
                ("push", (4, 18), "DOWN", 1),
                ("push", (4, 19), "LEFT", 2)]      # lands (2,19) = switch 1
FLAG_2F_BOULDER_HIDDEN = 0x058     # FLAG_HIDE_VICTORY_ROAD_2F_BOULDER: set = not yet dropped from 3F
VR3F_SWITCH_PUZZLE = [("strength", (32, 5)),
                      ("push", (32, 5), "UP", 2),
                      ("push", (32, 3), "LEFT", 21),
                      ("push", (11, 3), "DOWN", 1),
                      ("push", (11, 4), "LEFT", 5),
                      ("push", (6, 4), "DOWN", 3),
                      ("push", (6, 7), "RIGHT", 1)]  # lands (7,7) = the 3F switch


class VictoryRoad:
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
        self.deadline = time.time() + 3600

    # ── snap / battle / dialogue drains ────────────────────────────────────────────────────────────────
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

    # ── water/edge machinery (seafoam/giovanni verbatim; VR adds boulders to the obstacle set) ───────────
    def water_save(self, g):
        return {(bx - tv.MAP_OFFSET, by - tv.MAP_OFFSET) for bx, by in g.water}

    def sea_ok(self, g, wset):
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

    def live_npc_tiles(self):
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

    def live_boulders(self):
        return [ob["coord"] for ob in fm.scan_field_objects(self.b, {fm.GFX_BOULDER})]

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

    def sea_walk(self, goal_test, label, tries=14, avoid=(), allow=()):
        b = self.b
        budget = tries
        while budget > 0:
            budget -= 1
            if self.handle_interrupts():
                budget += 1
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            g = tv.Grid(b)
            wset = self.water_save(g)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - set(allow)
            npcs = self.live_npc_tiles() | {tuple(o[0]) for o in
                                            tv.read_object_templates(b)
                                            if o[2] and o[1] != fm.GFX_BOULDER}
            npcs |= {tuple(t) for t in self.live_boulders()}
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
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
            if tuple(tv.coords(b) or ()) != cur:
                budget += 1
        return goal_test(tuple(tv.coords(b) or ()))

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
                self.settle(120)
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            band.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]) + round_ * 7)
            tgt = band[min(round_, len(band) - 1)]
            if not self.sea_walk(lambda c, t=tgt: c == t, f"{label}-band"):
                self.settle(120)
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
                    self.settle(120)
                    self.log(f"   [{label}] EDGE {direction}: {m0} -> {tuple(tv.map_id(b))} "
                             f"@ {tv.coords(b)}")
                    return True
        self.log(f"!! [{label}] {direction} crossing never fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def tile_behavior(self, t):
        b = self.b
        try:
            ml = b.rd32(tv.GMAPHEADER)
            attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
            bw = b.rd32(tv.BACKUP_LAYOUT)
            mp0 = b.rd32(tv.BACKUP_LAYOUT + 8)
            mid = b.rd16(mp0 + ((t[1] + tv.MAP_OFFSET) * bw
                                + (t[0] + tv.MAP_OFFSET)) * 2) & 0x3FF
            base, idx = (attr[0], mid) if mid < tv.NUM_PRIMARY else (attr[1],
                                                                     mid - tv.NUM_PRIMARY)
            return b.rd32(base + idx * 4) & 0xFF
        except Exception:
            return 0

    def go_warp(self, tile, dest, label, avoid=()):
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        beh = self.tile_behavior(tile)
        arrow = ARROW_KEY.get(beh)
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in
               ((0, 1), (0, -1), (1, 0), (-1, 0))]
        if arrow:
            d = DELTA[arrow]
            nbs = [(tile[0] - d[0], tile[1] - d[1])]
        for attempt in range(4):
            if tuple(tv.coords(b) or ()) not in nbs and tuple(tv.coords(b) or ()) != tile:
                if not self.sea_walk(lambda c, s=set(nbs): c in s, f"{label}-approach",
                                     avoid=avoid):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile and not arrow:
                # standing ON a plain (beh-0) warp tile that never fires on step — the real exit is walking
                # OUT through the door frame beyond it (the gate-mat class). Try outward-first.
                g0 = tv.Grid(b)
                order = []
                if tile[1] >= g0.sy_hi - 2:
                    order.append("DOWN")
                if tile[1] <= 2:
                    order.append("UP")
                if tile[0] >= g0.sx_hi - 2:
                    order.append("RIGHT")
                if tile[0] <= 2:
                    order.append("LEFT")
                order += [k for k in ("DOWN", "UP", "LEFT", "RIGHT") if k not in order]
                for k2 in order:
                    b.press(k2, 26, 10, camp.render, owner="agent")
                    for _ in range(120):
                        b.run_frame()
                        if tuple(tv.map_id(b)) != m0:
                            break
                    if tuple(tv.map_id(b)) != m0:
                        break
                    if tuple(tv.coords(b) or ()) != tile:      # stepped off — go back
                        self.sea_walk(lambda c, t=tile: c == t, f"{label}-remount",
                                      allow=(tile,))
                if self.handle_interrupts():
                    continue
                if tuple(tv.map_id(b)) == dest:
                    self.settle(180)
                    self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)} (door walk-out)")
                    return True
                if tuple(tv.map_id(b)) != m0:
                    self.log(f"!! [{label}] warped to {tuple(tv.map_id(b))}, wanted {dest}")
                    self.settle(180)
                    return False
                continue
            key = (arrow if arrow and cur == tile
                   else KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1])) or arrow)
            if key is None:
                continue
            for _press in range(4 if arrow else 1):
                b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    break
            if self.handle_interrupts():
                continue
            if tuple(tv.map_id(b)) == dest:
                self.settle(180)
                self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)} (beh {hex(beh)})")
                return True
            if tuple(tv.map_id(b)) != m0:
                self.log(f"!! [{label}] warped to {tuple(tv.map_id(b))}, wanted {dest}")
                self.settle(180)
                return False
        self.log(f"!! [{label}] never fired (at {tv.map_id(b)}@{tv.coords(b)}, beh {hex(beh)})")
        self.snap(f"warpfail_{label[:16]}")
        return False

    def nearest_boulder(self, approx, radius=8):
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
        b, camp = self.b, self.camp
        if fm.read_flag(b, FLAG_STR_ACTIVE):
            return True
        bl = self.nearest_boulder(approx)
        if bl is None:
            self.log(f"!! [strength] no live boulder near {approx} on {tv.map_id(b)}")
            return False
        for attempt in range(3):
            nbs = [(bl[0] + dx, bl[1] + dy) for dx, dy in
                   ((0, 1), (0, -1), (1, 0), (-1, 0))]
            if not self.sea_walk(lambda c, s=set(nbs): c in s, "str-approach"):
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            face = KEY_OF.get((bl[0] - cur[0], bl[1] - cur[1]))
            if face is None:
                continue
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            self.settle(40)
            self.drain(key="A")
            self.settle(60)
            if fm.read_flag(b, FLAG_STR_ACTIVE):
                self.log(f"   [strength] ARMED (flag 0x805) at {tv.coords(b)}")
                return True
        self.log(f"!! [strength] flag 0x805 never set (boulder {bl})")
        self.snap("strength_fail")
        return False

    def push(self, approx, key, n, allow=()):
        b, camp = self.b, self.camp
        d = DELTA[key]
        for i in range(n):
            bl = self.nearest_boulder(approx)
            if bl is None:
                self.log(f"!! [push] boulder near {approx} vanished (i={i})")
                return False
            stand = (bl[0] - d[0], bl[1] - d[1])
            if not self.sea_walk(lambda c, s=stand: c == s, f"push-approach{i}",
                                 avoid={tuple(bl)}, allow=allow):
                self.log(f"!! [push] can't reach {stand} to push {bl} {key}")
                return False
            moved = False
            for _try in range(4):
                if self.handle_interrupts():
                    continue
                b.press(key, 40, 10, camp.render, owner="agent")
                self.settle(70)
                b2l = self.nearest_boulder((bl[0] + d[0], bl[1] + d[1]))
                if b2l != bl:
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

    def run_puzzle(self, ops, barrier_tile, label):
        for op in ops:
            while self.handle_interrupts():
                pass
            self.log(f"-- op {op} (map {tv.map_id(self.b)} @ {tv.coords(self.b)})")
            if op[0] == "strength":
                if not self.ensure_strength(op[1]):
                    return False
            elif op[0] == "push":
                if not self.push(op[1], op[2], op[3], allow=op[4] if len(op) > 4 else ()):
                    return False
        self.settle(150)                                   # switch scene (SE + map redraw)
        self.drain()
        g_now = tv.Grid(self.b)
        opened = g_now.col.get((barrier_tile[0] + tv.MAP_OFFSET,
                                barrier_tile[1] + tv.MAP_OFFSET), 1) == 0
        self.log(f"   [{label}] barrier {barrier_tile} open={opened}")
        self.snap(f"{label}_done")
        return opened

    def barrier_open(self, tile):
        g = tv.Grid(self.b)
        return g.col.get((tile[0] + tv.MAP_OFFSET, tile[1] + tv.MAP_OFFSET), 1) == 0

    def puzzle2_2f(self):
        # the row-19 boulder may sit ANYWHERE x14..33 after a partial chain — find it, push LEFT the
        # remaining distance onto the (14,19) switch
        if not self.ensure_strength((33, 19)):
            return False
        bl = None
        for ax in (33, 27, 21, 16):
            c = self.nearest_boulder((ax, 19), radius=6)
            if c and c[1] == 19 and 14 <= c[0] <= 33:
                bl = c
                break
        if bl is None:
            self.log("!! [2f-sw2] no boulder on row 19 — 3F reset detour needed")
            return False
        if bl[0] == 14:
            return True
        return self.push(bl, "LEFT", bl[0] - 14)

    def wedge(self, label, cap=4):
        b = self.b
        self.wedges[label] = self.wedges.get(label, 0) + 1
        # NS9: a post-battle EVOLUTION box (won mid-VR) JAMS overworld nav — dd_box does NOT flag it, so
        # drain()/handle_interrupts skip it and the go_warp step loop presses into a dead box forever. RAW
        # press-through (B, ungated by dd_box) on the early wedges to punch past it, then abort if stuck.
        if self.wedges[label] < cap:
            for _ in range(20):
                b.press("B", 8, 12, self.camp.render, owner="agent")
            return False
        self.log(f"!! [{label}] failed x{cap} — abort LOUD")
        self.snap(f"wedge_{label[:14]}")
        return True

    def lead_frac(self):
        b = self.b
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def badge8(self):
        return fm.read_flag(self.b, FLAG_BADGE_EARTH)

    # ── PHASE 0: teach EARTHQUAKE (TM26 -> Venusaur, over a droppable slot) — DEFAULT OFF ────────────────
    def _teach_eq(self):
        b, camp = self.b, self.camp
        eq_slot = next((s for s in range(6) if st.read_party_species(b, s) == 3), None)
        have = (st.read_party_moves(b, eq_slot) or []) if eq_slot is not None else []
        if eq_slot is None:
            self.log("   EQ teach SKIPPED — Venusaur (species 3) not in party")
            return
        if MOVE_EQ in have:
            self.log(f"   EQ already known (Venusaur slot {eq_slot}) — skipping teach")
            return
        if not TEACH_EQ:
            self.log(f"   EQ teach SKIPPED (POKEMON_TEACH_EQ off) — Venusaur keeps {have}; Razor Leaf "
                     f"carries VR")
            return
        try:
            import hm_teach as ht
            _PROTECT = {75, 70, MOVE_EQ}       # Razor Leaf, Strength (boulders), EQ
            forget_idx = next((have.index(m) for m in (15, 290) if m in have), None)
            if forget_idx is None:
                forget_idx = next((i for i, m in enumerate(have) if m not in _PROTECT), None)
            self.log(f"   [teach-eq] forget_idx={forget_idx} (moves before={have})")
            teacher = ht.TeachFlow(camp, log=self.log)
            r = teacher.teach("surf", eq_slot, forget_idx=forget_idx,
                              item_override=TM26_ITEM, move_override=MOVE_EQ)
            after = st.read_party_moves(b, eq_slot) or []
            self.log(f"   [teach-eq] -> {r}; moves now {after} (EQ={'YES' if MOVE_EQ in after else 'NO'})")
            self.drain(key="B")
            self.settle(60)
        except Exception as e:
            self.log(f"   [teach-eq] errored: {e} — continuing without EQ (LOUD)")

    # ── the strike ───────────────────────────────────────────────────────────────────────────────────
    def run(self):
        b, camp = self.b, self.camp
        self.log(f"   victory road strike: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"badge8={self.badge8()} lead={self.lead_frac():.0%}")
        if not self.badge8():
            self.log("!! badge 8 not held — wrong state, abort (this is a post-badge-8 vehicle)")
            return "stuck"
        if tuple(tv.map_id(b)) == INDIGO:
            self.log("   already at the Indigo Plateau — nothing to strike")
            return "reached_indigo"

        self._teach_eq()

        # ── PHASES 1-5: ONE WHITEOUT-TOLERANT DISPATCH LOOP (map-keyed; progress ratchets in the save) ──
        r23_logged = vr_logged = False
        while time.time() < self.deadline:
            if self.handle_interrupts():
                continue
            here = tuple(tv.map_id(b))
            if here == INDIGO:
                break
            if here == VIRIDIAN:
                if self.lead_frac() < 0.9:
                    camp.heal_nearest()
                    continue
                if not self.cross_edge("west", "to-r22") and self.wedge("viridian-west"):
                    return "stuck"
            elif here == R22:
                # westward crosses Gary's trigger col 33 (scene + battle fire mid-path; handle_interrupts
                # owns them; a loss whiteouts and this loop recovers via the center)
                if not self.go_warp((8, 5), GATE, "gate-south"):
                    if tuple(tv.map_id(b)) == R22 and self.wedge("gate-south"):
                        self.snap("gate_fail")
                        return "stuck"
            elif here == GATE:
                cands = [tuple(xy) for xy, d, _w in tv.read_warps(b) if tuple(d) == R23]
                if not cands:
                    self.log("!! no R23 warp inside the gate — abort")
                    return "stuck"
                cands.sort(key=lambda t: t[1])            # north side = lowest y
                if not self.go_warp(cands[0], R23, "gate-thru"):
                    self.drain()
                    if self.wedge("gate-thru", 6):
                        return "stuck"
            elif here == R23:
                cy = (tv.coords(b) or (0, 0))[1]
                if cy <= 30:                              # north side (past VR)
                    if not self.cross_edge("north", "to-indigo") and self.wedge("r23-north"):
                        return "stuck"
                else:
                    if not r23_logged:
                        self.log(f"   ROUTE 23 @ {tv.coords(b)} after {self.n_battles} battles "
                                 f"(Gary handled en route) [lead {self.lead_frac():.0%}]")
                        r23_logged = True
                    if not self.go_warp((5, 28), VR1F, "vr-door") and self.wedge("vr-door"):
                        return "stuck"
            elif here == VR1F:
                if not vr_logged:
                    self.log(f"   VICTORY ROAD 1F @ {tv.coords(b)} [lead {self.lead_frac():.0%}]")
                    vr_logged = True
                if not self.barrier_open((12, 14)):
                    if not self.run_puzzle(VR1F_PUZZLE, (12, 14), "1f-switch") \
                            and self.wedge("1f-switch", 3):
                        return "stuck"
                elif not self.go_warp((3, 2), VR2F, "1f-ladder") and self.wedge("1f-ladder"):
                    return "stuck"
            elif here == VR2F:
                cx, cy = tuple(tv.coords(b) or (0, 0))
                if cx >= 36 and cy <= 13:                 # east pocket (from 3F drop)
                    if (self.go_warp((48, 12), R23, "vr-exit")
                            or self.go_warp((47, 13), R23, "vr-exit-b")
                            or self.go_warp((49, 13), R23, "vr-exit-c")):
                        self.log(f"   VICTORY ROAD CLEARED -> R23 north @ {tv.coords(b)} "
                                 f"[lead {self.lead_frac():.0%}, battles {self.n_battles}]")
                    elif self.wedge("vr-exit"):
                        return "stuck"
                elif not self.barrier_open((13, 10)):
                    if not self.run_puzzle(VR2F_PUZZLE1, (13, 10), "2f-switch1") \
                            and self.wedge("2f-switch1", 3):
                        return "stuck"
                elif not self.barrier_open((33, 16)):
                    if fm.read_flag(b, FLAG_2F_BOULDER_HIDDEN):
                        # the row-19 boulder hasn't dropped from 3F yet — up the (34,9) ladder
                        if not self.go_warp((34, 9), VR3F, "2f-to-3f-detour") \
                                and self.wedge("2f-to-3f-detour"):
                            return "stuck"
                    elif self.puzzle2_2f():
                        self.settle(150)
                        self.drain()
                        self.log(f"   [2f-switch2] barrier (33,16) open={self.barrier_open((33, 16))}")
                    elif self.wedge("2f-switch2", 3):
                        return "stuck"
                elif not self.go_warp((36, 17), VR3F, "2f-to-3f") and self.wedge("2f-to-3f"):
                    return "stuck"
            elif here == VR3F:
                if fm.read_flag(b, FLAG_2F_BOULDER_HIDDEN):
                    # THE RESET/REVEAL DETOUR — leg 1: boulder (32,5) -> switch (7,7) opens the 3F barrier
                    # (12,12-13); leg 2: push (33,18) RIGHT into hole (34,18) (drops it to 2F (33,19), clears
                    # 0x058), then jump in after it -> lands 2F (34,19). NEVER push boulder (35,13).
                    if not self.barrier_open((12, 12)):
                        if not self.run_puzzle(VR3F_SWITCH_PUZZLE, (12, 12), "3f-switch") \
                                and self.wedge("3f-switch", 3):
                            return "stuck"
                    elif not self.ensure_strength((33, 18)):
                        if self.wedge("3f-drop-strength"):
                            return "stuck"
                    elif not self.push((33, 18), "RIGHT", 1):
                        if self.wedge("3f-drop"):
                            return "stuck"
                    else:
                        self.settle(120)
                        self.log(f"   [3f-drop] boulder down the hole (34,18) — 2F boulder "
                                 f"hidden={fm.read_flag(b, FLAG_2F_BOULDER_HIDDEN)}")
                        # the hole is a warp tile (0x66) — jump in after it
                        if not self.sea_walk(lambda c: c == (34, 18), "hole-jump",
                                             allow=((34, 18),)) and self.wedge("hole-jump"):
                            return "stuck"
                        self.settle(180)
                elif not self.go_warp((37, 10), VR2F, "3f-to-2f-east",
                                      # Ray+Tyra (38,13)/(39,13) are a trainerbattle_DOUBLE the battle agent
                                      # can't target — dodge their sight tiles via column 36
                                      avoid=((38, 14), (39, 14))) and self.wedge("3f-to-2f"):
                    return "stuck"
            else:
                # off-route (whiteout center interior, etc.) — exit to the overworld
                self.log(f"   off-route at {here} — exiting to the overworld")
                camp.enter_warp(prefer="south")
                self.settle(80)

        if tuple(tv.map_id(b)) != INDIGO:
            self.log(f"!! never reached Indigo Plateau (at {tv.map_id(b)}@{tv.coords(b)}) — deadline/exhausted")
            self.snap("70_fail")
            return "stuck"

        self.log(f"   INDIGO PLATEAU @ {tv.coords(b)} — healing at the League center")
        try:
            camp.heal_nearest()
        except Exception as e:
            self.log(f"   Indigo heal errored: {e} — continuing (LOUD)")
        self.drain(key="B")
        self.log(f"   INDIGO REACHED: pos {tv.map_id(b)}@{tv.coords(b)} | lead {self.lead_frac():.0%} | "
                 f"battles {self.n_battles} | money ${camp.money()}")
        self.snap("80_final")
        return "reached_indigo"


def run_strike(camp, log, dbg_dir=None):
    """Run the Victory-Road push (post-badge-8 -> Indigo Plateau) from wherever she stands, in ONE call.
    Resume-safe (map-keyed dispatch; progress ratchets in the save). Drives Viridian -> Route 22 (Gary) ->
    the gate -> Route 23 -> the Victory-Road boulder floors -> Route 23 north -> Indigo, heals at the League
    center. Returns 'reached_indigo' (success) | 'battle_loss' | 'stuck'."""
    return VictoryRoad(camp, log, dbg_dir).run()
