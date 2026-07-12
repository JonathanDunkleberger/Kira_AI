"""mansion_strike.py — THE SECRET KEY, in-loop (night shift, Blaine/badge-7 build).

The gate to Blaine's Cinnabar Gym: the gym door is LOCKED until the Secret Key is taken from the
Pokémon Mansion B1F. Post-Seafoam she's on Cinnabar but can't enter Blaine's gym — she must clear the
Mansion switch-puzzle, grab the key, and walk back out. The champion cleared it with the bespoke
recon_mansion.py strike; this is a FAITHFUL port of that proven script, driven by the live `camp` so the
general questline can call it as ONE decision (the same shape as beat_gym / safari_strike / silph_strike /
seafoam_strike). FireRed coords are isolated here (rule 14 — portability debt: a Kanto-Mansion fact table +
the statue-toggle MISSION script, swap per game).

Prereq (recognition-enforced): on Cinnabar (== the seafoam_strike OUTPUT state).

THE DERIVATION (recon_mansion.py header, all pret ground truth — layout bins + pokemon_mansion.inc): the
Mansion is ONE global switch state (FLAG 0x26C) whose statues toggle barrier sets on all four floors
(setmetatile arg 4 IS the collision bit). Route to the key (nodes = floor x state x region, edges = warps +
statue toggles):
  1F entrance -> TOGGLE statue (12,5) ON -> stairs -> 2F -> 3F -> fall (18,18) -> 1F balcony -> stair (25,27)
  -> B1F -> TOGGLE (24,29) OFF -> TOGGLE (27,5) ON -> SECRET KEY ball (5,7) -> re-open toggles -> exit stair
  (34,29) -> 1F -> SE back door (34,33) -> Cinnabar.
Statue actuation: face + A -> "press the switch?" YES (default) -> flag 0x26C flips + DrawWholeMapView
applies collision instantly. Key pickup verified by FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY (0x1A8).
Floors: 1F (1,59) 2F (1,60) 3F (1,61) B1F (1,62). Wilds (Koffing/Grimer/Rattata) + mansion trainers ride
the BattleAgent via the interrupt path.

STATUE ACTUATION NUANCE (recon_mansion): the Mansion statues are BG_EVENT_PLAYER_FACING_NORTH — they fire
ONLY from BELOW, facing UP; so each toggle op stands at (x, y+1) (the `stand` param) and toggle ops are
idempotent (skip if the switch is already in the wanted state).

run_strike returns 'got_key' | 'in_mansion' (key obtained but the exit didn't complete) |
'not_here' (no strike from here) | 'failed'.
"""
import os
import time

import firered_ram as ram
import field_moves as fm
import travel as tv
from dialogue_drive import box_open as dd_box

# ── FireRed Pokémon-Mansion fact table (game-knowledge layer; rule 14 portability debt) ────────────
CINNABAR = (3, 8)
M1F, M2F, M3F, MB1F = (1, 59), (1, 60), (1, 61), (1, 62)
MANSION_MAPS = {M1F, M2F, M3F, MB1F}
# Anchors for the questline strike dispatcher: Cinnabar (the strike STARTS there — the mansion front door
# is off the Cinnabar overworld) + every interior floor (so a mid-tour re-tick RESUMES the switch mission
# rather than dumping her).
MANSION_ANCHORS = {CINNABAR} | MANSION_MAPS
FLAG_SWITCH = 0x26C                  # the global Mansion switch state (statues toggle it)
FLAG_KEY_TAKEN = 0x1A8              # FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY — the crossing-done signal
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
# arrow warps (0x62-0x65) + DIRECTIONAL STAIR WARPS (0x6C-0x6F: UP_RIGHT / UP_LEFT / DOWN_RIGHT /
# DOWN_LEFT — fire only when WALKED INTO along their direction; the recon5 (10,13) wedge stood ON one).
ARROW_KEY = {0x62: "RIGHT", 0x63: "LEFT", 0x64: "UP", 0x65: "DOWN",
             0x6C: "RIGHT", 0x6D: "LEFT", 0x6E: "RIGHT", 0x6F: "LEFT"}
OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# (op, args...) — coords are floor-local SAVE coords. Verbatim from recon_mansion.py (proven on the
# champion climb). CORRECTED (recon6 + anchor truth): a warp EVENT with plain behavior is a LANDING ANCHOR
# and never fires (2F (27,17) = the 3F fall-holes' landing). Route uses only trigger-behavior tiles: stairs
# 0x6C/0x6F, falls 0x66, arrows 0x65.
MISSION = [
    ("door", (8, 3), M1F),          # mansion front door (Cinnabar overworld)
    ("door", (10, 13), M2F),        # 0x6C up-right stair
    ("door", (9, 3), M3F),          # 0x6C up-right stair
    ("toggle", (12, 5), True),      # 3F statue
    ("door", (18, 18), M1F),        # 0x66 fall hole -> 1F balcony (19,22)
    ("door", (25, 27), MB1F),       # 0x6F down-left stair
    ("toggle", (24, 29), False),
    ("toggle", (27, 5), True),
    ("pickup", (5, 7)),
    ("toggle", (27, 5), False),     # re-open the way back (ON seals the stair side)
    ("toggle", (24, 29), True),     # 1F SE pocket needs ON (Press opens (27-29,25))
    ("door", (34, 29), M1F),        # 0x6C stair up
    ("door", (34, 33), CINNABAR),   # the SE BACK DOOR (0x65) — the front hall is
]                                   # unreachable from the stair pocket in EITHER state


class MansionStrike:
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
        self.deadline = time.time() + 1800

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
        """'battle' / 'box' / False — walk() refunds budget ONLY for battles (a box that reopens every
        cycle must BURN budget or the loop has infinite fuel — the recon1 (20,6) 527-replan wedge)."""
        if self.fight_open():
            self.fight()
            self.drain()
            return "battle"
        if dd_box(self.b):
            self.drain()
            return "box"
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

    # ── navigation primitives (recon_mansion verbatim) ─────────────────────────────────────────────
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

    def coord_event_tiles(self):
        """Script-trigger tiles (MapHeader.events coordEvents, stride 0x10) — the recon1-3 Cinnabar wedge:
        (20,5) fires GymDoorLocked, boxes + bounces her; BFS must route AROUND script tiles like warps."""
        b = self.b
        try:
            ev = b.rd32(tv.GMAPHEADER + 0x04)
            n = b.rd8(ev + 0x02)
            arr = b.rd32(ev + 0x0C)
            if not (0 < n <= 32) or arr < 0x08000000:
                return set()
            return {(b.rds16(arr + i * 0x10), b.rds16(arr + i * 0x10 + 2))
                    for i in range(n)}
        except Exception:
            return set()

    def step_to(self, tile):
        b, camp = self.b, self.camp
        cur = tuple(tv.coords(b) or (0, 0))
        d = (tile[0] - cur[0], tile[1] - cur[1])
        if d in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            d = (d[0] // 2, d[1] // 2)
        key = KEY_OF.get(d)
        if key is None:
            return camp._step_to(tile)
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

    def walk(self, goal_test, label, tries=10, avoid=()):
        b = self.b
        budget = tries
        stuck = [None, 0]                       # same-coord replan wedge detector
        while budget > 0:
            budget -= 1
            it = self.handle_interrupts()
            if it == "battle":
                budget += 1
                continue
            if it:
                continue                        # box drained: burns budget
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            if cur == stuck[0]:
                stuck[1] += 1
                if stuck[1] >= 4:
                    self.log(f"!! [{label}] WEDGE: 4 same-coord replans at {cur} "
                             f"(dd_box={dd_box(b)} fight={self.fight_open()})")
                    self.snap(f"wedge_{label[:12]}_{cur[0]}_{cur[1]}")
                    return False
            else:
                stuck[0], stuck[1] = cur, 0
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            npcs = self.live_npc_tiles() | self.coord_event_tiles() | {
                tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: g.walkable(sx, sy) and (sx, sy) not in wts
                       and (sx, sy) not in npcs and (sx, sy) not in avoid)
            self.log(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                self.log(f"   [{label}] no path from {cur}")
                self.snap(f"nopath_{label[:12]}_{cur[0]}_{cur[1]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                it2 = self.handle_interrupts()
                if it2:
                    self.log(f"   [{label}] interrupt={it2} at {tuple(tv.coords(b) or ())} "
                             f"before step {tuple(t)}")
                    if it2 == "battle":
                        budget += 1
                    break
                if not self.step_to(tuple(t)):
                    self.log(f"   [{label}] step blocked {tuple(tv.coords(b) or ())} -> {tuple(t)}")
                    self.snap(f"blocked_{t[0]}_{t[1]}")
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
            if tuple(tv.coords(b) or ()) != cur:
                budget += 1
        return goal_test(tuple(tv.coords(b) or ()))

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

    def go_warp(self, tile, dest, label):
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
            if arrow and tuple(tv.coords(b) or ()) == tile:
                b.press(OPP[arrow], 26, 10, camp.render, owner="agent")  # step OFF first
                self.settle(40)
            if tuple(tv.coords(b) or ()) not in nbs and tuple(tv.coords(b) or ()) != tile:
                if not self.walk(lambda c, s=set(nbs): c in s, f"{label}-approach"):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
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

    def interact(self, tile, label, key="A", verify=None, tries=3, stand=None):
        """Face `tile` from an adjacent square, press A, drain (A = YES default). `stand`: required standing
        tile (the Mansion statues are bg events with BG_EVENT_PLAYER_FACING_NORTH — they fire ONLY from
        below, facing UP; recon4 interacted from the east forever)."""
        b, camp = self.b, self.camp
        nbs = ([stand] if stand else
               [(tile[0] + dx, tile[1] + dy) for dx, dy in
                ((0, 1), (0, -1), (1, 0), (-1, 0))])
        for attempt in range(tries):
            if tuple(tv.coords(b) or ()) not in nbs:
                if not self.walk(lambda c, s=set(nbs): c in s, f"{label}-approach", avoid={tile}):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            face = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            if face is None:
                continue
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            self.settle(30)
            self.drain(key=key)
            self.settle(30)
            if verify is None or verify():
                return True
            self.log(f"   [{label}] verify failed (attempt {attempt + 1})")
        self.log(f"!! [{label}] interaction never verified")
        self.snap(f"interact_fail_{label[:14]}")
        return False

    # ── the mission ────────────────────────────────────────────────────────────────────────────────
    def run(self):
        b, camp = self.b, self.camp
        here = tuple(tv.map_id(b))
        self.log(f"   mansion strike: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"money=${camp.money()} switch={int(fm.read_flag(b, FLAG_SWITCH))} "
                 f"key_taken={int(fm.read_flag(b, FLAG_KEY_TAKEN))}")

        key_already = fm.read_flag(b, FLAG_KEY_TAKEN)

        # PREFLIGHT (mirror recon_mansion 407-418): anchors guarantee she's on Cinnabar or an interior
        # floor. If she already has the key AND is back on Cinnabar, she's done — go straight to arrival.
        if key_already and here == CINNABAR:
            self.log("   Secret Key already in bag and back on Cinnabar — nothing to strike")
            return self._arrive_cinnabar()

        # Pre-mission heal — ONLY when starting fresh from Cinnabar (recon_mansion healed only after the
        # on-Cinnabar preflight; a mid-dungeon heal_nearest could route her out unexpectedly).
        if here == CINNABAR:
            try:
                r = camp.heal_nearest()
                self.log(f"   pre-mission heal_nearest -> {r}")
            except Exception as e:
                self.log(f"   pre-mission heal errored: {e} — entering as-is (LOUD)")
        while self.handle_interrupts():
            pass

        # RESUME-SAFE: if the key is already in the bag, skip the acquisition ops and run only the exit
        # tail (post-pickup toggles that re-open the way + the two doors back to Cinnabar) — mirror
        # seafoam's skip-to-exit on FLAG_B3F_CALM. Toggle ops are idempotent (skip if already set).
        if key_already:
            pick = next(i for i, op in enumerate(MISSION) if op[0] == "pickup")
            ops = MISSION[pick + 1:]
            self.log("   RESUME: Secret Key already in bag — running the exit tail only")
        else:
            ops = MISSION

        for op in ops:
            if time.time() > self.deadline:
                self.log("!! deadline inside the mansion")
                return self._interior_bail()
            while self.handle_interrupts():
                pass
            self.log(f"-- op {op} (map {tv.map_id(b)} @ {tv.coords(b)})")
            kind = op[0]
            if kind == "door":
                if not self.go_warp(op[1], op[2], f"door{op[1]}"):
                    return self._interior_bail()
            elif kind == "toggle":
                want = op[2]
                if fm.read_flag(b, FLAG_SWITCH) == want:
                    self.log(f"   [toggle] switch already {want}")
                    continue
                if not self.interact(op[1], f"statue{op[1]}", key="A",
                                     verify=lambda w=want: fm.read_flag(b, FLAG_SWITCH) == w,
                                     stand=(op[1][0], op[1][1] + 1)):    # FACING_NORTH class
                    return self._interior_bail()
                self.log(f"   [toggle] switch -> {int(fm.read_flag(b, FLAG_SWITCH))}")
            elif kind == "pickup":
                if not self.interact(op[1], "secret-key", key="B",
                                     verify=lambda: fm.read_flag(b, FLAG_KEY_TAKEN)):
                    return self._interior_bail()
                self.log("   [pickup] SECRET KEY in bag (flag 0x1A8)")

        # tour walked to its end — verify she's back on Cinnabar with the key.
        here = tuple(tv.map_id(b))
        if here == CINNABAR and fm.read_flag(b, FLAG_KEY_TAKEN):
            return self._arrive_cinnabar()
        return self._interior_bail()

    def _interior_bail(self):
        """A MISSION op failed. If the key is already taken, the objective is done — report in_mansion so
        the caller keeps the inside marker and retries the exit next tick; else it's a real failure."""
        if fm.read_flag(self.b, FLAG_KEY_TAKEN):
            self.log("   mansion op failed AFTER the key was taken (0x1A8 set) — exit owns the rest")
            return "in_mansion"
        return "failed"

    def _arrive_cinnabar(self):
        b, camp = self.b, self.camp
        self.log(f"   OUT with the key @ {tv.map_id(b)}{tv.coords(b)} after {self.n_battles} battles — healing")
        try:
            r = camp.heal_nearest()
            self.log(f"   heal_nearest -> {r}")
        except Exception as e:
            self.log(f"   Cinnabar arrival heal errored: {e} — continuing")
        self.snap("90_key_out")
        self.log(f"   MANSION DONE: key_taken={int(fm.read_flag(b, FLAG_KEY_TAKEN))} "
                 f"pos {tv.map_id(b)}@{tv.coords(b)} | battles {self.n_battles}")
        return "got_key"


def run_strike(camp, log, dbg_dir=None):
    """Run the Pokémon Mansion Secret-Key tour (Cinnabar -> Mansion switch-puzzle -> B1F key -> back out to
    Cinnabar) from wherever she stands, in ONE call. Returns:
      'got_key'    — Secret Key in bag (FLAG_KEY_TAKEN 0x1A8 set) and back on Cinnabar (healed); Blaine's
                     gym door is now openable.
      'in_mansion' — key obtained but the exit didn't complete (still inside / bounced); the objective is
                     done, the caller keeps the inside marker and retries the exit next tick.
      'not_here'   — no strike applies from here (caller falls through to the general layer).
      'failed'     — strike aborted before the key was obtained.
    Idempotent by state: already on Cinnabar with the key short-circuits to the arrival; a mid-tour re-tick
    with the key already taken resumes at the exit tail."""
    try:
        here = tuple(tv.map_id(camp.b))
    except Exception:
        return "failed"
    if here not in MANSION_ANCHORS:
        return "not_here"
    return MansionStrike(camp, log, dbg_dir).run()
