"""blaine_gym.py — THE CINNABAR GYM STRIKE (Blaine / badge 7), in-loop (night shift).

A FAITHFUL port of the proven recon_blaine.py strike into an in-loop module driven by the live `camp`
bridge, so the general questline can call it as ONE decision (the same shape as beat_gym / mansion_strike /
safari_strike / seafoam_strike). FireRed coords are isolated here (rule 14 — portability debt: a Cinnabar-
gym quiz fact table, swap per game).

Prereq (recognition-enforced): on Cinnabar island (== the mansion_strike OUTPUT state, Secret Key in bag so
Blaine's gym door is openable).

Ground truth (pret CinnabarGym_scripts.inc + CinnabarGym.json): the gym is SIX quiz machines (FACING_NORTH
bg-event pairs); each door opens on EITHER the correct YES/NO answer OR beating that room's trainer (a wrong
answer walks the trainer to you, and winning fires the same QuizNComplete) — fail-safe both ways, a botched
press only costs a battle we win. Correct answers: Q1 YES, Q2 NO, Q3 NO, Q4 NO, Q5 YES, Q6 NO. B advances
plain msgboxes AND selects NO on a YES/NO box, so each station drains on ONE key.

Blaine (5,4) face DOWN -> front (5,5) face UP; Arcanine L47 tops his roster. Post-win flags: 0x4B6 defeated +
0x826 BADGE 7; TM38 gift A-drains (no Y/N). Success = flag 0x826.

⚠️ THE BILL AMBUSH: beating Blaine sets VAR_MAP_SCENE_CINNABAR=1; the first transition back onto the island
fires a FORCED scene — Bill (spawn (20,7), the gym doorstep) runs up with a YES/NO "sail to One Island?"
where A/YES ships her to the Sevii Islands = an OFF-MAINLINE CATASTROPHE. The post-badge island drain is
B-ONLY (island_b_drain — B advances every box and answers NO) until stable, then heal.

run_gym returns one of the beat_gym contract strings: 'badge' | 'needs_heal' | 'battle_loss' | 'stuck'.
"""
import os
import time

import travel as tv
import pokemon_state as st
import firered_ram as ram
import field_moves as fm
from dialogue_drive import box_open as dd_box

# ── FireRed Cinnabar-Gym fact table (game-knowledge layer; rule 14 portability debt) ────────────────
ISLAND = (3, 8)
GYM = (12, 0)
BLAINE_FRONT = (5, 5)                # Blaine (5,4), face UP
FLAG_BADGE_VOLCANO = 0x826
FLAG_DEFEATED_BLAINE = 0x4B6
GYM_ENTRY_MATS = {(24, 23), (25, 23), (26, 23)}

# (label, quiz flag, [front tiles: stand here + face UP], drain key for the CORRECT answer)
QUIZ_CHAIN = [
    ("quiz1", 0x265, [(22, 11), (23, 11)], "A"),   # YES
    ("quiz2", 0x267, [(15, 3), (16, 3)], "B"),     # NO
    ("quiz3", 0x268, [(13, 11), (14, 11)], "B"),   # NO
    ("quiz4", 0x269, [(13, 18), (14, 18)], "B"),   # NO
    ("quiz5", 0x26A, [(1, 19), (2, 19)], "A"),     # YES
    ("quiz6", 0x26B, [(1, 11), (2, 11)], "B"),     # NO
]

# ── navigation key maps (mansion_strike machinery) ──────────────────────────────────────────────────
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
ARROW_KEY = {0x62: "RIGHT", 0x63: "LEFT", 0x64: "UP", 0x65: "DOWN",
             0x6C: "RIGHT", 0x6D: "LEFT", 0x6E: "RIGHT", 0x6F: "LEFT"}
OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


class BlaineGym:
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

    def drain(self, max_a=40, key="A"):
        """recon_blaine drain: B advances plain msgboxes AND answers NO on a YES/NO box, so a station
        drains on ONE key (the CORRECT answer's key). st.in_battle short-circuits (a wrong answer walks
        the room's trainer to you -> we win it via the interrupt path)."""
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_a):
            if st.in_battle(b):
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
        cycle must BURN budget or the loop has infinite fuel)."""
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

    # ── navigation primitives (mansion_strike verbatim) ────────────────────────────────────────────
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
        """Script-trigger tiles (MapHeader.events coordEvents, stride 0x10): the locked-door coord event
        (20,5) boxes + bounces her — BFS must route AROUND script tiles like warps."""
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

    # ── recon_blaine tour closures (ported to methods) ─────────────────────────────────────────────
    def badge(self):
        return fm.read_flag(self.b, FLAG_BADGE_VOLCANO)

    def lead_frac(self):
        b = self.b
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def walk_path_to(self, tile, label, tries=6):
        """The Silph/Sabrina deterministic same-map mover: static BFS, warps + template-NPC bodies
        masked; battles recompute; a step that fails outside battle after the spotting-wait is DEAD for
        this call. NO per-tile elevation law here — the opened quiz doorways are setmetatile'd to
        collision-0 elevation-0, and elev 0 is the game's wildcard; Grid.edge_open already enforces the
        real per-EDGE elevation rule inside bfs."""
        b, camp = self.b, self.camp
        dead = set()
        for _ in range(tries):
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile:
                return True
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
            npcs = ({tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
                    | dead) - {tile}
            p = tv.bfs(g, cur, lambda t: t == tile,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs)
            if not p:
                self.log(f"   [{label}] no NPC-free static path {cur} -> {tile} "
                         f"(dead={sorted(dead)})")
                return False
            for t in p[1:]:
                ok = camp._step_to(tuple(t))
                if st.in_battle(b):
                    self.log(f"   [{label}] battle mid-path -> {self.fight()}")
                    self.drain()
                    break
                if not ok:
                    for _ in range(120):
                        b.run_frame()
                    if dd_box(b):
                        self.drain()
                    if st.in_battle(b):
                        self.log(f"   [{label}] step was a trainer spotting -> {self.fight()}")
                        self.drain()
                        break
                    dead.add(tuple(t))
                    self.log(f"   [{label}] step into {tuple(t)} failed — dead-marked, recompute")
                    break
            if tuple(tv.coords(b) or ()) == tile:
                return True
        return tuple(tv.coords(b) or ()) == tile

    def engage(self, front, face, label, drains=1, key="A"):
        b, camp = self.b, self.camp
        if not self.walk_path_to(front, label):
            self.log(f"!! [{label}] couldn't reach {front} (at {tv.coords(b)})")
            return "unreached"
        out = "nothing"
        for _ in range(8):
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if st.in_battle(b):
                self.log(f"   [{label}] battle -> {self.fight()}")
                self.drain()
                return "battled"
            if dd_box(b):
                out = "talked"
                for _k in range(drains):
                    self.drain(key=key)
                    if st.in_battle(b):
                        self.log(f"   [{label}] battle -> {self.fight()}")
                        self.drain()
                        return "battled"
                    for _ in range(40):
                        b.run_frame()
                break
        return out

    def enter_to(self, dest, label):
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == dest]
        if not cands:
            self.log(f"!! [{label}] no warp on {m0} leads to {dest}")
            return False
        cur = tuple(tv.coords(b) or (0, 0))
        if cur in cands:
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            g6 = tv.Grid(b)
            for nb in ((cur[0], cur[1] + 1), (cur[0] + 1, cur[1]),
                       (cur[0] - 1, cur[1]), (cur[0], cur[1] - 1)):
                if nb not in wts and g6.walkable(nb[0], nb[1]):
                    camp._step_to(nb)
                    cur = tuple(tv.coords(b) or (0, 0))
                    break
        cands.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        for wt in cands:
            r = camp.enter_warp(pick=wt)
            if r == "need_heal":
                self.log(f"   [{label}] heal interrupt — healing, then retrying")
                camp.heal_nearest()
                r = camp.enter_warp(pick=wt)
            if st.in_battle(b):
                self.log(f"   [{label}] battle on approach -> {self.fight()}")
                self.drain()
                r = camp.enter_warp(pick=wt)
            if tuple(tv.map_id(b)) == dest:
                for _ in range(80):
                    b.run_frame()
                self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)}")
                return True
        self.log(f"!! [{label}] no candidate warp fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def island_b_drain(self, label, seconds=25.0):
        """THE BILL-AMBUSH DRAIN: on the island the on-frame scene locks her, walks Bill up, and asks a
        YES/NO where YES sails to One Island. B advances every box and answers NO. Patience windows cover
        the applymovement walks between boxes."""
        b, camp = self.b, self.camp
        end = time.time() + seconds
        quiet = 0
        while time.time() < end:
            if dd_box(b):
                quiet = 0
                b.press("B", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                quiet += 1
                for _ in range(45):
                    b.run_frame()
                if quiet >= 8:                      # ~6s of silence = the scene is over
                    break
        self.log(f"   [{label}] island B-drain done (quiet={quiet}, at {tv.coords(b)})")

    # ── the strike ─────────────────────────────────────────────────────────────────────────────────
    def run(self):
        b, camp = self.b, self.camp
        self.log(f"   blaine strike: boot map={tv.map_id(b)} coords={tv.coords(b)} "
                 f"badge7={self.badge()} lead={self.lead_frac():.0%} "
                 f"key_flag={fm.read_flag(b, 0x1A8)} "
                 f"blaine_done={fm.read_flag(b, FLAG_DEFEATED_BLAINE)}")
        if self.badge():
            self.log("   Volcano Badge already held — nothing to strike")
            return "badge"

        while time.time() < self.deadline and not self.badge():
            here = tuple(tv.map_id(b))
            if here == ISLAND:
                if self.lead_frac() < 0.6:
                    self.log(f"   lead at {self.lead_frac():.0%} — healing at the Cinnabar Center first")
                    camp.heal_nearest()
                    continue
                if not self.enter_to(GYM, "gym-door"):
                    self.wedges["door"] = self.wedges.get("door", 0) + 1
                    if self.wedges["door"] >= 2:
                        # the locked-door coord event only dies when OnTransition ran with the key flag —
                        # re-fire it via a Center round-trip, then retry
                        self.log("   gym door bounced — re-firing OnTransition via the Center")
                        camp.heal_nearest()
                    if self.wedges["door"] >= 4:
                        self.snap("10_no_gym_door")
                        self.log("!! can't enter the gym x4 — abort LOUD")
                        return "stuck"
                    self.drain(key="B")
                continue
            if here != GYM:
                self.log(f"   off-route at {here} (whiteout/heal interior?) — exiting to the overworld")
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
                if tuple(tv.map_id(b)) == here:
                    self.log(f"!! stuck off-route at {here}@{tv.coords(b)} — abort")
                    self.snap("11_offroute")
                    return "stuck"
                continue

            # hurt mid-gym: leave, heal, come back (flags/doors persist; trainers stay beaten)
            if self.lead_frac() < 0.5:
                self.wedges["heal_exit"] = self.wedges.get("heal_exit", 0) + 1
                if self.wedges["heal_exit"] >= 3:
                    self.snap("30_heal_exhaust")
                    self.log("   left the gym to heal x3 without a win — needs_heal (doors stay open)")
                    return "needs_heal"
                self.log(f"   lead at {self.lead_frac():.0%} — leaving to heal (doors stay open)")
                self.walk_path_to((25, 22), "heal-exit")
                self.enter_to(ISLAND, "exit-to-heal")
                continue

            # ── THE QUIZ CHAIN: open every door not already open ──
            progressed = False
            blocked = False
            for label, flag, fronts, key in QUIZ_CHAIN:
                if fm.read_flag(b, flag):
                    continue
                self.log(f"   [{label}] door closed — engaging (answer key {key})")
                done = False
                for front in fronts:
                    r = self.engage(front, "UP", label, drains=4, key=key)
                    self.drain(key=key)
                    for _ in range(60):
                        b.run_frame()
                    if fm.read_flag(b, flag):
                        self.log(f"   [{label}] DOOR OPEN ({r}) [lead {self.lead_frac():.0%}]")
                        done = True
                        break
                    self.log(f"   [{label}] engaged from {front} -> {r}, flag still unset")
                if not done:
                    self.wedges[label] = self.wedges.get(label, 0) + 1
                    if self.wedges[label] >= 3:
                        self.snap(f"20_{label}_wedge")
                        self.log(f"!! [{label}] door never opened x3 — abort LOUD")
                        return "stuck"
                    blocked = True
                progressed = True
                break                                   # re-evaluate the chain from the top
            if progressed:
                if blocked:
                    self.drain(key="B")
                continue

            # all six doors open — Blaine
            self.log(f"   quiz doors ALL OPEN — engaging Blaine [lead {self.lead_frac():.0%}]")
            nb0 = self.n_battles
            r = self.engage(BLAINE_FRONT, "UP", "blaine", drains=6)
            self.drain()
            for _ in range(240):                        # badge fanfare + TM38 gift pacing
                b.run_frame()
                if dd_box(b):
                    b.press("A", 8, 12, camp.render, owner="agent")
            self.drain()
            self.log(f"   BLAINE -> {r} (battles {nb0}->{self.n_battles}) badge7={self.badge()} "
                     f"lead={self.lead_frac():.0%}")
            if self.badge():
                self.snap("40_badge7")
                break
            if tuple(tv.map_id(b)) != GYM:
                # map left the gym interior with no badge -> she whited out to the city PC (a leader loss).
                # Propagate so the caller's blackout-recovery respawns + re-runs the gym (doors persist,
                # beaten juniors stay beaten). Matches beat_gym's leader-loss -> 'battle_loss'.
                self.log("   not in the gym after Blaine (whiteout) — leader loss; caller retries the gym")
                self.snap("55_blaine_blackout")
                return "battle_loss"
            self.wedges["blaine"] = self.wedges.get("blaine", 0) + 1
            if self.wedges["blaine"] >= 3:
                self.snap("50_blaine_wedge")
                self.log("!! Blaine engaged x3 without a badge — abort LOUD")
                return "stuck"

        if not self.badge():
            self.log(f"!! Volcano Badge NOT won (at {tv.map_id(b)}@{tv.coords(b)}) — deadline/exhausted")
            self.snap("70_fail")
            return "stuck"

        return self._exit_and_bill()

    def _exit_and_bill(self):
        """Post-badge: WALK OUT to the island + THE BILL AMBUSH B-DRAIN (A/YES ships her to Sevii — a
        catastrophe), then heal, then report the badge. If the walk-out is incomplete, still B-drain
        anyway (the badge holds; recovery owns the exit)."""
        b, camp = self.b, self.camp
        self.log("   badge in hand — walking out (B-ONLY on the island: the Bill ambush)")
        out_deadline = time.time() + 300
        while tuple(tv.map_id(b)) != ISLAND and time.time() < out_deadline:
            here = tuple(tv.map_id(b))
            if here == GYM:
                mats = sorted(GYM_ENTRY_MATS,
                              key=lambda t: abs(t[0] - (tv.coords(b) or (25, 12))[0]))
                tgt = (mats[0][0], mats[0][1] - 1)   # tile just above the mat row
                self.walk_path_to(tgt, "walk-out")
                if not self.enter_to(ISLAND, "out-door"):
                    self.drain(key="B")
            else:
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
        if tuple(tv.map_id(b)) != ISLAND:
            self.log(f"!! walk-out incomplete (at {tv.map_id(b)}) — the badge holds; recovery owns the exit")
        else:
            self.island_b_drain("bill-ambush")
            self.snap("60_post_bill")
            try:
                camp.heal_nearest()
            except Exception as e:
                self.log(f"   post-Bill heal errored: {e} — continuing (LOUD)")
            self.island_b_drain("post-heal", seconds=8.0)

        self.log(f"   VOLCANO BADGE: flag={self.badge()} | pos {tv.map_id(b)}@{tv.coords(b)} | "
                 f"lead {self.lead_frac():.0%} | battles {self.n_battles}")
        self.snap("80_final")
        return "badge"


def run_gym(camp, log, dbg_dir=None):
    """Run the Cinnabar-Gym (Blaine / badge 7) strike from wherever she stands, in ONE call. Walks the
    quiz-door chain (6 FACING_NORTH machines, correct answers Q1 YES / Q2-4 NO / Q5 YES / Q6 NO or beat
    the room's trainer — fail-safe both ways), beats Blaine, drives the badge/TM38 award, then WALKS OUT
    and B-drains the Bill ambush (A/YES would sail her to the Sevii Islands — off-mainline catastrophe),
    and heals. Returns one of the beat_gym contract strings:
      'badge'       — Blaine beaten, FLAG_BADGE_VOLCANO (0x826) set, Bill ambush B-drained, healed, back
                      on Cinnabar island. (Success.)
      'needs_heal'  — PP-famine / can't-win-now that a Center heal + retry fixes (juniors/doors stay done).
      'battle_loss' — blacked out mid-gym (map left the gym interior unexpectedly / lead fainted); caller
                      recovers + re-runs the gym.
      'stuck'       — a door never opened x3 / can't enter the gym x4 / off-route stuck / deadline."""
    return BlaineGym(camp, log, dbg_dir).run()
