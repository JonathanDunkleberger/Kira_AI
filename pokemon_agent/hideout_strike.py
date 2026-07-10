"""hideout_strike.py — THE ROCKET HIDEOUT STRIKE, in-loop (night shift #7).

The general questline lands her INSIDE the Celadon Game Corner (10,14) for the Silph Scope errand
(door-hint (34,21)->(10,14) works), but the blind room-tour + GO-DEEPER can NOT solve the hideout:
a fixed multi-floor RITUAL sits behind a POSTER-gate — press the poster to unlock the stairs, cross
arrow-tile spin mazes down to B4F, grab the Lift Key a grunt drops, climb back to the B2F elevator,
ride it into the sealed boss corridor, beat Giovanni, take the Scope, then ride/climb back OUT to the
Celadon street. The champion cleared this with two bespoke strike scripts (recon_hideout.py descent +
recon_hideout_exit.py exit); this is a FAITHFUL port of both, driven by the live `camp` so the general
questline can call it as ONE decision (the same shape as beat_gym). FireRed coords are isolated here
(rule 14 — portability debt: a Kanto-Rocket-Hideout fact table, swap per game).

Ground truth (disasm + the champion's proven runs):
  Game Corner (10,14): grunt (11,2) guards poster (11,1) — beat him, press the poster UP -> the
  stairs (15,2) unlock -> B1F (1,42). Descent stairs B1F->B4F: (17,2),(21,2),(15,18). B4F (1,45):
  Lift Key (id 356) from a grunt (4,2)/ball ~(3,2). Climb B4F->B3F (11,15) -> B2F (18,2); board the
  B2F elevator (28-29,16); ride DOWN to the B4F BOSS corridor (arrive ~(18-23,23)): door grunts
  (19,14)/(16,14), GIOVANNI (19,4) face UP, Silph Scope ball (20,5). Success = item 359 in Key Items
  (the 0x037 flag LIES — confirm the ITEM). EXIT: board B4F elevator (20-21,23), ride UP to B2F (1,43),
  spin-cross to (27,3), up-stairs (28,2) -> B1F (1,42), (12,2) -> GC (10,14), south mats -> Celadon.
"""
import os

import field_moves as fm
import firered_ram as ram
import hm_teach as ht
import pokemon_state as st
import travel as tv
from dialogue_drive import box_open as dd_box
from spin_nav import SpinNav

# ── FireRed Rocket-Hideout fact table (game-knowledge layer; rule 14 portability debt) ──────────
CELADON = (3, 6)
GC_DOOR = (34, 21)                     # Celadon -> Game Corner
GC = (10, 14)
GRUNT_FRONT, POSTER_FRONT = (11, 3), (11, 2)   # face UP at each
GC_STAIRS = (15, 2)                    # unlocked by the poster -> B1F
DESCENT = [(17, 2), (21, 2), (15, 18)]  # B1F->B2F->B3F->B4F down-stairs (ridden in order)
LIFT_KEY = 356
SILPH_SCOPE = 359
GIOVANNI_FRONT = (19, 5)               # boss at (19,4), face UP
# map ids (exit-script ground truth)
B4F, B3F, B2F, B1F, ELEV = (1, 45), (1, 44), (1, 43), (1, 42), (1, 46)
HIDEOUT_MAPS = {B1F, B2F, B3F, B4F, ELEV}


class HideoutStrike:
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
        self._nav = SpinNav(self.b, camp, camp.battle_runner, self.drain, log=log)

    # ── low-level helpers (ported verbatim from recon_hideout.py) ───────────────────────────────
    def fight(self):
        return self.camp.battle_runner()

    def key_items(self):
        return ht.pocket_items(self.b, ht.KEY_ITEMS_OFF, 30)

    def drain(self, max_a=30, key="A"):
        """Advance/close any open box; stop when stably closed."""
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

    def snap(self, name):
        if not self.dbg:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(os.path.join(self.dbg, name + ".png"))
        except Exception as e:
            self.log(f"   snap {name} failed: {e}")

    def goto(self, tile, label):
        b, camp = self.b, self.camp
        r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=200, max_seconds=90)
        if st.in_battle(b):                       # LoS trainer triggered en route
            self.log(f"   [{label}] battle en route -> {self.fight()}")
            self.drain()
            r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=200, max_seconds=90)
        if r != "arrived" and not camp._step_to(tile):
            return False
        return tuple(tv.coords(b) or ()) == tile

    def engage(self, front, face, label):
        """Stand at `front`, face `face`, press A; run the battle if one starts; drain text.
        Returns 'battled' | 'talked' | 'nothing'."""
        b, camp = self.b, self.camp
        if not self.goto(front, label):
            self.log(f"!! [{label}] couldn't reach {front} (at {tv.coords(b)})")
            return "nothing"
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
                self.drain()
                if st.in_battle(b):               # the talk escalated into the fight
                    self.log(f"   [{label}] battle -> {self.fight()}")
                    self.drain()
                    return "battled"
                break
        return out

    def spin_cross(self, done, label, rounds=3):
        return bool(self._nav.cross(done, label, rounds=rounds))

    # ── phase 1: descend the Game Corner poster gate down to the Scope ───────────────────────────
    def descend_and_take_scope(self):
        b, camp, L = self.b, self.camp, self.log

        # inside the Game Corner: the grunt guarding the poster, then the poster
        r = self.engage(GRUNT_FRONT, "UP", "poster-grunt")
        L(f"   grunt engage -> {r}")
        self.snap("10_post_grunt")
        for _ in range(3):
            r2 = self.engage(POSTER_FRONT, "UP", "poster")
            L(f"   poster press -> {r2}")
            self.drain()
            if r2 == "talked":
                break
            for _ in range(60):
                b.run_frame()
        self.snap("20_post_poster")

        # the revealed stairs -> B1F
        m0 = tuple(tv.map_id(b))
        if camp.enter_warp(pick=GC_STAIRS) != "warped" or tuple(tv.map_id(b)) == m0:
            self.snap("30_stairs_fail")
            L(f"!! stairs {GC_STAIRS} didn't warp (poster gate still shut?) — abort")
            return "failed"
        for _ in range(80):
            b.run_frame()
        L(f"   B1F: map={tv.map_id(b)} coords={tv.coords(b)}")

        # descend B1F -> B4F on the billed stairs (spin-cross fallback per floor)
        for i, stairs in enumerate(DESCENT, 1):
            m0 = tuple(tv.map_id(b))
            r = camp.enter_warp(pick=stairs)
            if r == "need_heal":
                L("   heal interrupt mid-floor — healing")
                camp.heal_nearest()
                r = camp.enter_warp(pick=stairs)
            if r != "warped" or tuple(tv.map_id(b)) == m0:
                L(f"   floor {i}: plain route to {stairs} failed — engaging the spin-tile crosser")
                if not self.spin_cross(lambda t, s=stairs: abs(t[0] - s[0]) + abs(t[1] - s[1]) <= 1,
                                       f"floor{i}-spin"):
                    self.snap(f"40_floor{i}_fail")
                    L(f"!! floor {i}: spin crossing failed from {tv.coords(b)} — abort")
                    return "failed"
                r = camp.enter_warp(pick=stairs)
            if r != "warped" or tuple(tv.map_id(b)) == m0:
                self.snap(f"40_floor{i}_fail")
                L(f"!! floor {i}: stairs {stairs} didn't warp from {m0}@{tv.coords(b)} — abort")
                return "failed"
            for _ in range(80):
                b.run_frame()
            L(f"   floor {i + 1} down: map={tv.map_id(b)} coords={tv.coords(b)}")

        # LIFT KEY on B4F — read the ball coord LIVE + beat the guarding grunt via the proven
        # LoS-retrigger. The old hardcoded fronts (grunt (4,3); ball fronts (3,3)/(4,2)/(2,2))
        # WEDGED (shift 7, frame-confirmed 55_liftkey_fail.png): all three billed ball-fronts were
        # unwalkable (two into decorative plants, one onto the STILL-STANDING grunt), and the grunt
        # only "talked" — a zero-step arrival opens his greeting, never the fight, so he never
        # cleared. Generalized below (fm.item_balls + camp._engage_trainer).
        import field_moves as fm
        self.snap("50_b4f")
        L(f"   B4F items={fm.item_balls(b)} trainers={camp._gym_trainers()} at {tv.coords(b)}")
        if LIFT_KEY not in self.key_items():
            if not self._grab_lift_key():
                self.snap("55_liftkey_fail")
                L(f"!! no Lift Key (key_items={self.key_items()}) — abort")
                return "failed"
        L(f"   LIFT KEY in bag: {self.key_items()}")

        # climb B4F->B3F (11,15) -> B2F (18,2)
        for stairs_up, label in (((11, 15), "B4F->B3F"), ((18, 2), "B3F->B2F")):
            m0 = tuple(tv.map_id(b))
            r = camp.enter_warp(pick=stairs_up)
            if r != "warped" or tuple(tv.map_id(b)) == m0:
                if not self.spin_cross(lambda t, s=stairs_up: abs(t[0] - s[0]) + abs(t[1] - s[1]) <= 1,
                                       f"{label}-spin"):
                    L(f"!! {label}: couldn't reach up-stairs {stairs_up} — abort")
                    return "failed"
                r = camp.enter_warp(pick=stairs_up)
            if r != "warped" or tuple(tv.map_id(b)) == m0:
                L(f"!! {label}: stairs {stairs_up} didn't warp — abort")
                return "failed"
            for _ in range(80):
                b.run_frame()
            L(f"   {label}: map={tv.map_id(b)} coords={tv.coords(b)}")

        # board the B2F elevator (28-29,16) via the spin maze, ride DOWN to the B4F boss corridor
        if not (self._board_elevator((28, 16)) or self._board_elevator((29, 16))):
            self.snap("57_no_elevator")
            L("!! couldn't board the B2F elevator — abort")
            return "failed"
        L(f"   in the elevator: map={tv.map_id(b)} coords={tv.coords(b)}")

        landed_b4f = False
        for downs in (2, 1, 0, 3):
            if not self._ride(downs, "DOWN"):
                L(f"   ride(downs={downs}): no exit — retrying")
                continue
            L(f"   ride(downs={downs}) landed: map={tv.map_id(b)} coords={tv.coords(b)}")
            cx, cy = tuple(tv.coords(b) or (0, 0))
            if abs(cy - 23) <= 3 and 18 <= cx <= 23:            # B4F elevator alcove
                landed_b4f = True
                break
            camp.enter_warp(prefer="nearest")                   # wrong floor: re-board
            for _ in range(80):
                b.run_frame()
        if not landed_b4f:
            self.snap("58_wrong_floor")
            L(f"!! elevator never landed in the B4F corridor (at {tv.map_id(b)}@{tv.coords(b)}) — abort")
            return "failed"

        # boss-door grunts, then GIOVANNI, then the Scope
        self.snap("59_corridor")
        for gx, gy in ((19, 14), (16, 14)):
            rg = self.engage((gx, gy + 1), "UP", f"door-grunt{gx}")
            L(f"   door grunt ({gx},{gy}) -> {rg}")
        # GIOVANNI — retry the engage: the beaten door-grunt BODIES wall the direct path so the
        # first goto to (19,5) fails from (16,15), but a re-travel from her shifted position DOES
        # reach it (shift 7, run4-confirmed: reached (19,5) on the 2nd approach from (20,7)). The
        # Scope only appears (script-given or a ball at (20,5)) AFTER he's beaten.
        for gtry in range(4):
            if SILPH_SCOPE in self.key_items():
                break
            r = self.engage(GIOVANNI_FRONT, "UP", "giovanni")
            L(f"   giovanni engage[{gtry}] -> {r}")
            self.drain()
            if r == "battled":
                break
        self.snap("60_post_giovanni")

        # the Scope: his beat-script usually hands it over; else grab the ball he leaves at (20,5)
        if SILPH_SCOPE not in self.key_items():
            for front, face in (((19, 5), "RIGHT"), ((20, 6), "UP"), ((21, 5), "LEFT"), ((20, 4), "DOWN")):
                self.engage(front, face, "scope-ball")
                if SILPH_SCOPE in self.key_items():
                    break
        got = SILPH_SCOPE in self.key_items()
        L(f"   SILPH SCOPE in Key Items: {got} (key_items={self.key_items()})")
        self.snap("70_scope")
        return "got_scope" if got else "failed"

    def _grab_ball(self, ball):
        """Reach one item ball and press A into it. Tries plain goto to each adjacent stand tile
        first, then the SPIN-TILE crosser (B4F's west region is an arrow-tile spinner maze that
        partitions the ball off — plain travel wedges at the ball's stand tile, shift 7). True iff
        an item box opened (bag change verified by the caller)."""
        b, camp = self.b, self.camp
        bx, by = ball["coord"]
        try:
            g = tv.Grid(b)
        except Exception:
            g = None
        # plain approach to each adjacent stand tile
        for d, key in fm._DELTA_KEY.items():
            sx, sy = bx - d[0], by - d[1]
            if g is not None and not g.walkable(sx, sy):
                continue
            if self.goto((sx, sy), "liftkey-ball"):
                b.press(key, 8, 10, camp.render, owner="agent")
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(30):
                    b.run_frame()
                self.drain()
                return True
        # spinner-partitioned: glide-cross to an adjacent tile, then face the ball and press
        if self.spin_cross(lambda t: abs(t[0] - bx) + abs(t[1] - by) == 1, "liftkey-ball-spin", rounds=4):
            cx, cy = tuple(tv.coords(b) or (0, 0))
            face = fm._DELTA_KEY.get((bx - cx, by - cy))
            if face:
                b.press(face, 8, 10, camp.render, owner="agent")
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(30):
                    b.run_frame()
                self.drain()
                return True
        return False

    def _grab_lift_key(self):
        """B4F Lift Key: grab each item ball (via _grab_ball's spin-aware approach), and if a ball
        stays unreachable, beat the nearest un-beaten trainer (LoS-retrigger) and retry. All coords
        read LIVE (fm.item_balls / _gym_trainers). Dumps the B4F grid once for the record."""
        b, camp, L = self.b, self.camp, self.log
        self.dump_grid("b4f")
        beaten = set()
        for _rnd in range(6):
            if LIFT_KEY in self.key_items():
                return True
            for ball in fm.item_balls(b):
                self._grab_ball(ball)
                if LIFT_KEY in self.key_items():
                    return True
            if LIFT_KEY in self.key_items():
                return True
            trs = [(i, c, f) for (i, c, f) in camp._gym_trainers() if i not in beaten]
            if not trs:
                break
            px, py = tv.coords(b) or (0, 0)
            trs.sort(key=lambda t: abs(t[1][0] - px) + abs(t[1][1] - py))
            idx, T, facing = trs[0]
            beaten.add(idx)
            L(f"   liftkey: engaging grunt obj{idx} at {T} (facing {facing})")
            camp._engage_trainer(T, facing)
            if st.in_battle(b):
                L(f"   liftkey grunt battle -> {self.fight()}")
                self.drain()
        return LIFT_KEY in self.key_items()

    def dump_grid(self, tag):
        """Behavior/walkability grid of the CURRENT map -> dbg dir, for maze postmortems."""
        if not self.dbg:
            return
        b, camp = self.b, self.camp
        try:
            g = tv.Grid(b)
            npc = set(camp.trav._npc_tiles())
            w_play = b.rd32(tv.BACKUP_LAYOUT) - 14
            h_play = b.rd32(tv.BACKUP_LAYOUT + 4) - 14
            lines = [f"map={tv.map_id(b)} coords={tv.coords(b)} dims={w_play}x{h_play} "
                     f"npcs={sorted(npc)}  (N=npc #=wall .=plain hex=behavior)"]
            for y in range(h_play):
                row = []
                for x in range(w_play):
                    v = camp._tile_behavior(x, y)
                    row.append(" N " if (x, y) in npc else
                               " # " if not g.walkable(x, y) else
                               (f"{v:02x} " if v else " . "))
                lines.append(f"y{y:02d} " + "".join(row))
            p = os.path.join(self.dbg, f"grid_{tag}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.log(f"   grid dumped -> {p}")
        except Exception as e:
            self.log(f"   grid dump failed: {e}")

    def _board_elevator(self, door_pick):
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        ax, ay = door_pick[0], door_pick[1] + 1
        self.spin_cross(lambda t: abs(t[0] - ax) + abs(t[1] - ay) <= 1, "to-elevator", rounds=4)
        camp.enter_warp(pick=door_pick)
        for _ in range(80):
            b.run_frame()
        return tuple(tv.map_id(b)) != m0

    def _ride(self, presses, key):
        """Board assumed done. Panel bg (0,2) (face (1,2)LEFT / (0,3)UP); floor multichoice
        (`presses` of `key`); exit the nearest door. self-correct on the LANDING."""
        b, camp = self.b, self.camp
        opened = False
        for front, face in (((1, 2), "LEFT"), ((0, 3), "UP"), ((0, 2), "UP")):
            if not self.goto(front, "panel"):
                continue
            for _ in range(4):
                b.press(face, 8, 10, camp.render, owner="agent")
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(40):
                    b.run_frame()
                    if dd_box(b):
                        opened = True
                        break
                if opened:
                    break
            if opened:
                break
        if not opened:
            return False
        b.press("A", 8, 12, camp.render, owner="agent")     # advance "Which floor?" -> menu
        for _ in range(30):
            b.run_frame()
        for _ in range(presses):
            b.press(key, 8, 10, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
        b.press("A", 8, 12, camp.render, owner="agent")
        self.drain()
        for _ in range(300):                                # the ride/shake
            b.run_frame()
        m0 = tuple(tv.map_id(b))
        camp.enter_warp(prefer="nearest")
        for _ in range(80):
            b.run_frame()
        return tuple(tv.map_id(b)) != m0

    # ── phase 2: walk her back OUT to the Celadon street (ported from recon_hideout_exit.py) ─────
    def exit_to_celadon(self):
        b, camp, L = self.b, self.camp, self.log
        L(f"   exit boot map={tv.map_id(b)} coords={tv.coords(b)}")
        if tuple(tv.map_id(b)) == B4F:
            boarded = False
            for door in ((20, 23), (21, 23)):
                m0 = tuple(tv.map_id(b))
                if camp.enter_warp(pick=door) == "warped" and tuple(tv.map_id(b)) != m0:
                    boarded = True
                    break
            if not boarded:
                self.snap("exit_10_no_board")
                L("!! couldn't board the B4F elevator — exit abort")
                return "failed"
            L(f"   in the elevator: {tv.map_id(b)}@{tv.coords(b)}")
            landed_b2f = False
            for ups in (1, 2, 0, 3):
                if not self._ride(ups, "UP"):
                    L(f"   ride(ups={ups}): no exit — retrying")
                    continue
                L(f"   ride(ups={ups}) landed: map={tv.map_id(b)} coords={tv.coords(b)}")
                if tuple(tv.map_id(b)) == B2F:
                    landed_b2f = True
                    break
                camp.enter_warp(prefer="nearest")
                for _ in range(80):
                    b.run_frame()
            if not landed_b2f:
                self.snap("exit_20_wrong_floor")
                L(f"!! never landed on B2F (at {tv.map_id(b)}@{tv.coords(b)}) — exit abort")
                return "failed"
        if tuple(tv.map_id(b)) == B2F:
            if not self.spin_cross(lambda t: abs(t[0] - 27) + abs(t[1] - 3) <= 1, "exit-b2f-top"):
                self.snap("exit_30_no_top")
                L("!! couldn't cross B2F back to the top corridor — exit abort")
                return "failed"
            m0 = tuple(tv.map_id(b))
            if camp.enter_warp(pick=(28, 2)) != "warped" or tuple(tv.map_id(b)) == m0:
                L("!! B2F up-stairs (28,2) didn't warp — exit abort")
                return "failed"
            for _ in range(80):
                b.run_frame()
            L(f"   B1F: {tv.map_id(b)}@{tv.coords(b)}")
        if tuple(tv.map_id(b)) == B1F:
            m0 = tuple(tv.map_id(b))
            if camp.enter_warp(pick=(12, 2)) != "warped" or tuple(tv.map_id(b)) == m0:
                L("!! B1F -> Game Corner stairs (12,2) didn't warp — exit abort")
                return "failed"
            for _ in range(80):
                b.run_frame()
            L(f"   Game Corner: {tv.map_id(b)}@{tv.coords(b)}")
        if tuple(tv.map_id(b)) == GC:
            m0 = tuple(tv.map_id(b))
            r = camp.enter_warp(prefer="south")
            if r != "warped" or tuple(tv.map_id(b)) == m0:
                for door in ((10, 13), (9, 13), (11, 13)):
                    if camp.enter_warp(pick=door) == "warped" and tuple(tv.map_id(b)) != m0:
                        break
            for _ in range(80):
                b.run_frame()
            L(f"   out: {tv.map_id(b)}@{tv.coords(b)}")
        if tuple(tv.map_id(b)) != CELADON:
            self.snap("exit_40_not_out")
            L(f"!! not on Celadon street (at {tv.map_id(b)}@{tv.coords(b)}) — exit incomplete")
            return "failed"
        hr = camp.heal_nearest()
        L(f"   heal -> {hr}; now {tv.map_id(b)}@{tv.coords(b)}")
        self.snap("exit_50_celadon")
        return "out"


def run_strike(camp, log, dbg_dir=None):
    """Run the Rocket Hideout strike from wherever she stands, in ONE call. Returns:
      'got_scope' — Silph Scope in Key Items AND back out on the Celadon street (healed).
      'in_hideout' — Scope in bag but the exit didn't complete (she's still below; caller retries).
      'not_here'  — no strike applies from here (caller falls through to the general layer).
      'failed'    — strike aborted mid-way (caller surfaces / recovery reacts).
    Idempotent by state: already-holding-and-out short-circuits; a partial run resumes by map."""
    b = camp.b
    hs = HideoutStrike(camp, log, dbg_dir=dbg_dir)
    here = tuple(tv.map_id(b))
    have = SILPH_SCOPE in hs.key_items()

    # already have the Scope: just get out (or already out)
    if have:
        if here == CELADON:
            return "not_here"                 # done — nothing to strike
        if here in HIDEOUT_MAPS or here == GC:
            return "got_scope" if hs.exit_to_celadon() == "out" else "in_hideout"
        return "not_here"                     # holding it and somewhere else — leave it to the loop

    # need the Scope: HEAL TO FULL first (HP + PP — a Center restores both), THEN descend. There is
    # NO Center below Giovanni's 3-mon boss fight, and the banked frontier can enter with a worn
    # lead (erika_done: Venusaur 76/135 = 56%). A worn lead + dungeon chip damage gets swept — the
    # loss the run4/5 STALL blamed on "underlevel" was really an un-healed L43 Venusaur (shift 7).
    try:
        lead_cur = b.rd16(ram.GPLAYER_PARTY + 0x56)
        lead_max = b.rd16(ram.GPLAYER_PARTY + 0x58)
        frac = (lead_cur / lead_max) if lead_max else 1.0
    except Exception:
        frac = 0.0
    if here in (GC, CELADON) and frac < 0.95:
        hr = camp.heal_nearest()
        log(f"   hideout strike: pre-dungeon heal (lead {frac:.0%}) -> {hr} "
            f"(now {tv.map_id(b)}@{tv.coords(b)})")
        here = tuple(tv.map_id(b))

    # descend from Celadon / the Game Corner
    if here == CELADON:
        m0 = here
        if camp.enter_warp(pick=GC_DOOR) != "warped" or tuple(tv.map_id(b)) == m0:
            log("!! hideout strike: couldn't enter the Game Corner from Celadon — abort")
            return "failed"
        for _ in range(80):
            b.run_frame()
        here = tuple(tv.map_id(b))
    if here == GC:
        r = hs.descend_and_take_scope()
        if r != "got_scope":
            return "failed"
        return "got_scope" if hs.exit_to_celadon() == "out" else "in_hideout"
    if here in HIDEOUT_MAPS:
        # mid-descent without the Scope (a prior tick failed partway): the descent isn't cleanly
        # resumable from an arbitrary floor — surface so the loop climbs her out and re-enters the GC.
        log(f"   hideout strike: mid-hideout at {here} without the Scope — not cleanly resumable; surfacing")
        return "failed"
    return "not_here"
