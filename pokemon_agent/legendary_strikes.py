"""legendary_strikes.py — the LEGENDARY HUNTS (2026-08-04, Jonny: "i want her catching mew or
mewtwo as a final endgame project. and all cool legendaries that are available before or after
the final 4 so she mops the floor with them").

Three hunts, each a walk-to-the-static-and-press-A mission; the BATTLE side is already solved
(battle_agent's _LEGENDARY_SPECIES careful-capture divert: weaken, never KO, throw balls — now
tiered best-ball-first, Master Ball allowed on Mewtwo only). This module is only the ROAD.

  ZAPDOS   — Power Plant (1,95) at (5,11). Route 10's water strip guards the door (7,40):
             sea_walk crosses it (giovanni_gym machinery verbatim). Pre-E4 the moment Surf
             is taught.
  ARTICUNO — Seafoam B4F (1,87) at (9,2). Gated on the boulders being TRULY down (hide flags
             0x046/0x047 cleared — the stamped 0x2D2 from the R21 reroute is NOT enough: the
             B4F water still rips without the fallen boulders, see seafoam_strike). Descends
             the ladder chain from either F1 door; candidates per floor, live BFS picks.
  MEWTWO   — Cerulean Cave B1F (1,74) at (7,12). THE final endgame project: the cave guard
             only steps aside post-champion (FLAG_SYS_GAME_CLEAR 0x82C). Cerulean's cave
             mouth (1,12) is across the river; inside is L46-70 wilds — her E4-winning team's
             victory lap.
  (Mew is EVENT-ONLY distribution hardware — not obtainable in FRLG by playing; Mewtwo IS
  the Mew-class prize this cartridge can give.)

GROUND TRUTH (pret/pokefirered map JSONs + flags.h, fetched 2026-08-04):
  Zapdos   : OBJ (5,11) PowerPlant | FLAG_HIDE_ZAPDOS 0x05D | FLAG_FOUGHT_ZAPDOS 0x2BF
             doors: R10 (7,40)->PLANT dw1 | exits (4,39)/(5,38)/(6,39)->R10
  Articuno : OBJ (9,2) B4F | FLAG_HIDE_ARTICUNO 0x082 | FLAG_FOUGHT_ARTICUNO 0x2BE
             descent candidates (down-ladders per floor, west chain first):
             F1 (10,6)/(28,19)/(31,4) -> B1F | B1F (7,3)/(17,9)/(32,14)/(25,19) -> B2F |
             B2F (7,17)/(31,17)/(32,4) -> B3F | B3F (6,18)/(9,18)/(29,5)/(12,9) -> B4F
  Mewtwo   : OBJ (7,12) CeruleanCave_B1F | FLAG_HIDE_MEWTWO 0x081 | FLAG_FOUGHT_MEWTWO 0x2BC
             mouth: Cerulean (1,12)->Cave1F dw0 (lands (33,21)) | 1F (1,7)->B1F (lands (5,7))
             1F exit (33,21)->Cerulean

SUCCESS SEMANTICS: 'caught' (dex owned bit) | 'battled' (FOUGHT/HIDE flag set — the encounter
is SPENT: KO'd statics never respawn, so the step must satisfy either way or she'd loop on a
ghost) | 'not_here' | 'failed' (fail-safe: leaves her fightable/walkable, the errand retries).
"""
import time

import field_moves as fm
import firered_ram as ram
import travel as tv
from giovanni_gym import GiovanniGym, KEY_OF

# ── fact table (game-knowledge layer; rule 14 portability debt) ────────────────────────────
R10, PLANT = (3, 28), (1, 95)
CERULEAN = (3, 3)
CAVE1F, CAVE2F, CAVEB1F = (1, 72), (1, 73), (1, 74)
F1, B1F, B2F, B3F, B4F = (1, 83), (1, 84), (1, 85), (1, 86), (1, 87)
R20 = (3, 38)

FLAG_SYS_GAME_CLEAR = 0x82C          # champion — the Cerulean Cave guard steps aside
FLAG_B3F_CALM = 0x2D2                # crossing signal (may be STAMPED by the R21 reroute!)
FLAG_HIDE_B3F_BOULDER_1 = 0x046      # cleared = boulder truly fell -> B4F water is safe
FLAG_HIDE_B3F_BOULDER_2 = 0x047

ZAPDOS = dict(name="Zapdos", species=145, tile=(5, 11), map=PLANT, hide=0x05D, fought=0x2BF)
ARTICUNO = dict(name="Articuno", species=144, tile=(9, 2), map=B4F, hide=0x082, fought=0x2BE)
MEWTWO = dict(name="Mewtwo", species=150, tile=(7, 12), map=CAVEB1F, hide=0x081, fought=0x2BC)

ZAPDOS_ANCHORS = {R10, PLANT}
ARTICUNO_ANCHORS = {R20, F1, B1F, B2F, B3F, B4F}
MEWTWO_ANCHORS = {CERULEAN, CAVE1F, CAVE2F, CAVEB1F}

# (floor, [candidate down-warp tiles in preference order], dest floor) — the eevee_fetch _ride
# doctrine: the leg for WHEREVER she stands, live BFS decides which candidate is reachable.
ARTICUNO_DESCENT = [
    (F1, [(10, 6), (28, 19), (31, 4)], B1F),
    (B1F, [(7, 3), (17, 9), (32, 14), (25, 19)], B2F),
    (B2F, [(7, 17), (31, 17), (32, 4)], B3F),
    (B3F, [(6, 18), (9, 18), (29, 5), (12, 9)], B4F),
]


class LegendaryHunt(GiovanniGym):
    """Shared hunt chassis on the giovanni_gym base: fight/drain/sea_walk/mount/step_to are
    the PROVEN water-aware primitives (recon_seafoam verbatim, live on the badge-8 sea road).
    Subclasses set QUARRY and implement run()."""

    QUARRY = None

    def __init__(self, camp, log, dbg_dir=None):
        super().__init__(camp, log, dbg_dir)
        self.deadline = time.time() + 1500

    # ── outcome truth ────────────────────────────────────────────────────────────────────
    def spent(self):
        """The encounter no longer exists to start: caught, or battled-and-gone."""
        q = self.QUARRY
        try:
            if ram.pokedex_owns(self.b, q["species"]) is True:
                return True
            return bool(fm.read_flag(self.b, q["fought"]) or fm.read_flag(self.b, q["hide"]))
        except Exception:
            return False

    def outcome(self):
        q = self.QUARRY
        try:
            if ram.pokedex_owns(self.b, q["species"]) is True:
                return "caught"
            if fm.read_flag(self.b, q["fought"]) or fm.read_flag(self.b, q["hide"]):
                return "battled"
        except Exception:
            pass
        return None

    # ── movement ─────────────────────────────────────────────────────────────────────────
    def enter_step(self, tile, dest, label):
        """Walk BESIDE a warp tile (sea_walk excludes warp tiles from its own paths), then STEP
        onto it — doors, ladders, cave mouths and holes all fire on the step; a door that wants
        its arrow press gets one more directional hold. Verified by the map-id flip."""
        b = self.b
        if tuple(tv.map_id(b)) == dest:
            return True
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1))]
        for _att in range(3):
            if time.time() > self.deadline:
                return False
            if tuple(tv.coords(b) or ()) not in nbs:
                if not self.sea_walk(lambda c, s=set(nbs): c in s, f"{label}-approach"):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            key = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            m0 = tuple(tv.map_id(b))
            self.step_to(tile)
            for _ in range(160):
                b.run_frame()
                if tuple(tv.map_id(b)) != m0:
                    break
            if tuple(tv.map_id(b)) == m0 and key:
                b.press(key, 26, 10, self.camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
            self.drain()
            if tuple(tv.map_id(b)) == dest:
                for _ in range(60):
                    b.run_frame()
                self.log(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)}")
                return True
        return tuple(tv.map_id(b)) == dest

    def ride(self, table, goal_map, label):
        """Leg-by-leg floor descent (eevee_fetch._ride doctrine, candidates per floor)."""
        for _hop in range(len(table) + 2):
            while self.handle_interrupts():
                pass
            here = tuple(tv.map_id(self.b))
            if here == goal_map:
                return True
            if time.time() > self.deadline:
                self.log(f"!! [{label}] deadline mid-ride")
                return False
            leg = next((l for l in table if l[0] == here), None)
            if leg is None:
                self.log(f"!! [{label}] no leg from {here} — off the mission rails")
                return False
            _, cands, dest = leg
            if not any(self.enter_step(t, dest, f"{label}{t}") for t in cands):
                self.log(f"!! [{label}] every candidate warp failed on {here}")
                return False
        return tuple(tv.map_id(self.b)) == goal_map

    # ── the press ────────────────────────────────────────────────────────────────────────
    def press_quarry(self):
        """Stand beside the static, face it, A — cry — the battle. The campaign battle runner
        owns the fight (its legendary divert does the careful capture)."""
        q, b = self.QUARRY, self.b
        tile = q["tile"]
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1))]
        try:
            self.camp.on_event(f"there it is. {q['name']}. okay — deep breath, balls ready. "
                               f"we are NOT blowing this.", kind="legendary", tier=3)
        except Exception:
            pass
        for _att in range(4):
            if self.spent():
                return True
            if time.time() > self.deadline:
                return False
            if tuple(tv.coords(b) or ()) not in nbs:
                if not self.sea_walk(lambda c, s=set(nbs): c in s, "quarry-approach"):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            key = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            if key:
                b.press(key, 8, 10, self.camp.render, owner="agent")
            b.press("A", 8, 12, self.camp.render, owner="agent")
            for _ in range(90):
                b.run_frame()
                if self.fight_open():
                    break
            if not self.fight_open():
                self.drain(key="A")           # the pre-battle cry/text advances on A
            if self.fight_open():
                self.fight()
                self.drain()
                return True
        return self.spent()


class ZapdosHunt(LegendaryHunt):
    QUARRY = ZAPDOS

    def run(self):
        b = self.b
        here = tuple(tv.map_id(b))
        if self.spent() and here != PLANT:
            return self.outcome() or "battled"
        if here == R10:
            if not self.enter_step((7, 40), PLANT, "plant-door"):
                return "failed"
        if tuple(tv.map_id(b)) != PLANT:
            return "not_here"
        if not self.spent() and not self.press_quarry():
            return "failed"
        out = self.outcome() or "failed"
        # walk out (best-effort — a wedged exit never voids the banked outcome; the campaign's
        # travel/stuck machinery owns any leftover interior walking)
        self.enter_step((5, 38), R10, "plant-exit")
        return out


class ArticunoHunt(LegendaryHunt):
    QUARRY = ARTICUNO

    def run(self):
        b = self.b
        here = tuple(tv.map_id(b))
        if self.spent() and here not in ARTICUNO_ANCHORS - {R20}:
            return self.outcome() or "battled"
        # HARD SAFETY: without BOTH fallen boulders the B4F water still rips (the R21-reroute
        # stamp sets 0x2D2 without them — see seafoam_strike) — refuse rather than get swept.
        if fm.read_flag(b, FLAG_HIDE_B3F_BOULDER_1) or fm.read_flag(b, FLAG_HIDE_B3F_BOULDER_2):
            self.log("   [articuno] boulders NOT truly down (hide flags still set) — B4F is "
                     "rip-current water; refusing the descent (the gate should have caught this)")
            return "failed"
        if here == R20:
            # either F1 door works; try east (60,8) then west (72,14) side by proximity — the
            # doors warp on step, sea_walk crosses whichever sea band she's on
            for door in ((60, 8), (72, 14)):
                if self.enter_step(door, F1, "seafoam-door"):
                    break
        if tuple(tv.map_id(b)) not in {F1, B1F, B2F, B3F, B4F}:
            return "not_here"
        if tuple(tv.map_id(b)) != B4F and not self.ride(ARTICUNO_DESCENT, B4F, "descent"):
            return "failed"
        if not self.spent() and not self.press_quarry():
            return "failed"
        out = self.outcome() or "failed"
        # walk out: back up the same ladder pairs (B4F tiles pair with B3F warps 3/4/7/8)
        ascent = [(B4F, [(8, 17), (9, 17), (15, 9), (32, 5)], B3F),
                  (B3F, [(8, 14), (31, 16), (31, 4)], B2F),
                  (B2F, [(7, 4), (17, 9), (32, 14), (25, 19)], B1F),
                  (B1F, [(10, 6), (28, 19), (31, 4)], F1),
                  (F1, [(6, 21), (32, 21)], R20)]
        self.ride(ascent, R20, "ascent")
        return out


class MewtwoHunt(LegendaryHunt):
    QUARRY = MEWTWO

    def run(self):
        b = self.b
        here = tuple(tv.map_id(b))
        if self.spent() and here not in {CAVE1F, CAVE2F, CAVEB1F}:
            return self.outcome() or "battled"
        if not fm.read_flag(b, FLAG_SYS_GAME_CLEAR):
            self.log("   [mewtwo] not champion yet — the cave guard won't move (gate leak?)")
            return "failed"
        if here == CERULEAN:
            if not self.enter_step((1, 12), CAVE1F, "cave-mouth"):
                return "failed"
        here = tuple(tv.map_id(b))
        if here == CAVE2F:
            # dropped onto 2F somehow — any down-warp returns to 1F ((33,4) pairs with 1F (34,2))
            for t in ((33, 4), (13, 4), (7, 14), (26, 9), (23, 10), (5, 6)):
                if self.enter_step(t, CAVE1F, "back-to-1f"):
                    break
        if tuple(tv.map_id(b)) == CAVE1F:
            if not self.enter_step((1, 7), CAVEB1F, "b1f-ladder"):
                return "failed"
        if tuple(tv.map_id(b)) != CAVEB1F:
            return "not_here"
        if not self.spent() and not self.press_quarry():
            return "failed"
        out = self.outcome() or "failed"
        # walk out: B1F (5,7) -> 1F, then the long 1F crossing back to the mouth (33,21)
        self.enter_step((5, 7), CAVE1F, "b1f-out")
        if tuple(tv.map_id(b)) == CAVE1F:
            self.enter_step((33, 21), CERULEAN, "cave-out")
        return out


def _dispatch(cls, anchors, camp, log, dbg_dir):
    try:
        here = tuple(tv.map_id(camp.b))
    except Exception:
        return "failed"
    if here not in anchors:
        return "not_here"
    return cls(camp, log, dbg_dir).run()


def run_zapdos(camp, log, dbg_dir=None):
    return _dispatch(ZapdosHunt, ZAPDOS_ANCHORS, camp, log, dbg_dir)


def run_articuno(camp, log, dbg_dir=None):
    return _dispatch(ArticunoHunt, ARTICUNO_ANCHORS, camp, log, dbg_dir)


def run_mewtwo(camp, log, dbg_dir=None):
    return _dispatch(MewtwoHunt, MEWTWO_ANCHORS, camp, log, dbg_dir)
