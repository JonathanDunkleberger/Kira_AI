"""legendary_strikes.py — the LEGENDARY HUNTS (2026-08-04, Jonny: "i want her catching mew or
mewtwo as a final endgame project. and all cool legendaries that are available before or after
the final 4 so she mops the floor with them").

Four hunts, each a walk-to-the-static-and-press-A mission; the BATTLE side is already solved
(battle_agent's _LEGENDARY_SPECIES careful-capture divert: weaken, never KO, throw balls — now
tiered best-ball-first, Master Ball allowed on Mewtwo only). This module is only the ROAD.

  ZAPDOS   — Power Plant (1,95) at (5,11). Route 10's water strip guards the door (7,40):
             sea_walk crosses it (giovanni_gym machinery verbatim). Pre-E4 the moment Surf
             is taught.
  ARTICUNO — Seafoam B4F (1,87) at (9,2). Gated on the boulders being TRULY down (hide flags
             0x046/0x047 cleared — the stamped 0x2D2 from the R21 reroute is NOT enough: the
             B4F water still rips without the fallen boulders, see seafoam_strike). Descends
             the ladder chain from either F1 door; candidates per floor, live BFS picks.
  MOLTRES  — Mt. Ember summit (1,101) at (9,6), on ONE ISLAND (Sevii). The only hunt with a
             FERRY in it: Bill's post-Blaine offer is the ride out (declining parks him in
             the Cinnabar Center where he re-offers forever — pret CinnabarIsland scripts),
             and the ride HOME is story-gated on the Lostelle detour (rescue her in Three
             Island's Berry Forest, hand Celio's Meteorite to her dad on Two Island — ONLY
             that delivery sets the One-Island-Center scene var to 2, arming Bill's
             sail-home trigger; pret seagallop.inc refuses Vermilion until the return).
             Kindle Road is a sea road (Surf); the exterior + summit are Strength boulder
             puzzles (solver-verified push plans below).
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
  Moltres  : OBJ (9,6) MtEmber_Summit | FLAG_HIDE_MOLTRES 0x052 | FLAG_FOUGHT_MOLTRES 0x2BD
             maps: OneIsland (3,12) | KindleRoad (3,45) | Ember exterior (1,97) |
             SummitPath 1F/2F/3F (1,98)/(1,99)/(1,100) | Summit (1,101) | One-Island PC
             (32,0) / Harbor (32,4) | TwoIsland (3,13) / GameCorner (33,0) / Harbor (33,4) |
             ThreeIsland (3,14) / Port (3,49) / Harbor (38,0) | BondBridge (3,48) |
             BerryForest (1,109). Doors: Kindle (11,6)/(12,6)->exterior | ext (14,24)->1F |
             1F (11,1)->2F | 2F (39,6)->3F | 3F (11,8)->ext upper (39,19) | ext (29,7)->
             summit (lands (9,15)). Bill: Cinnabar PC obj (11,5), hide 0x0A2 (SET at new
             game, CLEARED by the doorstep decline, RE-SET when she sails). Lostelle:
             BerryForest (4,8) (Hypno L30 script battle, then auto-warp to GameCorner);
             Daddy (5,5); Meteorite = item 280, removed on delivery.
  Mewtwo   : OBJ (7,12) CeruleanCave_B1F | FLAG_HIDE_MEWTWO 0x081 | FLAG_FOUGHT_MEWTWO 0x2BC
             mouth: Cerulean (1,12)->Cave1F dw0 (lands (33,21)) | 1F (1,7)->B1F (lands (5,7))
             1F exit (33,21)->Cerulean

SUCCESS SEMANTICS: 'caught' (dex owned bit) | 'battled' (FOUGHT/HIDE flag set — the encounter
is SPENT: KO'd statics never respawn, so the step must satisfy either way or she'd loop on a
ghost) | 'not_here' | 'failed' (fail-safe: leaves her fightable/walkable, the errand retries).
"""
import time

import boulder_puzzle as bp
import field_moves as fm
import firered_ram as ram
import pokemon_state as pst
import travel as tv
from dialogue_drive import box_open as dd_box
from giovanni_gym import GiovanniGym, KEY_OF

DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# ── fact table (game-knowledge layer; rule 14 portability debt) ────────────────────────────
R10, PLANT = (3, 28), (1, 95)
CERULEAN = (3, 3)
CAVE1F, CAVE2F, CAVEB1F = (1, 72), (1, 73), (1, 74)
F1, B1F, B2F, B3F, B4F = (1, 83), (1, 84), (1, 85), (1, 86), (1, 87)
R20 = (3, 38)
# Sevii (Moltres) — town/route maps in group 3, dungeons group 1, interiors groups 32/33/38
CINNABAR = (3, 8)
CINNABAR_PC = (12, 5)
ONE_ISLAND, ONE_PC, ONE_HARBOR = (3, 12), (32, 0), (32, 4)
KINDLE = (3, 45)
EMBER_EXT, EMBER_1F, EMBER_2F, EMBER_3F, EMBER_SUMMIT = ((1, 97), (1, 98), (1, 99),
                                                         (1, 100), (1, 101))
TWO_ISLAND, TWO_HARBOR, GAME_CORNER = (3, 13), (33, 4), (33, 0)
THREE_ISLAND, THREE_PORT, THREE_HARBOR = (3, 14), (3, 49), (38, 0)
BOND_BRIDGE, BERRY_FOREST = (3, 48), (1, 109)

FLAG_SYS_GAME_CLEAR = 0x82C          # champion — the Cerulean Cave guard steps aside
FLAG_B3F_CALM = 0x2D2                # crossing signal (may be STAMPED by the R21 reroute!)
FLAG_HIDE_B3F_BOULDER_1 = 0x046      # cleared = boulder truly fell -> B4F water is safe
FLAG_HIDE_B3F_BOULDER_2 = 0x047
FLAG_HIDE_CINNABAR_PC_BILL = 0x0A2   # CLEAR = Bill waits in the Cinnabar Center (trip open)
FLAG_RESCUED_LOSTELLE = 0x2A3        # Berry Forest rescue done (Hypno beaten)
FLAG_STR_ACTIVE = 0x805              # FLAG_SYS_USE_STRENGTH — resets per map load
ITEM_METEORITE = 280                 # Celio's parcel; REMOVED from the bag on delivery

ZAPDOS = dict(name="Zapdos", species=145, tile=(5, 11), map=PLANT, hide=0x05D, fought=0x2BF)
ARTICUNO = dict(name="Articuno", species=144, tile=(9, 2), map=B4F, hide=0x082, fought=0x2BE)
MOLTRES = dict(name="Moltres", species=146, tile=(9, 6), map=EMBER_SUMMIT,
               hide=0x052, fought=0x2BD)
MEWTWO = dict(name="Mewtwo", species=150, tile=(7, 12), map=CAVEB1F, hide=0x081, fought=0x2BC)

ZAPDOS_ANCHORS = {R10, PLANT}
ARTICUNO_ANCHORS = {R20, F1, B1F, B2F, B3F, B4F}
MOLTRES_ANCHORS = {CINNABAR, CINNABAR_PC, ONE_ISLAND, ONE_PC, ONE_HARBOR, KINDLE,
                   EMBER_EXT, EMBER_1F, EMBER_2F, EMBER_3F, EMBER_SUMMIT,
                   TWO_ISLAND, TWO_HARBOR, GAME_CORNER,
                   THREE_ISLAND, THREE_PORT, THREE_HARBOR, BOND_BRIDGE, BERRY_FOREST}
MEWTWO_ANCHORS = {CERULEAN, CAVE1F, CAVE2F, CAVEB1F}

# Solver-verified Strength push plans (BFS over pret map.bin collision+elevation with the
# The Mt. Ember boulder boards live in boulder_puzzle.py now (2026-08-05 #3, the volcano
# loop): bp.EMBER_ASCENT / bp.EMBER_DESCENT / bp.EMBER_SUMMIT_BOARD — same pret map.json
# templates, but solved by the IDEMPOTENT chain engine (live readback per push, resume
# mid-chain without over-pushing, fail-in-place, door-reset LAST resort, checkpoint-per-push).

# Seagallop harbor menus while the detour is live (VAR_MAP_SCENE_CINNABAR_ISLAND < 4 —
# pret seagallop.inc: no Vermilion row until Bill has sailed her home): row order per pier.
SAIL_ROWS = {ONE_HARBOR: [TWO_HARBOR, THREE_HARBOR],
             TWO_HARBOR: [ONE_HARBOR, THREE_HARBOR],
             THREE_HARBOR: [ONE_HARBOR, TWO_HARBOR]}

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
    # THE FREE RETRY (2026-08-05, standing at Moltres with 6 Ultras): a failed catch —
    # fainted quarry, balls exhausted, even a whiteout — reloads the 'pre-<quarry>'
    # checkpoint banked seconds before the press. Savestates restore FULL RAM: the fought/
    # hide flags read clear again, the spent balls are back in the bag, and she is standing
    # topped-up in front of the bird. Each reload is a fresh 6-ball, sleep-assisted attempt;
    # bounded so a truly cursed RNG night still ends (LOUD, and the errand books 'battled').
    LEGEND_CATCH_RETRIES = 4

    def __init__(self, camp, log, dbg_dir=None):
        super().__init__(camp, log, dbg_dir)
        self.deadline = time.time() + 1500
        self._catch_retries = 0

    def _retry_failed_catch(self):
        """After a quarry battle resolves: True = the pre-quarry checkpoint was RELOADED (the
        caller loops into a fresh attempt), False = nothing to retry (caught / encounter still
        live / budget spent / no checkpoint). Never raises — a reload fault reads as 'accept
        the outcome' (the errand's bounded attempts own any further recovery)."""
        q = self.QUARRY or {}
        try:
            if ram.pokedex_owns(self.b, q["species"]) is True:
                return False                                   # CAUGHT — nothing to retry
            if not (fm.read_flag(self.b, q["fought"]) or fm.read_flag(self.b, q["hide"])):
                return False                                   # encounter still live/unspent
        except Exception:
            return False
        name = (q.get("name") or "quarry").lower()
        if self._catch_retries >= self.LEGEND_CATCH_RETRIES:
            if not getattr(self, "_retry_exhaust_logged", False):
                self._retry_exhaust_logged = True
                self.log(f"   [hunt] !! {name} catch FAILED and the retry budget is spent "
                         f"({self._catch_retries}/{self.LEGEND_CATCH_RETRIES}) — accepting "
                         f"'battled' (LOUD; the ball war-chest restock is the road back)")
            return False
        # verified reload (2026-08-05, the poisoned 'pre-moltres' bank): the campaign's hunt
        # door checks the fought/hide flags in the LOADED RAM and ratchets to older banks.
        if not self.camp._reload_hunt_checkpoint(name):
            self.log(f"   [hunt] !! {name} catch failed but no VERIFIED 'pre-{name}' bank "
                     f"reloadable (all candidates poisoned or missing) — accepting the "
                     f"outcome (LOUD)")
            return False
        self._catch_retries += 1
        self.log(f"   [hunt] !!!! FREE RETRY {self._catch_retries}/{self.LEGEND_CATCH_RETRIES}: "
                 f"'pre-{name}' reloaded — fought flag clear, balls restored, standing at the "
                 f"{name} again (LOUD)")
        try:
            self.camp.on_event(f"no. we are NOT losing {q.get('name', 'it')} like that. "
                               f"rewinding to right before the fight — fresh balls, fresh plan.",
                               kind="legendary", tier=3)
        except Exception:
            pass
        return True

    def spent_final(self):
        """spent() with THE FREE RETRY offered first: a spent-but-UNCAUGHT quarry reloads its
        'pre-<quarry>' bank (bounded) before any hunt leg may treat 'battled' as the end of
        the road. False after a reload — the encounter is live again and the caller re-runs
        the press. Every spent()-keyed decision in the hunts routes through this, so a flee/
        faint can never silently flip a hunt into its homebound/exit flow with retries left."""
        if not self.spent():
            return False
        if self._retry_failed_catch():
            return False
        return True

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

    # ── milestone durability (2026-08-05 #3, 'respawn right where she is') ──────────────
    def strike_checkpoint(self, reason=None):
        """Bank a checkpoint + refresh the recent-good at a climb milestone, so ANY recovery
        (watchdog escape hatch, region-local reload) resumes seconds back ON THIS MAP with the
        puzzle state intact (savestates capture full RAM — pushed boulders survive). Also
        refunds the lap's bounded-attempt counter for this quarry: a banked milestone IS
        progress, so the errand can never honest-skip while the climb is genuinely advancing
        (the GREEN law, applied to the hunt). Returns True (composable); never raises."""
        camp = self.camp
        key = ((self.QUARRY or {}).get("name") or "hunt").lower()
        label = reason or f"{key}-leg"
        # POISONED-BANK LAW (2026-08-05 URGENT): a 'pre-<quarry>' bank must contain the LIVE
        # encounter. Re-banking after a fled/fainted fight wrote the fought flag INTO the
        # newest 'pre-moltres' — the rewind then loaded an empty summit ("gone forever").
        # Fresh flag read here; spent + uncaught -> REFUSE the bank, loudly.
        if str(label).startswith("pre-"):
            q = self.QUARRY or {}
            try:
                if (q.get("species")
                        and ram.pokedex_owns(self.b, q["species"]) is not True
                        and (fm.read_flag(self.b, q.get("fought", 0))
                             or fm.read_flag(self.b, q.get("hide", 0)))):
                    self.log(f"   [ckpt] !! REFUSED to bank '{label}': the quarry is already "
                             f"battled-away in THIS state (fought/hide flag set, uncaught) — "
                             f"a pre-encounter bank must contain the live encounter "
                             f"(poisoned-bank law, LOUD)")
                    return False
            except Exception:
                pass
        try:
            camp._bank_milestone(label)
            (getattr(camp, "_lap_fails", None) or {}).pop(key, None)
        except Exception as e:
            self.log(f"   [ckpt] strike milestone '{label}' skipped: {e}")
        return True

    def _chip_pp_audit(self):
        """PRE-ENCOUNTER CHIP-PP AUDIT (2026-08-05 LIVE, the one-Bite ball-burn: the climb
        drained Bite to 1 PP, the chip phase died after one swing and Ultras flew at a
        near-full bird). Before pressing A, read every healthy mon's moves + CURRENT PP from
        the party struct (pst.read_party_moves/read_party_pp — the encrypted Attacks
        substructure, fresh RAM) and log the chip picture LOUDLY. No Ether/Elixir bag rail
        exists in the harness, so the audit's job is the honest headline: the in-battle PP
        ladder (per-swing PP recheck -> bench chipper switch) does the adapting, and the
        FREE-RETRY savestate restores this exact PP picture on every attempt — a
        thin-but-nonzero audit stays sufficient forever. Zero damaging PP on the WHOLE party
        is the one truly bad state; scream it (sleep+throw only). Returns True when any
        healthy mon still carries damaging PP; never raises."""
        try:
            any_pp = False
            rows = []
            for s, hp, mx, frac in (self.camp.party_health() or []):
                if hp <= 0:
                    continue
                ids = pst.read_party_moves(self.b, s) or []
                pps = pst.read_party_pp(self.b, s) or []
                dmg = sum(int(p) for m, p in zip(ids, pps)
                          if m and p and (pst.move_info_full(self.b, m)[1] or 0) > 0)
                any_pp = any_pp or dmg > 0
                rows.append(f"slot{s} hp {hp}/{mx} dmgPP {dmg} "
                            f"({', '.join(f'{pst.MOVE_NAMES.get(m, m)}:{p}' for m, p in zip(ids, pps) if m)})")
            self.log("   [hunt] CHIP-PP AUDIT — " + ("; ".join(rows) or "party unreadable"))
            if not any_pp:
                self.log("   [hunt] !! CHIP-PP AUDIT: ZERO damaging PP on the whole party — "
                         "the encounter will be sleep+throw only (no Ether rail exists; a "
                         "Center visit restores PP but costs the summit) (LOUD)")
            return any_pp
        except Exception as e:
            self.log(f"   [hunt] chip-PP audit skipped: {e}")
            return True

    # ── field healing (2026-08-05, the Mt. Ember climb) ─────────────────────────────────
    def field_heal_seam(self, top_up=False):
        """OUT-OF-BATTLE HEAL SEAM: a hunt owns the loop for up to ~47 minutes with NO roam
        tick and (on Mt. Ember) no Center — wild/trainer chip accumulates unanswered. Between
        legs (and as the pre-press TOP-UP) hand the turn to the campaign's [fieldheal]
        doctrine: ace under 50% (near-full for top_up) drinks the cheapest-adequate potion
        via the proven bag rails. Best-effort, bounded by the doctrine's own backoff —
        a heal wedge can never void the hunt."""
        try:
            self.camp.field_heal_check(reason="strike", top_up=top_up)
        except Exception as e:
            self.log(f"   [fieldheal] strike seam skipped: {e}")

    # ── movement ─────────────────────────────────────────────────────────────────────────
    def enter_step(self, tile, dest, label):
        """Walk BESIDE a warp tile (sea_walk excludes warp tiles from its own paths), then STEP
        onto it — doors, ladders, cave mouths and holes all fire on the step; a door that wants
        its arrow press gets one more directional hold. Verified by the map-id flip."""
        b = self.b
        if tuple(tv.map_id(b)) == dest:
            return True
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1))]
        # ARROW-MAT LAW (2026-08-05, the One-Island PC exit / the teleport-back incident):
        # a directional MB_*_ARROW_WARP mat (every PC exit mat = SOUTH arrow) fires ONLY when
        # stepped in the arrow's direction — walked across sideways it is plain floor. The old
        # nearest-neighbor approach reached (10,9)/(8,9) beside the (9,9) mat, pressed LEFT/
        # RIGHT across it, never warped, and burned every strike try. When the mat's behavior
        # byte reads directional, the ONLY legal approach tile is the one OPPOSITE the arrow
        # (the arrow press then falls out of the same KEY_OF math below). Behavior None
        # (unreadable) keeps the old any-neighbor fan — bound, don't blind.
        arrow = tv.ARROW_WARP_STEP.get(tv.behavior_at(b, *tile))
        if arrow:
            adx, ady = DELTA[arrow]
            nbs = [(tile[0] - adx, tile[1] - ady)]
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
            self.field_heal_seam()          # between-legs drink (battles just drained above)
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
        # PRE-LEGENDARY TOP-UP (2026-08-05): the static fight starts at HER choice of moment —
        # a real player tops the ace to (near-)full BEFORE pressing A, not after turn 1.
        self.field_heal_seam(top_up=True)
        # PRE-ENCOUNTER CHIP-PP AUDIT (2026-08-05, the one-Bite ball-burn): the PP picture
        # about to be frozen into 'pre-<quarry>' is what every FREE RETRY replays — confess it.
        self._chip_pp_audit()
        # PRE-LEGENDARY CHECKPOINT (2026-08-05 addendum): standing in front of the bird, topped
        # up, board solved — the exact moment Jonny wants a recovery (or a manual
        # PROMOTE_TARGET pin) to respawn into. Named 'pre-<quarry>' in the inventory.
        self.strike_checkpoint(f"pre-{(q.get('name') or 'quarry').lower()}")
        try:
            self.camp.on_event(f"there it is. {q['name']}. okay — deep breath, balls ready. "
                               f"we are NOT blowing this.", kind="legendary", tier=3)
        except Exception:
            pass
        for _att in range(4 + self.LEGEND_CATCH_RETRIES):
            if self.spent():
                # spent-but-not-caught: offer THE FREE RETRY before accepting 'battled'
                # (covers a resume landing here post-battle too — the reload restores the
                # un-fought world and the loop presses again).
                if self._retry_failed_catch():
                    continue
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
                # the battle resolved — a failed catch (fainted / out of balls / whiteout)
                # reloads 'pre-<quarry>' and loops into a fresh attempt (bounded).
                if self._retry_failed_catch():
                    continue
                return True
        return self.spent()


class ZapdosHunt(LegendaryHunt):
    QUARRY = ZAPDOS

    def run(self):
        b = self.b
        here = tuple(tv.map_id(b))
        if self.spent_final() and here != PLANT:
            return self.outcome() or "battled"
        here = tuple(tv.map_id(b))          # a FREE-RETRY reload may have moved her — re-read
        if here == R10:
            if not self.enter_step((7, 40), PLANT, "plant-door"):
                return "failed"
        if tuple(tv.map_id(b)) != PLANT:
            return "not_here"
        if not self.spent_final() and not self.press_quarry():
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
        if self.spent_final() and here not in ARTICUNO_ANCHORS - {R20}:
            return self.outcome() or "battled"
        here = tuple(tv.map_id(b))          # a FREE-RETRY reload may have moved her — re-read
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
        if not self.spent_final() and not self.press_quarry():
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


class MoltresHunt(LegendaryHunt):
    """MT. EMBER MOLTRES — the Sevii strike, a stage machine keyed on the CURRENT MAP so any
    entry point resumes cleanly (whiteout respawn is the One-Island Center — pret setrespawn
    on the forced arrival walk-in, so a blackout mid-climb resumes ON the archipelago).

    OUTBOUND (bird alive): Cinnabar PC -> Bill YES (the auto-sail cutscene ends standing
    free in the One-Island Center, Tri Pass + Meteorite banked by the forced Celio scene) ->
    east edge -> Kindle Road sea leg north -> Ember exterior boulder corridor -> SummitPath
    1F/2F/3F -> exterior upper ledge -> summit boulder puzzle -> press_quarry.

    HOMEBOUND (bird SPENT): the ride home is story-gated (seagallop refuses Vermilion and the
    Center's leave-trigger sleeps until the Meteorite is DELIVERED) — descend, sail Three,
    port -> town (the biker gauntlet fires trainerbattle_no_intro scripts; the interrupt
    machinery fights them), Bond Bridge -> Berry Forest -> Lostelle (Hypno script battle,
    auto-warp to the Two-Island Game Corner) -> hand Daddy the Meteorite (bag delta 280 is
    the proof) -> sail One -> Center -> cross the x=12 trigger column -> Bill sails her home
    to Cinnabar. Routing predicate per island: Meteorite in bag + Lostelle unrescued -> Three;
    in bag + rescued -> Two (Game Corner); delivered -> One (the trigger walk)."""

    QUARRY = MOLTRES

    def __init__(self, camp, log, dbg_dir=None):
        super().__init__(camp, log, dbg_dir)
        self.deadline = time.time() + 2800   # the longest hunt: two ferries + the detour

    # ── boulder machinery (seafoam_strike verbatim-adapted) ────────────────────────────────
    def live_boulders(self):
        return [ob["coord"] for ob in fm.scan_field_objects(self.b, {fm.GFX_BOULDER})]

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
            self.drain(key="A")                 # info text + YES/NO (YES default)
            self.settle(30)
            if fm.read_flag(b, FLAG_STR_ACTIVE):
                return True
        self.log(f"!! [strength] flag 0x805 never set (boulder {bl})")
        return False

    def push(self, approx, key, n, allow=()):
        b, camp = self.b, self.camp
        d = DELTA[key]
        for i in range(n):
            bl = self.nearest_boulder(approx)
            if bl is None:
                # 2026-08-05 #3: 'absent -> assuming pushed' was a LIE — the GBA unloads
                # off-camera object events AND a failed approach also reads absent, so the
                # old assumption green-lit unsolved boards. The chain engine owns absence
                # (walk-near + verified look); an absent boulder HERE is an honest failure.
                self.log(f"!! [push] boulder near {approx} absent (i={i}) — NOT assuming; "
                         f"failing LOUD (the solver re-derives from live truth)")
                return False
            stand = (bl[0] - d[0], bl[1] - d[1])
            if not self.sea_walk(lambda c, s=stand: c == s, f"push-approach{i}",
                                 avoid={tuple(bl)}):
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
                return False
            approx = (bl[0] + d[0], bl[1] + d[1])
            self.log(f"   [push] {bl} -> {approx} ({key}, {i + 1}/{n})")
            self.settle(30)
        return True

    def board_mission(self, room):
        """One boulder room via the SHARED chain engine (boulder_puzzle.solve_room): live
        readback per push, idempotent mid-chain resume (never over-push a solved board),
        fail-IN-PLACE retries, the door-reset LAST resort, and a milestone checkpoint after
        every verified push (savestates keep pushed boulders — recoveries resume mid-board)."""
        return bp.solve_room(self, room, checkpoint=self.strike_checkpoint, log=self.log)

    # ── talk / scene primitives ──────────────────────────────────────────────────────────
    def poke(self, tile, label, avoid=()):
        """Stand beside an NPC/scene tile, face it, A, A-drain the boxes (A answers YES on
        a YES/NO — every box on this mission wants YES)."""
        b = self.b
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1))
               if (tile[0] + dx, tile[1] + dy) not in set(avoid)]
        for _att in range(3):
            if time.time() > self.deadline:
                return False
            if tuple(tv.coords(b) or ()) not in nbs:
                if not self.sea_walk(lambda c, s=set(nbs): c in s, f"{label}-approach"):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            key = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            if key:
                b.press(key, 8, 10, self.camp.render, owner="agent")
            b.press("A", 8, 12, self.camp.render, owner="agent")
            self.settle(40)
            if dd_box(b) or self.fight_open():
                self.drain(key="A")
                return True
        return False

    def wait_scene(self, done, label, timeout=300, quiet_n=4):
        """Ride out a long scripted scene (ferry, forced walks, fanfares): A-drain every box,
        fight anything that opens, until `done()` and the boxes go quiet. quiet_n tunes the
        silence bar — the Bill arrival/return cutscenes have LONG boxless forced-walk gaps,
        so their callers demand ~9s of quiet before believing the scene is over."""
        b = self.b
        end = time.time() + timeout
        quiet = 0
        while time.time() < end:
            if self.fight_open():
                self.fight()
                self.drain()
                quiet = 0
                continue
            if dd_box(b):
                quiet = 0
                b.press("A", 8, 12, self.camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
                continue
            for _ in range(45):
                b.run_frame()
            quiet += 1
            if done() and quiet >= quiet_n:     # ~0.75s of silence per unit, in goal state
                self.log(f"   [{label}] scene done at {tv.map_id(b)}@{tv.coords(b)}")
                return True
        self.log(f"!! [{label}] scene never settled (at {tv.map_id(b)}@{tv.coords(b)})")
        return done()

    def sail(self, want):
        """Seagallop hop: talk to the pier sailor (8,6) from (8,7), pick the row (DOWN x row
        + A; the menu is [row0/row1/Exit] per SAIL_ROWS while the detour is live), ride the
        ferry (map flip). A missed row self-corrects: run() re-dispatches from wherever the
        boat actually landed, and every wrong island's menu still contains the right one."""
        b, camp = self.b, self.camp
        here = tuple(tv.map_id(b))
        rows = SAIL_ROWS.get(here)
        if not rows or want not in rows:
            return False
        row = rows.index(want)
        if not self.sea_walk(lambda c: c == (8, 7), "sailor-approach"):
            return False
        b.press("UP", 8, 10, camp.render, owner="agent")
        b.press("A", 8, 12, camp.render, owner="agent")
        self.settle(80)                          # "Where do you want to sail?" + multichoice
        for _ in range(row):
            b.press("DOWN", 8, 10, camp.render, owner="agent")
            self.settle(20)
        b.press("A", 8, 12, camp.render, owner="agent")
        ok = self.wait_scene(lambda: tuple(tv.map_id(b)) != here, "ferry", timeout=60)
        self.settle(60)
        got = tuple(tv.map_id(b))
        if got != want:
            self.log(f"   [sail] {here} -> {got} (wanted {want}) — "
                     f"{'re-routing from the wrong pier' if ok else 'ferry never left'}")
        else:
            self.log(f"   [sail] {here} -> {got}")
        # a wrong pier is PROGRESS, not a wedge: every island's menu contains the right
        # destination, so the stage router self-corrects on the next leg_home dispatch
        return ok

    def heal_here(self, nurse_tile):
        """Best-effort Center heal (the nurse YES/NO answers YES on A-drain).
        COUNTER LAW (2026-08-05, the One-Island 'nurse-approach no path' log): every Center
        nurse stands BEHIND a counter — her 1-tile neighbors are furniture, so a poke() to an
        adjacent tile no-paths forever. The talk happens ACROSS the counter: stand two south
        of the nurse, face UP, press A (exactly how the campaign's heal_at_center does it)."""
        b = self.b
        try:
            front = (nurse_tile[0], nurse_tile[1] + 2)
            if tuple(tv.coords(b) or ()) != front and not self.sea_walk(
                    lambda c, f=front: c == f, "nurse-front", tries=6):
                self.log(f"   [nurse] counter front {front} unreachable — skipping the "
                         f"top-up (best-effort; the strike marches on)")
                return
            b.press("UP", 8, 10, self.camp.render, owner="agent")
            b.press("A", 8, 12, self.camp.render, owner="agent")
            self.settle(120)
            self.drain(key="A")
            self.settle(60)
            self.drain()
        except Exception as e:
            self.log(f"   [nurse] heal errored: {e} — continuing (LOUD)")

    def meteorite_in_bag(self):
        try:
            return self.camp.bag_count(ITEM_METEORITE) > 0
        except Exception:
            return False

    # ── outbound stages ──────────────────────────────────────────────────────────────────
    def board_with_bill(self):
        """Cinnabar PC: Bill (11,5) YES -> the whole auto-sail (island exit walk, ferry,
        One-Island harbor walk-out, the forced Center walk-in, the Celio Meteorite/Tri-Pass
        scene). Ends standing free in the One-Island Center."""
        b = self.b
        if tuple(tv.map_id(b)) == CINNABAR:
            if not self.enter_to(CINNABAR_PC, "cinnabar-pc"):
                return False
        if fm.read_flag(b, FLAG_HIDE_CINNABAR_PC_BILL):
            self.log("!! [moltres] Bill is NOT in the Cinnabar Center (hide 0x0A2 set) — "
                     "no ride to One Island; the gate should have suppressed this")
            return False
        try:
            self.camp.on_event("Bill's waiting in the Pokémon Center — and this time the "
                               "answer is YES. One Island, here we come: MOLTRES is on that "
                               "mountain.", kind="legendary", tier=3)
        except Exception:
            pass
        if not self.poke((11, 5), "bill-yes"):
            return False
        return self.wait_scene(lambda: tuple(tv.map_id(b)) == ONE_PC, "sail-to-one",
                               quiet_n=12)

    def leg_to_summit(self, here):
        """One stage of the climb, keyed by map; run() loops until the summit press."""
        b = self.b
        if here == ONE_PC:
            self.heal_here((5, 2))
            return self.enter_step((9, 9), ONE_ISLAND, "pc-out")
        if here == ONE_ISLAND:
            return self.cross_edge("east", "to-kindle")
        if here == KINDLE:
            return (self.enter_step((11, 6), EMBER_EXT, "ember-door")
                    or self.enter_step((12, 6), EMBER_EXT, "ember-door2"))
        if here == EMBER_EXT:
            y = (tv.coords(b) or (0, 0))[1]
            if y > 30:                           # lower corridor — the 6-push ascent board
                if not self.board_mission(bp.EMBER_ASCENT):
                    return False
                return self.enter_step((14, 24), EMBER_1F, "1f-door")
            return self.enter_step((29, 7), EMBER_SUMMIT, "summit-door")
        if here == EMBER_1F:
            return self.enter_step((11, 1), EMBER_2F, "2f-door")
        if here == EMBER_2F:
            return self.enter_step((39, 6), EMBER_3F, "3f-door")
        if here == EMBER_3F:
            return self.enter_step((11, 8), EMBER_EXT, "ext-upper")
        if here == EMBER_SUMMIT:
            if not self.board_mission(bp.EMBER_SUMMIT_BOARD):
                return False
            return self.press_quarry()
        return False

    # ── homebound stages ─────────────────────────────────────────────────────────────────
    def leg_home(self, here):
        """One stage of the ride home, keyed by map + the detour predicate."""
        b = self.b
        met, resc = self.meteorite_in_bag(), fm.read_flag(b, FLAG_RESCUED_LOSTELLE)
        if here == EMBER_SUMMIT:
            return self.enter_step((9, 15), EMBER_EXT, "summit-out")
        if here == EMBER_EXT:
            y = (tv.coords(b) or (0, 0))[1]
            if y <= 30:
                return self.enter_step((39, 19), EMBER_3F, "3f-upper-door")
            if not self.board_mission(bp.EMBER_DESCENT):
                return False
            return (self.enter_step((28, 48), KINDLE, "kindle-door")
                    or self.enter_step((29, 48), KINDLE, "kindle-door2"))
        if here == EMBER_3F:
            return self.enter_step((2, 4), EMBER_2F, "3f-down")
        if here == EMBER_2F:
            return self.enter_step((8, 39), EMBER_1F, "2f-down")
        if here == EMBER_1F:
            return self.enter_step((2, 15), EMBER_EXT, "1f-down")
        if here == KINDLE:
            return self.cross_edge("west", "to-one-island")
        if here == ONE_ISLAND:
            if met:
                return self.enter_step((12, 18), ONE_HARBOR, "one-harbor")
            return self.enter_step((14, 5), ONE_PC, "one-pc")
        if here == ONE_PC:
            self.heal_here((5, 2))
            if met:
                return self.enter_step((9, 9), ONE_ISLAND, "pc-out")
            # Meteorite delivered -> the x=12 trigger column (y 6-9, scene var == 2) fires
            # Bill's leave scene: walks, the sail, the Cinnabar return cutscene.
            try:
                self.camp.on_event("detour done, Meteorite delivered — Bill's sailing us "
                                   "home to Cinnabar. what a trip.", kind="legendary", tier=2)
            except Exception:
                pass
            if not self.sea_walk(lambda c: c[0] == 12 and 6 <= c[1] <= 9, "leave-trigger"):
                return False
            return self.wait_scene(lambda: tuple(tv.map_id(b)) == CINNABAR, "sail-home",
                                   quiet_n=12)
        if here == ONE_HARBOR:
            if met:
                return self.sail(THREE_HARBOR if not resc else TWO_HARBOR)
            return self.enter_step((8, 2), ONE_ISLAND, "harbor-out")
        if here == THREE_HARBOR:
            if met and not resc:
                return self.enter_step((8, 2), THREE_PORT, "three-port")
            return self.sail(ONE_HARBOR if not met else TWO_HARBOR)
        if here == THREE_PORT:
            if met and not resc:
                return self.cross_edge("north", "to-three-town")
            return self.enter_step((12, 13), THREE_HARBOR, "port-harbor")
        if here == THREE_ISLAND:
            # the biker gauntlet (y=26/27 triggers, x 7-11) fires trainerbattle_no_intro
            # scripts on the way north — sea_walk's interrupt handling fights them through
            if met and not resc:
                return self.cross_edge("west", "to-bond-bridge")
            return self.cross_edge("south", "to-three-port")
        if here == BOND_BRIDGE:
            if met and not resc:
                return (self.enter_step((12, 6), BERRY_FOREST, "forest-door")
                        or self.enter_step((13, 6), BERRY_FOREST, "forest-door2"))
            return self.cross_edge("east", "to-three-town")
        if here == BERRY_FOREST:
            if not resc:
                # Lostelle (4,8): the talk runs into the Hypno L30 script battle, then the
                # script AUTO-WARPS her to the Two-Island Game Corner (pret BerryForest)
                if not self.poke((4, 8), "lostelle"):
                    return False
                return self.wait_scene(lambda: tuple(tv.map_id(b)) == GAME_CORNER,
                                       "hypno-and-warp", timeout=240, quiet_n=8)
            return self.enter_step((43, 41), BOND_BRIDGE, "forest-out")
        if here == GAME_CORNER:
            # OnFrame reunion scene (scene var 2) plays first — drain it, then Daddy (5,5);
            # the delivery is PROVEN by the Meteorite leaving the bag (item 280 removed)
            self.wait_scene(lambda: True, "lostelle-reunion", timeout=45)
            if met:
                for _att in range(3):
                    self.poke((5, 5), "daddy", avoid=((6, 5),))
                    self.settle(60)
                    self.drain(key="A")
                    if not self.meteorite_in_bag():
                        break
                if self.meteorite_in_bag():
                    self.log("!! [moltres] Daddy never took the Meteorite — wedged delivery")
                    return False
                self.log("   [moltres] Meteorite DELIVERED — the ride home is armed "
                         "(One-Island Center scene var -> 2)")
            return self.enter_step((5, 8), TWO_ISLAND, "corner-out")
        if here == TWO_ISLAND:
            if met and resc:
                return self.enter_step((39, 9), GAME_CORNER, "to-corner")
            return self.enter_step((10, 8), TWO_HARBOR, "two-harbor")
        if here == TWO_HARBOR:
            if met and not resc:
                return self.sail(THREE_HARBOR)
            if met and resc:
                return self.enter_step((8, 2), TWO_ISLAND, "harbor-out")
            return self.sail(ONE_HARBOR)
        return False

    # ── the run ──────────────────────────────────────────────────────────────────────────
    def run(self):
        b = self.b
        here = tuple(tv.map_id(b))
        if self.spent_final() and here not in MOLTRES_ANCHORS - {CINNABAR, CINNABAR_PC}:
            return self.outcome() or "battled"
        here = tuple(tv.map_id(b))          # a FREE-RETRY reload may have moved her — re-read
        if here not in MOLTRES_ANCHORS:
            return "not_here"
        # WEDGE-MARK HYGIENE (2026-08-05 #3): today's menu-frozen windows banked phantom
        # wedge marks on the climb maps (frozen coords made real tiles look like traps, 12h
        # TTL) — drop every mark on the strike's map set before the first step; a REAL trap
        # re-marks itself in seconds, a phantom one poisons the router for the whole climb.
        if not getattr(self, "_hygiene_done", False):
            self._hygiene_done = True
            try:
                self.camp._release_wedge_marks_on(
                    (ONE_ISLAND, KINDLE, EMBER_EXT, EMBER_1F, EMBER_2F, EMBER_3F,
                     EMBER_SUMMIT), "moltres strike start")
            except Exception as e:
                self.log(f"   [moltres] wedge hygiene skipped: {e}")
        last_ckpt_map = None
        for _stage in range(64):
            while self.handle_interrupts():
                pass
            # CLIMB MILESTONE (2026-08-05 addendum): every leg boundary of the climb (each
            # map of the volcano set) banks a checkpoint — a recovery resumes seconds back on
            # THIS map, boulders intact, instead of five minutes down the mountain.
            _m = tuple(tv.map_id(b))
            if _m != last_ckpt_map and _m in (EMBER_EXT, EMBER_1F, EMBER_2F, EMBER_3F,
                                              EMBER_SUMMIT):
                last_ckpt_map = _m
                self.strike_checkpoint("moltres-leg")
            # STRIKE-LEG HEAL SEAM (2026-08-05): the climb's battles resolve inside the legs/
            # interrupts above; this is the 'control is back on the overworld' moment — drink
            # if the doctrine says so (no Center between One Island town and the summit).
            self.field_heal_seam()
            if time.time() > self.deadline:
                self.log("!! [moltres] deadline — surfacing (stage machine resumes by map)")
                return "failed"
            here = tuple(tv.map_id(b))
            # spent_final: a fled/fainted-but-UNCAUGHT bird reloads 'pre-moltres' RIGHT HERE
            # (bounded) instead of flipping the stage machine homebound — the 2026-08-05
            # emergency law: she finishes or honestly fails the encounter before any leg home.
            if self.spent_final():
                if here in (CINNABAR, CINNABAR_PC):
                    self.log("   [moltres] SPENT + home at Cinnabar — the hunt is complete")
                    return self.outcome() or "battled"
                if here not in MOLTRES_ANCHORS:
                    return self.outcome() or "battled"
                if not self.leg_home(here):
                    self.log(f"!! [moltres] home leg wedged on {here} @ {tv.coords(b)}")
                    return "failed"
            else:
                if here in (CINNABAR, CINNABAR_PC):
                    if not self.board_with_bill():
                        return "failed"
                elif here in (ONE_PC, ONE_ISLAND, KINDLE, EMBER_EXT, EMBER_1F,
                              EMBER_2F, EMBER_3F, EMBER_SUMMIT):
                    if not self.leg_to_summit(here):
                        self.log(f"!! [moltres] climb leg wedged on {here} @ {tv.coords(b)}")
                        return "failed"
                elif here == ONE_HARBOR:
                    if not self.enter_step((8, 2), ONE_ISLAND, "harbor-out"):
                        return "failed"
                else:
                    # unfought but drifted onto Two/Three (bizarre resume) — sail back to One
                    if here in (TWO_HARBOR, THREE_HARBOR):
                        if not self.sail(ONE_HARBOR):
                            return "failed"
                    elif here == TWO_ISLAND:
                        if not self.enter_step((10, 8), TWO_HARBOR, "two-harbor"):
                            return "failed"
                    elif here == THREE_PORT:
                        if not self.enter_step((12, 13), THREE_HARBOR, "port-harbor"):
                            return "failed"
                    elif here == THREE_ISLAND:
                        if not self.cross_edge("south", "to-three-port"):
                            return "failed"
                    elif here == BOND_BRIDGE:
                        if not self.cross_edge("east", "to-three-town"):
                            return "failed"
                    elif here == BERRY_FOREST:
                        if not self.enter_step((43, 41), BOND_BRIDGE, "forest-out"):
                            return "failed"
                    elif here == GAME_CORNER:
                        if not self.enter_step((5, 8), TWO_ISLAND, "corner-out"):
                            return "failed"
                    else:
                        return "not_here"
        return "failed"


class MewtwoHunt(LegendaryHunt):
    QUARRY = MEWTWO

    def run(self):
        b = self.b
        here = tuple(tv.map_id(b))
        if self.spent_final() and here not in {CAVE1F, CAVE2F, CAVEB1F}:
            return self.outcome() or "battled"
        here = tuple(tv.map_id(b))          # a FREE-RETRY reload may have moved her — re-read
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
        if not self.spent_final() and not self.press_quarry():
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


def run_moltres(camp, log, dbg_dir=None):
    return _dispatch(MoltresHunt, MOLTRES_ANCHORS, camp, log, dbg_dir)


def run_mewtwo(camp, log, dbg_dir=None):
    return _dispatch(MewtwoHunt, MEWTWO_ANCHORS, camp, log, dbg_dir)
