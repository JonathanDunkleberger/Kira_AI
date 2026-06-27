"""campaign.py - the continuous CAMPAIGN DRIVER (the skeleton). Executes a KNOWN
FireRed objective sequence Pallet -> Pewter -> Brock end-to-end via reusable handlers,
never exiting on arrival. The harness knows WHERE to go and WHAT the objectives are
(documented game route/gates); Kira's existing self provides ALL battle decisions +
commentary + personality (UNTOUCHED). Engine layer (battle_agent, _pokemon_react,
voice/mood/bond) is NOT modified here. Stuck-detection is LOUD.

OBJECTIVE TYPES (each a reusable handler, built once for the whole game):
  WALK_TO_MAP(map, dir)   - travel to a connected map by crossing its dir edge (BFS,
                            NPC-aware, grass-aware - all already in travel.Traveler)
  WALK_TO_COORD(x, y)     - BFS to a specific tile on the current map
  ENTER_WARP(target_map)  - step onto a door/warp tile, confirm the map flip
  INTERACT(dir)           - face dir, press A, advance the dialogue to its end
  BATTLE()                - hand the pad to the 5/5 battle engine, resume after
  GRIND(map, level)       - battle in grass until our lead mon reaches `level`

A gate that needs a savestate (a scripted quest the harness can't yet script) is
flagged GATE_NEEDS_STATE so the run stops LOUD with exactly what to hand-play.

Run:  .venv\\Scripts\\python.exe pokemon_agent\\campaign.py --headless
"""
import argparse
import os
import sys
import time
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge        # noqa: E402
import firered_ram as ram        # noqa: E402
import pokemon_state as st       # noqa: E402
import travel as tv              # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from dialogue_drive import DialogueDriver, box_open as dd_box_open  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
# ── THREE SAVE LINEAGES (Part A) ──────────────────────────────────────────────
# states/workshop/  = the SMALL set of real sherpa checkpoints WE use to jump up/down the game to
#                     lay hooks (beginning, brock_done, mtmoon_done, misty_done) + any state a live
#                     segment loads. WORKSHOP runs resume-from-any here AND bank here.
# states/kira/      = her SACRED playthrough save(s). ONLY a SHOW run may write here; a WORKSHOP run
#                     is PHYSICALLY FORBIDDEN to write into it (see _save_checkpoint guard). Old runs
#                     are timestamp-archived under kira/archive_<ts>/ (kira_run.py), never clobbered.
# states/archive/   = every handicapped-era TEACHING capture + orphaned scratch/recon state. Out of
#                     the live path so nothing mistakes archaeology for current. Not deleted.
# resolve_state() finds a named state across the live buckets (workshop -> kira -> flat -> archive),
# so boots/gates keep loading after the archive sweep regardless of which bucket holds the file.
STATES_WORKSHOP = os.path.join(STATES, "workshop")
STATES_KIRA = os.path.join(STATES, "kira")
STATES_ARCHIVE = os.path.join(STATES, "archive")


def resolve_state(name):
    """Find a named savestate across the live buckets, preferring the live path over archaeology:
    workshop/ -> kira/ -> flat states/ -> archive/. Returns the full path or None."""
    for d in (STATES_WORKSHOP, STATES_KIRA, STATES, STATES_ARCHIVE):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def kira_checkpoints():
    """The seg_*.state checkpoints present in states/kira/ (a Kira run is in progress iff non-empty)."""
    if not os.path.isdir(STATES_KIRA):
        return []
    return sorted(f for f in os.listdir(STATES_KIRA) if f.endswith(".state"))


def archive_kira_save(stamp):
    """Timestamp-archive the current Kira playthrough so a FRESH show run never clobbers her save:
    move states/kira/*.state -> states/kira/archive_<stamp>/. `stamp` is passed in (caller stamps the
    clock). Returns the archive dir, or None if there was no save to move."""
    import shutil
    saves = kira_checkpoints()
    if not saves:
        return None
    dst = os.path.join(STATES_KIRA, f"archive_{stamp}")
    os.makedirs(dst, exist_ok=True)
    for f in saves:
        shutil.move(os.path.join(STATES_KIRA, f), os.path.join(dst, f))
    return dst

# ── known FireRed map IDs (group, num) - documented, not reconned ────────────
PALLET, VIRIDIAN, PEWTER = (3, 0), (3, 1), (3, 2)
ROUTE1, ROUTE2, ROUTE22 = (3, 19), (3, 20), (3, 41)
ROUTE3, ROUTE4 = (3, 21), (3, 22)      # Pewter->Route 3 (east), Mt Moon->Route 4 (Cerulean side)
# Viridian Forest + gym interiors live in other groups; resolved at runtime on entry.
# Viridian Pokemon Center (RED-roof bldg, reconned 2026-06-24): town door (26,26) -> interior
# (5,4); nurse at (7,2), talked to from (7,4) ACROSS the counter; entrance mat at (7,8).
VIRIDIAN_PC_DOOR, VIRIDIAN_PC_MAP, NURSE_FRONT, PC_ENTRY = (26, 26), (5, 4), (7, 4), (7, 8)
# Pewter Gym (reconned 2026-06-24): town door (15,16) -> interior (6,2); climb the centre column
# (x=6) to Brock's front tile (6,6); the lone gym trainer auto-engages off the centre path.
PEWTER_GYM_DOOR, BROCK_FRONT = (15, 16), (6, 6)
# Pewter Pokemon Center door (disasm PewterCity warp_events: x17 y25 -> POKEMON_CENTER_1F). The
# nearest heal once on Route 3 (cross WEST to Pewter) — PC interiors share ONE layout so the generic
# heal_at_center(door) works for any city. (Pewter Mart door = (28,18), banked for later TM/shopping.)
PEWTER_PC_DOOR, PEWTER_MART_DOOR = (17, 25), (28, 18)
# Oak's Parcel quest (disasm: ViridianCity/PalletTown + their interiors). Viridian Mart town door
# (36,19) -> interior; clerk at (2,3) -> talk from the front tile (2,4) facing UP. Oak's Lab town
# door (16,13) in Pallet -> interior; Oak at (6,3) -> talk from (6,4) UP. The 5 Poke Balls are GIVEN
# by the lab ReceiveDexScene (giveitem ITEM_POKE_BALL,5) on delivery — NO shop-buy. FLAG_SYS_POKEDEX_
# GET=0x829 is set on delivery (the "parcel delivered" proof, alongside ball count 0->5).
VIRIDIAN_MART_DOOR, MART_CLERK_FRONT = (36, 19), (2, 4)
OAK_LAB_DOOR, OAK_FRONT = (16, 13), (6, 4)
FLAG_POKEDEX_GET = 0x829
# Mt Moon entrance (disasm Route4 warp_events): the 1F mouth is a warp at (19,5) ON ROUTE 4 — Route 3
# has NO Mt Moon warp; it connects UP to Route 4 (offset 60), and the cave is entered from Route 4.
MT_MOON_ENTRANCE = (19, 5)
ROUTE4_PC_DOOR = (12, 5)        # Route 4 has its own Pokemon Center (disasm Route4 warp_events)
# Cerulean Gym (reconned 2026-06-26, recon_gymscan): town door (31,21) -> interior; Misty (8,6,
# trainerType=0) is fought from the front tile (8,7). TWO junior trainers gate her: (10,12) and a
# WANDERER (4,7..7,7) - both trainerType!=0. badge flag 0x821 (Cascade) follows Boulder (0x820).
CERULEAN, CERULEAN_GYM_DOOR, MISTY_FRONT = (3, 3), (31, 21), (8, 7)
# FLAG_BADGE0x_GET in the SaveBlock1 flag array (base + 0x0EE0): Boulder 0x820, Cascade 0x821.
FLAG_BADGE_BOULDER, FLAG_BADGE_CASCADE = 0x820, 0x821

# ── GYM REGISTRY: one row per leader, so beat_gym is data-driven + general (gyms gate the leader
# behind junior trainers - beat all juniors, THEN the leader). reserve = move-slots to free for an
# expected level-up double-learn (Brock's L15); leader_dir = key that faces the leader from the
# front tile (UP for Brock + Misty - the leader sits directly above their front tile). ──
GymSpec = namedtuple("GymSpec", ["name", "city", "door", "leader_front", "badge_flag",
                                 "reserve", "leader_dir"])
GYMS = {
    "Brock": GymSpec("Brock", PEWTER, PEWTER_GYM_DOOR, BROCK_FRONT, FLAG_BADGE_BOULDER, 2, "UP"),
    "Misty": GymSpec("Misty", CERULEAN, CERULEAN_GYM_DOOR, MISTY_FRONT, FLAG_BADGE_CASCADE, 0, "UP"),
}
# party-mon cached HP (unencrypted): current 0x56, max 0x58 (off GPLAYER_PARTY)
P_HP, P_MAXHP = 0x56, 0x58

# ── THE CAMPAIGN: Pallet -> Pewter -> Brock, as an objective list ─────────────
# Each objective: (TYPE, *args, label). The driver executes them in order, never
# exiting on arrival. Gates the harness can't script yet are GATE_NEEDS_STATE.
def build_objectives():
    return [
        ("WALK_TO_MAP", VIRIDIAN, "north", "Route 1 -> Viridian City"),
        # Oak's Parcel STORY BEAT (Skip 4, AUTO): Viridian Mart clerk gives OAK'S PARCEL -> carry it
        # to Oak in Pallet -> Pokedex + 5 Poke Balls (the lab ReceiveDexScene's giveitem, NOT a shop
        # buy). The balls are what unblock the Route 3 catch. Interior nav + dialogue only.
        ("DELIVER_PARCEL", "Viridian Mart -> OAK'S PARCEL -> Oak's lab -> Pokedex + 5 Poke Balls"),
        ("ADVANCE_NORTH", PEWTER, "Viridian -> Route 2 -> Forest -> Pewter (auto walk/warp)"),
        ("LEVEL_CHECK", 13, "Brock-readiness (need ~Lv13 for Vine Whip + to survive Onix)"),
        ("BEAT_GYM", "Brock", "Pewter Gym -> Brock -> Boulder Badge (beat_gym enters + fights)"),
    ]


# ── SEGMENT MANIFEST: the resumable spine that grows to the credits ──────────
# A SEGMENT is a named run of objectives ending at a stable OVERWORLD checkpoint. run_segments()
# plays them in order and AUTO-SAVES seg_<name>.state after each, so a continuous "Go" RESUMES from
# the furthest checkpoint instead of replaying the whole game. Hand-play-only gates stay
# GATE_NEEDS_STATE. Each future route/gym is ONE more line here. Additive — build_objectives()/run()
# are untouched, so --fast and the proven Brock arc keep working exactly as before.
Segment = namedtuple("Segment", ["name", "objectives", "checkpoint"])


def build_segments():
    """The whole-game spine as ordered, resumable segments. TODAY: the proven Pallet->Brock arc as
    segment 1, checkpointing the post-badge Pewter overworld. Later segments append below — each its
    OWN recon+build (east-edge travel, Mt Moon cave nav, Mart buy, catch, Misty), so we never ship
    a leg blind. Uncomment/extend as each is built and proven."""
    return [
        # THE AUTONOMOUS OPENING (replaces the old after_pick_bulbasaur hand-bank): from a FRESH ROM
        # boot, Kira names herself + rival + picks gender -> bedroom -> Pallet -> Oak's lab -> picks
        # her starter -> rival battle -> Pallet. Run segment 0 from a FRESH boot (no --boot state);
        # its checkpoint seg_opening.state (= a clean post-opening Pallet) feeds pallet_to_brock.
        Segment("the_opening", [
            ("OPENING", "KIRA", "GARY",
             "fresh new game -> name self+rival (her choice) -> pick starter -> rival battle -> Pallet"),
        ], "seg_opening.state"),
        Segment("pallet_to_brock", build_objectives(), "seg_boulder_badge.state"),
        # Pewter -> Route 3: AUTO (Skip 1 solved). The "collision paradox" was a travel bug, not a game
        # wall: E/W map connections load a DEEP overlap of the neighbour's tiles past the edge (Pewter
        # east reaches x=55, past sx_hi=48; disasm offset=10 -> Route 3). The old "x==sx_hi" exit goal
        # trapped the agent the instant it stepped into the overlap (BFS dragged it back west). travel
        # now crosses on "x>=sx_hi" and presses east THROUGH the overlap until the map flips. Control:
        # brock_done.state Pewter(3,2)@(15,16) -> walk east -> MAP TRANSITION -> Route 3(3,21)@(0,11).
        Segment("pewter_to_route3", [
            ("WALK_TO_MAP", ROUTE3, "east", "Pewter -> Route 3 (cross the east connection band)"),
        ], "seg_route3_entered.state"),
        # CATCH = AUTO (Skip 2 + Skip 4). The "agent can't drive the battle menu" wall was a stale
        # claim (the phantom-A bug was fixed 2026-06-25; catch_pokemon is control-proven party N->N+1).
        # catch_one wanders Route 3 grass (warp-avoiding), heals the lone starter at Pewter between
        # gauntlet trainers, and commits Poke Balls to the wild — using the 5 balls Oak's Parcel
        # (Skip 4) put in the bag. ROSTER_REACT then voices her new teammate. Control: from a Route 3
        # state with balls, catch_one -> party 1->2 (caught a Pidgey).
        Segment("route3_catch", [
            ("CATCH", "wander Route 3 grass + catch a teammate (whatever appears — her journey, not a script)"),
            ("ROSTER_REACT", "Kira reacts to her new caught teammate, in her own voice"),
        ], "seg_route3_caught.state"),
        # Route 3 -> Mt Moon -> Route 4: the cave clear is PROVEN (recon_mtmoon_clear: warp chain +
        # fossil/Miguel + east exit, mtmoon_cleared/cerulean_caught banked) but lives in a recon
        # Route 3 -> Mt Moon -> Route 4 -> Cerulean: FULLY AUTO now (Skip 3). The "entrance seam" was
        # a geography misread — Route 3 has NO Mt Moon warp; it connects UP to Route 4, and the cave
        # mouth is a warp at (19,5) ON ROUTE 4 (disasm). So: cross north to Route 4, step into the Mt
        # Moon entrance, then hand to the PROVEN clear_mt_moon (untouched nav; now flees wilds / fights
        # only trainers in the cave, per the solo-Ivysaur reality). Cave-clear emerges on Route 4 and
        # crosses east to Cerulean itself.
        Segment("route3_to_cerulean", [
            ("WALK_TO_MAP", ROUTE4, "north", "Route 3 -> Route 4 (north connection)"),
            ("HEAL_NEAREST", "top up at the Route 4 Pokemon Center before the cave (survive Miguel)"),
            ("ENTER_WARP", MT_MOON_ENTRANCE, "step into the Mt Moon entrance (Route 4 (19,5))"),
            ("CLEAR_MT_MOON", "follow the saved route (run_plan) Mt Moon -> Route 4 -> Cerulean; "
             "flee wilds, fight trainers; a Miguel death -> blackout-recovery retries the segment"),
        ], "seg_cerulean.state"),
        # Misty gym: FULLY AUTO + proven (general beat_gym - clears the 2 junior trainers, beats
        # Misty, dialogue-paced award -> Cascade flag). From cerulean_caught (Route 4) walk EAST into
        # Cerulean, then beat the gym.
        Segment("beat_misty", [
            ("WALK_TO_MAP", CERULEAN, "east", "Route 4 -> Cerulean City"),
            ("BEAT_GYM", "Misty", "Cerulean Gym -> Cascade Badge"),
        ], "seg_cascade_badge.state"),
    ]


def log(m):
    print(f"   [campaign] {m}", flush=True)


# ── reusable objective handlers ──────────────────────────────────────────────
class Campaign:
    def __init__(self, bridge, battle_runner, on_event=None, beat=None, render=None):
        self.b = bridge
        self.battle_runner = battle_runner
        self.on_event = on_event or (lambda s: log(f"[event] {s}"))
        self.beat = beat or (lambda s: None)
        self.render = render or (lambda: None)
        # heal-when-low: the pause fires ONLY during forward traversal, never during a SERVICE
        # navigation (the heal-return south, the PC routing) - else it would re-trigger mid-heal.
        self._suppress_heal = False
        # CLIMAX guard: True only inside beat_gym's scripted award/TM A-drain. The soul-on render
        # (play_live) reads it and STOPS polling/holding dialogue there, so the scripted cutscene
        # drains exactly like the proven headless path (a reading-pace hold injected into the award
        # was freezing the badge moment ~30-100s). Default False -> normal reactive perception.
        self.draining_award = False
        # MODE (Part A): SHOW vs WORKSHOP. SHOW = fresh/canonical spine, banks to states/kira/, any
        # GATE_NEEDS_STATE load is a LOUD violation (target 0). WORKSHOP = resume-from-any scratch,
        # banks to states/workshop/, never writes states/kira/. ckpt_dir is the write/resume target.
        self.show_mode = False
        self._show_violations = 0
        self.ckpt_dir = STATES_WORKSHOP
        # BATCH 2 SCAFFOLD: her Pokémon-self (wants + roster bonds + move-learn beats). Routes every
        # reaction OUT through on_event (the core-Kira seam); decisions are seams Batch 2 fills. The
        # firewall (_pokemon_react/voice/mood/bond/bridge) is NEVER touched - this is mode plumbing.
        try:
            from pokemon_soul import PokemonSoul
            self.soul = PokemonSoul(emit=self.on_event, choose=getattr(self, "_soul_choose", None))
        except Exception:
            self.soul = None
        # one Traveler reused for every WALK leg (BFS + NPC-aware + grass-aware + handoff)
        self.trav = tv.Traveler(bridge, battle_runner=battle_runner, render=self.render,
                                on_event=self.on_event, beat=self.beat,
                                pause_check=lambda: self.needs_heal() and not self._suppress_heal)

    def _advance_dialogue(self, taps=12):
        """Press A to advance/close any open textbox until movement is free again."""
        for _ in range(taps):
            if tv.coords(self.b) is not None and not st.in_battle(self.b):
                # heuristic: if a direction press moves us, dialogue is closed
                pass
            self.b.press("A", 6, 6, self.render, owner="agent")

    def walk_to_map(self, target_map, direction):
        # travel.Traveler now crosses N/S rows AND E/W columns (east = the Pewter->Route 3 unlock).
        if direction not in ("north", "south", "east", "west"):
            log(f"   (unknown edge '{direction}', defaulting 'north')")
            direction = "north"
        # HEAL-AND-RETRY: a lone starter crossing a gauntlet route drops below the heal line mid-leg;
        # travel yields need_heal. Heal at the NEAREST center (location-aware) and retry, so the leg
        # actually completes instead of stopping short (the Route 3 -> Route 4 stall).
        for _ in range(8):
            if tv.map_id(self.b)[0] != 3:           # stuck INSIDE a building (a fragile heal-excursion
                self._exit_to_overworld()           # left her inside) -> get to the overworld first, or
                #                                     travel can never path to a route edge from an interior
            r = self.trav.travel(target_map=target_map, edge=direction)
            if r == "need_heal":
                if self.heal_nearest() == "stuck":
                    log("   !! WALK: heal failed mid-leg - LOUD"); return "stuck"
                continue
            if r in ("no_path", "stuck"):
                # transient NPC-wedge (often a PC door after a heal) — retry; the traveler waits for
                # NPCs each attempt and a fresh plan usually clears once they step off. Bounded by the
                # loop, so a genuine wall still stops loud.
                log(f"   WALK: {r} - retrying (transient NPC-wedge?)")
                continue
            return r
        return "stuck"

    def heal_nearest(self):
        """Heal at the Pokemon Center nearest the CURRENT map (location-aware — the pre-Pewter arc
        heals at Viridian, Route 3 at Pewter one cross west, Route 4 at its own on-map PC). Returns
        'ok' | 'stuck'. Generalises the Viridian-only return_to_center for the wider early game."""
        m = tv.map_id(self.b)
        if m == ROUTE3:
            return "ok" if self._heal_excursion(PEWTER, PEWTER_PC_DOOR, "west",
                                                ROUTE3, "east") == "ok" else "stuck"
        if m == ROUTE4:
            return "ok" if self.heal_at_center(ROUTE4_PC_DOOR) in ("healed", "healed_stuck_inside") else "stuck"
        # default (Route 1/Pallet/Route 2/Forest): the direction-aware Viridian return
        return "ok" if self.return_to_center() not in ("stuck", "battle_loss") else "stuck"

    def enter_warp(self, prefer="nearest", pick=None):
        """REAL warp entry: find door/warp tiles (behavior 0x6x), walk to the tile just
        beside a chosen one, step INTO it, and confirm the map flips. `pick` (x,y) targets a
        specific door; else `prefer` = 'nearest' / 'north' (forward-progress gates are the
        northmost door, approached from the south, stepped UP) / 'south' (the heal-return exit,
        the southmost door, approached from the north, stepped DOWN)."""
        before = tv.map_id(self.b)
        doors = self._door_tiles()
        if not doors:
            log("   no door/warp tiles (0x6x) found on this map"); return "no_warp"
        cur = tv.coords(self.b)
        if pick is not None:
            order = [pick]
        elif prefer == "north":
            order = sorted(doors, key=lambda t: (t[1], abs(t[0] - cur[0])))
        elif prefer == "south":
            order = sorted(doors, key=lambda t: (-t[1], abs(t[0] - cur[0])))
        else:
            order = sorted(doors, key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        # approach from the side OPPOSITE the travel direction, then step INTO the door.
        # north/nearest: stand south of the door, step UP. south: stand north, step DOWN.
        approach_dy, step_key = (-1, "DOWN") if prefer == "south" else (1, "UP")
        log(f"   {len(doors)} door(s); candidates (chosen order): {order}")
        for door in order:
            approach = (door[0], door[1] + approach_dy)  # stand just beside the door
            # FAST reachability pre-check (BFS only) so unreachable doors are skipped
            # instantly instead of a slow walk-then-fail. Uses grass-ALLOWED collision
            # (grid.walkable, not _safe): grass-heavy maps like Viridian Forest are
            # wall-to-wall tall grass, so a grass-free pre-check wrongly skips every door.
            grid = tv.Grid(self.b)
            if not tv.bfs(grid, tv.coords(self.b), lambda t: t == approach,
                          walkable=grid.walkable):
                continue
            log(f"   reachable door {door}: walking to approach {approach}")
            r = self.trav.travel(target_map=None, arrive_coord=approach, max_steps=300)
            if r == "need_heal":
                return "need_heal"          # heal interrupt during the approach -> let caller heal
            if r != "arrived":
                continue
            for _ in range(5):                           # step INTO the doorway -> warp
                self.b.press(step_key, 8, 8, self.render, owner="agent")
                self._advance_dialogue(taps=2)           # gate buildings may print a line
                if tv.map_id(self.b) != before:
                    log(f"   WARPED {before} -> {tv.map_id(self.b)} via door {door}")
                    return "warped"
        log(f"   no reachable door warped (entry geometry?) - LOUD")
        return "no_warp"

    def _door_tiles(self):
        ml = self.b.rd32(0x02036DFC)
        attr = (self.b.rd32(self.b.rd32(ml + 0x10) + 0x14),
                self.b.rd32(self.b.rd32(ml + 0x14) + 0x14))
        w = self.b.rd32(tv.BACKUP_LAYOUT); h = self.b.rd32(tv.BACKUP_LAYOUT + 4)
        mp = self.b.rd32(tv.BACKUP_LAYOUT + 8)
        out = []
        for by in range(h):
            for bx in range(w):
                e = self.b.rd16(mp + (bx + w * by) * 2); mid = e & 0x3FF
                base, idx = (attr[0], mid) if mid < 640 else (attr[1], mid - 640)
                if 0x60 <= (self.b.rd32(base + idx * 4) & 0xFF) <= 0x6F:
                    out.append((bx - 7, by - 7))
        return out

    def advance_north(self, target_map, max_legs=60):
        """Generic 'head north until target_map' - auto-picks WALK (cross a route edge)
        vs WARP (step through a gate-house door) at each map, so the route->gate-house->
        route->forest chain self-discovers instead of being hardcoded leg by leg. The
        Traveler handles BFS/maze/NPC/grass/battle-handoff; this only sequences legs."""
        for leg in range(max_legs):
            m = tv.map_id(self.b)
            if m == target_map:
                return "arrived"
            out = self.trav.travel(target_map=target_map, max_steps=800)
            if out == "arrived":
                continue                       # crossed an edge (maybe more legs north)
            if out == "battle_loss":
                return "battle_loss"
            if out == "need_heal":             # heal-when-low: go south, heal, resume north
                r = self.return_to_center()
                if r in ("stuck", "battle_loss"):
                    return r
                continue
            if out in ("no_path", "stuck"):
                # no walkable north edge here -> it's an interior/gate house: warp north
                log(f"   leg {leg}: no north edge on map {m} - warping north")
                w = self.enter_warp(prefer="north")
                if w == "need_heal":            # got low crossing toward the door -> heal, retry
                    r = self.return_to_center()
                    if r in ("stuck", "battle_loss"):
                        return r
                    continue
                if w != "warped":
                    log(f"   !! ADVANCE stuck on map {m} (no north edge, no north warp)")
                    return "stuck"
        return "timeout"

    HEAL_HP_FRAC = 0.75     # heal-return when lead HP drops below this fraction of max. The
                            # return FLEES wild fights (see _flee_runner) so the retreat costs
                            # ~0 HP - but forest TRAINERS can't be fled, so we keep the trigger
                            # high enough that she meets a return-trainer near full HP and wins
                            # (esp. the fragile first foray at L8).

    def needs_heal(self):
        """True when the lead mon is low enough to risk a blackout. HP-gated: HP depletes
        faster than Tackle's 35 PP, so an HP heal-return tops up PP as a side effect (keeping
        Tackle her Forest move and the Vine-Whip swap a Brock-only tool)."""
        hp, mx = self.lead_hp()
        return mx > 0 and hp < self.HEAL_HP_FRAC * mx

    def _heal_excursion(self, city, pc_door, out_edge, back_map, back_edge):
        """Low HP on a ROUTE with no PC of its own: cross to an ADJACENT city, heal at its PC, cross
        back to the route. Generic city-heal for routes (return_to_center is the Viridian-south
        special case for the pre-Pewter arc). Flees wilds both ways so the trip costs ~0 HP. Returns
        'ok' (back on back_map, healed) | 'stuck'."""
        saved, saved_runner = self._suppress_heal, self.trav.battle_runner
        self._suppress_heal = True                    # don't re-trigger heal WHILE going to heal
        self.trav.battle_runner = self._flee_runner
        try:
            log(f"   HEAL-EXCURSION: lead at {self.lead_hp()} - crossing {out_edge} to {city} to heal")
            for _ in range(6):
                if tv.map_id(self.b) == city:
                    break
                if self.trav.travel(target_map=city, edge=out_edge,
                                    max_steps=400, max_seconds=120) != "arrived":
                    break
            if tv.map_id(self.b) != city:
                log(f"   !! HEAL-EXCURSION: didn't reach {city} (at {tv.map_id(self.b)})")
                return "stuck"
            self.heal_at_center(pc_door)
            healed = self.lead_hp()[0] >= self.lead_hp()[1]    # HP topped up = the heal succeeded
            for _ in range(6):
                if tv.map_id(self.b) == back_map:
                    break
                if self.trav.travel(target_map=back_map, edge=back_edge,
                                    max_steps=400, max_seconds=120) != "arrived":
                    break
            if tv.map_id(self.b) == back_map:
                log(f"   HEAL-EXCURSION: healed + back on {back_map} at {tv.coords(self.b)}")
                return "ok"
            # The HEAL succeeded; the cross-back wedged (typically an NPC parked at the PC door). That
            # is NOT a hard stall — return ok so the caller re-navs toward the real goal FROM HERE
            # (moving away from the door), retrying as the NPC steps off. Heal is best-effort transport.
            log(f"   HEAL-EXCURSION: healed (HP {self.lead_hp()}) but still on {tv.map_id(self.b)} - "
                f"non-fatal, caller re-navs")
            return "ok" if healed else "stuck"
        finally:
            self._suppress_heal, self.trav.battle_runner = saved, saved_runner

    def return_to_center(self):
        """Heal-return to the Viridian Center (the only PC before Pewter), then heal (restores HP +
        PP). DIRECTION-AWARE: Viridian is NORTH when we're on Route 1 / in Pallet (the parcel arc) and
        SOUTH when we're up in Route 2 / the Forest (the Brock arc). The old hard-coded 'south' bounced
        the agent toward Pallet whenever it got low on Route 1. Cross that edge (or warp it) until we're
        on the Viridian map, then heal_at_center (walks to the PC, heals, returns to the pre-heal spot)."""
        up = tv.map_id(self.b) in (PALLET, ROUTE1)     # Viridian is NORTH of us -> head north to heal
        edge = "north" if up else "south"
        log(f"   HEAL-RETURN: lead at {self.lead_hp()} - routing {edge} to Viridian to heal")
        saved = self._suppress_heal
        saved_runner = self.trav.battle_runner
        self._suppress_heal = True              # don't re-trigger heal WHILE returning to heal
        self.trav.battle_runner = self._flee_runner   # RETREAT: flee wild fights on the way out
        try:
            for leg in range(20):
                m = tv.map_id(self.b)
                if m == VIRIDIAN:
                    break
                out = self.trav.travel(target_map=VIRIDIAN, edge=edge, max_steps=800)
                if out == "arrived":
                    continue                       # crossed an edge toward Viridian
                if out == "battle_loss":
                    return "battle_loss"           # blacked out en route -> auto-heals at Viridian
                log(f"   HEAL-RETURN leg {leg}: no {edge} edge on map {m} - warping {edge}")
                if self.enter_warp(prefer=edge) != "warped":
                    log(f"   !! HEAL-RETURN stuck on map {m} (no {edge} edge, no {edge} warp)")
                    return "stuck"
            if tv.map_id(self.b) != VIRIDIAN:
                log(f"   !! HEAL-RETURN did not reach Viridian (at {tv.map_id(self.b)})")
                return "stuck"
            return self.heal_at_center()
        finally:
            self._suppress_heal = saved
            self.trav.battle_runner = saved_runner

    def _flee_runner(self):
        """Battle handler used DURING the heal-return: flee wild fights (retreat costs ~0 HP),
        win forced trainers. Built fresh per battle; routes events to the campaign's voice."""
        return BattleAgent(self.b, on_event=lambda s, **k: self.on_event(s),
                           render=self.render, log=lambda m: None).flee(max_seconds=90)

    def level_check(self, min_level, leader="Brock"):
        """LOUD Brock-readiness check at the Forest exit / before the gym. Reads the lead
        Pokemon's level; flags underleveled with a grind recommendation rather than
        silently walking into a gym loss. (Brock: Geodude L12 / Onix L14.)"""
        lvl = self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        if lvl >= min_level:
            log(f"   LEVEL CHECK: lead Pokemon Lv{lvl} >= {min_level} - {leader}-ready")
            return "ok"
        # NON-FATAL: a fresh flee-to-survive run reaches Pewter ~Lv12 (one short of the conservative
        # Lv13 gate). Bulbasaur's Vine Whip is 2x on Brock's Rock/Ground (Boulder badge proven at this
        # level), and a gym LOSS is caught by the segment's blackout-recovery (respawn + retry) — never
        # a hard stop. So WARN + PROCEED rather than stalling the whole run on a soft readiness gate.
        # (grind() is a built capability for explicit/aggressive leveling, not auto-run here — a full
        # level headless is slow, and it must never block the 'walk away' spine.)
        log(f"   LEVEL CHECK: lead Lv{lvl} < {min_level} for {leader} — proceeding (Vine Whip 2x; a "
            f"loss is recoverable), not stalling on the soft gate")
        self.on_event(f"I'm only level {lvl}, but Vine Whip should still do the job on {leader}")
        return "ok"

    # ── move-learn handler (the empty-slot auto-learn fallback) ────────────────
    # WHY: a mid-battle level-up with 4 moves shows "Delete a move? / Stop learning?" Y/N boxes.
    # Those boxes are UN-ACTUATABLE in libmgba's continuous core (diagnosed 2026-06-24/25: the
    # key reaches KEYINPUT and a FRESH core confirms 6/6, but a long-running core can't land the
    # press at ANY timing/phase, and save+load doesn't reset the hidden core-instance state - only
    # a brand-new core does). So instead of fighting the box, we make it NEVER appear: reserve
    # open move slots so the level-up move AUTO-LEARNS. Bulbasaur learns TWO moves at L15
    # (PoisonPowder + Sleep Powder), so we reserve 2. Keeps her strongest moves (Tackle + Vine
    # Whip); the new moves fill the open slots with no prompt.
    def _attacks_block(self):
        base = ram.GPLAYER_PARTY
        key = self.b.rd32(base) ^ self.b.rd32(base + 4)        # PID ^ OTID
        a = st._SUBSTRUCT_ORDER[self.b.rd32(base) % 24].index("A")
        return base, key, base + 32, a                         # base, key, data_addr, A-substruct idx

    def _lead_moves(self):
        base, key, data, a = self._attacks_block()
        w0 = self.b.rd32(data + a * 12) ^ key
        w1 = self.b.rd32(data + a * 12 + 4) ^ key
        return [w0 & 0xFFFF, (w0 >> 16) & 0xFFFF, w1 & 0xFFFF, (w1 >> 16) & 0xFFFF]

    def _lead_pps(self):
        base, key, data, a = self._attacks_block()
        w = self.b.rd32(data + a * 12 + 8) ^ key               # the 4 PP bytes (one u32)
        return [w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF]

    def _set_lead_moves(self, moves, pps):
        """Write the lead's 4 move IDs + 4 PP bytes wholesale, keeping the Gen-3 checksum valid
        (decrypt the 48-byte data block, overwrite the A-substruct's 3 words, recompute the u16-sum
        checksum, re-encrypt). Used to COMPACT moves (no empty gaps between real moves)."""
        base, key, data, a = self._attacks_block()
        words = [self.b.rd32(data + i * 4) ^ key for i in range(12)]
        words[a * 3] = (moves[0] & 0xFFFF) | ((moves[1] & 0xFFFF) << 16)
        words[a * 3 + 1] = (moves[2] & 0xFFFF) | ((moves[3] & 0xFFFF) << 16)
        words[a * 3 + 2] = ((pps[0] & 0xFF) | ((pps[1] & 0xFF) << 8)
                            | ((pps[2] & 0xFF) << 16) | ((pps[3] & 0xFF) << 24))
        chk = 0
        for w in words:
            chk = (chk + (w & 0xFFFF) + ((w >> 16) & 0xFFFF)) & 0xFFFF
        for i in range(12):
            self.b.core.memory.u32.raw_write(data + i * 4, words[i] ^ key)
        self.b.core.memory.u16.raw_write(base + 0x1C, chk)

    def _reserve_move_slots(self, n):
        """Free `n` move slots on the lead so upcoming level-up moves AUTO-LEARN with no (un-
        actuatable) Y/N box. We COMPACT, not delete-in-place: keep the (4-n) strongest moves in the
        TOP slots and push the empties to the END. WHY compact: the in-battle FIGHT menu displays
        real moves CONTIGUOUSLY, but BattleAgent navigates by RAM slot index - so a gap (e.g.
        [Tackle, _, _, VineWhip]) made it walk the cursor to grid-3 while Vine Whip showed at grid-1,
        and the super-effective move NEVER FIRED (it could only Tackle). Compacting makes RAM order
        == menu order, so the chosen move is actually navigable; trailing empties still feed the
        L15 auto-learn (new moves land in the lowest empty slot, i.e. the end). Keeps >= 2 moves."""
        moves, pps = self._lead_moves(), self._lead_pps()
        real = [(i, m) for i, m in enumerate(moves) if m]      # (slot, moveId) of real moves
        # MOVE-LEARN-AS-A-BEAT (Batch 2) is SCAFFOLDED in pokemon_soul.move_drop_decision but NOT wired
        # here yet: its first cut DROPPED the super-effective move (broke Brock), so per the safety
        # rail we keep the PROVEN keep-strongest-by-power reserve until the beat is debugged.
        keep_count = max(2, len(real) - n)                     # free up to n, never drop below 2
        ranked = sorted(real, key=lambda im: st.move_info(self.b, im[1])[1], reverse=True)
        keep = sorted(ranked[:keep_count], key=lambda im: im[0])   # strongest kept, in slot order
        new_moves = [m for _, m in keep] + [0] * (4 - len(keep))
        new_pps = [pps[i] for i, _ in keep] + [0] * (4 - len(keep))
        self._set_lead_moves(new_moves, new_pps)
        for _ in range(4):
            self.b.run_frame()

    def has_badge(self, flag):
        """Read any FLAG_BADGE0x_GET from the SaveBlock1 flag array (base + 0x0EE0)."""
        sb1 = self.b.rd32(0x03005008)                          # gSaveBlock1Ptr (DMA-shuffled target)
        fa = sb1 + 0x0EE0 + (flag >> 3)
        return bool(self.b.rd8(fa) & (1 << (flag & 7)))

    def has_boulder_badge(self):
        return self.has_badge(FLAG_BADGE_BOULDER)

    # ── GENERAL gym-trainer gauntlet (a gym LEADER is gated behind the junior trainers) ──────────
    _OB, _SZ = 0x02036E38, 0x24
    _DELTA = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}     # facing nibble -> look direction
    _TOWARD = {(1, 0): "RIGHT", (-1, 0): "LEFT", (0, 1): "DOWN", (0, -1): "UP"}

    def _gym_trainers(self):
        """Every loaded TRAINER object event (trainerType 1/2): (object index, coord, facing). We
        track 'beaten' by INDEX, not coord - a gym trainer can WANDER (Cerulean's 2nd swimmer roams
        (4,7)->(7,7)), and a DEFEATED trainer still scans as trainerType, so coord-tracking would
        both re-engage a wanderer and lose track of who's done. The index is stable."""
        out = []
        for i in range(1, 16):
            o = self._OB + i * self._SZ
            if (self.b.rd8(o) & 1) and self.b.rd8(o + 0x07) in (1, 2):
                c = (self.b.rds16(o + 0x10) - 7, self.b.rds16(o + 0x12) - 7)
                out.append((i, c, self.b.rd8(o + 0x18) & 0x0F))
        return out

    def _engage_trainer(self, T, facing):
        """Walk to a tile adjacent to a junior trainer (its facing-front first, then any reachable
        side), face it, A -> battle. Its line of sight may auto-fire mid-approach (caught)."""
        order = [self._DELTA.get(facing, (0, 1)), (0, 1), (0, -1), (-1, 0), (1, 0)]
        seen = set()
        for adj in order:
            if adj in seen:
                continue
            seen.add(adj)
            front = (T[0] + adj[0], T[1] + adj[1])
            self.trav.travel(target_map=None, arrive_coord=front, max_steps=200, max_seconds=90)
            if st.in_battle(self.b):
                return True                                    # LoS fired during the walk-in
            if tv.coords(self.b) != front:
                continue                                       # couldn't reach this side - try next
            face = self._TOWARD[(-adj[0], -adj[1])]            # turn back toward the trainer body
            for _ in range(4):
                if st.in_battle(self.b):
                    return True
                self.b.press(face, 8, 8, self.render, owner="agent")
                self.b.press("A", 6, 12, self.render, owner="agent")
                for _ in range(20):
                    self.b.run_frame()
            if st.in_battle(self.b):
                return True
        return False

    def _drain_overworld(self, label="dlg"):
        """Drive any overworld dialogue box (a trainer's post-battle line, an NPC, the badge award)
        to a clean close at a watchable pace via the general primitive. No box -> returns at once."""
        self.b.set_input_owner("agent")
        return DialogueDriver(self.b, render=self.render,
                              log=lambda m: log(m)).drive(label=label)

    def _clear_gym_trainers(self, leader_front, max_rounds=10):
        """Beat EVERY junior trainer (re-scanning each round, since far ones are proximity-loaded
        and one may wander), THEN return so the caller engages the gated leader. General gym
        mechanic - reusable for Surge/Erika/etc."""
        beaten = set()                                         # object indices already cleared
        for rnd in range(max_rounds):
            self.b.set_input_owner("agent")
            trs = [(i, c, f) for (i, c, f) in self._gym_trainers() if i not in beaten]
            if trs:
                px, py = tv.coords(self.b) or (0, 0)
                trs.sort(key=lambda t: abs(t[1][0] - px) + abs(t[1][1] - py))
                idx, T, facing = trs[0]
                log(f"   GYM: engaging junior trainer obj{idx} at {T} (facing {facing})")
                self._engage_trainer(T, facing)
                if st.in_battle(self.b):
                    log(f"   GYM: junior trainer -> {self.battle_runner()}")
                    self._drain_overworld(label="trainer")
                beaten.add(idx)                                # cleared (or unreachable) -> don't reloop
                continue
            # none loaded -> advance toward the leader to load/trigger far trainers (LoS may fire)
            self.trav.travel(target_map=None, arrive_coord=leader_front,
                             max_steps=200, max_seconds=90)
            if st.in_battle(self.b):
                log(f"   GYM: en-route LoS trainer -> {self.battle_runner()}")
                self._drain_overworld(label="trainer")
                nt = [(i, c, f) for (i, c, f) in self._gym_trainers() if i not in beaten]
                if nt:
                    px, py = tv.coords(self.b) or (0, 0)
                    nt.sort(key=lambda t: abs(t[1][0] - px) + abs(t[1][1] - py))
                    beaten.add(nt[0][0])
                continue
            if [(i, c, f) for (i, c, f) in self._gym_trainers() if i not in beaten]:
                continue                                       # a far trainer just loaded -> engage it
            log(f"   GYM: all junior trainers cleared (beaten obj {sorted(beaten)})")
            return
        log(f"   !! GYM: hit {max_rounds} clear-rounds with trainers still loaded - proceeding LOUD")

    def beat_gym(self, name):
        """GENERAL gym handler (gyms gate the leader behind their junior trainers): reserve move
        slots if this gym has a level-up double-learn, enter, BEAT EVERY JUNIOR TRAINER, then engage
        the gated leader, beat them with the current team, drive the badge/TM award to a clean close
        at a watchable pace, and confirm the badge flag. Data-driven via GYMS - one row per leader."""
        gym = GYMS.get(name)
        if gym is None:
            log(f"   !! GYM: no spec for '{name}'"); return "stuck"
        if gym.reserve:
            self._reserve_move_slots(gym.reserve)
            log(f"   GYM: reserved {gym.reserve} slot(s) for a level-up learn; moves now "
                f"{[st.MOVE_NAMES.get(m, '#' + str(m)) for m in self._lead_moves()]}")
        if tv.map_id(self.b) == gym.city:
            if self.enter_warp(pick=gym.door) != "warped":
                log(f"   !! GYM: couldn't enter the {name} gym"); return "stuck"
        for _ in range(45):
            self.b.run_frame()
        gym_map = tv.map_id(self.b)                             # the gym interior we just entered
        log(f"   GYM: inside {gym_map} at {tv.coords(self.b)} - clearing junior trainers")
        # 1) BEAT THE JUNIOR TRAINERS FIRST (the leader is gated until they're all down)
        self._clear_gym_trainers(gym.leader_front)
        # BLACKOUT during the juniors -> she whited out + respawned in the city PC (the map left the
        # gym interior). _clear_gym_trainers can't tell (no trainers load in the PC), so detect it here
        # and propagate -> the segment's blackout-recovery respawns + RE-RUNS the gym (beaten juniors
        # stay beaten in-game, so only the one that won re-fights; a fresh full-HP, on-level run wins).
        if tv.map_id(self.b) != gym_map:
            log(f"   GYM: no longer in the gym ({tv.map_id(self.b)} != {gym_map}) - blackout during "
                f"the juniors; caller recovers + retries the gym")
            return "battle_loss"
        # 2) engage the LEADER (now ungated): walk to the front tile, face them + A to initiate,
        # then DRIVE the pre-battle challenge speech INTO the battle (Brock's "So, you're here..."
        # is a multi-box speech that a few A-taps don't fully clear; the primitive advances it and
        # stops the instant the battle starts). Misty starts the battle near-immediately - same path.
        self.b.set_input_owner("agent")
        self.trav.travel(target_map=None, arrive_coord=gym.leader_front, max_steps=200, max_seconds=90)
        _lvl = self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        _sp = st.SPECIES_NAME.get(st.read_party_species(self.b, 0), "?")
        log(f"   GYM: at {tv.coords(self.b)} (leader front {gym.leader_front}) - engaging {name} "
            f"[lead {_sp} Lv{_lvl}, party={self.b.rd8(ram.GPLAYER_PARTY_CNT)}]")
        for _ in range(6):
            if st.in_battle(self.b) or dd_box_open(self.b):
                break
            self.b.press(gym.leader_dir, 8, 8, self.render, owner="agent")
            self.b.press("A", 8, 8, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        if not st.in_battle(self.b):                           # advance the challenge speech -> battle
            DialogueDriver(self.b, render=self.render, log=lambda m: log(m)).drive(
                stop_when=lambda: st.in_battle(self.b), label=f"{name}-challenge")
        if st.in_battle(self.b):
            res = self.battle_runner()
            log(f"   GYM: {name.upper()} -> {res}")
            # LOSING the leader is a SURVIVAL blackout (she whites out -> respawns at the city PC),
            # not a nav stuck. Propagate it so the segment's blackout-recovery respawns + retries the
            # whole gym (re-walk, junior trainers stay beaten, fresh full-HP leader attempt). Without
            # this, beat_gym fell through to the award drain on a loss -> 'stuck' inside the PC.
            if res == "loss":
                log(f"   GYM: lost to {name} (blackout) - caller recovers + retries the gym")
                return "battle_loss"
        # 3) AWARD (the climax): drive the badge + TM gift to a clean close at a WATCHABLE pace via
        # the general dialogue primitive. draining_award tells the soul-on render (play_live) to stop
        # polling/holding here so nothing competes with the drain. The proven brisk A-mash stays as a
        # BACKSTOP - the climax must NEVER freeze, so if the paced drive hasn't set the badge we fall
        # back to it immediately (preserves the regression-green reliability).
        self.draining_award = True
        try:
            self._drain_overworld(label=f"{name}-award")
            if not self.has_badge(gym.badge_flag):
                log("   !! GYM: paced award didn't set the badge - A-mash backstop")
                for k in range(160):
                    self.b.press("A", 8, 8, self.render, owner="agent")
                    for _ in range(16):
                        self.b.run_frame()
                    if self.has_badge(gym.badge_flag):
                        break
                for _ in range(30):                            # clear any TM gift -> overworld
                    self.b.press("A", 8, 8, self.render, owner="agent")
                    for _ in range(16):
                        self.b.run_frame()
        finally:
            self.draining_award = False
        if self.has_badge(gym.badge_flag):
            log(f"   GYM: *** {name.upper()} BADGE obtained ***")
            self.on_event(f"I beat {name} - that's the badge!")
            if tv.map_id(self.b)[0] != 3:                      # still in the gym interior -> leave to
                self._exit_to_overworld()                      # the city via the general hardened exit
                log(f"   GYM: exit -> now {tv.map_id(self.b)}@{tv.coords(self.b)}")
            return "badge"
        log(f"   !! GYM: {name} not beaten / no badge flag"); return "stuck"

    # ── new-game CHARACTER CREATION (the autonomous opening debut) ──────────────
    def create_character(self, player_name, rival_name, girl=True):
        """Drive FireRed's new-game intro AUTONOMOUSLY to the bedroom: Oak's welcome -> GENDER pick
        -> name HERSELF -> name the RIVAL. CAPABILITY-not-script: the names + gender are passed in
        (Kira's choices / a Batch-2 seam), NEVER hardcoded. Assumes the boot is already past the
        title into the intro (the opening driver does title->New Game). Returns 'bedroom'|'stuck'.
        Proven: KIRA/GARY/GIRL -> bedroom, committed playerName+gender+rival all correct."""
        from naming import name_entry
        names = [player_name, rival_name]
        ni = 0
        for _ in range(700):
            if tv.map_id(self.b) == (4, 1):                    # reached the bedroom
                return "bedroom"
            low = self._read_overworld_text().lower()
            if "boy" in low and "girl" in low:                 # GENDER select (default BOY at top)
                if girl:
                    self.b.press("DOWN", 8, 10, self.render, owner="agent")
                    for _ in range(10):
                        self.b.run_frame()
                self.b.press("A", 8, 10, self.render, owner="agent")
                for _ in range(24):
                    self.b.run_frame()
                continue
            if ("name" in low and "?" in low) and ni < 2:      # name prompt -> open kb + type
                for _ in range(6):
                    if not dd_box_open(self.b):
                        break
                    self.b.press("A", 4, 10, self.render, owner="agent")
                    for _ in range(16):
                        self.b.run_frame()
                for _ in range(40):
                    self.b.run_frame()
                if not dd_box_open(self.b):                     # keyboard is up
                    name_entry(self.b, names[ni], render=self.render)
                    log(f"   OPENING: named {'self' if ni == 0 else 'rival'} = {names[ni]!r}")
                    ni += 1
                    continue
            self.b.press("A", 4, 10, self.render, owner="agent")  # advance the intro dialogue
            for _ in range(12):
                self.b.run_frame()
        return "stuck"

    def _read_overworld_text(self):
        from dialogue_reader import DialogueReader
        try:
            return DialogueReader(self.b)._read_buffer()
        except Exception:
            return ""

    def _adv_dialogue(self, n=50):
        """Advance overworld dialogue (A) until the box closes or n presses."""
        for _ in range(n):
            if not dd_box_open(self.b):
                break
            self.b.press("A", 4, 10, self.render, owner="agent")
            for _ in range(12):
                self.b.run_frame()

    def _title_to_newgame(self):
        """From a FRESH ROM boot: mash A/START through the title + copyright screens into a New Game,
        stopping once Oak's intro dialogue is rolling (so drive_opening can take over)."""
        from dialogue_reader import DialogueReader
        dr = DialogueReader(self.b)
        for k in range(80):
            low = dr._read_buffer().lower()
            if "welcome" in low or "boy" in low:
                return
            self.b.press("A" if k % 2 else "START", 4, 8, self.render, owner="agent")
            for _ in range(10):
                self.b.run_frame()

    def drive_opening(self, player_name, rival_name, girl=True, starter_choice=0):
        """THE AUTONOMOUS OPENING (the personality debut), assuming the boot is already at Oak's
        intro (the caller does title->New Game). Character creation (her gender + names) -> bedroom
        -> downstairs (FireRed stairs fire on walking INTO them from the SIDE, not step-on) -> Mom
        -> Pallet -> walk north -> Oak intercepts to the lab -> PICK her starter -> the rival battle
        -> walk back out to Pallet. CAPABILITY-not-script: gender/names/starter are HER choices
        (params/seams), never hardcoded. Returns the final map id."""
        if self.create_character(player_name, rival_name, girl) != "bedroom":
            log("   !! OPENING: character creation didn't reach the bedroom"); return tv.map_id(self.b)
        self._adv_dialogue()                                   # bedroom "legend unfold" + NES cutscene
        log(f"   OPENING: in bedroom {tv.coords(self.b)} - heading downstairs")
        self.trav.travel(target_map=None, arrive_coord=(11, 2), max_steps=60, max_seconds=40)
        for _ in range(4):                                     # walk WEST into the stairs (side-trigger)
            self.b.press("LEFT", 8, 8, self.render, owner="agent")
            for _ in range(8):
                self.b.run_frame()
            if tv.map_id(self.b) != (4, 1):
                break
        self._adv_dialogue()                                   # Mom (1F)
        if not self._exit_to_overworld():                      # general hardened building-exit
            log("   !! OPENING: couldn't exit the house"); return tv.map_id(self.b)
        self._adv_dialogue()
        log(f"   OPENING: in Pallet {tv.coords(self.b)} - walking north (Oak will intercept)")
        for _ in range(160):                                   # north -> Oak intercept -> lab (4,3)
            if tv.map_id(self.b) == (4, 3):
                break
            if dd_box_open(self.b):
                self.b.press("A", 4, 10, self.render, owner="agent")
                for _ in range(12):
                    self.b.run_frame()
                continue
            out = self.advance_north(PALLET if tv.map_id(self.b) != PALLET else ROUTE1, max_legs=2)
            if out in ("stuck", "timeout"):
                self.b.press("UP", 8, 8, self.render, owner="agent")
                for _ in range(8):
                    self.b.run_frame()
        if tv.map_id(self.b) != (4, 3):
            log("   !! OPENING: Oak intercept didn't reach the lab"); return tv.map_id(self.b)
        log("   OPENING: in Oak's lab - picking her starter")
        self._starter_choice = starter_choice
        if self.pick_starter() != "picked":
            log("   !! OPENING: starter pick failed"); return tv.map_id(self.b)
        for _ in range(60):                                    # advance until the RIVAL battle starts
            if st.in_battle(self.b):
                break
            self.b.press("A", 6, 10, self.render, owner="agent")
            for _ in range(14):
                self.b.run_frame()
        if st.in_battle(self.b):
            log(f"   OPENING: RIVAL battle -> {self.battle_runner()}")    # win OR lose - game goes on
            self.b.set_input_owner("agent")
        # walk out to Pallet - ROBUST to win OR loss (the post-battle dialogue + door differ a little
        # by outcome): drain dialogue + retry the south exit until we're actually out of the lab.
        for _ in range(4):
            if tv.map_id(self.b) != (4, 3):
                break
            self._drain_overworld(label="post-rival")
            self.b.set_input_owner("agent")
            self._exit_to_overworld()
        log(f"   OPENING: *** DONE -> {tv.map_id(self.b)}@{tv.coords(self.b)} "
            f"party={self.b.rd8(ram.GPLAYER_PARTY_CNT)} ***")
        return tv.map_id(self.b)

    # ── starter pick (Oak's lab) - CAPABILITY not DECISION ──────────────────────
    # The three Poké Balls on Oak's table (recon_starter / starter.state): left->right.
    STARTER_BALLS = {0: (8, 4), 1: (9, 4), 2: (10, 4)}         # ball tiles (face UP from below)
    STARTER_SPECIES = {0: 1, 1: 4, 2: 7}                       # Bulbasaur / Charmander / Squirtle

    def choose_starter(self):
        """DECISION SEAM - which starter Kira takes. Capability-not-script: the HANDS (pick_starter)
        work for ANY of the three; WHICH she picks is HER choice, set on self._starter_choice (Batch
        2 routes this through her soul). NOT hardcoded into the pick mechanics. Default 0 (Bulbasaur)
        is only a fallback when nothing set the choice - the point is the seam exists + is honoured."""
        c = getattr(self, "_starter_choice", None)
        return c if c in (0, 1, 2) else 0

    def choose_nickname(self):
        """DECISION SEAM - does Kira nickname her new Pokemon, and what? Returns a name str to name
        it (via name_entry) or None to DECLINE. Capability-not-script: set self._nickname (Batch 2
        routes this through her soul). Default None = decline cleanly (NOT A-spam)."""
        nk = getattr(self, "_nickname", None)
        return nk if isinstance(nk, str) and nk.strip() else None

    def _handle_nickname(self):
        """The 'Give a nickname to <X>?' Yes/No prompt after acquiring a Pokemon. Her choice: name it
        (name_entry on the SAME keyboard as self/rival naming) or decline (No). Fixes the A-spam that
        nicknamed the starter 'AAAA'."""
        from naming import name_entry
        nick = self.choose_nickname()
        for _ in range(30):                                    # advance toward the nickname prompt
            low = self._read_overworld_text().lower()
            if "nickname" in low:
                if nick:
                    for _ in range(8):                                     # YES, then advance the
                        if not dd_box_open(self.b):                        # Yes/No -> the keyboard
                            break                                          # (opens after a few A's)
                        self.b.press("A", 8, 10, self.render, owner="agent")
                        for _ in range(16):
                            self.b.run_frame()
                    for _ in range(40):                                    # SETTLE: the fresh keyboard
                        self.b.run_frame()                                 # eats the first tap otherwise
                    name_entry(self.b, nick, render=self.render)           # type her chosen name
                    log(f"   STARTER: nicknamed it {nick!r} (her choice)")
                else:
                    self.b.press("B", 8, 10, self.render, owner="agent")   # NO -> decline (no A-spam)
                    for _ in range(20):
                        self.b.run_frame()
                    log("   STARTER: declined a nickname (her choice)")
                return
            self.b.press("A", 4, 10, self.render, owner="agent")
            for _ in range(12):
                self.b.run_frame()

    def pick_starter(self, idx=None):
        """HANDS: walk below the chosen ball, face it, take it + confirm YES, drive Oak's dialogue
        until the party grows, then handle the nickname prompt (her choice). Choice-agnostic (ball
        0/1/2). Returns 'picked' | 'stuck'. WORKS when the lab is reached CONTINUOUSLY (Oak's full
        intro grants pick-permission) - the dead starter.state mid-dialogue capture was the old wall."""
        if idx is None:
            idx = self.choose_starter()
        bx, by = self.STARTER_BALLS[idx]
        log(f"   STARTER: ball {idx} at {(bx, by)} (her choice; hands are choice-agnostic)")
        # 1) advance Oak's intro speech ("Those are POKé BALLS...") until control returns
        DialogueDriver(self.b, render=self.render, log=lambda m: log(m)).drive(label="oak-intro")
        # 2) walk below the chosen ball + open the "Do you want X?" prompt
        self._step_to((bx, by + 1))                            # stand directly below the ball
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        for _ in range(6):
            self.b.press("UP", 8, 8, self.render, owner="agent")
            self.b.press("A", 8, 8, self.render, owner="agent")
            for _ in range(18):
                self.b.run_frame()
            if dd_box_open(self.b) or self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                break
        # 3) confirm YES (default cursor) + drive Oak's dialogue until the party grows
        for _ in range(40):
            if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:         # stop the instant we own it
                break
            self.b.press("A", 8, 8, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
            sp = st.read_party_species(self.b, 0)
            log(f"   STARTER: *** picked {st.SPECIES_NAME.get(sp, '?')} (#{sp}) ***")
            self.on_event(f"I'll go with {st.SPECIES_NAME.get(sp, 'this one')}")
            self._handle_nickname()                            # the "give a nickname?" prompt - her choice
            return "picked"
        log("   !! STARTER: no Pokemon obtained - pick stuck (LOUD)")
        return "stuck"

    # ── Oak's Parcel quest (the story beat that yields the Pokedex + the 5 catch-enabling balls) ──
    def _ball_count(self):
        """Poke Balls (item id 4) in the bag's balls pocket; quantity XOR'd with the SaveBlock2 key
        (mirrors BattleAgent._ball_count). 0 -> nothing to throw (the catch gate before this quest)."""
        sb1 = self.b.rd32(ram.GSAVEBLOCK1_PTR)
        key = self.b.rd32(self.b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
        for i in range(16):
            iid = self.b.rd16(sb1 + 0x430 + i * 4)
            if iid == 0:
                break
            if iid == 4:
                return self.b.rd16(sb1 + 0x430 + i * 4 + 2) ^ key
        return 0

    def _talk(self, front, face_dir, label):
        """Interior NPC interaction: drain any cutscene already running, step to the front tile,
        face the NPC, A, then drive the dialogue/cutscene to a clean close (the giveitem_msg lines)."""
        self.b.set_input_owner("agent")
        if dd_box_open(self.b):
            self._drain_overworld(label=label + "-pre")
        self._step_to(front)
        for _ in range(6):
            if dd_box_open(self.b) or st.in_battle(self.b):
                break
            self.b.press(face_dir, 8, 8, self.render, owner="agent")
            self.b.press("A", 8, 10, self.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()
        self._drain_overworld(label=label)

    def deliver_parcel(self):
        """Oak's Parcel STORY BEAT (disasm-seeded), converting the viridian_parcel_done GATE -> AUTO.
        From Viridian: enter the Mart -> the clerk gives OAK'S PARCEL; carry it south to Oak's lab in
        Pallet -> deliver -> the ReceiveDexScene grants the Pokedex (FLAG 0x829) + 5 Poke Balls
        (giveitem, NOT a shop buy). All dialogue + interior nav. Returns 'done'|'stuck'. The real
        proof is the ball count going 0 -> 5 (that's what unblocks the Route 3 catch)."""
        b0 = self._ball_count()
        if b0 >= 5 and self.has_badge(FLAG_POKEDEX_GET):       # idempotent: a retry after it's DONE
            log("   PARCEL: already delivered (Pokedex + balls present) - done"); return "done"
        log(f"   PARCEL: starting Oak's Parcel quest (balls={b0}, map={tv.map_id(self.b)})")
        # This is a SHORT, low-danger errand that passes through Viridian (which HAS a PC). Suppress
        # the heal-when-low BOUNCE for the whole errand and instead heal ONCE explicitly at the
        # Viridian PC up front — else every Route 1 wild dropping the lone starter below 75% would
        # yield need_heal and abort each travel leg (the south-to-Pallet stall).
        saved = self._suppress_heal
        saved_runner = self.trav.battle_runner
        self._suppress_heal = True
        # FLEE wild encounters for the whole fetch errand — Route 1/2 have no required trainers before
        # the parcel, and a just-picked Lv5 starter can't tank the round-trip's wilds (it blacked out
        # in Part C). Fleeing costs ~0 HP. A genuine flee-fail death PROPAGATES as battle_loss so the
        # segment's blackout-recovery retries it (instead of being swallowed into a non-recoverable 'stuck').
        self.trav.battle_runner = self._flee_runner
        try:
            for _ in range(5):                                 # 1) reach Viridian (the Mart is here)
                if tv.map_id(self.b) == VIRIDIAN:
                    break
                if self.advance_north(VIRIDIAN, max_legs=3) == "battle_loss":
                    return "battle_loss"
            if tv.map_id(self.b) != VIRIDIAN:
                log(f"   !! PARCEL: not at Viridian (at {tv.map_id(self.b)})"); return "stuck"
            if self.needs_heal():                              # 1b) top up so the round trip is safe
                self.heal_at_center(VIRIDIAN_PC_DOOR)
            if self.enter_warp(pick=VIRIDIAN_MART_DOOR) != "warped":   # 2) Mart -> clerk -> OAK'S PARCEL
                log("   !! PARCEL: couldn't enter the Viridian Mart"); return "stuck"
            for _ in range(60):
                self.b.run_frame()
            self._talk(MART_CLERK_FRONT, "UP", "parcel-pickup")
            log("   PARCEL: collected from the Mart counter; routing to Oak in Pallet")
            self._exit_to_overworld()                          # exit Mart -> Viridian (general exit)
            for _ in range(5):                                 # 3) south to Pallet
                if tv.map_id(self.b) == PALLET:
                    break
                r = self.trav.travel(target_map=PALLET, edge="south", max_steps=600, max_seconds=200)
                if r == "battle_loss":
                    return "battle_loss"
                if r != "arrived":
                    break
            if tv.map_id(self.b) != PALLET:
                log(f"   !! PARCEL: didn't reach Pallet (at {tv.map_id(self.b)})"); return "stuck"
            if self.enter_warp(pick=OAK_LAB_DOOR) != "warped":  # into Oak's lab
                log("   !! PARCEL: couldn't enter Oak's lab"); return "stuck"
            for _ in range(60):
                self.b.run_frame()
            self._talk(OAK_FRONT, "UP", "parcel-deliver")       # 4) deliver -> Pokedex + 5 balls
            for _ in range(2):                                  # the ReceiveDexScene is multi-box
                self._drain_overworld(label="dex-scene")
            b1 = self._ball_count()
            dex = self.has_badge(FLAG_POKEDEX_GET)
            log(f"   PARCEL: after delivery balls {b0}->{b1}  pokedex={dex}")
            self._exit_to_overworld()                           # 5) exit lab (general) -> north to Viridian
            for _ in range(5):
                if tv.map_id(self.b) == VIRIDIAN:
                    break
                if self.advance_north(VIRIDIAN, max_legs=3) == "arrived":
                    continue
                break
        finally:
            self._suppress_heal = saved
            self.trav.battle_runner = saved_runner
        if b1 > b0 and dex:
            self.on_event("delivered Oak's parcel — got the Pokedex and a handful of Poke Balls")
            return "done"
        log(f"   !! PARCEL: quest incomplete (balls {b0}->{b1}, pokedex={dex}) - LOUD"); return "stuck"

    def catch_one(self, max_seconds=300):
        """AUTO catch a teammate (converts the route3_catch hand-play GATE): WANDER in the grass
        until a WILD encounter, then catch it (BattleAgent.catch_pokemon - weaken/status then commit
        to throws). A trainer's line of sight en route is fought (trainer mons can't be caught), then
        we keep wandering. Returns 'caught' | 'no_balls' | 'timeout' | 'battle_loss'."""
        t0 = time.time()
        cur0 = tv.coords(self.b)
        # Grid.grass is in BUFFER coords (save + MAP_OFFSET); travel's arrive_coord + coords() are
        # SAVE coords. Convert here so the nearest-grass sort AND the walk targets share one space —
        # the old code fed buffer-coord grass straight to travel (a 7-tile miss that never stepped
        # onto grass -> no wild encounter -> the catch looked "impossible").
        off = tv.MAP_OFFSET
        grass = sorted(((g[0] - off, g[1] - off) for g in tv.Grid(self.b).grass),
                       key=lambda g: abs(g[0] - cur0[0]) + abs(g[1] - cur0[1]))
        if not grass:
            log("   !! CATCH: no grass on this map - can't find a wild Pokemon here")
            return "no_grass"
        # Warp/door tiles to keep OUT of: pathing across a door warps us into a building (the Route 3
        # PC / Mt Moon mouth) and the catch wedges on an NPC inside (the (5,4) trap). Treat doors as
        # permanent blocks for the whole wander — same avoid-param pattern as cave warp-to-warp nav.
        doors = frozenset(self._door_tiles())
        log(f"   CATCH: {len(grass)} grass tiles (nearest {grass[0]}); avoiding {len(doors)} warp tile(s)"
            f" - traversing grass to catch one")
        p0 = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        caught = [False]
        out_of_balls = [False]

        def catch_runner():
            """Travel hands every encounter here. Wild -> CATCH (commit); trainer -> normal fight.
            Always returns 'win' on a wild so travel keeps walking the grass for more encounters."""
            if self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08:           # trainer: can't catch -> fight
                log("   CATCH: a trainer engaged en route - fighting through")
                return self.battle_runner()
            log("   CATCH: WILD encounter - committing to the catch")
            res = BattleAgent(self.b, on_event=self.on_event, render=self.render,
                              log=lambda m: log(m)).catch_pokemon(max_seconds=160)
            self.b.set_input_owner("agent")
            log(f"   CATCH: catch_pokemon -> {res}")
            if res == "caught":
                caught[0] = True
                self.on_event("I caught a new teammate!")
            elif res == "no_balls":
                out_of_balls[0] = True              # STOP the wander — wandering ball-less just hits
            return "win"                            # more wilds forever (the 8636-loop). a break_free is fine

        # Reuse the BFS Traveler (pathfinds around walls AND walks through grass, handing each
        # encounter to our catch_runner) - pace between near/far grass tiles to keep stepping in it.
        orig = self.trav.battle_runner
        self.trav.battle_runner = catch_runner
        try:
            waypoints = [grass[0], grass[-1], grass[len(grass) // 2]]
            wi = 0
            while time.time() - t0 < max_seconds and not caught[0] and not out_of_balls[0]:
                r = self.trav.travel(target_map=None, arrive_coord=waypoints[wi % len(waypoints)],
                                     max_steps=120, max_seconds=80, avoid=doors)
                if r == "need_heal":
                    # a lone starter can't tank Route 3's trainer gauntlet — heal at Pewter (one west
                    # cross) and resume. Beaten trainers don't re-engage, so each cycle makes progress.
                    if self._heal_excursion(PEWTER, PEWTER_PC_DOOR, "west", ROUTE3, "east") == "stuck":
                        log("   !! CATCH: heal excursion failed - LOUD"); break
                    continue                                  # retry the same waypoint, now healed
                wi += 1
                if self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
                    caught[0] = True
        finally:
            self.trav.battle_runner = orig
            self.b.set_input_owner("agent")
        if caught[0] or self.b.rd8(ram.GPLAYER_PARTY_CNT) > p0:
            return "caught"
        if out_of_balls[0]:
            log("   !! CATCH: ran out of Poké Balls before a catch - LOUD"); return "no_balls"
        log("   !! CATCH: no catch within budget - LOUD")
        return "timeout"

    def _cave_runner(self):
        """Cave battle policy (CLARIFICATION 1): FLEE wild encounters — a solo Ivysaur can't grind
        every Zubat/Geodude through Mt Moon and survive — but FIGHT forced trainers (can't flee them).
        This is how the cave was cleared autonomously; the default fight-everything runner would
        attrition the lone starter to a blackout."""
        if self.b.rd32(ram.GBATTLE_TYPE_FLAGS) & 0x08:         # trainer battle -> must fight
            return self.battle_runner()
        return self._flee_runner()                             # wild -> flee (costs ~0 HP)

    # Mt Moon fossil stand-tiles (face UP): touching a fossil triggers the Super Nerd (Miguel), who
    # battles you for it; beating him + the pick MOVES him, opening the gated B2F->B1F warp (5,10).
    MTMOON_FOSSILS = [((13, 8), "UP"), ((14, 8), "UP")]

    def clear_mt_moon(self):
        """Clear Mt Moon by FOLLOWING the saved warp route (states/mtmoon_plan.json) via the proven
        warp navigator (CaveNav._enter, the same primitive run_plan uses) — PLUS the one step the plan
        omits: the B2F FOSSIL/MIGUEL event, which moves Miguel to open the gated (5,10) exit (confirmed
        by nav-isolation: even win+heal can't enter (5,10) until the fossil event fires). Battle policy:
        FLEE wilds / FIGHT trainers; heal suppressed (no PC in a cave). A death -> 'blackout' (caller's
        blackout-recovery respawns + retries). Returns 'cleared' | 'blackout' | 'stuck'."""
        import json
        from cave_nav import CaveNav
        plan_path = resolve_state("mtmoon_plan.json") or os.path.join(STATES, "mtmoon_plan.json")
        if not os.path.exists(plan_path):
            log("   !! MTMOON: saved route mtmoon_plan.json not found - LOUD"); return "stuck"
        plan = json.load(open(plan_path))
        saved, saved_heal = self.trav.battle_runner, self._suppress_heal
        self.trav.battle_runner = self._cave_runner
        self._suppress_heal = True       # no PC in a cave -> survive-or-blackout, never excursion out
        try:
            nav = CaveNav(self.b, self, fog_path=None, on_event=self.on_event,
                          render=self.render, log=lambda m: log(m))
            # ENTRY NORMALIZATION: the real Route 4 warp lands on 1F at ~(18,37); the plan starts at
            # (15,30). Walk there so the first warp is reachable from EITHER entry. Navigation, not a skip.
            if tv.map_id(self.b) == (1, 1) and tv.coords(self.b) != (15, 30):
                self.trav.travel(target_map=None, arrive_coord=(15, 30), max_steps=400, max_seconds=150)
            for exp_map, wxy in plan:                          # follow the saved warp sequence
                cur = tv.map_id(self.b)
                if tuple(cur) != tuple(exp_map):
                    log(f"   !! MTMOON: PLAN MISMATCH on {cur}, expected {exp_map}"); return "stuck"
                if tuple(cur) == (1, 3):                        # B2F: clear Miguel BEFORE the gated (5,10)
                    if self._mtmoon_fossil_event() == "blackout":
                        log("   MTMOON: blacked out at the fossil/Miguel (SURVIVAL)"); return "blackout"
                    if tv.map_id(self.b)[0] != 1:               # left the cave (PC respawn) = a blackout
                        log("   MTMOON: left B2F during the Miguel fight (blackout) - caller retries")
                        return "blackout"
                r = nav._enter(tuple(wxy))
                # NUDGE-RETRY: the exit warp (45,4) is finicky to approach from (39,4) — the proven
                # port nudged one tile toward it (44,4) and retried up to 4x. run_plan does a single
                # _enter, which dropped that; re-add it so the final exit isn't a one-shot stall.
                tries = 0
                while r is None and tries < 4 and tuple(tv.map_id(self.b)) == tuple(exp_map):
                    tries += 1
                    self.trav.travel(target_map=None, arrive_coord=(wxy[0] - 1, wxy[1]),
                                     max_steps=200, max_seconds=90)
                    r = nav._enter(tuple(wxy))
                if r == "BLACKOUT":
                    log("   MTMOON: blacked out mid-cave (SURVIVAL) - caller retries"); return "blackout"
                if r is None:
                    log(f"   !! MTMOON: warp {tuple(wxy)} unenterable after {tries} nudge(s) "
                        f"(map={tv.map_id(self.b)})"); return "stuck"
            if not nav._fwd_reachable("east"):                 # last warp done -> cross out to Cerulean
                log(f"   !! MTMOON: warps done at {tv.map_id(self.b)} but east edge unreachable"); return "stuck"
            self.trav.travel(target_map=(99, 99), edge="east", max_steps=400, max_seconds=200)
            m = tv.map_id(self.b)
            if m[0] != 1 and m != (3, 22):
                log(f"   MTMOON: *** CLEARED via the saved route -> emerged on {m} ***")
                self.on_event("made it through Mt Moon")
                return "cleared"
            log(f"   !! MTMOON: not fully out (map={m})"); return "stuck"
        finally:
            self.trav.battle_runner, self._suppress_heal = saved, saved_heal

    def _mtmoon_fossil_event(self):
        """Touch a B2F fossil -> the Super Nerd (Miguel) battles you for it; win + the pick MOVES him,
        opening the gated (5,10) exit warp. _cave_runner FIGHTS him (trainer). Returns 'ok'|'blackout'."""
        for stand, facing in self.MTMOON_FOSSILS:
            self.trav.travel(target_map=None, arrive_coord=stand, max_steps=600, max_seconds=300)
            if tv.coords(self.b) == stand:
                log(f"   MTMOON: at fossil stand {stand} - triggering the Miguel event")
                self.b.press(facing, 8, 8, self.render, owner="agent")   # FACE the fossil ONCE
                for _ in range(16):
                    # A-ONLY from here — never re-press the d-pad, or grabbing the fossil lets the
                    # next UP walk off the stand-tile into the warp above it (the map=(16,0) bug).
                    self.b.press("A", 8, 8, self.render, owner="agent")
                    for _ in range(8):
                        self.b.run_frame(); self.render()
                    if st.in_battle(self.b):                              # Miguel challenges for it
                        r = self._cave_runner()
                        if r == "loss" or tv.map_id(self.b) != (1, 3):    # lost -> whited out off B2F
                            return "blackout"
                self._drain_overworld(label="fossil-pick")
                return "ok"
        log("   !! MTMOON: couldn't reach a fossil stand-tile (Miguel uncleared) - (5,10) will gate")
        return "ok"

    # ── healing (the ordinary survival loop) ──────────────────────────────────
    def lead_hp(self):
        return (self.b.rd16(ram.GPLAYER_PARTY + P_HP),
                self.b.rd16(ram.GPLAYER_PARTY + P_MAXHP))

    def _step_to(self, tile, steps=30):
        """Obstacle-aware greedy stepper for tiny INTERIORS (the Traveler refuses an NPC's
        facing tile - e.g. a nurse/counter - so interiors need a manual stepper). Tries the
        dominant axis; if a press doesn't move us, tries the other; nudges sideways if boxed."""
        for _ in range(steps):
            cur = tv.coords(self.b)
            if cur == tile:
                return True
            if cur is None:
                for _ in range(8):
                    self.b.run_frame()
                continue
            dx, dy = tile[0] - cur[0], tile[1] - cur[1]
            if abs(dx) >= abs(dy):
                order = ([("RIGHT" if dx > 0 else "LEFT")] if dx else []) + \
                        ([("DOWN" if dy > 0 else "UP")] if dy else [])
            else:
                order = ([("DOWN" if dy > 0 else "UP")] if dy else []) + \
                        ([("RIGHT" if dx > 0 else "LEFT")] if dx else [])
            moved = False
            for key in order:
                self.b.press(key, 8, 8, self.render, owner="agent")
                if tv.coords(self.b) != cur:
                    moved = True
                    break
            if not moved:
                self.b.press("RIGHT", 8, 8, self.render, owner="agent")
        return tv.coords(self.b) == tile

    def heal_at_center(self, pc_door=VIRIDIAN_PC_DOOR):
        """AUTHENTIC Pokemon Center heal (no RAM poking): route to a Pokemon Center door (default
        Viridian; pass any city's PC door for NEAREST-center healing), enter, walk to the nurse
        counter, drive the YES heal dialogue, verify HP -> max, exit back to the city. PC interiors
        share ONE layout so only the overworld door differs. Returns 'healed' or 'stuck' (LOUD)."""
        h0 = self.lead_hp()
        city = tv.map_id(self.b)                  # the city we heal in (generic; was Viridian-only)
        if h0[0] >= h0[1]:
            log(f"   HEAL: already full ({h0[0]}/{h0[1]}) - skipping"); return "healed"
        return_to = tv.coords(self.b)             # heal is TRANSPARENT: come back here after
        log(f"   HEAL: lead at {h0[0]}/{h0[1]} -> routing to the Viridian Pokemon Center "
            f"(will return to {return_to})")
        # 1) to the PC door + step in
        if self.trav.travel(target_map=None, arrive_coord=(pc_door[0],
                            pc_door[1] + 1), max_steps=400, max_seconds=120) != "arrived":
            log(f"   !! HEAL: couldn't reach the PC door (at {tv.coords(self.b)})"); return "stuck"
        before = tv.map_id(self.b)
        for _ in range(6):
            self.b.press("UP", 8, 10, self.render, owner="agent")
            for _ in range(20):
                self.b.run_frame()
            if tv.map_id(self.b) != before:
                break
        if tv.map_id(self.b) == city:
            log(f"   !! HEAL: did not enter the Center (still on {tv.map_id(self.b)})"); return "stuck"
        for _ in range(90):                       # entrance auto-walk settle (input lock)
            self.b.run_frame()
        self.b.set_input_owner("agent")
        # 2) to the nurse counter + drive the YES heal dialogue
        self._step_to(NURSE_FRONT)
        self.b.press("UP", 6, 8, self.render, owner="agent")     # face the nurse
        for _ in range(26):
            # YES/NO box eats the first press; UP engages (YES is top, can't move off), A=YES
            self.b.press("UP", 4, 4, self.render, owner="agent")
            self.b.press("A", 6, 10, self.render, owner="agent")
            for _ in range(30):
                self.b.run_frame()
            if self.lead_hp()[0] == self.lead_hp()[1]:
                break
        h1 = self.lead_hp()
        if h1[0] != h1[1] or h1[1] == 0:
            log(f"   !! HEAL: nurse dialogue did not complete (HP {h1[0]}/{h1[1]})"); return "stuck"
        log(f"   HEAL: restored {h0[0]}/{h0[1]} -> {h1[0]}/{h1[1]}")
        # 3) EXIT: the post-heal script holds control for a beat and the exit mat only fires when you
        # STEP onto it. PATIENTLY clear the nurse's closing text (B, harmless in the overworld) and
        # walk DOWN the counter column onto the mat, retrying until we're back in the city. The old
        # single DOWN-burst was too impatient and froze at the counter (the Cerulean heal-exit bug).
        for _ in range(6):                          # clear the nurse's closing text (B = harmless)
            self.b.press("B", 3, 8, self.render, owner="agent")
        self._exit_to_overworld()                   # the general, stress-tested building-exit (south
        #                                             door / DOWN-mat fallback). The old DOWN-only loop
        #                                             wedged inside some PCs ('stuck inside' at (5,4)).
        log(f"   HEAL: exited Center -> map={tv.map_id(self.b)} coords={tv.coords(self.b)}")
        if tv.map_id(self.b)[0] != 3:               # still not on the overworld -> genuinely stuck
            return "healed_stuck_inside"
        # walk back to the pre-heal spot so HEAL is transparent (the next objective resumes
        # from where it expected to be, not the PC doorstep)
        if return_to and tv.coords(self.b) != return_to:
            self.b.set_input_owner("agent")
            r = self.trav.travel(target_map=None, arrive_coord=return_to,
                                 max_steps=400, max_seconds=120)
            log(f"   HEAL: returned toward {return_to} -> {r} (now {tv.coords(self.b)})")
        return "healed"

    def grind(self, target_level):
        """Train the lead to target_level in the grass, healing when low. Self-sufficient gym-readiness
        capability (the 'walk away' vision needs autonomous leveling). From a CITY with no grass
        (Pewter), cross EAST to the adjacent route (Route 3: grass + a trainer gauntlet = fast XP),
        grind, then cross back. Wilds are FOUGHT (XP), not fled. Returns 'ok' | 'battle_loss'."""
        def lvl():
            return self.b.rd8(ram.GPLAYER_PARTY + 0x54)
        if lvl() >= target_level:
            log(f"   GRIND: already Lv{lvl()} >= {target_level}"); return "ok"
        off = tv.MAP_OFFSET
        home = tv.map_id(self.b)

        def grass_save():
            return [(x - off, y - off) for (x, y) in tv.Grid(self.b).grass]
        log(f"   GRIND: Lv{lvl()} < {target_level} - heading to the grass to train")
        if not grass_save() and home == PEWTER:               # city has no grass -> Route 3 (east)
            self.walk_to_map(ROUTE3, "east")
        t0 = time.time()
        while lvl() < target_level and time.time() - t0 < 480:
            gs = grass_save()
            if not gs:
                log(f"   !! GRIND: no grass reachable on {tv.map_id(self.b)} - stopping LOUD"); break
            cur = tv.coords(self.b) or (0, 0)
            gs.sort(key=lambda g: abs(g[0] - cur[0]) + abs(g[1] - cur[1]))
            doors = frozenset(self._door_tiles())
            for wp in (gs[0], gs[-1], gs[len(gs) // 2]):
                if lvl() >= target_level:
                    break
                r = self.trav.travel(target_map=None, arrive_coord=wp,
                                     max_steps=120, max_seconds=80, avoid=doors)
                if r == "battle_loss":
                    return "battle_loss"
                if r == "need_heal":
                    self.heal_nearest()
        log(f"   GRIND: trained to Lv{lvl()} (target {target_level})")
        if tv.map_id(self.b) != home and home == PEWTER:      # back to Pewter for the gym
            self.walk_to_map(PEWTER, "west")
        return "ok"

    def roster_react(self):
        """SOUL beat after a hand-play-banked catch (the agent can't drive this core's battle menus,
        so acquisition is hand-played — but the RELATIONSHIP is live). Emit a NEUTRAL roster-change
        event so Kira reacts to her new teammate IN HER OWN VOICE via the existing seam (fulfilling
        her want for a partner). Touches NO core (on_event -> _pokemon_react -> her self)."""
        cnt = self.b.rd8(ram.GPLAYER_PARTY_CNT)
        if cnt <= 1:
            log("   ROSTER_REACT: party size 1 — no new teammate to react to"); return "reacted"
        name = st.SPECIES_NAME.get(st.read_party_species(self.b, cnt - 1), "a new Pokemon")
        log(f"   ROSTER_REACT: party now {cnt} — newest teammate is {name}")
        self.on_event(f"you've got a new teammate now — a {name} that's going to fight alongside you")
        return "reacted"

    # ── the continuous driver ────────────────────────────────────────────────
    def run(self, objectives):
        for i, obj in enumerate(objectives):
            kind, label = obj[0], obj[-1]
            _t0 = time.time()
            log(f"OBJECTIVE {i+1}/{len(objectives)}: {kind} - {label}  "
                f"[at map={tv.map_id(self.b)} coords={tv.coords(self.b)}]")
            if kind == "OPENING":
                self._title_to_newgame()             # fresh ROM -> title -> New Game -> Oak's intro
                m = self.drive_opening(obj[1], obj[2])    # her char-creation + starter + rival
                out = "complete" if m == PALLET else "stuck"
            elif kind == "WALK_TO_MAP":
                out = self.walk_to_map(obj[1], obj[2])
            elif kind == "ADVANCE_NORTH":
                out = self.advance_north(obj[1])
            elif kind == "ENTER_WARP":
                spec = obj[1]                        # (x,y) door pick | "north"/"nearest" | None
                out = (self.enter_warp(pick=spec) if isinstance(spec, tuple)
                       else self.enter_warp(prefer=spec or "nearest"))
            elif kind == "LEVEL_CHECK":
                out = self.level_check(obj[1])
            elif kind == "HEAL":
                out = self.heal_at_center()
            elif kind == "GRIND":
                out = self.grind(obj[1])
            elif kind == "BEAT_GYM":
                out = self.beat_gym(obj[1])
            elif kind == "CATCH":
                out = self.catch_one()
            elif kind == "DELIVER_PARCEL":
                out = self.deliver_parcel()
            elif kind == "CLEAR_MT_MOON":
                out = self.clear_mt_moon()
            elif kind == "HEAL_NEAREST":
                out = "ok" if self.heal_nearest() == "ok" else "ok"   # heal is best-effort, never a stop
            elif kind == "ROSTER_REACT":
                out = self.roster_react()
            elif kind == "GATE_NEEDS_STATE":
                state, what = obj[1], obj[2]
                path = self._resolve_gate(state)
                if self.show_mode:
                    # A banked-state fallback in SHOW mode is a HARD-FAILURE we want SURFACED, not
                    # silently skipped: log LOUD + count. The run still loads (if the file exists) so
                    # ONE show run reveals EVERY remaining gate; violations>0 = the spine isn't AUTO.
                    self._show_violations += 1
                    log(f"   !!!! SHOW-MODE SKIP VIOLATION #{self._show_violations}: {label} "
                        f"(would load banked sherpa state {state}) — this skip is NOT yet AUTO")
                if path:
                    log(f"   gate savestate present ({state}) - loading past the gate")
                    with open(path, "rb") as f:
                        self.b.load_state(f.read())
                    for _ in range(40):
                        self.b.run_frame()
                    out = "loaded"
                else:
                    log(f"   !! GATE NEEDS SAVESTATE: {state}")
                    log(f"      hand-play once: {what}")
                    log(f"      then save it to states/workshop/{state} and re-run.")
                    return f"blocked_gate:{state}"
            else:
                out = "unknown"
            log(f"   -> objective result: {out} in {time.time()-_t0:.0f}s "
                f"(now map={tv.map_id(self.b)} coords={tv.coords(self.b)})")
            if out in ("stuck", "no_path", "battle_loss", "no_warp", "underleveled", "blackout"):
                log(f"!! CAMPAIGN STOP (loud) at objective {i+1} ({kind}): {out}")
                return f"stopped:{kind}:{out}"
        log("CAMPAIGN COMPLETE (all objectives done)")
        return "complete"

    # ── SEGMENT MANIFEST driver (the resumable whole-game spine) ──────────────────
    def _resolve_gate(self, name):
        """Find a banked sherpa/gate state across the live buckets (see resolve_state). Returns the
        full path or None. (Each gate converted to AUTO drops its load entirely.)"""
        return resolve_state(name)

    def _save_checkpoint(self, name) -> bool:
        """Auto-save a resumable overworld checkpoint between segments into self.ckpt_dir. LOUD on
        failure (Constraint #3). HARD GUARD: a WORKSHOP run is physically forbidden to write into
        states/kira/ (her canonical saves) — only a SHOW run banks there."""
        if not name:
            return False
        target_dir = self.ckpt_dir
        kira_abs = os.path.abspath(STATES_KIRA)
        if not self.show_mode and os.path.abspath(target_dir).startswith(kira_abs):
            log("   !! CHECKPOINT REFUSED: WORKSHOP mode may not write into states/kira/ (canonical)")
            return False
        try:
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, name)
            data = self.b.save_state()
            with open(path, "wb") as f:
                f.write(data)
            where = "kira" if self.show_mode else "workshop"
            log(f"   CHECKPOINT saved -> {where}/{name}  "
                f"(map={tv.map_id(self.b)} coords={tv.coords(self.b)})")
            return True
        except Exception as e:
            log(f"   !! CHECKPOINT SAVE FAILED ({name}): {e}")
            return False

    MAX_BLACKOUT_RETRIES = 12     # per segment — a thin solo roster needs several Miguel attempts;
                                  # each retry re-walks (more trainer XP) + re-heals, so it converges.
                                  # "die as many times as needed and still finish" — survival variance.

    @staticmethod
    def _is_blackout(result):
        """A segment result that is a SURVIVAL death (recoverable by respawn+retry), not a nav stuck.
        run() encodes these as 'stopped:<kind>:battle_loss' or 'stopped:<kind>:blackout'."""
        return isinstance(result, str) and (result.endswith("battle_loss") or result.endswith("blackout"))

    def _settle_after_blackout(self):
        """After a blackout the game whites out + respawns at the last-healed Center (auto-heals the
        party). Advance past the respawn animation + any closing text so we're back on the overworld,
        ready to re-run the segment from the Center."""
        self.b.set_input_owner("agent")
        for _ in range(180):                  # let the whiteout + respawn walk-in play out
            self.b.run_frame(); self.render()
        self._drain_overworld(label="blackout-respawn")
        for _ in range(40):
            self.b.run_frame()
        # The respawn drops her INSIDE a building (the Pallet house / any Pokémon Center). Get her
        # back to the walkable overworld before the segment re-nav (which can't path out of an
        # interior). General capability — works for EVERY PC/town (game-wide for the 24/7 vision).
        self._exit_to_overworld()
        log(f"   BLACKOUT-RECOVERY: respawned at {tv.map_id(self.b)}@{tv.coords(self.b)} "
            f"HP={self.lead_hp()}")

    def _exit_to_overworld(self, max_tries=5):
        """From INSIDE any building (PC / house / Mart / lab — map group != 3, the Kanto overworld),
        return to the walkable overworld. The general recovery primitive every blackout respawn needs.
        Tries the south exit door; falls back to walking DOWN onto the entrance mat (PC/house mats fire
        on a step-on from the north). Returns True once on the overworld (group 3)."""
        self.b.set_input_owner("agent")
        for _ in range(max_tries):
            if tv.map_id(self.b)[0] == 3:
                return True
            before = tv.map_id(self.b)
            if self.enter_warp(prefer="south") == "warped" and tv.map_id(self.b) != before:
                log(f"   EXIT-BUILDING: warped {before} -> {tv.map_id(self.b)}@{tv.coords(self.b)}")
                continue
            for _ in range(10):                       # fallback: walk DOWN onto the exit mat
                self.b.press("DOWN", 8, 8, self.render, owner="agent")
                for _ in range(8):
                    self.b.run_frame()
                if tv.map_id(self.b) != before:
                    break
            if tv.map_id(self.b) != before:
                log(f"   EXIT-BUILDING: walked out {before} -> {tv.map_id(self.b)}@{tv.coords(self.b)}")
        if tv.map_id(self.b)[0] != 3:
            log(f"   !! EXIT-BUILDING: still inside {tv.map_id(self.b)}@{tv.coords(self.b)} - LOUD")
        return tv.map_id(self.b)[0] == 3

    def run_segments(self, segments, resume=True, mode="workshop"):
        """Play SEGMENTS in order, auto-checkpointing after each.
        mode='workshop' (default): resume-from-furthest in states/workshop/, bank there, NEVER touch
            states/kira/. Our skip-elimination scaffolding — jump anywhere, scratch freely.
        mode='show': the canonical/recordable spine. Banks into states/kira/ and resumes from a kira
            save if present (else fresh boot). Any GATE_NEEDS_STATE fallback logs a LOUD SHOW-MODE
            SKIP VIOLATION and is counted — a clean SHOW run has ZERO violations + zero skips."""
        self.show_mode = (mode == "show")
        self.ckpt_dir = STATES_KIRA if self.show_mode else STATES_WORKSHOP
        self._show_violations = 0
        log(f"RUN_SEGMENTS: ** {'SHOW' if self.show_mode else 'WORKSHOP'} MODE ** "
            f"-> banks to {os.path.basename(self.ckpt_dir)}/"
            + ("; any GATE load = LOUD violation (target 0)" if self.show_mode
               else "; states/kira/ is READ-ONLY here"))
        start, resume_cp = 0, None
        if resume:
            for i, seg in enumerate(segments):
                cp = os.path.join(self.ckpt_dir, seg.checkpoint) if seg.checkpoint else None
                if cp and os.path.exists(cp):
                    start, resume_cp = i + 1, cp
        if start >= len(segments):
            log(f"RUN_SEGMENTS: all {len(segments)} segment checkpoint(s) already present — done")
            return "all_segments_complete"
        if resume_cp:
            log(f"RUN_SEGMENTS: RESUME — checkpoint {os.path.basename(resume_cp)} present, skipping "
                f"{start} done segment(s); loading past them")
            with open(resume_cp, "rb") as f:
                self.b.load_state(f.read())
            for _ in range(40):
                self.b.run_frame()
            self.b.set_input_owner("agent")
        for i in range(start, len(segments)):
            seg = segments[i]
            log(f"==== SEGMENT {i + 1}/{len(segments)}: {seg.name}  "
                f"({len(seg.objectives)} objective(s)) ====")
            # BLACKOUT-RECOVERY (core resilience): a death (Miguel/cave/Route 3 with a thin solo
            # roster) is NOT a dead end — the game respawns + heals at the last Center, and we RE-RUN
            # the segment (the objectives re-navigate from the respawn point). "Die as many times as
            # needed and still finish." A PERMANENT stuck (can't proceed, not dying) still stops loud.
            attempts = 0
            while True:
                result = self.run(seg.objectives)
                if result == "complete":
                    break
                if self._is_blackout(result) and attempts < self.MAX_BLACKOUT_RETRIES:
                    attempts += 1
                    log(f"   ~~~~ BLACKOUT-RECOVERY: '{seg.name}' died ({result}); retry "
                        f"{attempts}/{self.MAX_BLACKOUT_RETRIES} — respawned + healed, re-running segment")
                    self.on_event("knocked out — back to the Center, dust off, and right back at it")
                    self._settle_after_blackout()
                    continue
                log(f"!! RUN_SEGMENTS STOP at segment '{seg.name}': {result}"
                    + (f" (blackout retries exhausted: {attempts})" if self._is_blackout(result) else ""))
                return f"segment_stopped:{seg.name}:{result}"
            self._save_checkpoint(seg.checkpoint)
            log(f"==== SEGMENT '{seg.name}' COMPLETE"
                + (f" (survived {attempts} blackout(s))" if attempts else "") + " ====")
        if self.show_mode:
            verdict = "ZERO skips — clean AUTO spine" if self._show_violations == 0 \
                else f"{self._show_violations} SKIP VIOLATION(S) — NOT yet zero-skip"
            log(f"RUN_SEGMENTS: SHOW MODE COMPLETE — {verdict}")
        log("RUN_SEGMENTS: ALL SEGMENTS COMPLETE")
        return "all_segments_complete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--boot", default="after_pick_bulbasaur.state")
    args = ap.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    b = Bridge(ROM)
    boot_path = resolve_state(args.boot) or os.path.join(STATES, args.boot)
    with open(boot_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    log(f"booted {args.boot}: map={tv.map_id(b)} coords={tv.coords(b)}")

    def battle_runner():
        return BattleAgent(b, on_event=lambda s, **k: log(f"[battle] {s}"),
                           render=lambda: None).run(max_seconds=120)

    camp = Campaign(b, battle_runner=battle_runner)
    outcome = camp.run(build_objectives())
    print("\n" + "=" * 60)
    print(f"   CAMPAIGN RESULT: {outcome}")
    print(f"   final: map={tv.map_id(b)} coords={tv.coords(b)}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
