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

# ── known FireRed map IDs (group, num) - documented, not reconned ────────────
PALLET, VIRIDIAN, PEWTER = (3, 0), (3, 1), (3, 2)
ROUTE1, ROUTE2, ROUTE22 = (3, 19), (3, 20), (3, 41)
# Viridian Forest + gym interiors live in other groups; resolved at runtime on entry.
# Viridian Pokemon Center (RED-roof bldg, reconned 2026-06-24): town door (26,26) -> interior
# (5,4); nurse at (7,2), talked to from (7,4) ACROSS the counter; entrance mat at (7,8).
VIRIDIAN_PC_DOOR, VIRIDIAN_PC_MAP, NURSE_FRONT, PC_ENTRY = (26, 26), (5, 4), (7, 4), (7, 8)
# Pewter Gym (reconned 2026-06-24): town door (15,16) -> interior (6,2); climb the centre column
# (x=6) to Brock's front tile (6,6); the lone gym trainer auto-engages off the centre path.
PEWTER_GYM_DOOR, BROCK_FRONT = (15, 16), (6, 6)
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
        # Oak's Parcel gate: the old man ('I absolutely forbid you from going through')
        # blocks Route 2 until the parcel errand is done. Scripting it needs Viridian
        # Mart interior + Oak's lab interior coords (town-interior nav). Flagged for a
        # one-time hand-played savestate unless/until we script the interiors.
        ("GATE_NEEDS_STATE", "viridian_parcel_done.state",
         "Oak's Parcel quest done + the north gatekeeper cleared "
         "(Viridian Mart -> deliver to Oak in Pallet)"),
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
        Segment("pallet_to_brock", build_objectives(), "seg_boulder_badge.state"),
        # Pewter -> Route 3: the EAST map-edge seam is an unsolved collision PARADOX (every static
        # source — collision/elevation/behavior/objects — reads walkable, yet real movement traps the
        # agent in a 1-tile prison at the connection edge). Per the manifest's own design (each leg is
        # automated OR hand-play-once-bank-skip), this seam is a GATE_NEEDS_STATE: hand-play across it
        # once, bank seg_route3_start.state, and the run resumes on Route 3 (3,21) at (9,9). Checkpoint
        # is a DISTINCT name so the auto-save never overwrites the hand-played gate input.
        Segment("pewter_to_route3", [
            ("GATE_NEEDS_STATE", "seg_route3_start.state",
             "hand-play ONCE across the Pewter east seam onto Route 3, then F5-save: "
             ".venv\\Scripts\\python.exe pokemon_agent\\handplay.py --boot brock_done.state "
             "--save seg_route3_start.state  (autonomous crossing deferred — collision paradox)"),
        ], "seg_route3_entered.state"),
        # CATCH = HAND-PLAY-BANK (the agent CANNOT drive this core's battle menus — confirmed by
        # replicating Jonny's exact human input; injection is dead too). So catching follows the
        # seam pattern: hand-catch a teammate once, bank route3_caught.state (party=2: Ivysaur +
        # Rattata), and ROSTER_REACT makes Kira react to the new teammate IN HER VOICE on resume —
        # acquisition hand-played, RELATIONSHIP live. The run continues with a REAL roster.
        Segment("route3_catch", [
            ("GATE_NEEDS_STATE", "route3_caught.state",
             "hand-play ONCE: catch a teammate on Route 3 (open BAG -> Poke Ball -> throw by hand; "
             "the agent can't drive the battle menu), then F5-save route3_caught.state via "
             ".venv\\Scripts\\python.exe pokemon_agent\\handplay.py --boot seg_route3_start.state "
             "--save route3_caught.state"),
            ("ROSTER_REACT", "Kira reacts to her new caught teammate, in her own voice"),
        ], "seg_route3_caught.state"),
        # ── NEXT (staged build order — each gated on its own recon; see the plan) ──────────────
        # Segment("route3_to_cerulean", [
        #     ("WALK_TO_MAP", ?MT_MOON?, "north", "Route 3 -> Mt Moon entrance (recon N->(3,22) first)"),
        #     ("CAVE_NAV",    MT_MOON_EXIT,       "Mt Moon maze -> exit (NET-NEW: general cave nav)"),
        #     ("WALK_TO_MAP", CERULEAN, "east",   "Route 4 -> Cerulean City"),
        # ], "seg_cerulean.state"),
        # Segment("beat_misty", [
        #     ("LEVEL_CHECK", 18, "Misty-readiness (lean on Bulbasaur grass vs her water)"),
        #     ("BEAT_GYM", "Misty", "Cerulean Gym -> Cascade Badge"),
        # ], "seg_cascade_badge.state"),
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
        return self.trav.travel(target_map=target_map, edge=direction)

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

    def return_to_center(self):
        """Heal-return: travel SOUTH back to Viridian (the only Center before Pewter), then heal
        at the PC (restores HP + PP). Mirrors advance_north but southward: cross a south edge,
        else warp south, until we're on the Viridian map; then heal_at_center (walks to the PC,
        heals, returns to the pre-heal spot). The traveler fights any encounter en route."""
        log(f"   HEAL-RETURN: lead at {self.lead_hp()} - routing south to Viridian to heal")
        saved = self._suppress_heal
        saved_runner = self.trav.battle_runner
        self._suppress_heal = True              # don't re-trigger heal WHILE returning to heal
        self.trav.battle_runner = self._flee_runner   # RETREAT: flee wild fights on the way out
        try:
            for leg in range(20):
                m = tv.map_id(self.b)
                if m == VIRIDIAN:
                    break
                out = self.trav.travel(target_map=VIRIDIAN, edge="south", max_steps=800)
                if out == "arrived":
                    continue                       # crossed a south edge toward Viridian
                if out == "battle_loss":
                    return "battle_loss"           # blacked out en route -> auto-heals at Viridian
                log(f"   HEAL-RETURN leg {leg}: no south edge on map {m} - warping south")
                if self.enter_warp(prefer="south") != "warped":
                    log(f"   !! HEAL-RETURN stuck on map {m} (no south edge, no south warp)")
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
        log(f"   !! LEVEL CHECK: lead Pokemon Lv{lvl} < {min_level} - UNDERLEVELED for "
            f"{leader} (Geodude L12 / Onix L14). GRIND in the grass before the gym.")
        self.on_event(f"I'm only level {lvl} - I need to train more before I take on {leader}")
        return "underleveled"

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
        log(f"   GYM: inside {tv.map_id(self.b)} at {tv.coords(self.b)} - clearing junior trainers")
        # 1) BEAT THE JUNIOR TRAINERS FIRST (the leader is gated until they're all down)
        self._clear_gym_trainers(gym.leader_front)
        # 2) engage the LEADER (now ungated): walk to the front tile, face them + A to initiate,
        # then DRIVE the pre-battle challenge speech INTO the battle (Brock's "So, you're here..."
        # is a multi-box speech that a few A-taps don't fully clear; the primitive advances it and
        # stops the instant the battle starts). Misty starts the battle near-immediately - same path.
        self.b.set_input_owner("agent")
        self.trav.travel(target_map=None, arrive_coord=gym.leader_front, max_steps=200, max_seconds=90)
        log(f"   GYM: at {tv.coords(self.b)} (leader front {gym.leader_front}) - engaging {name}")
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
            log(f"   GYM: {name.upper()} -> {self.battle_runner()}")
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
            return "badge"
        log(f"   !! GYM: {name} not beaten / no badge flag"); return "stuck"

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
        for _ in range(40):
            if tv.map_id(self.b) == city:
                break
            self.b.press("B", 3, 8, self.render, owner="agent")
            self.b.press("DOWN", 8, 12, self.render, owner="agent")
            for _ in range(10):
                self.b.run_frame()
        log(f"   HEAL: exited Center -> map={tv.map_id(self.b)} coords={tv.coords(self.b)}")
        if tv.map_id(self.b) != city:
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
        log("   GRIND: handler not built yet"); return "stuck"   # TODO: build after heal verify

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
            if kind == "WALK_TO_MAP":
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
            elif kind == "ROSTER_REACT":
                out = self.roster_react()
            elif kind == "GATE_NEEDS_STATE":
                state, what = obj[1], obj[2]
                path = os.path.join(STATES, state)
                if os.path.exists(path):
                    log(f"   gate savestate present ({state}) - loading past the gate")
                    with open(path, "rb") as f:
                        self.b.load_state(f.read())
                    for _ in range(40):
                        self.b.run_frame()
                    out = "loaded"
                else:
                    log(f"   !! GATE NEEDS SAVESTATE: {state}")
                    log(f"      hand-play once: {what}")
                    log(f"      then save it to states/{state} and re-run.")
                    return f"blocked_gate:{state}"
            else:
                out = "unknown"
            log(f"   -> objective result: {out} in {time.time()-_t0:.0f}s "
                f"(now map={tv.map_id(self.b)} coords={tv.coords(self.b)})")
            if out in ("stuck", "no_path", "battle_loss", "no_warp", "underleveled"):
                log(f"!! CAMPAIGN STOP (loud) at objective {i+1} ({kind}): {out}")
                return f"stopped:{kind}:{out}"
        log("CAMPAIGN COMPLETE (all objectives done)")
        return "complete"

    # ── SEGMENT MANIFEST driver (the resumable whole-game spine) ──────────────────
    def _save_checkpoint(self, name) -> bool:
        """Auto-save a resumable overworld checkpoint between segments. LOUD on failure
        (Constraint #3 — a silently-missing checkpoint would replay the whole game)."""
        if not name:
            return False
        path = os.path.join(STATES, name)
        try:
            data = self.b.save_state()
            with open(path, "wb") as f:
                f.write(data)
            log(f"   CHECKPOINT saved -> {name}  (map={tv.map_id(self.b)} coords={tv.coords(self.b)})")
            return True
        except Exception as e:
            log(f"   !! CHECKPOINT SAVE FAILED ({name}): {e}")
            return False

    def run_segments(self, segments, resume=True):
        """Play SEGMENTS in order, auto-checkpointing after each. With resume=True, RESUMES from the
        furthest existing checkpoint so a continuous 'Go' never replays finished segments. Stops LOUD
        on the first segment that doesn't complete (its objectives' own loud stop is preserved)."""
        start, resume_cp = 0, None
        if resume:
            for i, seg in enumerate(segments):
                cp = os.path.join(STATES, seg.checkpoint) if seg.checkpoint else None
                if cp and os.path.exists(cp):
                    start, resume_cp = i + 1, seg.checkpoint
        if start >= len(segments):
            log(f"RUN_SEGMENTS: all {len(segments)} segment checkpoint(s) already present — done")
            return "all_segments_complete"
        if resume_cp:
            log(f"RUN_SEGMENTS: RESUME — checkpoint {resume_cp} present, skipping {start} done "
                f"segment(s); loading past them")
            with open(os.path.join(STATES, resume_cp), "rb") as f:
                self.b.load_state(f.read())
            for _ in range(40):
                self.b.run_frame()
            self.b.set_input_owner("agent")
        for i in range(start, len(segments)):
            seg = segments[i]
            log(f"==== SEGMENT {i + 1}/{len(segments)}: {seg.name}  "
                f"({len(seg.objectives)} objective(s)) ====")
            result = self.run(seg.objectives)
            if result != "complete":
                log(f"!! RUN_SEGMENTS STOP at segment '{seg.name}': {result}")
                return f"segment_stopped:{seg.name}:{result}"
            self._save_checkpoint(seg.checkpoint)
            log(f"==== SEGMENT '{seg.name}' COMPLETE ====")
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
    with open(os.path.join(STATES, args.boot), "rb") as f:
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
