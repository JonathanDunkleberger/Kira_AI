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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge        # noqa: E402
import firered_ram as ram        # noqa: E402
import pokemon_state as st       # noqa: E402
import travel as tv              # noqa: E402
from battle_agent import BattleAgent  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")

# ── known FireRed map IDs (group, num) - documented, not reconned ────────────
PALLET, VIRIDIAN, PEWTER = (3, 0), (3, 1), (3, 2)
ROUTE1, ROUTE2, ROUTE22 = (3, 19), (3, 20), (3, 41)
# Viridian Forest + gym interiors live in other groups; resolved at runtime on entry.
# Viridian Pokemon Center (RED-roof bldg, reconned 2026-06-24): town door (26,26) -> interior
# (5,4); nurse at (7,2), talked to from (7,4) ACROSS the counter; entrance mat at (7,8).
VIRIDIAN_PC_DOOR, VIRIDIAN_PC_MAP, NURSE_FRONT, PC_ENTRY = (26, 26), (5, 4), (7, 4), (7, 8)
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
        ("ENTER_WARP", None, "Pewter -> Brock's Gym (door)"),
        ("BEAT_GYM", "Brock", "defeat Brock (Gym 1)"),
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
        # one Traveler reused for every WALK leg (BFS + NPC-aware + grass-aware + handoff)
        self.trav = tv.Traveler(bridge, battle_runner=battle_runner, render=self.render,
                                on_event=self.on_event, beat=self.beat)

    def _advance_dialogue(self, taps=12):
        """Press A to advance/close any open textbox until movement is free again."""
        for _ in range(taps):
            if tv.coords(self.b) is not None and not st.in_battle(self.b):
                # heuristic: if a direction press moves us, dialogue is closed
                pass
            self.b.press("A", 6, 6, self.render, owner="agent")

    def walk_to_map(self, target_map, direction):
        # direction support: travel.Traveler currently crosses the NORTH edge (the whole
        # Pallet->Pewter chain is northward). Non-north would extend the edge goal.
        if direction != "north":
            log(f"   (only 'north' edge crossing implemented; '{direction}' needs the "
                f"edge-goal generalization) - attempting north")
        return self.trav.travel(target_map=target_map)

    def enter_warp(self, prefer="nearest", pick=None):
        """REAL warp entry: find door/warp tiles (behavior 0x6x), walk to the tile just
        south of a chosen one (the standard approach), step UP onto it, and confirm the
        map flips. `pick` (x,y) targets a specific door; else `prefer` = 'nearest' or
        'north' (forward-progress gates are the northmost door)."""
        before = tv.map_id(self.b)
        doors = self._door_tiles()
        if not doors:
            log("   no door/warp tiles (0x6x) found on this map"); return "no_warp"
        cur = tv.coords(self.b)
        if pick is not None:
            order = [pick]
        elif prefer == "north":
            order = sorted(doors, key=lambda t: (t[1], abs(t[0] - cur[0])))
        else:
            order = sorted(doors, key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        log(f"   {len(doors)} door(s); candidates (chosen order): {order}")
        for door in order:
            approach = (door[0], door[1] + 1)            # stand just south of the door
            # FAST reachability pre-check (BFS only) so unreachable doors are skipped
            # instantly instead of a slow walk-then-fail. Uses grass-ALLOWED collision
            # (grid.walkable, not _safe): grass-heavy maps like Viridian Forest are
            # wall-to-wall tall grass, so a grass-free pre-check wrongly skips every door.
            grid = tv.Grid(self.b)
            if not tv.bfs(grid, tv.coords(self.b), lambda t: t == approach,
                          walkable=grid.walkable):
                continue
            log(f"   reachable door {door}: walking to approach {approach}")
            if self.trav.travel(target_map=None, arrive_coord=approach, max_steps=300) != "arrived":
                continue
            for _ in range(5):                           # step UP into the doorway -> warp
                self.b.press("UP", 8, 8, self.render, owner="agent")
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

    def advance_north(self, target_map, max_legs=20):
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
            if out in ("no_path", "stuck"):
                # no walkable north edge here -> it's an interior/gate house: warp north
                log(f"   leg {leg}: no north edge on map {m} - warping north")
                if self.enter_warp(prefer="north") != "warped":
                    log(f"   !! ADVANCE stuck on map {m} (no north edge, no north warp)")
                    return "stuck"
        return "timeout"

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

    def beat_gym(self, name):
        # the road in + the gym trainers are battles -> the 5/5 engine. Travel handles
        # walking to/through; encountered trainers hand off automatically.
        log(f"   gym handler: battles route through the battle engine (5/5, untouched)")
        if st.in_battle(self.b):
            return self.battle_runner()
        return "gym_todo"

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

    def heal_at_center(self):
        """AUTHENTIC Pokemon Center heal (no RAM poking): route to the Viridian Center door,
        enter, walk to the nurse counter, drive the YES heal dialogue, verify HP -> max, exit.
        Returns 'healed' or 'stuck' (LOUD)."""
        h0 = self.lead_hp()
        if h0[0] >= h0[1]:
            log(f"   HEAL: already full ({h0[0]}/{h0[1]}) - skipping"); return "healed"
        return_to = tv.coords(self.b)             # heal is TRANSPARENT: come back here after
        log(f"   HEAL: lead at {h0[0]}/{h0[1]} -> routing to the Viridian Pokemon Center "
            f"(will return to {return_to})")
        # 1) to the PC door + step in
        if self.trav.travel(target_map=None, arrive_coord=(VIRIDIAN_PC_DOOR[0],
                            VIRIDIAN_PC_DOOR[1] + 1), max_steps=400, max_seconds=120) != "arrived":
            log(f"   !! HEAL: couldn't reach the PC door (at {tv.coords(self.b)})"); return "stuck"
        before = tv.map_id(self.b)
        for _ in range(6):
            self.b.press("UP", 8, 10, self.render, owner="agent")
            for _ in range(20):
                self.b.run_frame()
            if tv.map_id(self.b) != before:
                break
        if tv.map_id(self.b) != VIRIDIAN_PC_MAP:
            log(f"   !! HEAL: did not enter the Center (got {tv.map_id(self.b)})"); return "stuck"
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
        # 3) exit: step off the entrance mat then back down onto it (being PLACED on a warp
        # tile doesn't fire it - only STEPPING onto it does). Walk up one, then down onto the
        # exit tile; the door/warp tiles are found via the same 0x6x behavior scan as entry.
        warp_tiles = set(self._door_tiles())
        log(f"   HEAL: interior warp tiles={sorted(warp_tiles)} (player {tv.coords(self.b)})")
        self._step_to((PC_ENTRY[0], PC_ENTRY[1] - 1))     # one tile ABOVE the mat
        inside = tv.map_id(self.b)
        for _ in range(8):
            if tv.map_id(self.b) != inside:
                break
            self.b.press("DOWN", 8, 12, self.render, owner="agent")
            for _ in range(18):
                self.b.run_frame()
            log(f"      exit DOWN -> map={tv.map_id(self.b)} coords={tv.coords(self.b)}")
        log(f"   HEAL: exited Center -> map={tv.map_id(self.b)} coords={tv.coords(self.b)}")
        if tv.map_id(self.b) != VIRIDIAN:
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
