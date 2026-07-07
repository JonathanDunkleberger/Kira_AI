"""recon_cinnabar.py — THE SEA ROAD: Fuchsia -> Route 19 -> Route 20 (past Seafoam) -> Cinnabar.

The badge-7 leg. Prereq canonical = surf_taught (Lapras knows Surf, badge 5+ in hand —
Surf's gate is FLAG_BADGE05_GET, source-confirmed in field_moves.py).

THE NEW CAPABILITY THIS PROVES: WATER IS A ROAD WITH A MOUNT TOLL.
  - Grid.water (travel.py, additive surfable-water layer 0x10/0x12/0x15) becomes WALKABLE
    for planning: sea_ok() = land-walkable OR surfable-water.
  - The land->water boundary step is NOT a step: face the water, press A — the game offers
    "would you like to SURF?" (GetInteractedWaterScript, YES default) — A confirms, the
    mount animation hops her on. water->land auto-dismounts (plain step).
  - "Am I surfing?" is BEHAVIORAL ground truth: her save-coord is in the water set
    (no gPlayerAvatar reader exists; do not invent one until a wall demands it).
  - Seafoam's surface strips are land mid-route: the stepper re-mounts at each strip edge,
    which is why mount lives in the STEP EXECUTOR, not in a one-shot "start surfing" phase.
ROUTE (Kanto ground truth): Fuchsia south edge -> Route 19 (vertical water), south, then
the route bends WEST -> Route 20 (E-W past the twin Seafoam islands) -> west edge ->
Cinnabar Island. Swimmers + Tentacool ride the normal BattleAgent.
Success = arrival on a NEW city map west of Route 20 with a Pokemon Center (Cinnabar),
heal, bank -> banked_CINNABAR (label cinnabar_reach).

STATUS: DRAFT — written night shift #2 while the Safari strike ran; UNRUN until the
safari_hms + surf_taught banks land. Expect first-run walls at: the Fuchsia south-edge
band, the R19->R20 west band, Seafoam strip re-mounts.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_cinnabar.py
"""
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.environ.get("WATCH") != "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import firered_ram as ram            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_cinnabar")
BANK = os.path.join(SCRATCH, "banked_CINNABAR")
DBG = os.path.join(SCRATCH, "cinnabar_probe")

FUCHSIA = (3, 7)
MOVE_SURF = 57


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    n_battles = [0]

    def fight():
        n_battles[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=240)

    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    os.makedirs(STAGE, exist_ok=True)
    os.makedirs(DBG, exist_ok=True)

    def snap(name):
        try:
            b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, name + ".png"))
        except Exception as e:
            L(f"   snap {name} failed: {e}")

    def _stage_save(reason="tick"):
        try:
            with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
                f.write(b.save_state())
            return True
        except Exception as e:
            L(f"!! STAGE SAVE FAILED [{reason}]: {e}")
            return False

    def _stage_continuity():
        try:
            camp.world.save(os.path.join(STAGE, "world_model.json"))
            camp.strat.save(os.path.join(STAGE, "strat_memory.json"))
            if camp.soul is not None:
                camp.soul.save(os.path.join(STAGE, "soul.json"))
            with open(os.path.join(STAGE, "journey_core.json"), "w", encoding="utf-8") as jf:
                json.dump(camp._journey_narrative(), jf, ensure_ascii=False, indent=2)
        except Exception as e:
            L(f"!! stage continuity failed: {e}")

    camp._save_campaign = _stage_save
    camp._continuity_save = _stage_continuity
    camp._continuity_load = lambda *a, **k: None
    for loader, path in ((camp.world.load, C.WORLD_JSON), (camp.strat.load, C.STRAT_JSON)):
        try:
            loader(path)
        except Exception:
            pass
    try:
        if camp.soul is not None:
            camp.soul.load(os.path.join(CANON, "soul.json"))
    except Exception:
        pass

    def fight_open():
        return ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))

    def drain(max_n=40, key="B"):
        stable = 0
        for _ in range(max_n):
            if fight_open():
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

    def handle_interrupts():
        if fight_open():
            fight()
            drain()
            return True
        if dd_box(b):
            drain()
            return True
        return False

    # ── the water layer ──────────────────────────────────────────────────────
    def water_save(g):
        """Grid.water holds BUFFER coords; planning runs in save coords (-MAP_OFFSET)."""
        return {(bx - tv.MAP_OFFSET, by - tv.MAP_OFFSET) for bx, by in g.water}

    def sea_ok(g, wset):
        def ok(sx, sy):
            return g.walkable(sx, sy) or (sx, sy) in wset
        return ok

    def on_water():
        g = tv.Grid(b)
        return tuple(tv.coords(b) or (99, 99)) in water_save(g)

    KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}

    def mount(face_key):
        """Face the water, A -> 'want to SURF?' YES (default) -> she hops on.
        Verified by the BEHAVIORAL read: coord lands in the water set."""
        for attempt in range(4):
            b.press(face_key, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(40):
                b.run_frame()
            drain(key="A")                       # YES is the default cursor
            for _ in range(240):                 # mount animation
                b.run_frame()
                if on_water():
                    break
            if fight_open():
                fight()
                drain()
            if on_water():
                L(f"   [surf] MOUNTED at {tv.coords(b)} (attempt {attempt + 1})")
                return True
        L(f"!! [surf] mount failed at {tv.coords(b)} facing {face_key}")
        return False

    def step_to(tile):
        """One-tile step with the tap-turn law + the mount toll at a land->water edge."""
        cur = tuple(tv.coords(b) or (0, 0))
        d = (tile[0] - cur[0], tile[1] - cur[1])
        key = KEY_OF.get(d)
        if key is None:
            return camp._step_to(tile)
        g = tv.Grid(b)
        wset = water_save(g)
        if tile in wset and cur not in wset:
            return mount(key)                    # the toll: A-prompt, not a step
        return camp._step_to(tile)

    def sea_walk(goal_test, label, tries=10):
        """BFS over land+water, execute with step_to. Battles/boxes don't spend tries."""
        budget = tries
        while budget > 0:
            budget -= 1
            if handle_interrupts():
                budget += 1
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            g = tv.Grid(b)
            wset = water_save(g)
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
            ok0 = sea_ok(g, wset)
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: ok0(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs)
            if not p:
                L(f"   [{label}] no sea path from {cur}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                if handle_interrupts():
                    budget += 1
                    break
                if not step_to(tuple(t)):
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True                  # crossed an edge mid-walk — re-dispatch
            if goal_test(tuple(tv.coords(b) or ())):
                return True
        return goal_test(tuple(tv.coords(b) or ()))

    def cross_edge(direction, label):
        """Cross a map connection: sea_walk to the extreme row/col in `direction`, then
        hold the key until the map id flips (connection bands are partial — if the hold
        doesn't cross, slide along the edge and retry)."""
        m0 = tuple(tv.map_id(b))
        key = {"south": "DOWN", "north": "UP", "west": "LEFT", "east": "RIGHT"}[direction]
        for round_ in range(6):
            g = tv.Grid(b)
            wset = water_save(g)
            ok0 = sea_ok(g, wset)
            if direction in ("south", "north"):
                extreme = g.sy_hi if direction == "south" else 0
                band = [(x, extreme) for x in range(g.sx_lo, g.sx_hi + 1) if ok0(x, extreme)]
            else:
                extreme = g.sx_hi if direction == "east" else 0
                band = [(extreme, y) for y in range(g.sy_lo, g.sy_hi + 1) if ok0(extreme, y)]
            if not band:
                L(f"!! [{label}] no {direction}-edge band on {m0}")
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            band.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]) + round_ * 7)
            tgt = band[min(round_, len(band) - 1)]
            if not sea_walk(lambda c, t=tgt: c == t, f"{label}-band"):
                continue
            for _hold in range(4):
                cur2 = tuple(tv.coords(b) or (0, 0))
                nxt = {"south": (cur2[0], cur2[1] + 1), "north": (cur2[0], cur2[1] - 1),
                       "west": (cur2[0] - 1, cur2[1]), "east": (cur2[0] + 1, cur2[1])}[direction]
                g2 = tv.Grid(b)
                w2 = water_save(g2)
                if nxt in w2 and cur2 not in w2:
                    if not mount(key):
                        break
                    continue
                b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(90):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if handle_interrupts():
                    continue
                if tuple(tv.map_id(b)) != m0:
                    for _ in range(120):
                        b.run_frame()
                    L(f"   [{label}] EDGE {direction}: {m0} -> {tuple(tv.map_id(b))} "
                      f"@ {tv.coords(b)}")
                    return True
        L(f"!! [{label}] {direction} crossing never fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    # ── preflight ────────────────────────────────────────────────────────────
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} money=${camp.money()}")
    if tuple(tv.map_id(b)) != FUCHSIA:
        L(f"!! not in Fuchsia (at {tv.map_id(b)}) — abort")
        return 1
    n = b.rd8(ram.GPLAYER_PARTY_CNT)
    surfer = next((s for s in range(n)
                   if MOVE_SURF in (st.read_party_moves(b, s) or [])), None)
    if surfer is None:
        L("!! nobody in the party knows Surf — run recon_surf_teach first; abort")
        return 1
    L(f"   surfer = slot {surfer} ({st.SPECIES_NAME.get(st.read_party_species(b, surfer))})")

    # ── the legs ─────────────────────────────────────────────────────────────
    deadline = time.time() + 1800
    legs = 0
    seen_maps = [tuple(tv.map_id(b))]
    while time.time() < deadline:
        here = tuple(tv.map_id(b))
        if handle_interrupts():
            continue
        if here != seen_maps[-1]:
            seen_maps.append(here)
            _stage_save(f"map_{here[0]}_{here[1]}")
        # arrival heuristic: a NEW map after >=2 water crossings that is NOT a route-sized
        # sliver — confirmed by a Center door on the map (Cinnabar has one; routes don't).
        if len(seen_maps) >= 3 and here not in (FUCHSIA,):
            pc = None
            try:
                pc = C.CITY_PC_DOORS.get(here)
            except Exception:
                pass
            if pc:
                L(f"   ARRIVED: {here} has a Center — treating as Cinnabar")
                break
        if here == FUCHSIA:
            if not cross_edge("south", "fuchsia-south"):
                snap("fail_fuchsia_south")
                return 1
            legs += 1
            continue
        # Route 19: keep going SOUTH over water while a south band exists, then WEST.
        # Route 20: WEST all the way. The generic policy: try WEST first once off
        # Fuchsia+R19-north, fall back to SOUTH (the bend self-discovers).
        if cross_edge("west", f"leg{legs}-west"):
            legs += 1
            continue
        if cross_edge("south", f"leg{legs}-south"):
            legs += 1
            continue
        L(f"!! no crossing fired on {here} — wedged (legs={legs}, seen={seen_maps})")
        snap("fail_wedged")
        return 1

    here = tuple(tv.map_id(b))
    L(f"   sea road done: at {here}@{tv.coords(b)} after {legs} legs, "
      f"{n_battles[0]} battles (maps: {seen_maps})")
    r = camp.heal_nearest()
    L(f"   heal_nearest -> {r}")
    snap("80_final")
    _stage_save("cinnabar_reach")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} cinnabar_reach")
    return 0


if __name__ == "__main__":
    sys.exit(main())
