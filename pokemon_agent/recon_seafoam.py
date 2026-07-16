"""recon_seafoam.py — THE SEAFOAM CROSSING: Fuchsia -> R19 -> R20east -> the Seafoam
interior (STRENGTH boulder chain -> becalmed B3F surf) -> R20west -> CINNABAR.

Prereq canonical = surf_taught (Lapras knows Surf, Venusaur knows Strength, badge 6).

THE DERIVATION (recon_seafoam_plan.py, all pret ground truth — layout bins + map.json +
scripts.inc, cached G:\\temp\\longrun\\pret\\):
  - R20's surface is SEVERED at Seafoam (dual-flood proven, night shift 2). The crossing
    is the interior. Land+water meta-BFS over the 5 floors finds NO route with currents
    active: the west-exit cluster {F1r3 <- B1Fr3 <- B2Fr10 <- B3F east block} is sealed
    behind the B3F current field.
  - THE MECHANISM (scripts.inc): B3F's current stops when BOTH B3F boulders are PRESENT
    (FLAG_HIDE_SEAFOAM_B3F_BOULDER_1/2 0x046/0x047 cleared) -> FLAG_STOPPED_SEAFOAM_B3F_
    CURRENT (0x2D2) -> setmaplayoutindex LAYOUT_..._CURRENT_STOPPED (currents -> calm
    water). Boulders cascade DOWN THE HOLE CHAIN (MB_FALL_WARP 0x66): 1F pushes drop
    them to B1F, B1F pushes to B2F, B2F pushes to B3F. Falling into B3F with <2 boulders
    present = the FORCED CURRENT RIDE to B4F (27,21) — so the last fall must come after
    both boulders are down.
  - Warp-event tiles at the hole LANDING spots ((21,8)/(29,8) B1F class) are one-way
    anchors with PLAIN behavior — safe to stand on; only 0x66 tiles fall.
  - With B3F stopped, the meta-BFS route is: fall B2F hole -> B3F becalmed pocket ->
    surf east/south -> ladder (31,16) -> B2F (31,17) -> ladder (32,14) -> B1F (32,14)
    -> ladder (28,19) -> F1 (28,19) -> exit door (32,21) -> R20 (72,14) = WEST SEA ->
    surf west -> Cinnabar (3,8).

THE MISSION (floors: F1 (1,83) B1F (1,84) B2F (1,85) B3F (1,86)):
  F1:  Strength on; b1 (22,12) UP x4 + LEFT x1 -> hole (21,8); b2 (32,9) UP x1 +
       LEFT x2 -> hole (30,8); fall (21,8).
  B1F: Strength; b1 (22,8) RIGHT x1 -> hole (23,8); fall (23,8).
  B2F: Strength; b1 (22,8) RIGHT x2 -> hole (24,8)   [boulder 1 lands B3F (23,8)]
       climb west ladders (7,4)->B1F, (10,6)->F1; fall (30,8) -> B1F (29,8).
  B1F: Strength; b2 (30,8) LEFT x2 -> hole (28,8); fall (28,8) -> B2F (29,8).
  B2F: Strength; b2 (30,8) LEFT x3 -> hole (27,8)    [boulder 2 lands B3F (24,8);
       flag 0x2D2 arms on next B3F transition]
       fall (27,8) -> B3F (24,9) ON CALM WATER (EnterByFalling sees 2 boulders ->
       CurrentBlocked; no ride).
  B3F: verify 0x2D2; sea_walk to ladder (31,16) -> B2F -> (32,14) -> B1F -> (28,19)
       -> F1 -> exit (32,21) -> R20 west -> Cinnabar.

STRENGTH ACTUATION (first live verify of the class): face boulder, A -> "Want to use
STRENGTH?" YES (default) -> drain; VERIFIED by FLAG_SYS_USE_STRENGTH (0x805). Resets on
every map change -> re-arm per floor. A push: stand opposite, hold the direction; the
boulder slides one tile (player stays); VERIFIED by the live gObjectEvents coord
(field_moves.scan_field_objects — behavioral ground truth, not the template table).

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_seafoam.py
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
# travel.py logs a "🌊 SURF MOUNT" line; piped to a cp1252 console that crashes the run mid-heal
# (NS5). Force utf-8 stdout so any emoji in shared-plumbing logs can't kill a headless strike.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import firered_ram as ram            # noqa: E402
import field_moves as fm             # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_seafoam")
BANK = os.path.join(SCRATCH, "banked_CINNABAR")
DBG = os.path.join(SCRATCH, "seafoam_probe")

FUCHSIA, R19, R20, CINNABAR = (3, 7), (3, 37), (3, 38), (3, 8)
F1, B1F, B2F, B3F = (1, 83), (1, 84), (1, 85), (1, 86)
MOVE_SURF, MOVE_STRENGTH = 57, 70
FLAG_STR_ACTIVE = 0x805
FLAG_B3F_CALM = 0x2D2
DOOR_EAST = (60, 8)        # R20 -> F1 (6,21)
EXIT_WEST = (32, 21)       # F1 -> R20 (72,14)
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# (op, args...) — coords are floor-local SAVE coords; push boulder ids are the boulder's
# position at op time (matched to the nearest live object).
MISSION = [
    ("strength", (22, 12)),
    ("push", (22, 12), "UP", 4), ("push", (22, 8), "LEFT", 1),   # b1 -> hole (21,8)
    ("push", (32, 9), "UP", 1), ("push", (32, 8), "LEFT", 2),    # b2 -> hole (30,8)
    ("fall", (21, 8), B1F),
    ("strength", (22, 8)), ("push", (22, 8), "RIGHT", 1),        # b1 -> hole (23,8)
    ("fall", (23, 8), B2F),
    ("strength", (22, 8)), ("push", (22, 8), "RIGHT", 2),        # b1 -> B3F (23,8)
    ("ladder", (7, 4), B1F), ("ladder", (10, 6), F1),
    ("fall", (30, 8), B1F),
    ("strength", (30, 8)), ("push", (30, 8), "LEFT", 2),         # b2 -> hole (28,8)
    ("fall", (28, 8), B2F),
    ("strength", (30, 8)), ("push", (30, 8), "LEFT", 3),         # b2 -> B3F (24,8)
    ("fall", (27, 8), B3F),
    ("verify_calm",),
    ("ladder", (31, 16), B2F),
    ("ladder", (32, 14), B1F),
    ("ladder", (28, 19), F1),
    ("ladder", EXIT_WEST, R20),
]


def _resolve_state(name):
    """Resolve a state BASENAME/path -> (state_path, sidecar_dir, sidecar_prefix)."""
    if not name:
        return (os.path.join(CANON, "kira_campaign.state"), CANON, "kira_campaign")
    for cand in (name, os.path.join(_HERE, "states", name),
                 os.path.join(_HERE, "states", "workshop", name),
                 os.path.join(_HERE, "states", name + ".state"),
                 os.path.join(_HERE, "states", "workshop", name + ".state")):
        if os.path.exists(cand):
            d = os.path.dirname(cand)
            base = os.path.basename(cand)
            pref = base[:-6] if base.endswith(".state") else base
            return (cand, d, pref)
    return (name, os.path.dirname(name) or CANON, "kira_campaign")


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("SEAFOAM_STATE", ""))
    b = Bridge(ROM)
    with open(state_path, "rb") as f:
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
    _w_side = os.path.join(sc_dir, sc_pref + ".world_model.json")
    _s_side = os.path.join(sc_dir, sc_pref + ".strat_memory.json")
    _soul_side = os.path.join(sc_dir, sc_pref + ".soul.json")
    for loader, path, fallback in (
            (camp.world.load, _w_side, C.WORLD_JSON),
            (camp.strat.load, _s_side, C.STRAT_JSON)):
        try:
            loader(path if os.path.exists(path) else fallback)
        except Exception:
            pass
    try:
        if camp.soul is not None:
            camp.soul.load(_soul_side if os.path.exists(_soul_side)
                           else os.path.join(CANON, "soul.json"))
    except Exception:
        pass
    L(f"boot state = {state_path} map={tv.map_id(b)} coords={tv.coords(b)}")

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

    # ── water machinery (recon_cinnabar verbatim) ────────────────────────────
    def water_save(g):
        return {(bx - tv.MAP_OFFSET, by - tv.MAP_OFFSET) for bx, by in g.water}

    def sea_ok(g, wset):
        def ok(sx, sy):
            bx, by = sx + tv.MAP_OFFSET, sy + tv.MAP_OFFSET
            if not (0 <= bx < g.w and 0 <= by < g.h):
                return False
            if g.col.get((bx, by), 1) != 0:
                return False
            return g.walkable(sx, sy) or (sx, sy) in wset
        return ok

    def on_water():
        g = tv.Grid(b)
        return tuple(tv.coords(b) or (99, 99)) in water_save(g)

    def mount(face_key):
        if on_water():
            return True
        for attempt in range(4):
            b.press(face_key, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(40):
                b.run_frame()
            drain(key="A")
            for _ in range(240):
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

    def live_boulders():
        return [ob["coord"] for ob in fm.scan_field_objects(b, {fm.GFX_BOULDER})]

    def live_npc_tiles():
        """Live object-event BODY tiles (travel._npc_tiles pattern) — run4 truth: a
        WANDERING swimmer parked on the planned tile blocks the same step forever if
        only static templates are masked. Body tile only, never the facing tile."""
        OB, SZ = 0x02036E38, 0x24
        out = set()
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.add((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                     b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    def step_to(tile, wset=None):
        cur = tuple(tv.coords(b) or (0, 0))
        d = (tile[0] - cur[0], tile[1] - cur[1])
        if d in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            d = (d[0] // 2, d[1] // 2)
        key = KEY_OF.get(d)
        if key is None:
            return camp._step_to(tile)
        if wset is None:
            wset = water_save(tv.Grid(b))
        if tile in wset and cur not in wset and not on_water():
            return mount(key)
        for _attempt in range(3):
            b.press(key, 8, 6, camp.render, owner="agent")
            for _ in range(50):
                b.run_frame()
                if tuple(tv.coords(b) or ()) == tile:
                    break
            if fight_open() or dd_box(b):
                return True
            if tuple(tv.coords(b) or ()) == tile:
                return True
        return False

    def sea_walk(goal_test, label, tries=10, avoid=()):
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
            # boulders: LIVE positions only — templates go stale after a push (the
            # template coord becomes empty floor and must not mask the path).
            # NPCs: live body tiles FIRST (wanderers move off-template — run4), plus
            # non-boulder templates for distance-culled far spawns.
            npcs = live_npc_tiles() | {tuple(o[0]) for o in
                                       tv.read_object_templates(b)
                                       if o[2] and o[1] != fm.GFX_BOULDER}
            ok0 = sea_ok(g, wset)
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: ok0(sx, sy) and (sx, sy) not in wts
                       and (sx, sy) not in npcs and (sx, sy) not in avoid)
            L(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                L(f"   [{label}] no path from {cur}")
                snap(f"nopath_{label[:12]}_{cur[0]}_{cur[1]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                if handle_interrupts():
                    budget += 1
                    break
                if not step_to(tuple(t), wset):
                    L(f"   [{label}] step blocked {tuple(tv.coords(b) or ())} -> {tuple(t)} "
                      f"(facing {b.rd8(0x02036E38 + 0x18) & 0xF}, "
                      f"npcs {sorted(live_npc_tiles())[:6]})")
                    snap(f"blocked_{t[0]}_{t[1]}")
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
            if tuple(tv.coords(b) or ()) != cur:
                budget += 1
        return goal_test(tuple(tv.coords(b) or ()))

    def _s32(v):
        return v - (1 << 32) if v >= (1 << 31) else v

    DIRN_OF = {"south": 1, "north": 2, "west": 3, "east": 4}

    def connections():
        out = {}
        hdr = b.rd32(tv.GMAPHEADER + 0x0C)
        if not hdr or hdr < 0x02000000:
            return out
        n = _s32(b.rd32(hdr))
        arr = b.rd32(hdr + 4)
        if not (0 < n < 16) or arr < 0x02000000:
            return out
        for i in range(n):
            c = arr + i * 0xC
            out.setdefault(b.rd8(c), []).append(_s32(b.rd32(c + 4)))
        return out

    def cross_edge(direction, label):
        m0 = tuple(tv.map_id(b))
        conns = connections().get(DIRN_OF[direction])
        if not conns:
            L(f"   [{label}] no {direction} connection on {m0} — skip")
            return False
        off = conns[0]
        key = {"south": "DOWN", "north": "UP", "west": "LEFT", "east": "RIGHT"}[direction]
        for round_ in range(6):
            g = tv.Grid(b)
            wset = water_save(g)
            ok0 = sea_ok(g, wset)
            if direction in ("south", "north"):
                extreme = g.sy_hi if direction == "south" else 0
                band = [(x, extreme) for x in range(max(g.sx_lo, off), g.sx_hi + 1)
                        if ok0(x, extreme)]
            else:
                extreme = g.sx_hi if direction == "east" else 0
                band = [(extreme, y) for y in range(max(g.sy_lo, off), g.sy_hi + 1)
                        if ok0(extreme, y)]
            if not band:
                L(f"!! [{label}] no {direction}-edge band on {m0}")
                for _ in range(120):
                    b.run_frame()
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            band.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]) + round_ * 7)
            tgt = band[min(round_, len(band) - 1)]
            if not sea_walk(lambda c, t=tgt: c == t, f"{label}-band"):
                for _ in range(120):
                    b.run_frame()
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

    # ── the interior primitives ──────────────────────────────────────────────
    def settle(n=90):
        for _ in range(n):
            b.run_frame()

    def nearest_boulder(approx, radius=8):
        """Live gObjectEvents are DISTANCE-CULLED — a far boulder reads as absent.
        If no live boulder is near `approx`, walk toward approx first, then re-scan."""
        for _attempt in range(3):
            bs = [t for t in live_boulders()
                  if abs(t[0] - approx[0]) + abs(t[1] - approx[1]) <= radius]
            if bs:
                return min(bs, key=lambda t: abs(t[0] - approx[0]) + abs(t[1] - approx[1]))
            cur = tuple(tv.coords(b) or (0, 0))
            if abs(cur[0] - approx[0]) + abs(cur[1] - approx[1]) <= 3:
                return None                     # close enough to trust the empty scan
            if not sea_walk(lambda c, a=approx: abs(c[0] - a[0]) + abs(c[1] - a[1]) <= 3,
                            "boulder-approach"):
                return None
        return None

    def ensure_strength(approx):
        """Face the target boulder, A -> YES; verified by FLAG_SYS_USE_STRENGTH."""
        if fm.read_flag(b, FLAG_STR_ACTIVE):
            return True
        bl = nearest_boulder(approx)
        if bl is None:
            L(f"!! [strength] no live boulder near {approx} on {tv.map_id(b)}")
            return False
        for attempt in range(3):
            nbs = [(bl[0] + dx, bl[1] + dy) for dx, dy in
                   ((0, 1), (0, -1), (1, 0), (-1, 0))]
            if not sea_walk(lambda c, s=set(nbs): c in s, "str-approach"):
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            face = KEY_OF.get((bl[0] - cur[0], bl[1] - cur[1]))
            if face is None:
                continue
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            settle(30)
            drain(key="A")                      # info text + YES/NO (YES default)
            settle(30)
            if fm.read_flag(b, FLAG_STR_ACTIVE):
                L(f"   [strength] ACTIVE (attempt {attempt + 1}) at {cur} facing {face}")
                return True
        L(f"!! [strength] flag 0x805 never set (boulder {bl})")
        snap("strength_fail")
        return False

    def push(approx, key, n):
        """n Strength pushes; each verified by the live boulder coord moving."""
        d = DELTA[key]
        for i in range(n):
            bl = nearest_boulder(approx)
            if bl is None:
                L(f"!! [push] boulder near {approx} vanished (i={i}) — treating as fallen")
                return True                     # last push dropped it down a hole
            stand = (bl[0] - d[0], bl[1] - d[1])
            if not sea_walk(lambda c, s=stand: c == s, f"push-approach{i}",
                            avoid={tuple(bl)}):
                L(f"!! [push] can't reach {stand} to push {bl} {key}")
                return False
            moved = False
            for _try in range(4):
                if handle_interrupts():
                    continue
                b.press(key, 40, 10, camp.render, owner="agent")
                settle(70)                      # push animation ~1s
                b2l = nearest_boulder((bl[0] + d[0], bl[1] + d[1]))
                if b2l != bl:                   # moved or fell (vanished/None)
                    moved = True
                    break
            if not moved:
                L(f"!! [push] {bl} would not move {key} (player {tv.coords(b)})")
                snap(f"push_fail_{bl[0]}_{bl[1]}")
                return False
            approx = (bl[0] + d[0], bl[1] + d[1])
            L(f"   [push] {bl} -> {approx} ({key}, {i + 1}/{n})")
            settle(30)
        return True

    ARROW_KEY = {0x62: "RIGHT", 0x63: "LEFT", 0x64: "UP", 0x65: "DOWN"}

    def tile_behavior(t):
        """Live metatile behavior via the backup layout + tileset attrs (probe read)."""
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

    def go_warp(tile, dest, label):
        """Walk adjacent to a warp/hole/door tile, then step ON it; verify map flip.
        ARROW-WARP class (run9: the F1 west exit (32,21) is MB_SOUTH_ARROW_WARP 0x65 —
        fires ONLY by pressing its arrow direction while standing on the mat): approach
        from the arrow-opposite side and hold the arrow key through the tile."""
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        beh = tile_behavior(tile)
        arrow = ARROW_KEY.get(beh)
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in
               ((0, 1), (0, -1), (1, 0), (-1, 0))]
        if arrow:
            d = DELTA[arrow]
            nbs = [(tile[0] - d[0], tile[1] - d[1])]   # walk in along the arrow
        for attempt in range(4):
            if tuple(tv.coords(b) or ()) not in nbs and tuple(tv.coords(b) or ()) != tile:
                if not sea_walk(lambda c, s=set(nbs): c in s, f"{label}-approach"):
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
            if handle_interrupts():
                continue
            if tuple(tv.map_id(b)) == dest:
                settle(180)                     # warp settle + map scripts
                L(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)} (beh {hex(beh)})")
                _stage_save(label)
                return True
            if tuple(tv.map_id(b)) != m0:
                L(f"!! [{label}] warped to {tuple(tv.map_id(b))}, wanted {dest}")
                settle(180)
                return False
        L(f"!! [{label}] never fired (at {tv.map_id(b)}@{tv.coords(b)}, beh {hex(beh)})")
        snap(f"warpfail_{label[:16]}")
        return False

    # ── preflight ────────────────────────────────────────────────────────────
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} money=${camp.money()}")
    n = b.rd8(ram.GPLAYER_PARTY_CNT)
    have = {m for s in range(n) for m in (st.read_party_moves(b, s) or [])}
    if MOVE_SURF not in have or MOVE_STRENGTH not in have:
        L(f"!! party lacks Surf/Strength (have surf={MOVE_SURF in have} "
          f"str={MOVE_STRENGTH in have}) — abort")
        return 1
    L(f"   b3f-calm flag pre-read: {fm.read_flag(b, FLAG_B3F_CALM)} "
      f"boulder hide flags 040-047: "
      f"{[int(fm.read_flag(b, f)) for f in range(0x40, 0x48)]}")

    # ── PHASE 1: the sea road to the Seafoam east door ──────────────────────
    # NS9: a grind-bank start arrives with DEPLETED PP + a worn party — she WHITED OUT on R20's
    # wild gauntlet mid-crossing (runs 1/2: Lapras move-slot-0 hit 0 PP, fodder fainted, blackout
    # warped her to a Center = the (11,5) wedge). Heal to FULL before the sea road so the lead's
    # moves don't run dry. Harmless from a fresh canonical start (already full).
    try:
        L("   pre-crossing heal (full PP/HP for the R19/R20 wild gauntlet)")
        camp.heal_nearest()
        L(f"   healed @ {tv.map_id(b)} {tv.coords(b)}")
    except Exception as e:
        L(f"   pre-crossing heal errored: {e}")
    deadline = time.time() + 2400
    while tuple(tv.map_id(b)) != R20 and time.time() < deadline:
        here = tuple(tv.map_id(b))
        if handle_interrupts():
            continue
        if here == FUCHSIA:
            # ROBUST START (NS5 kit line): the Safari strike leaves her at (33,32) by the Warden's
            # door, where the bespoke south-band sea_walk can't navigate out. The general campaign
            # traveler crosses Fuchsia->R19 fine from anywhere (proven this shift), so use it FIRST;
            # cross_edge stays the fallback for the credits-canonical start position.
            _wr = camp.walk_to_map(R19, "south")
            if tuple(tv.map_id(b)) != R19 and not cross_edge("south", "fuchsia-south"):
                L(f"!! Fuchsia->R19 failed (walk_to_map={_wr})")
                return 1
        elif here == R19:
            if not cross_edge("west", "r19-west"):
                return 1
        else:
            # ROBUST START from a nearby grind spot (NS9: bench-grind banks at Route 18 (3,36),
            # west of Fuchsia). Route to Fuchsia via the general traveler FIRST — it crosses the
            # R18/R15 gatehouses fine — then the FUCHSIA branch takes the proven south sea road.
            L(f"!! off-route map {here} — routing to Fuchsia via the general traveler")
            try:
                camp.walk_to_map(FUCHSIA, "east")
            except Exception as e:
                L(f"   walk_to_map(Fuchsia) errored: {e}")
            if tuple(tv.map_id(b)) != FUCHSIA and not (cross_edge("west", "reroute-west")
                                                       or cross_edge("south", "reroute-south")):
                L(f"!! could not route off {here} toward Fuchsia")
                return 1
        settle(180)
        _stage_save("leg")
    if tuple(tv.map_id(b)) != R20:
        L("!! never reached Route 20")
        return 1
    L(f"   ON ROUTE 20 @ {tv.coords(b)} — heading for the east door {DOOR_EAST}")
    if not go_warp(DOOR_EAST, F1, "east-door"):
        return 1

    # ── PHASE 2: the interior mission ────────────────────────────────────────
    for op in MISSION:
        if time.time() > deadline:
            L("!! deadline inside the interior")
            return 1
        while handle_interrupts():
            pass
        kind = op[0]
        L(f"-- op {op} (map {tv.map_id(b)} @ {tv.coords(b)})")
        if kind == "strength":
            if not ensure_strength(op[1]):
                return 1
        elif kind == "push":
            if not push(op[1], op[2], op[3]):
                return 1
        elif kind in ("fall", "ladder"):
            if not go_warp(op[1], op[2], f"{kind}{op[1]}"):
                return 1
        elif kind == "verify_calm":
            calm = fm.read_flag(b, FLAG_B3F_CALM)
            L(f"   [calm] FLAG_STOPPED_SEAFOAM_B3F_CURRENT = {calm}; "
              f"on_water={on_water()} @ {tv.coords(b)}")
            snap("b3f_becalmed")
            if not calm:
                L("!! current NOT stopped — boulder chain incomplete")
                return 1

    # ── PHASE 3: west sea -> Cinnabar ────────────────────────────────────────
    L(f"   WEST SEA @ {tv.map_id(b)}{tv.coords(b)} — surfing for Cinnabar")
    settle(180)
    _stage_save("r20_west")
    legs = 0
    while time.time() < deadline:
        here = tuple(tv.map_id(b))
        if handle_interrupts():
            continue
        if here == CINNABAR:
            break
        if not cross_edge("west", f"west{legs}"):
            if not cross_edge("south", f"south{legs}"):
                L(f"!! wedged on {here}")
                snap("fail_west_wedge")
                return 1
        legs += 1
        settle(180)
        _stage_save(f"westleg{legs}")
    if tuple(tv.map_id(b)) != CINNABAR:
        L("!! never reached Cinnabar")
        return 1

    L(f"   CINNABAR ISLAND @ {tv.coords(b)} after {n_battles[0]} battles — healing")
    _stage_save("cinnabar_arrival")          # bank the arrival BEFORE the heal (crash-safety, NS5)
    r = camp.heal_nearest()
    L(f"   heal_nearest -> {r}")
    snap("90_cinnabar")
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
