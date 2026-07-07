"""recon_victory.py — THE ROAD TO THE PLATEAU: EQ teach + R22 Gary + the badge gauntlet
+ Victory Road + Indigo Plateau bank.

Ground truth (pret, cached G:\\temp\\longrun\\pret\\: Route22/23 + VictoryRoad_* maps/
scripts/bins + vr_solve.py offline meta-BFS):
- Phase 0 (Viridian): teach TM26 EARTHQUAKE (item 314, move 89) to Venusaur slot 0 over
  Secret Power (move slot 3) — 100-power ground STAB-neutral carry for the E4.
- Phase 1: west edge -> Route 22 (3,41). GARY trigger = col 33 rows 4-6 (var==3, armed
  by badge 8): forced scene -> his strongest pre-E4 team; handle_interrupts owns it.
  A loss whiteouts to Viridian — the loop heals + re-crosses (trigger stays armed).
- Phase 2: gate (8-9,5) -> Route22_NorthEntrance (28,0) -> north side -> Route 23 (3,42).
- Phase 3: R23 south leg northward: six badge-guard lockall scenes (Cascade y149 ->
  Volcano y61) + Earth guard (y31-36) — all msgbox drains, she holds all 8 badges.
  VICTORY ROAD door (5,28) -> VR 1F (1,39).
- Phase 4 (offline-solved ELEVATION-AWARE, vr1f_probe*.py — vr_solve.py's "1F/2F-east
  no puzzle" was an elevation-blind artifact; victory_run2 died on it): every floor
  barrier opens ONLY by pushing a boulder onto its 0x20 STRENGTH_BUTTON switch
  (boulder-lands-on-switch fires the coord event — field_control_avatar.c:1076).
  1F: barrier (12,14-15) gates the ladder; push (7,18) -> switch (20,16) [chain
  below; the (11,20) stand is the entrance arrow tile 0x65 — fires on DOWN only,
  we press UP there] -> ladder (3,2). 2F: puzzle1 (6,17) D,LL,D,LL -> switch (2,19)
  opens barrier1 (13,10-11); then the row-19 boulder (33,19) (present from game
  start, FLAG 0x058 clear — the 3F hole (34,18) is the game's reset-insurance for
  a botched push, not a required reveal) LEFT x19 -> switch (14,19) opens barrier2
  (33,16-17) -> walk (36,17) -> 3F (39,17) pocket -> (37,10) -> 2F east (38,9) ->
  (48,12) -> R23 north (18,28).
- Phase 5: north edge -> Indigo Plateau Exterior (3,9) -> heal at the League center ->
  BANK indigo_reach. (E4 strike = the NEXT vehicle: shop Full Restores first.)
Bank -> %TEMP%/longrun/banked_VICTORY.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_victory.py   (WATCH=1 = window)
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
import field_moves as fm             # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402
import hm_teach as ht                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_victory")
BANK = os.path.join(SCRATCH, "banked_VICTORY")
DBG = os.path.join(SCRATCH, "victory_probe")

VIRIDIAN = (3, 1)
R22 = (3, 41)
R23 = (3, 42)
GATE = (28, 0)
VR1F, VR2F, VR3F = (1, 39), (1, 40), (1, 41)
INDIGO = (3, 9)
FLAG_STR_ACTIVE = 0x805
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
TM26_ITEM, MOVE_EQ = 314, 89

# the three-switch truth, offline-derived elevation-aware (vr1f_probe6.py verified
# every stand tile reachable at its step). Op = (kind, tile[, KEY, n[, allow-stands]]).
VR1F_PUZZLE = [("strength", (7, 18)),
               ("push", (7, 18), "DOWN", 1),
               ("push", (7, 19), "RIGHT", 4),
               # stand (11,20) = the entrance arrow tile (0x65: warps on DOWN
               # press only; this push presses UP) — sea_walk must allow it
               ("push", (11, 19), "UP", 1, ((11, 20),)),
               ("push", (11, 18), "RIGHT", 1),
               ("push", (12, 18), "UP", 1),
               ("push", (12, 17), "RIGHT", 7),
               ("push", (19, 17), "UP", 2),
               ("push", (19, 15), "RIGHT", 1),
               ("push", (20, 15), "DOWN", 1)]      # lands (20,16) = the switch
VR2F_PUZZLE1 = [("strength", (6, 17)),
                ("push", (6, 17), "DOWN", 1),
                ("push", (6, 18), "LEFT", 2),
                ("push", (4, 18), "DOWN", 1),
                ("push", (4, 19), "LEFT", 2)]      # lands (2,19) = switch 1
VR2F_PUZZLE2 = [("strength", (33, 19)),
                ("push", (33, 19), "LEFT", 19)]    # lands (14,19) = switch 2


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    render_fn = lambda: None                       # noqa: E731
    if os.environ.get("WATCH") == "1":
        import pygame
        pygame.init()
        _scale = 3
        _win = (b.width * _scale, b.height * _scale)
        _screen = pygame.display.set_mode(_win)
        pygame.display.set_caption("Kira — VICTORY ROAD (live watch)")

        def render_fn():
            for _ev in pygame.event.get():
                if _ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            _surf = pygame.image.fromstring(b.frame_rgb().tobytes(),
                                            (b.width, b.height), "RGB")
            _screen.blit(pygame.transform.scale(_surf, _win), (0, 0))
            pygame.display.flip()

        _fc = [0]
        _orig_rf = b.run_frame

        def _rf_watch():
            _orig_rf()
            _fc[0] += 1
            if _fc[0] % 4 == 0:
                render_fn()
        b.run_frame = _rf_watch

    n_battles = [0]

    def fight():
        n_battles[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=render_fn,
                           log=lambda m: print(m, flush=True)).run(max_seconds=420)

    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=render_fn)
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

    def lead_frac():
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

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

    def settle(n=90):
        for _ in range(n):
            b.run_frame()

    # ── water/edge machinery (seafoam verbatim) ──────────────────────────────
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

    def live_npc_tiles():
        OB, SZ = 0x02036E38, 0x24
        out = set()
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.add((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                     b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    def live_boulders():
        return [ob["coord"] for ob in fm.scan_field_objects(b, {fm.GFX_BOULDER})]

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

    def sea_walk(goal_test, label, tries=14, avoid=(), allow=()):
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
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - set(allow)
            npcs = live_npc_tiles() | {tuple(o[0]) for o in
                                       tv.read_object_templates(b)
                                       if o[2] and o[1] != fm.GFX_BOULDER}
            npcs |= {tuple(t) for t in live_boulders()}
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
                      f"(npcs {sorted(live_npc_tiles())[:6]})")
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
                settle(120)
                continue
            cur = tuple(tv.coords(b) or (0, 0))
            band.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]) + round_ * 7)
            tgt = band[min(round_, len(band) - 1)]
            if not sea_walk(lambda c, t=tgt: c == t, f"{label}-band"):
                settle(120)
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
                    settle(120)
                    L(f"   [{label}] EDGE {direction}: {m0} -> {tuple(tv.map_id(b))} "
                      f"@ {tv.coords(b)}")
                    return True
        L(f"!! [{label}] {direction} crossing never fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    ARROW_KEY = {0x62: "RIGHT", 0x63: "LEFT", 0x64: "UP", 0x65: "DOWN"}

    def tile_behavior(t):
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
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        beh = tile_behavior(tile)
        arrow = ARROW_KEY.get(beh)
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in
               ((0, 1), (0, -1), (1, 0), (-1, 0))]
        if arrow:
            d = DELTA[arrow]
            nbs = [(tile[0] - d[0], tile[1] - d[1])]
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
                settle(180)
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

    def nearest_boulder(approx, radius=8):
        for _attempt in range(3):
            bs = [t for t in live_boulders()
                  if abs(t[0] - approx[0]) + abs(t[1] - approx[1]) <= radius]
            if bs:
                return min(bs, key=lambda t: abs(t[0] - approx[0]) + abs(t[1] - approx[1]))
            cur = tuple(tv.coords(b) or (0, 0))
            if abs(cur[0] - approx[0]) + abs(cur[1] - approx[1]) <= 3:
                return None
            if not sea_walk(lambda c, a=approx: abs(c[0] - a[0]) + abs(c[1] - a[1]) <= 3,
                            "boulder-approach"):
                return None
        return None

    def ensure_strength(approx):
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
            settle(40)
            drain(key="A")
            settle(60)
            if fm.read_flag(b, FLAG_STR_ACTIVE):
                L(f"   [strength] ARMED (flag 0x805) at {tv.coords(b)}")
                return True
        L(f"!! [strength] flag 0x805 never set (boulder {bl})")
        snap("strength_fail")
        return False

    def push(approx, key, n, allow=()):
        d = DELTA[key]
        for i in range(n):
            bl = nearest_boulder(approx)
            if bl is None:
                L(f"!! [push] boulder near {approx} vanished (i={i})")
                return False
            stand = (bl[0] - d[0], bl[1] - d[1])
            if not sea_walk(lambda c, s=stand: c == s, f"push-approach{i}",
                            avoid={tuple(bl)}, allow=allow):
                L(f"!! [push] can't reach {stand} to push {bl} {key}")
                return False
            moved = False
            for _try in range(4):
                if handle_interrupts():
                    continue
                b.press(key, 40, 10, camp.render, owner="agent")
                settle(70)
                b2l = nearest_boulder((bl[0] + d[0], bl[1] + d[1]))
                if b2l != bl:
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

    # ── preflight ────────────────────────────────────────────────────────────
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} badges8={fm.read_flag(b, 0x827)} "
      f"lead={lead_frac():.0%} money=${camp.money()}")
    if not fm.read_flag(b, 0x827):
        L("!! badge 8 not held — wrong canonical, abort")
        return 1
    deadline = time.time() + 3600

    # ── PHASE 0: TEACH EARTHQUAKE (TM26 -> Venusaur slot 0, over Secret Power) ─
    have = st.read_party_moves(b, 0) or []
    if MOVE_EQ in have:
        L("   EQ already known — skipping teach")
    else:
        forget_idx = 3 if 290 in have else None      # Secret Power's slot (run logs)
        teacher = ht.TeachFlow(camp, log=lambda m: print(m, flush=True))
        r = teacher.teach("surf", 0, forget_idx=forget_idx,
                          item_override=TM26_ITEM, move_override=MOVE_EQ)
        after = st.read_party_moves(b, 0) or []
        L(f"   [teach-eq] -> {r}; moves now {after} (EQ={'YES' if MOVE_EQ in after else 'NO'})")
        drain(key="B")
        settle(60)
        if MOVE_EQ in after:
            _stage_save("eq_taught")

    # ── PHASES 1-5: ONE WHITEOUT-TOLERANT DISPATCH LOOP ──────────────────────
    # victory_run3 truth: a silent field whiteout (faint boxes B-drained by
    # handle_interrupts) respawned her in the Viridian center mid-2F and the
    # linear phase chain aborted. But progress RATCHETS: Gary's var, the
    # gauntlet scenes and every VR switch var persist in the save — a whiteout
    # costs half the money and a re-cross, never solved ground. So: dispatch
    # on the CURRENT map every iteration; skip switches whose barriers are
    # already open; retreat-heal at Viridian before entering VR hurt (the
    # R22/R23-south road is battle-free once Gary is beaten).
    def run_puzzle(ops, barrier_tile, label):
        for op in ops:
            while handle_interrupts():
                pass
            L(f"-- op {op} (map {tv.map_id(b)} @ {tv.coords(b)})")
            if op[0] == "strength":
                if not ensure_strength(op[1]):
                    return False
            elif op[0] == "push":
                if not push(op[1], op[2], op[3],
                            allow=op[4] if len(op) > 4 else ()):
                    return False
        settle(150)                                   # switch scene (SE + map redraw)
        drain()
        g_now = tv.Grid(b)
        opened = g_now.col.get((barrier_tile[0] + tv.MAP_OFFSET,
                                barrier_tile[1] + tv.MAP_OFFSET), 1) == 0
        L(f"   [{label}] barrier {barrier_tile} open={opened}")
        snap(f"{label}_done")
        _stage_save(label)
        return opened

    def barrier_open(tile):
        g = tv.Grid(b)
        return g.col.get((tile[0] + tv.MAP_OFFSET, tile[1] + tv.MAP_OFFSET), 1) == 0

    def puzzle2_2f():
        # the row-19 boulder may sit ANYWHERE x14..33 after a partial chain —
        # find it, push LEFT the remaining distance onto the (14,19) switch
        if not ensure_strength((33, 19)):
            return False
        bl = None
        for ax in (33, 27, 21, 16):
            c = nearest_boulder((ax, 19), radius=6)
            if c and c[1] == 19 and 14 <= c[0] <= 33:
                bl = c
                break
        if bl is None:
            L("!! [2f-sw2] no boulder on row 19 — 3F reset detour needed: (34,9) "
              "ladder -> push (32,5) onto (7,7) -> push (33,18) into hole (34,18)")
            return False
        if bl[0] == 14:
            return True
        return push(bl, "LEFT", bl[0] - 14)

    wedges = {}

    def wedge(label, cap=4):
        wedges[label] = wedges.get(label, 0) + 1
        if wedges[label] >= cap:
            L(f"!! [{label}] failed x{cap} — abort LOUD")
            snap(f"wedge_{label[:14]}")
            return True
        return False

    retreating = False
    r23_logged = vr_logged = False
    while time.time() < deadline:
        if handle_interrupts():
            continue
        here = tuple(tv.map_id(b))
        if here == INDIGO:
            break
        if here == VIRIDIAN:
            if lead_frac() < 0.9:
                camp.heal_nearest()
                continue
            retreating = False
            if not cross_edge("west", "to-r22") and wedge("viridian-west"):
                return 1
        elif here == R22:
            # eastward = home to heal; westward crosses Gary's trigger col 33
            # (the scene + battle fire mid-path; handle_interrupts owns them;
            # a loss whiteouts and this loop recovers)
            if retreating:
                if not cross_edge("east", "r22-home") and wedge("r22-home"):
                    return 1
            elif not go_warp((8, 5), GATE, "gate-south"):
                if tuple(tv.map_id(b)) == R22 and wedge("gate-south"):
                    snap("gate_fail")
                    return 1
        elif here == GATE:
            dest = R22 if retreating else R23
            cands = [tuple(xy) for xy, d, _w in tv.read_warps(b) if tuple(d) == dest]
            if not cands:
                L(f"!! no {dest} warp inside the gate — abort")
                return 1
            # north side = lowest y; south side = highest y
            cands.sort(key=lambda t: t[1], reverse=retreating)
            if not go_warp(cands[0], dest, "gate-thru"):
                drain()
        elif here == R23:
            cy = (tv.coords(b) or (0, 0))[1]
            if cy <= 30:                              # north side (past VR)
                if not cross_edge("north", "to-indigo") and wedge("r23-north"):
                    return 1
            elif lead_frac() < 0.5 and not retreating:
                L(f"   [retreat] lead {lead_frac():.0%} at R23 south — healing at "
                  f"Viridian before the Road (battle-free road home)")
                retreating = True
            elif retreating:
                # the gate pair is (8,153) arrow-warp + (9,154) PHANTOM anchor on a
                # col-1 tile — only walkable warp tiles are real entries
                g_r23 = tv.Grid(b)
                gates = [tuple(xy) for xy, d, _w in tv.read_warps(b)
                         if tuple(d) == GATE
                         and g_r23.col.get((xy[0] + tv.MAP_OFFSET,
                                            xy[1] + tv.MAP_OFFSET), 1) == 0]
                gates.sort(key=lambda t: t[1], reverse=True)
                if not gates or not go_warp(gates[0], GATE, "r23-to-gate"):
                    if wedge("r23-to-gate"):
                        return 1
            else:
                if not r23_logged:
                    L(f"   ROUTE 23 @ {tv.coords(b)} after {n_battles[0]} battles "
                      f"(Gary handled en route)")
                    _stage_save("r23_south")
                    r23_logged = True
                if not go_warp((5, 28), VR1F, "vr-door") and wedge("vr-door"):
                    return 1
        elif here == VR1F:
            if not vr_logged:
                L(f"   VICTORY ROAD 1F @ {tv.coords(b)} [lead {lead_frac():.0%}]")
                vr_logged = True
            if not barrier_open((12, 14)):
                if not run_puzzle(VR1F_PUZZLE, (12, 14), "1f-switch") \
                        and wedge("1f-switch", 3):
                    return 1
            elif not go_warp((3, 2), VR2F, "1f-ladder") and wedge("1f-ladder"):
                return 1
        elif here == VR2F:
            cx, cy = tuple(tv.coords(b) or (0, 0))
            if cx >= 36 and cy <= 13:                 # east pocket (from 3F drop)
                if (go_warp((48, 12), R23, "vr-exit")
                        or go_warp((47, 13), R23, "vr-exit-b")
                        or go_warp((49, 13), R23, "vr-exit-c")):
                    L(f"   VICTORY ROAD CLEARED -> R23 north @ {tv.coords(b)} "
                      f"[lead {lead_frac():.0%}, battles {n_battles[0]}]")
                    _stage_save("vr_cleared")
                elif wedge("vr-exit"):
                    return 1
            elif not barrier_open((13, 10)):
                if not run_puzzle(VR2F_PUZZLE1, (13, 10), "2f-switch1") \
                        and wedge("2f-switch1", 3):
                    return 1
            elif not barrier_open((33, 16)):
                if puzzle2_2f():
                    settle(150)
                    drain()
                    _stage_save("2f-switch2")
                    L(f"   [2f-switch2] barrier (33,16) open={barrier_open((33, 16))}")
                elif wedge("2f-switch2", 3):
                    return 1
            elif not go_warp((36, 17), VR3F, "2f-to-3f") and wedge("2f-to-3f"):
                return 1
        elif here == VR3F:
            if not go_warp((37, 10), VR2F, "3f-to-2f-east") and wedge("3f-to-2f"):
                return 1
        else:
            # off-route (whiteout center interior, etc.) — exit to the overworld
            L(f"   off-route at {here} — exiting to the overworld")
            camp.enter_warp(prefer="south")
            settle(80)
    if tuple(tv.map_id(b)) != INDIGO:
        L("!! never reached Indigo Plateau")
        return 1

    L(f"   INDIGO PLATEAU @ {tv.coords(b)} — healing at the League center")
    r = camp.heal_nearest()
    L(f"   heal_nearest -> {r}")
    drain(key="B")

    L(f"   INDIGO BANKED: pos {tv.map_id(b)}@{tv.coords(b)} | lead {lead_frac():.0%} | "
      f"battles {n_battles[0]} | money ${camp.money()} | "
      f"EQ={'YES' if MOVE_EQ in (st.read_party_moves(b, 0) or []) else 'NO'}")
    snap("80_final")
    _stage_save("indigo_reach")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} indigo_reach")
    return 0


if __name__ == "__main__":
    sys.exit(main())
