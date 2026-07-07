"""recon_giovanni.py — THE VIRIDIAN GYM STRIKE (badge 8): the sea road home + Giovanni.

Ground truth (pret maps/scripts, cached G:\\temp\\longrun\\pret\\ViridianGym*.json/.inc):
- THE ROAD: Cinnabar -> Viridian is FIVE consecutive NORTH edge crossings
  (Cinnabar -> R21South -> R21North -> Pallet -> Route1 -> Viridian), sea legs surfed
  (Lapras has SURF; recon_seafoam's mount/sea_walk/cross_edge machinery verbatim).
- THE DOOR: ViridianCity OnTransition unlocks the gym when badges 2-7 are held (she
  has 1-7) — the locked-door coord event (36,11) dies on her first city transition.
  Gym warp (36,10) -> gym map (5,1).
- THE GYM: no doors/quizzes — a SPIN-TILE floor maze (the Rocket-Hideout class;
  spin_nav.SpinNav = the proven glide crosser, wired here for its second customer).
  8 juniors WITH sight 2-3 (spotting battles are expected and fine — Razor Leaf is
  x2 into Giovanni's ground/rock rosters, and the SE-chunk sleep-lock covers Nido
  poison). Giovanni (2,2) face DOWN -> front (2,3), face UP.
- Post-win: FLAG_DEFEATED_LEADER_GIOVANNI + BADGE 8 = flag 0x827 + TM26 gift
  (A-drain, no Y/N) + Giovanni removeobject fade. No exit ambush (the badge-8 Gary
  fight arms on ROUTE 22, westbound — the NEXT objective's problem).
Success = flag 0x827. Bank -> %TEMP%/longrun/banked_GIOVANNI.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_giovanni.py   (WATCH=1 = window)
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
from spin_nav import SpinNav         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_giovanni")
BANK = os.path.join(SCRATCH, "banked_GIOVANNI")
DBG = os.path.join(SCRATCH, "giovanni_probe")

CINNABAR = (3, 8)
VIRIDIAN = (3, 1)
GYM = (5, 1)
GIOVANNI_FRONT = (2, 3)              # Giovanni (2,2), face UP
FLAG_BADGE_EARTH = 0x827
FLAG_DEFEATED_GIOVANNI = 0x4B7       # FLAG_DEFEATED_LEADER_GIOVANNI (unused if wrong; badge is truth)
GYM_EXIT_BAND = {(16, 21), (17, 21), (18, 21)}   # above the entrance mats (16-18,22)
KEY_OF = {(0, 1): "DOWN", (0, -1): "UP", (1, 0): "RIGHT", (-1, 0): "LEFT"}


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
        pygame.display.set_caption("Kira — VIRIDIAN GYM STRIKE (live watch)")

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
                           log=lambda m: print(m, flush=True)).run(max_seconds=300)

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

    def badge():
        return fm.read_flag(b, FLAG_BADGE_EARTH)

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

    # ── water/edge machinery (recon_seafoam verbatim) ────────────────────────
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
            npcs = live_npc_tiles() | {tuple(o[0]) for o in
                                       tv.read_object_templates(b) if o[2]}
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

    def settle(n=90):
        for _ in range(n):
            b.run_frame()

    def enter_to(dest, label):
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == dest]
        if not cands:
            L(f"!! [{label}] no warp on {m0} leads to {dest}")
            return False
        cur = tuple(tv.coords(b) or (0, 0))
        cands.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        for wt in cands:
            r = camp.enter_warp(pick=wt)
            if r == "need_heal":
                L(f"   [{label}] heal interrupt — healing, then retrying")
                camp.heal_nearest()
                r = camp.enter_warp(pick=wt)
            if st.in_battle(b):
                L(f"   [{label}] battle on approach -> {camp.battle_runner()}")
                drain()
                r = camp.enter_warp(pick=wt)
            if tuple(tv.map_id(b)) == dest:
                for _ in range(80):
                    b.run_frame()
                L(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)}")
                return True
        L(f"!! [{label}] no candidate warp fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    nav = SpinNav(b, camp, fight, drain, log=L)

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} badge8={badge()} "
      f"lead={lead_frac():.0%} badge7={fm.read_flag(b, 0x826)}")
    if badge():
        L("Earth Badge already held — nothing to strike")
        return 0

    deadline = time.time() + 2400
    wedges = {}

    # ── PHASE A: THE SEA ROAD HOME (five north crossings) ────────────────────
    legs = 0
    while time.time() < deadline and tuple(tv.map_id(b)) != VIRIDIAN:
        if handle_interrupts():
            continue
        here = tuple(tv.map_id(b))
        if here == GYM:
            break                                    # already there somehow
        if not cross_edge("north", f"leg{legs}"):
            wedges["road"] = wedges.get("road", 0) + 1
            if wedges["road"] >= 3:
                snap("10_road_wedge")
                L(f"!! northbound crossing wedged x3 at {here}@{tv.coords(b)} — abort")
                return 1
            settle(120)
            continue
        wedges.pop("road", None)
        legs += 1
        settle(180)
        _stage_save(f"leg{legs}")
    if tuple(tv.map_id(b)) == VIRIDIAN:
        L(f"   VIRIDIAN reached after {legs} legs, {n_battles[0]} battles — healing first")
        camp.heal_nearest()
        _stage_save("viridian")

    # ── PHASE B: THE GYM (spin maze -> Giovanni) ─────────────────────────────
    while time.time() < deadline and not badge():
        if handle_interrupts():
            continue
        here = tuple(tv.map_id(b))
        if here == VIRIDIAN:
            if lead_frac() < 0.6:
                camp.heal_nearest()
                continue
            if not enter_to(GYM, "gym-door"):
                wedges["door"] = wedges.get("door", 0) + 1
                if wedges["door"] >= 3:
                    snap("20_no_gym_door")
                    L("!! can't enter the gym x3 — abort LOUD")
                    return 1
                drain()
            continue
        if here != GYM:
            L(f"   off-route at {here} (whiteout?) — exiting to the overworld")
            camp.enter_warp(prefer="south")
            settle(80)
            if tuple(tv.map_id(b)) == here:
                L(f"!! stuck off-route at {here}@{tv.coords(b)} — abort")
                snap("21_offroute")
                return 1
            continue

        if lead_frac() < 0.5:
            L(f"   lead at {lead_frac():.0%} — leaving to heal (beaten trainers stay beaten)")
            if nav.cross(lambda c: c in GYM_EXIT_BAND, "heal-exit"):
                enter_to(VIRIDIAN, "exit-to-heal")
            continue

        L(f"   gym floor — spin-crossing to Giovanni's front {GIOVANNI_FRONT} "
          f"[lead {lead_frac():.0%}]")
        if not nav.cross(lambda c: c == GIOVANNI_FRONT, "to-giovanni"):
            wedges["cross"] = wedges.get("cross", 0) + 1
            if wedges["cross"] >= 4:
                snap("30_cross_wedge")
                L(f"!! spin crosser never reached Giovanni x4 (at {tv.coords(b)}) — abort")
                return 1
            # THE BEATEN-BODY SEAL (run1, (10,5)): a sight-walking trainer STOPS adjacent
            # to engage and his body STAYS there after losing — in a 1-wide corridor that
            # seals the walk route. The game's own reset: objects respawn at TEMPLATE
            # positions on map reload (beaten trainers stay beaten) — exit + re-enter,
            # then replan on a clean floor.
            L("   cross blocked — beaten-body seal suspected: exit + re-enter to reset "
              "object positions to templates")
            snap(f"31_seal_{wedges['cross']}")
            if nav.cross(lambda c: c in GYM_EXIT_BAND, "seal-reset-out"):
                enter_to(VIRIDIAN, "seal-reset")
            else:
                drain()
            continue

        L(f"   Giovanni's front reached — engaging [lead {lead_frac():.0%}]")
        nb0 = n_battles[0]
        r = "nothing"
        for _ in range(8):
            b.press("UP", 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if fight_open():
                L(f"   [giovanni] battle -> {fight()}")
                drain(key="A")
                r = "battled"
                break
            if dd_box(b):
                drain(key="A")
                if fight_open():
                    L(f"   [giovanni] battle -> {fight()}")
                    drain(key="A")
                    r = "battled"
                    break
                r = "talked"
        for _ in range(300):                        # badge fanfare + TM26 + his exit fade
            b.run_frame()
            if dd_box(b):
                b.press("A", 8, 12, camp.render, owner="agent")
        drain(key="A")
        L(f"   GIOVANNI -> {r} (battles {nb0}->{n_battles[0]}) badge8={badge()} "
          f"lead={lead_frac():.0%}")
        if badge():
            _stage_save("badge8")
            snap("40_badge8")
            break
        if tuple(tv.map_id(b)) != GYM:
            continue
        wedges["gio"] = wedges.get("gio", 0) + 1
        if wedges["gio"] >= 3:
            snap("50_gio_wedge")
            L("!! Giovanni engaged x3 without a badge — abort LOUD")
            return 1

    if not badge():
        L(f"!! Earth Badge NOT won (at {tv.map_id(b)}@{tv.coords(b)}) — NOT banking")
        snap("70_fail")
        return 1

    # ── walk out, heal, bank ─────────────────────────────────────────────────
    L("   badge in hand — walking out")
    out_deadline = time.time() + 300
    while tuple(tv.map_id(b)) != VIRIDIAN and time.time() < out_deadline:
        if handle_interrupts():
            continue
        here = tuple(tv.map_id(b))
        if here == GYM:
            if nav.cross(lambda c: c in GYM_EXIT_BAND, "walk-out"):
                enter_to(VIRIDIAN, "out-door")
            else:
                drain()
        else:
            camp.enter_warp(prefer="south")
            settle(80)
    if tuple(tv.map_id(b)) == VIRIDIAN:
        camp.heal_nearest()
    else:
        L(f"!! walk-out incomplete (at {tv.map_id(b)}) — banking anyway (the badge holds)")

    L(f"   EARTH BADGE: flag={badge()} | pos {tv.map_id(b)}@{tv.coords(b)} | "
      f"lead {lead_frac():.0%} | battles {n_battles[0]}")
    snap("80_final")
    _stage_save("giovanni_badge8")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} giovanni_badge8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
