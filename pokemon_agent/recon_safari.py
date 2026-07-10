"""recon_safari.py — THE SAFARI ZONE STRIKE: HM03 Surf + Gold Teeth -> Warden -> HM04 Strength.

The badge-7 unlock (Blaine is on Cinnabar; the sea road needs Surf; Victory Road later needs
Strength — BOTH live behind the Safari Zone). Disasm ground truth (pret, fetched 2026-07-07):
  Fuchsia (3,7): Safari entrance door (24,5); Warden's House door (33,31).
  Entrance interior: city doors (3-5,7); ENTRY TRIGGERS at (3-5,3) — walking north fires the
  $500 join prompt (YES default, A pays, 30 balls); warp (4,1) -> SAFARI CENTER.
  Center: arrival (26,30); exit warps (25-27,30) -> entrance; EAST doors (43,15-17).
  ⚠ THE POND TRUTH (strikes 7-11, probe + pret map.bin): the Center's pond splits it into
  TWO components — the WEST doors (8,17-19) + NORTH doors (25-27,5) sit on the far SHELF,
  UNREACHABLE on foot from the entrance pocket. The tour chain is the classic Safari loop:
  Center -EAST(43,15-17)-> Area 1 East [arrive (8,26-28); its NW doors (8,9-11)] ->
  Area 2 North [arrive (48,31-33); its S doors (10-12,34)/(20-22,34)] ->
  Area 3 West [arrive (30-32,5)/(37-39,5) top row]. Return = REVERSE the chain (West's
  (40,26-28) doors land on the Center SHELF — wrong side for the exit mats).
  West (Area 3): GOLD TEETH item ball (28,14) (FLAG_HIDE_SAFARI_ZONE_WEST_GOLD_TEETH);
  SECRET HOUSE door (12,7); house: HM03 attendant (6,5) (engage from (6,6) face UP);
  house exit (3-5,9) -> West.
  Warden's house: give Gold Teeth (auto on talk) -> HM04 STRENGTH.
  HM item ids: HM03 Surf = 341, HM04 Strength = 342 (TM-case pocket); Gold Teeth = key item.
SAFARI RULES: 600-step limit (running out = scripted warp back to the entrance — the strike
treats an unexpected entrance/city map as RE-ENTER-AND-RESUME, objectives are flag-idempotent);
wild encounters use the BALL/BAIT/ROCK/RUN menu (battle_agent's FIGHT menu doesn't exist) —
the handler throws balls at NEW species (dex doctrine: cheap first-of-a-kind attempts) then
flees via cursor combos; ALL battle-end drains use B (the catch flow ends in the nickname
keyboard — the AAAAAAAAAA class).
Success = items 341 + 342 in the TM pocket. Bank -> %TEMP%/longrun/banked_SAFARI.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_safari.py
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
import hm_teach as ht                # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_safari")
BANK = os.path.join(SCRATCH, "banked_SAFARI")
DBG = os.path.join(SCRATCH, "safari_probe")

FUCHSIA = (3, 7)
ENTRANCE_DOOR = (24, 5)              # on Fuchsia
WARDEN_DOOR = (33, 31)               # on Fuchsia
GOLD_TEETH_BALL = (28, 14)           # West
SECRET_DOOR_WEST = (12, 7)           # West -> Secret House
HM03, HM04 = 341, 342


def _resolve_state(name):
    """Resolve a state BASENAME/path -> (state_path, sidecar_dir, sidecar_prefix). Kit fixtures live
    in states/workshop as <name>.state + <name>.<sidecar>.json; the credits canonical is
    states/campaign/kira_campaign.state with bare-named sidecars. Also accepts a bank dir's bundle."""
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

    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("SAFARI_STATE", ""))
    L(f"boot state = {state_path}")
    b = Bridge(ROM)
    with open(state_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    n_battles = [0]

    def trainer_fight():
        n_battles[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=240)

    camp = Campaign(b, battle_runner=trainer_fight,
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

    def tm_pocket():
        return ht.pocket_items(b, ht.TM_CASE_OFF, 64)

    def key_items():
        return ht.pocket_items(b, ht.KEY_ITEMS_OFF, 30)

    def have_surf():
        return HM03 in tm_pocket()

    def have_strength():
        return HM04 in tm_pocket()

    def dest_of(xy):
        for wxy, d, _w in tv.read_warps(b):
            if tuple(wxy) == xy:
                return tuple(d)
        return None

    def fight_open():
        """Battle-open gate that SEES safari encounters. st.in_battle sanity-checks
        gBattleMons[0] — the PLAYER'S battle mon — which a safari battle never fields
        (strike3 truth: she froze mid-grass in an encounter the gate read as False).
        The battle-resources pointer alone is the truth: valid EWRAM only in battle."""
        return ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))

    # ── SAFARI BATTLE HANDLER (BALL/BAIT/ROCK/RUN menu — not the FIGHT menu) ─────────────
    thrown_species = set()             # ball economy (strike5 truth): 2 throws at EVERY
    #                                    re-encounter of an uncatchable species drains 30
    #                                    balls in ~3 min and ends the game mid-crossing —
    #                                    the dex doctrine is ONE cheap attempt per species.

    def safari_battle():
        n_battles[0] += 1
        sp = st.read_enemy_species(b, 0)
        new = (sp and ram.pokedex_owns(b, sp) is False
               and sp not in thrown_species)
        if new:
            thrown_species.add(sp)
        nm = st.SPECIES_NAME.get(sp, f"#{sp}")
        L(f"   [safari] wild {nm} (new={new}) @ {tv.coords(b)}")

        def settle(n):
            for _ in range(n):
                b.run_frame()
                if not fight_open():
                    return False
            return True

        def bdrain():
            for _ in range(30):
                if fight_open() or not dd_box(b):
                    return
                b.press("B", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()

        deadline = time.time() + 90
        threw = 0
        boxed = 0
        while fight_open() and time.time() < deadline:
            # drain intro text on B — but the safari ACTION MENU itself can read as an
            # open box, so cap the drains and fall through to the press logic
            if dd_box(b) and boxed < 5:
                boxed += 1
                b.press("B", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
                continue
            boxed = 0
            if new and threw < 2:
                b.press("A", 8, 12, camp.render, owner="agent")   # cursor home = BALL
                threw += 1
                if not settle(500):
                    break
                continue
            # flee: RUN lives bottom-right of the 2x2 — try cursor combos until the
            # battle ends (a wrong landing throws bait/rock: harmless, costs a turn)
            for combo in (("DOWN", "RIGHT"), ("RIGHT", "DOWN"), ("DOWN",), ("RIGHT",), ()):
                for k in combo:
                    b.press(k, 8, 10, camp.render, owner="agent")
                    for _ in range(12):
                        b.run_frame()
                b.press("A", 8, 12, camp.render, owner="agent")
                if not settle(320):
                    break
                if dd_box(b):
                    bdrain()
                if not fight_open():
                    break
        # end-of-battle drains are ALWAYS B (catch flow -> nickname keyboard trap)
        for _ in range(40):
            if not dd_box(b):
                break
            b.press("B", 8, 12, camp.render, owner="agent")
            for _ in range(20):
                b.run_frame()
        L(f"   [safari] battle over (in_battle={fight_open()}, threw {threw})")
        return "safari"

    def drain(max_a=40, key="B"):
        stable = 0
        for _ in range(max_a):
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

    def on_battle():
        # inside safari maps every battle is a safari encounter; elsewhere use the agent
        if tuple(tv.map_id(b)) in safari_maps:
            return safari_battle()
        return trainer_fight()

    KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}

    def elev_of(sx, sy):
        """Map-grid ELEVATION nibble (bits 12-15) — the Sabrina/Silph class: void strips
        read collision-0/elev-0 beside elev-3 floor and the game blocks cross-elevation
        steps (East's y<=8 phantom-open strip, strike15 truth). 0xF = multi-level pass."""
        w = b.rd32(tv.BACKUP_LAYOUT)
        h = b.rd32(tv.BACKUP_LAYOUT + 4)
        mp = b.rd32(tv.BACKUP_LAYOUT + 8)
        bx, by = sx + tv.MAP_OFFSET, sy + tv.MAP_OFFSET
        if not (0 <= bx < w and 0 <= by < h):
            return -1
        return (b.rd16(mp + (by * w + bx) * 2) >> 12) & 0xF

    def safari_step(t):
        """One-tile step, tap-turn aware, NO SIDEWAYS NUDGE. camp._step_to's short verify
        window reads grass steps as failures and its nudges BURN SAFARI STEPS + bounce her
        onto adjacent door warps (strike 12/13 truth: the tour died to the 600-step limit
        on nudge waste). Returns 'ok' | 'battle' | 'blocked' | 'warped'."""
        m0 = tuple(tv.map_id(b))
        cur = tuple(tv.coords(b) or (0, 0))
        d = (t[0] - cur[0], t[1] - cur[1])
        if d in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            d = (d[0] // 2, d[1] // 2)         # LEDGE HOP: one press, the game jumps 2
        key = KEY_OF.get(d)
        if key is None:
            return "blocked"
        for _attempt in range(3):              # tap 1 may only TURN her (tap-turn law)
            b.press(key, 8, 6, camp.render, owner="agent")
            for _ in range(60):                # grass rustle / ledge-hop anim settle
                b.run_frame()
                if tuple(tv.coords(b) or ()) == t:
                    break
            if fight_open():
                return "battle"
            if tuple(tv.map_id(b)) != m0:
                return "warped"
            if tuple(tv.coords(b) or ()) == t:
                return "ok"
        if dd_box(b):
            return "battle"                    # a box opened — let the caller drain
        return "blocked"

    def safari_bfs(g, start, goal_test, tile_ok):
        """tv.bfs + THE PER-EDGE ELEVATION LAW (strike18 truth: she wandered onto the
        West teeth-plateau (e4) and every e4->e3 edge off it is game-refused; exits are
        the e0 stair tiles). A step is legal iff elevations match or either side is
        0/0xF; ledge hops are exempt (jumping down changes elevation legally)."""
        from collections import deque
        came = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if goal_test(cur):
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = came[cur]
                return path[::-1]
            cx, cy = cur
            ec = elev_of(cx, cy)
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if g.ledge_dir(cx + dx, cy + dy) == (dx, dy):
                    nx, ny = cx + 2 * dx, cy + 2 * dy
                else:
                    nx, ny = cx + dx, cy + dy
                    if not g.edge_open(cx, cy, dx, dy):
                        continue
                    en = elev_of(nx, ny)
                    if ec not in (0, 0xF) and en not in (0, 0xF) and en != ec:
                        continue
                if not (g.sx_lo <= nx <= g.sx_hi and g.sy_lo <= ny <= g.sy_hi):
                    continue
                if (nx, ny) in came or not tile_ok(nx, ny):
                    continue
                came[(nx, ny)] = cur
                q.append((nx, ny))
        return None

    def walk_path_to(tile, label, tries=8):
        dead = set()
        budget = tries
        hops = 0
        while budget > 0:
            hops += 1
            if hops > 400:            # hard cap — the deadline is the real guard
                L(f"   [{label}] hop cap hit at {tv.coords(b)}")
                return False
            budget -= 1
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile:
                return True
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
            npcs = ({tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
                    | dead) - {tile}
            # LEDGE TILES ARE NEVER STANDING TILES (strike14 truth): tv.bfs special-cases
            # a ledge only when approached in jump direction — any other approach walks
            # THROUGH it as floor, which the game refuses. Exclude them from standing;
            # bfs's own hop logic still jumps them in the legal direction.
            p = safari_bfs(g, cur, lambda t: t == tile,
                           lambda sx, sy: g.walkable(sx, sy)
                           and g.ledge_dir(sx, sy) is None
                           and (sx, sy) not in wts and (sx, sy) not in npcs)
            L(f"   [{label}] replan at {cur} -> {tile} (len {len(p) if p else 0}, "
              f"budget {budget}, head {[tuple(x) for x in (p or [])[1:4]]})")
            if not p:
                L(f"   [{label}] no NPC-free static path {cur} -> {tile} "
                  f"(dead={sorted(dead)})")
                return False
            for t in p[1:]:
                r = safari_step(tuple(t))
                if r == "battle":
                    on_battle()
                    drain()
                    budget += 1          # a battle is not a failed try (grass roads have
                    break                # an encounter every few steps — strike4 truth)
                if r == "warped":
                    return False         # step limit / door — the main loop re-dispatches
                if r == "blocked":
                    dead.add(tuple(t))   # genuinely refused (pond shore / fence)
                    break
            if tuple(tv.coords(b) or ()) == tile:
                return True
            if tuple(tv.map_id(b)) != start_map[0]:
                return False                    # warped mid-walk (step limit) — re-dispatch
            if tuple(tv.coords(b) or ()) != cur:
                budget += 1        # she MOVED — progress is never a consumed try (grass
                #                    steps read as failed in _step_to's verify window and
                #                    were eating one try per TILE — strike12 truth)
        return tuple(tv.coords(b) or ()) == tile

    def engage(front, face, label, drains=3, key="B"):
        if not walk_path_to(front, label):
            L(f"!! [{label}] couldn't reach {front} (at {tv.coords(b)})")
            return "nothing"
        out = "nothing"
        for _ in range(8):
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if fight_open():
                on_battle()
                drain()
                return "battled"
            if dd_box(b):
                out = "talked"
                for _k in range(drains):
                    drain(key=key)
                    for _ in range(40):
                        b.run_frame()
                break
        return out

    def step_warp(wt, label, tries=3):
        """Deliberately step onto a warp tile (the ride_pad pattern): approach a free
        neighbor via walk_path_to (grass INCLUDED — strike2 truth: the Safari's west half
        is only reachable through tall grass, so travel's grass-free planner reads
        no_route and its fallback wanders onto the EXIT mats), then one LONG-HOLD press
        onto the warp. Success = map changed."""
        m0 = tuple(tv.map_id(b))
        g = tv.Grid(b)
        wts = {tuple(w[0]) for w in tv.read_warps(b)}
        npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
        cur0 = tuple(tv.coords(b) or (0, 0))
        e0 = elev_of(*cur0)
        cands = []
        for nb, kin in (((wt[0] - 1, wt[1]), "RIGHT"), ((wt[0] + 1, wt[1]), "LEFT"),
                        ((wt[0], wt[1] - 1), "DOWN"), ((wt[0], wt[1] + 1), "UP")):
            if nb in wts or not g.walkable(nb[0], nb[1]):
                continue
            p = safari_bfs(g, cur0, lambda t, a=nb: t == a,
                           lambda sx, sy: g.walkable(sx, sy)
                           and g.ledge_dir(sx, sy) is None
                           and (sx, sy) not in wts and (sx, sy) not in npcs) \
                if cur0 != nb else [cur0]
            if p:
                cands.append((len(p), nb, kin))
        for _len, nb, kin in sorted(cands):
            if tuple(tv.coords(b) or ()) != nb and not walk_path_to(nb, f"{label}-approach",
                                                                    tries=14):
                if tuple(tv.map_id(b)) != m0:
                    return True                    # a battle/limit warp moved us anyway
                continue
            for _try in range(tries):
                b.press(kin, 26, 10, camp.render, owner="agent")
                for _ in range(150):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if fight_open():
                    on_battle()
                    drain()
                if tuple(tv.map_id(b)) != m0:
                    for _ in range(60):
                        b.run_frame()
                    L(f"   [{label}] stepped warp {wt}: {m0} -> {tuple(tv.map_id(b))} "
                      f"@ {tv.coords(b)}")
                    return True
            break
        if not cands:
            L(f"   [{label}] ZERO approach candidates for {wt}: grid {g.w}x{g.h}, "
              f"cur {cur0} walkable={g.walkable(*cur0)}; "
              + "; ".join(f"nb {nb}: walk={g.walkable(*nb)} warp={nb in wts}"
                          for nb in ((wt[0] - 1, wt[1]), (wt[0] + 1, wt[1]),
                                     (wt[0], wt[1] - 1), (wt[0], wt[1] + 1))))
        L(f"!! [{label}] warp step {wt} failed (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def enter_to(dest, label):
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == dest]
        if not cands:
            L(f"!! [{label}] no warp on {m0} leads to {dest}")
            return False
        cur = tuple(tv.coords(b) or (0, 0))
        if cur in cands:
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            g6 = tv.Grid(b)
            for nb in ((cur[0], cur[1] + 1), (cur[0] + 1, cur[1]),
                       (cur[0] - 1, cur[1]), (cur[0], cur[1] - 1)):
                if nb not in wts and g6.walkable(nb[0], nb[1]):
                    camp._step_to(nb)
                    cur = tuple(tv.coords(b) or (0, 0))
                    break
        cands.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        for wt in cands:
            r = camp.enter_warp(pick=wt)
            if fight_open():
                on_battle()
                drain()
                r = camp.enter_warp(pick=wt)
            if tuple(tv.map_id(b)) == dest:
                for _ in range(80):
                    b.run_frame()
                L(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)}")
                return True
        L(f"!! [{label}] no candidate warp fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    # ── resolve the safari map ids live (warp-table truth, no hardcoded group/nums) ──────
    start_map = [tuple(tv.map_id(b))]
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} surf={have_surf()} "
      f"strength={have_strength()} money=${camp.money()}")
    if tuple(tv.map_id(b)) != FUCHSIA:
        L(f"!! not in Fuchsia (at {tv.map_id(b)}) — abort")
        return 1
    ENTRANCE = dest_of(ENTRANCE_DOOR)
    WARDEN_HOUSE = dest_of(WARDEN_DOOR)
    if not ENTRANCE or not WARDEN_HOUSE:
        L(f"!! entrance/warden door not on the city warp table — abort "
          f"(entrance={ENTRANCE}, warden={WARDEN_HOUSE})")
        return 1
    L(f"   resolved: entrance={ENTRANCE} warden_house={WARDEN_HOUSE}")
    CENTER = EAST = NORTH = WEST = SECRET = None
    safari_maps = set()

    wedges = {}
    deadline = time.time() + 1500

    def wedge(k, cap, msg):
        wedges[k] = wedges.get(k, 0) + 1
        if wedges[k] >= cap:
            snap(f"wedge_{k}")
            L(f"!! {msg} x{cap} — abort LOUD")
            return True
        drain()
        for _ in range(150):     # PACE the retries: strike10 burned 12 attempts in 0.3s,
            b.run_frame()        # all inside one warp-transition window
        return False

    last_map = [None]
    while time.time() < deadline:
        if have_surf() and have_strength():
            break
        here = tuple(tv.map_id(b))
        start_map[0] = here
        if here != last_map[0]:
            last_map[0] = here
            _stage_save(f"arrive_{here[0]}_{here[1]}")   # probe-ability on every map
        if fight_open():
            on_battle()
            drain()
            continue
        if dd_box(b):
            drain()
            continue

        if here == FUCHSIA:
            if have_surf() and not have_strength():
                # ── the Warden leg ──
                if not enter_to(WARDEN_HOUSE, "warden-door"):
                    if wedge("warden_door", 3, "can't enter the Warden's house"):
                        return 1
                continue
            if not enter_to(ENTRANCE, "safari-entrance"):
                if wedge("entrance", 3, "can't enter the Safari building"):
                    return 1
            continue

        if here == ENTRANCE:
            if have_surf():
                # done inside — leave to the city (doors at the south row)
                if not enter_to(FUCHSIA, "exit-to-city"):
                    if wedge("exit_city", 3, "can't leave the entrance building"):
                        return 1
                continue
            # ── pay + go in: the ENTRY TRIGGER sits ON (3-5,3) — stepping onto it fires
            # the join prompt (strike1 truth: it froze enter_to's walk and the B-drain
            # DECLINED the join). Step onto (4,3) deliberately, A-drain (YES pays $500 +
            # 30 balls), then the (4,1) warp -> Center. ──
            if CENTER is None:
                CENTER = dest_of((4, 1))
                if CENTER:
                    safari_maps.add(CENTER)
            walk_path_to((4, 4), "to-trigger")
            camp._step_to((4, 3))
            camp._step_to((4, 3))              # tap-turn: same key again finishes the step
            for _ in range(90):                # let the trigger script open its box
                b.run_frame()
            drain(key="A")                     # the join prompt: A = YES (default)
            for _ in range(60):
                b.run_frame()
            drain(key="A")
            # the pay script itself can WARP her in (strike4/5 truth) — check before stepping
            if tuple(tv.map_id(b)) != CENTER:
                step_warp((4, 1), "into-center")
            if tuple(tv.map_id(b)) == CENTER:
                for _ in range(180):     # strike9/10 truth: SB1 map ids flip FIRST; the
                    b.run_frame()        # header/backup layout rebuild lags — planning
                #                          instantly reads the PREVIOUS city's grid
                wedges.pop("into_center", None)
                _stage_save("safari_in")
            elif wedge("into_center", 6, "can't reach the Safari Center"):
                return 1
            continue

        if CENTER and here == CENTER:
            if EAST is None:
                EAST = dest_of((43, 15))
                if EAST:
                    safari_maps.add(EAST)
            if have_surf():
                # leave: the exit warps (25-27,30) -> entrance (a leave prompt may fire)
                if not any(step_warp(w, "center-exit") for w in ((26, 30), (25, 30), (27, 30))):
                    if wedge("center_exit", 4, "can't exit the Safari Center"):
                        return 1
                continue
            # POND TRUTH: west doors unreachable on foot — the tour goes EAST first
            if not any(step_warp(w, "to-east") for w in ((43, 16), (43, 15), (43, 17))):
                if wedge("to_east", 4, "can't reach Area 1 (East)"):
                    return 1
            continue

        if EAST and here == EAST:
            if NORTH is None:
                NORTH = dest_of((8, 9))
                if NORTH:
                    safari_maps.add(NORTH)
            if have_surf():
                if not any(step_warp(w, "east-back") for w in ((8, 27), (8, 26), (8, 28))):
                    if wedge("east_back", 4, "can't get back to the Center"):
                        return 1
                continue
            if not any(step_warp(w, "to-north") for w in ((8, 10), (8, 9), (8, 11))):
                if wedge("to_north", 4, "can't reach Area 2 (North)"):
                    return 1
            continue

        if NORTH and here == NORTH:
            if WEST is None:
                WEST = dest_of((10, 34))
                if WEST:
                    safari_maps.add(WEST)
            if have_surf():
                if not any(step_warp(w, "north-back") for w in ((48, 32), (48, 31), (48, 33))):
                    if wedge("north_back", 4, "can't get back to Area 1"):
                        return 1
                continue
            # WEST IS SPLIT TOO (strike19 dual-flood): the (10-12,34) doors land at
            # (30-32,5) INSIDE the teeth/Secret-House component; the (20-22,34) doors
            # land at (37-39,5) in the arrival/east component with NO legal crossing.
            if not any(step_warp(w, "to-west") for w in ((11, 34), (10, 34), (12, 34),
                                                         (21, 34), (20, 34), (22, 34))):
                if wedge("to_west", 4, "can't reach Area 3 (West)"):
                    return 1
            continue

        if WEST and here == WEST:
            if SECRET is None:
                SECRET = dest_of(SECRET_DOOR_WEST)
                if SECRET:
                    safari_maps.add(SECRET)
            teeth_ball = next((o for o in tv.read_object_templates(b)
                               if tuple(o[0]) == GOLD_TEETH_BALL), None)
            if teeth_ball is not None and teeth_ball[2]:
                # ── Gold Teeth first (nearer the arrival doors) ──
                got0 = set(key_items())
                for face, front in (("UP", (GOLD_TEETH_BALL[0], GOLD_TEETH_BALL[1] + 1)),
                                    ("DOWN", (GOLD_TEETH_BALL[0], GOLD_TEETH_BALL[1] - 1)),
                                    ("RIGHT", (GOLD_TEETH_BALL[0] - 1, GOLD_TEETH_BALL[1])),
                                    ("LEFT", (GOLD_TEETH_BALL[0] + 1, GOLD_TEETH_BALL[1]))):
                    engage(front, face, "gold-teeth", drains=2)
                    tb = next((o for o in tv.read_object_templates(b)
                               if tuple(o[0]) == GOLD_TEETH_BALL), None)
                    if tb is None or not tb[2]:
                        break
                    if tuple(tv.map_id(b)) != WEST:
                        break
                tb = next((o for o in tv.read_object_templates(b)
                           if tuple(o[0]) == GOLD_TEETH_BALL), None)
                if tb is not None and tb[2]:
                    if wedge("teeth", 4, "Gold Teeth ball not collected"):
                        return 1
                else:
                    L(f"   GOLD TEETH banked (key items {sorted(set(key_items()) - got0)} "
                      f"added)")
                    _stage_save("gold_teeth")
                continue
            if not have_surf():
                if not step_warp(SECRET_DOOR_WEST, "secret-house"):
                    if wedge("secret", 4, "can't reach the Secret House"):
                        return 1
                continue
            # have surf — reverse the chain via the NORTH top doors (the (40,26-28)
            # doors land on the Center's pond-locked SHELF — no path to the exit mats)
            if not any(step_warp(w, "west-back") for w in ((31, 5), (30, 5), (32, 5),
                                                           (38, 5), (37, 5), (39, 5))):
                if wedge("west_back", 4, "can't get back to Area 2"):
                    return 1
            continue

        if SECRET and here == SECRET:
            if not have_surf():
                r = engage((6, 6), "UP", "hm03-man", drains=8)
                drain(key="B")
                L(f"   HM03 attendant -> {r} (surf in bag: {have_surf()})")
                if have_surf():
                    _stage_save("hm03")
                    snap("30_hm03")
                elif wedge("hm03", 3, "HM03 not granted"):
                    return 1
                continue
            if not step_warp((4, 9), "house-exit"):
                if wedge("house_exit", 3, "can't leave the Secret House"):
                    return 1
            continue

        if here == WARDEN_HOUSE:
            if not have_strength():
                # the Warden sits mid-room; find the nearest present NPC template
                npcs = [o for o in tv.read_object_templates(b) if o[2]]
                tgt = tuple(npcs[0][0]) if npcs else (4, 4)
                r = engage((tgt[0], tgt[1] + 1), "UP", "warden", drains=8)
                drain(key="B")
                L(f"   WARDEN -> {r} (strength in bag: {have_strength()})")
                if have_strength():
                    _stage_save("hm04")
                    snap("40_hm04")
                elif wedge("warden", 3, "HM04 not granted"):
                    return 1
                continue
            if not enter_to(FUCHSIA, "warden-exit"):
                if wedge("warden_exit", 3, "can't leave the Warden's house"):
                    return 1
            continue

        # anywhere else (rest house detour, step-limit bounce mid-map, North/East drift):
        L(f"   off-route at {here}@{tv.coords(b)} — stepping out south/known warps")
        if not (CENTER and enter_to(CENTER, "reroute-center")):
            camp.enter_warp(prefer="south")
            for _ in range(80):
                b.run_frame()
            if tuple(tv.map_id(b)) == here:
                if wedge(("offroute", here), 3, f"stuck off-route at {here}"):
                    return 1

    if not (have_surf() and have_strength()):
        L(f"!! strike incomplete (surf={have_surf()} strength={have_strength()}) — NOT banking")
        snap("70_fail")
        return 1

    # make sure she's back on the city, heal, bank
    out_deadline = time.time() + 300
    while tuple(tv.map_id(b)) != FUCHSIA and time.time() < out_deadline:
        here = tuple(tv.map_id(b))
        if here == ENTRANCE:
            enter_to(FUCHSIA, "final-exit")
        elif CENTER and here == CENTER:
            step_warp((26, 30), "final-center-exit")
        elif WEST and here == WEST:
            any(step_warp(w, "final-west-exit") for w in ((31, 5), (30, 5), (38, 5)))
        elif NORTH and here == NORTH:
            any(step_warp(w, "final-north-exit") for w in ((48, 32), (48, 31), (48, 33)))
        elif EAST and here == EAST:
            any(step_warp(w, "final-east-exit") for w in ((8, 27), (8, 26), (8, 28)))
        elif SECRET and here == SECRET:
            step_warp((4, 9), "final-house-exit")
        elif here == WARDEN_HOUSE:
            enter_to(FUCHSIA, "final-warden-exit")
        else:
            camp.enter_warp(prefer="south")
            for _ in range(80):
                b.run_frame()
        if fight_open():
            on_battle()
            drain()
    if tuple(tv.map_id(b)) == FUCHSIA:
        camp.heal_nearest()

    L(f"   SURF={have_surf()} STRENGTH={have_strength()} key_items={key_items()} | "
      f"pos {tv.map_id(b)}@{tv.coords(b)} | battles {n_battles[0]}")
    snap("80_final")
    _stage_save("safari_hms")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} safari_hms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
