"""recon_e4.py — THE LAST VEHICLE: League mart stock-up + Elite Four + Champion -> CREDITS.

Ground truth (pret, fetched 2026-07-07 night shift #8):
- IndigoPlateau_PokemonCenter_1F: clerk (0,7) behind the counter (stand (2,7) FACE LEFT);
  nurse (13,10); League door (4,1) -> Lorelei; exterior door (11,16). Door guard (5,3) is
  flavor-only pre-National-Dex (verbatim script: msgbox GoodLuck, release).
- Mart rows (0-based): 0 Ultra Ball, 1 Great Ball, 2 FULL RESTORE(19), 3 Max Potion(20),
  4 REVIVE(24), 5 FULL HEAL(23), 6 Max Repel. Buy = tm_errand's true-index engine
  (row+scroll @0x02039940/42), each unit verified by money drop + bag qty.
- E4 rooms are ONE TEMPLATE: arrive south warp (6,12), trainer at (6,5) (talk-triggered
  trainerbattle_no_intro), north door (6,2) OPENS on FLAG_DEFEATED_*; on_frame entry
  scene auto-walks her in (drained). Chain: Lorelei -> Bruno -> Agatha -> Lance ->
  ChampionsRoom (arrive (6,19), GARY at (6,8)) -> HALL OF FAME (6,2) -> CREDITS.
- Battle plan: sleep-lock + EQ/Razor Leaf; the battle agent's item instinct uses the
  bought Full Restores (pref list has 19 FIRST) at the matchup-aware threshold.
- Whiteout = respawn at THIS center (last heal) -> the dispatch loop re-heals and
  re-enters; DEFEATED flags persist so cleared rooms pass through.
- CREDITS: entering the Hall of Fame is the point of no return (auto-save + credits).
  Bank stage -> banked_E4 at every room fall; banked_CREDITS on the HoF map.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_e4.py   (WATCH=1 = window)
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
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# The anti-wedge forensic frame dump is env-gated — set it HERE, not in the launcher shell
# (runs 7-8 aborted dozens of times with ZERO frames because the env didn't ride the launch).
os.environ.setdefault("BATTLE_DEBUG_DIR", os.path.join(
    os.environ.get("TEMP", _HERE), "longrun", "e4_probe"))

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
STAGE = os.path.join(SCRATCH, "stage_e4")
BANK = os.path.join(SCRATCH, "banked_E4")
BANK_CREDITS = os.path.join(SCRATCH, "banked_CREDITS")
DBG = os.path.join(SCRATCH, "e4_probe")

INDIGO_EXT = (3, 9)
FULL_RESTORE, MAX_POTION, REVIVE, FULL_HEAL = 19, 20, 24, 23
# (item, mart true-index row, want, unit price)
# SPEND THE WAD (shift-15, run18 economy postmortem): items PERSIST through a whiteout,
# unspent cash gets HALVED — the old plan left $24k unspent and the surplus evaporated
# across attempts ($63k -> $36k -> $10k -> $0 by attempt 4). Wealth stored as Full
# Restores is whiteout-proof; the stock_up scaler already clamps to money per line.
# POVERTY-FIRST ORDER (shift-17, run20 postmortem): the whiteout treadmill pays $2-6k/lap
# (E4 payouts, halved on the next whiteout) — FR-first ate the whole budget when rich and
# bought NOTHING when poor, so by attempt 5 the bag was empty and every lap ran itemless.
# Revives first ($1500 = the comeback cycle: the ace back at half HP, and fainting cleared
# its sleep), Full Heals second (the $600 Jynx sleep counter), Full Restores with the rest.
# CAPS REBALANCED (shift-18, run22 laps 1-2): the $12k steady-state lap bought 8 Revives and
# then couldn't afford ONE Full Heal — laps ran with NO sleep cure into Jynx/Hypnosis, and the
# ace sat at half HP mid-Gary with FR x0 while the worthy-gate (rightly) refused to burn turns
# reviving L39 bench bodies. 5 Revives ($7.5k) + 3 Full Heals ($1.8k) + FRs with the rest
# (~$3k -> 1 per lap): ace UPTIME beats a deeper pile of one-turn bench corpses.
SHOPPING = [(REVIVE, 4, 5, 1500), (FULL_HEAL, 5, 3, 600), (FULL_RESTORE, 2, 16, 3000)]
CLERK_STAND = (2, 7)
LEAGUE_DOOR = (4, 1)
CENTER_EXIT = (11, 16)
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}


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

    b = Bridge(ROM)
    # E4_BOOT=dir -> boot a ratchet bank (banked_E4) instead of canonical, so a run that
    # died MID-CHAIN resumes with its DEFEATED flags (cleared rooms stay cleared). Booting
    # a bank lands inside a room: the dispatch loop's off-route branch walks her south to
    # the lobby (beaten trainers don't re-fight), learns the center, and re-enters the chain.
    # E4_STATE=<kit basename> boots the kit line (single .state + <pref>.*.json sidecars);
    # E4_BOOT=<dir> (the ratchet bank) OVERRIDES it for a mid-chain resume (cleared rooms
    # stay cleared) and reads dir-style sidecars (world_model.json).
    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("E4_STATE", ""))
    boot_dir = os.environ.get("E4_BOOT")
    if boot_dir:
        state_path, sc_dir, sc_pref = (
            os.path.join(boot_dir, "kira_campaign.state"), boot_dir, "")
    with open(state_path, "rb") as f:
        b.load_state(f.read())
    if boot_dir or os.environ.get("E4_STATE"):
        print(f"BOOT from: {state_path}", flush=True)
    for _ in range(40):
        b.run_frame()

    render_fn = lambda: None                       # noqa: E731
    if os.environ.get("WATCH") == "1":
        import pygame
        pygame.init()
        _scale = 3
        _win = (b.width * _scale, b.height * _scale)
        _screen = pygame.display.set_mode(_win)
        pygame.display.set_caption("Kira — THE ELITE FOUR (live watch)")

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

    # e4_run1 truth: the item instinct is ORACLE-GATED (_maybe_use_item returns False
    # when choose is None) and recon vehicles never passed a chooser — so the whole
    # Full Restore kit sat unused while Lorelei's Lapras wiped the party (FR x10
    # intact through a whiteout). Headless E4 policy = what a competent player does:
    # ALWAYS take the offered heal/cure. choose() only routes "battle_item".
    def _choose(ptype, offers, ctx):
        # run9 truth: the Bruno collapse offered use_revive SIX times while the bench died
        # one-by-one with 6 Revives in the bag — this list must know EVERY instinct the
        # engine can offer (revive + ether shipped shift-13), or the kit rides to a whiteout.
        for k in ("use_potion", "use_cure", "use_ether", "use_revive"):
            if k in offers:
                return k
        return "keep_fighting"

    def fight():
        n_battles[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=render_fn,
                           log=lambda m: print(m, flush=True),
                           choose=_choose).run(max_seconds=600)

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
    _pfx = (sc_pref + ".") if sc_pref else ""
    _w_side = os.path.join(sc_dir, _pfx + "world_model.json")
    _s_side = os.path.join(sc_dir, _pfx + "strat_memory.json")
    _soul_side = os.path.join(sc_dir, _pfx + "soul.json")
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

    def bank(dst, label):
        _stage_save(label)
        _stage_continuity()
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(STAGE, dst)
        L(f"BANKED [{label}] -> {dst}")

    def lead_frac():
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def party_alive():
        n = 0
        for i in range(6):
            base = ram.GPLAYER_PARTY + i * 100
            if b.rd16(base + 0x56) > 0 and b.rd16(base + 0x58) > 0:
                n += 1
        return n

    def fight_open():
        # pointer validity + LIVENESS (gMain.callback2) — a stale res_ptr after a whiteout
        # is a corpse, not a fight (the run3 phantom re-attach livelock)
        return (ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))
                and not ram.battle_cb2_dead(b))

    def drain(max_n=60, key="B"):
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

    def live_npc_tiles():
        OB, SZ = 0x02036E38, 0x24
        out = []
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.append((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                        b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    def walk(goal_test, label, tries=12, allow=()):
        budget = tries
        frozen, last_pos = 0, None
        map_start = tuple(tv.map_id(b))
        while budget > 0:
            budget -= 1
            if handle_interrupts():
                budget += 1
                continue
            if tuple(tv.map_id(b)) != map_start:
                # a whiteout (or any warp) mid-walk: the goal coords belong to the
                # OLD map — bail and let the dispatch loop re-route (e4_run1 burned
                # 17s planning room-coords on the center map)
                L(f"   [{label}] map changed {map_start} -> {tuple(tv.map_id(b))} — bail")
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            # FREEZE ARMOR (e4_run1: post-whiteout she stood at (13,12) planning
            # len-20 paths for 22 replans without moving ONE tile — the signature
            # of a hidden modal box eating inputs that dd_box doesn't see, the
            # bright-tileset class). 3 no-move replans -> snap the truth + blind
            # B/A drain, then keep walking.
            if cur == last_pos:
                frozen += 1
                if frozen >= 3:
                    L(f"!! [{label}] FROZEN at {cur} x{frozen} replans — hidden box? "
                      f"blind B/A drain + frame to e4_probe/")
                    snap(f"frozen_{label[:12]}")
                    for k in ("B", "B", "A", "B"):
                        b.press(k, 8, 12, camp.render, owner="agent")
                        for _ in range(30):
                            b.run_frame()
                    drain()
                    frozen = 0
                    budget += 1
            else:
                frozen = 0
            last_pos = cur
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - set(allow)
            npcs = set(live_npc_tiles())
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs)
            L(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                L(f"   [{label}] no path from {cur}")
                snap(f"nopath_{label[:12]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                if handle_interrupts():
                    budget += 1
                    break
                if not camp._step_to(tuple(t)):
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
        return goal_test(tuple(tv.coords(b) or ()))

    def go_warp(tile, label):
        """Walk to a neighbor of `tile`, step in; True on any map change."""
        m0 = tuple(tv.map_id(b))
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in
               ((0, 1), (0, -1), (1, 0), (-1, 0))]
        for _attempt in range(4):
            cur = tuple(tv.coords(b) or ())
            if cur != tile and cur not in nbs:
                if not walk(lambda c, s=set(nbs): c in s, f"{label}-approach",
                            allow=(tile,)):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            key = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            for _press in range(4):
                if key:
                    b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    break
            if handle_interrupts():
                continue
            if tuple(tv.map_id(b)) != m0:
                settle(180)
                L(f"   [{label}] {m0} -> {tuple(tv.map_id(b))} @ {tv.coords(b)}")
                return True
        L(f"!! [{label}] never fired (at {tv.map_id(b)}@{tv.coords(b)})")
        snap(f"warpfail_{label[:16]}")
        return False

    # ── the mart engine (tm_errand's true-index machinery, Items-pocket verify) ──
    def shop_index():
        return b.rd16(0x02039940) + b.rd16(0x02039942)

    def list_live():
        c0 = shop_index()
        b.press("DOWN", 8, 10, camp.render, owner="agent")
        for _ in range(20):
            b.run_frame()
        if shop_index() != c0:
            return True
        b.press("UP", 8, 10, camp.render, owner="agent")
        for _ in range(12):
            b.run_frame()
        return False

    def shop_goto_index(target, tries=20):
        for _ in range(tries):
            idx = shop_index()
            if idx == target:
                for _ in range(20):
                    b.run_frame()
                if shop_index() == target:
                    return True
                continue
            b.press("DOWN" if idx < target else "UP", 8, 10, camp.render, owner="agent")
            for _ in range(20):
                b.run_frame()
        return shop_index() == target

    def stock_up():
        need = [(iid, row, want - camp.bag_count(iid), price)
                for iid, row, want, price in SHOPPING
                if camp.bag_count(iid) < want]
        # afford check, priority order; drop the tail if money is short
        money = camp.money()
        plan = []
        # NO cash reserve (shift-17): money is HALVED on whiteout, items aren't — every
        # reserved dollar is half-wasted. The old $2000 floor left poverty laps ($2-6k)
        # unable to buy even one Revive.
        for iid, row, n, price in need:
            n = min(n, max(0, money // price))
            if n > 0:
                plan.append((iid, row, n, price))
                money -= n * price
        if not plan:
            if need:
                L(f"   [shop] kit SHORT but can't afford a single unit "
                  f"(money ${camp.money()}) — riding on what's aboard (LOUD)")
            else:
                L(f"   [shop] stocked already (money ${camp.money()})")
            return True
        L(f"   [shop] plan {plan} (money ${camp.money()})")
        if not walk(lambda c: c == CLERK_STAND, "clerk-approach"):
            return False
        opened = False
        for _ in range(8):
            b.press("LEFT", 8, 8, camp.render, owner="agent")
            b.press("A", 8, 10, camp.render, owner="agent")
            for _ in range(40):
                b.run_frame()
                if dd_box(b):
                    opened = True
                    break
            if opened:
                break
        if not opened:
            L(f"!! [shop] clerk never opened a dialog (coords {tv.coords(b)})")
            snap("shop_no_greeting")
            return False
        stable = 0
        for _ in range(30):
            if dd_box(b):
                stable = 0
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    break
                for _ in range(30):
                    b.run_frame()
        if not list_live():
            b.press("A", 8, 10, camp.render, owner="agent")   # BUY (top of BUY/SELL)
            for _ in range(120):
                b.run_frame()
            if not list_live():
                L("!! [shop] BUY list didn't confirm — abort shop")
                snap("shop_entry_fail")
                return False
        for iid, row, n, price in plan:
            for u in range(n):
                q0, m0 = camp.bag_count(iid), camp.money()
                if not shop_goto_index(row) or shop_index() != row:
                    L(f"!! [shop] couldn't hold index {row} — abort shop")
                    return False
                got = camp._mart_buy_one()
                q1 = camp.bag_count(iid)
                if got <= 0 or got > price + 500 or q1 != q0 + 1:
                    L(f"!! [shop] buy-verify FAILED item {iid} (price={got}, "
                      f"bag x{q0}->x{q1}) — abort shop")
                    snap("shop_buy_fail")
                    return False
            L(f"   [shop] bought {n} x item {iid} (bag x{camp.bag_count(iid)}, "
              f"money ${camp.money()})")
        for _ in range(8):
            b.press("B", 6, 12, camp.render, owner="agent")
            for _ in range(14):
                b.run_frame()
        drain()
        _stage_save("shopped")
        return True

    # ── preflight ────────────────────────────────────────────────────────────
    import field_moves as fm
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} badges8={fm.read_flag(b, 0x827)} "
      f"lead={lead_frac():.0%} alive={party_alive()} money=${camp.money()} "
      f"FR x{camp.bag_count(FULL_RESTORE)}")
    if not fm.read_flag(b, 0x827):
        L("!! badge 8 not held — wrong canonical, abort")
        return 1
    # 4h window (shift-15): a process restart resets the ace to canonical L66 — the XP
    # curve (L66->71 across run18's first two attempts) only COMPOUNDS within one live
    # process, so give the loop room to converge (level curve + whiteout-proof kit).
    deadline = time.time() + 14400

    # ── the dispatch loop: center -> shop+heal -> door chain -> CREDITS ──────
    # The League center's map id is KB ground truth (learned live in e4_run3). Pre-seeding
    # it lets an E4_BOOT mid-chain resume dispatch straight into the room logic instead of
    # spinning the off-route branch (run4: enter_warp with no adjacent warp fails instantly
    # -> a 10Hz "off-route" livelock).
    center = (13, 0)
    shopped = False
    healed = False
    rooms_cleared = 0
    seen_rooms = []               # map ids in door-chain order (Lorelei..Champion)
    hof_banked = False
    prev_here = None
    while time.time() < deadline:
        if handle_interrupts():
            continue
        here = tuple(tv.map_id(b))
        came_from, prev_here = prev_here, here
        if here == center and seen_rooms and came_from is not None \
                and came_from not in (center, INDIGO_EXT):
            # back at the center FROM the chain = a whiteout — the respawn heals
            # the party but the Full Restore kit is drained: re-check the shop
            # (money-aware; no-op if the bag is still stocked)
            L(f"   back at the center from the chain (whiteout) — re-checking the kit "
              f"[money ${camp.money()}, FR x{camp.bag_count(FULL_RESTORE)}]")
            shopped = False
            # XP RATCHET (shift-15): every attempt levels the ace (+5 over run18's first
            # two) but that XP lived only in process memory — a crash lost it. Post-
            # whiteout at the center is a clean, healed, resumable state: bank it so the
            # level curve survives a process death (E4_BOOT resumes from banked_E4).
            bank(BANK, "whiteout_center")
        warps = [(tuple(xy), tuple(d)) for xy, d, _w in tv.read_warps(b)]
        if here == INDIGO_EXT:
            # enter the center (its door is the only warp here)
            if not warps:
                settle(60)
                continue
            tgt = min(warps, key=lambda w: abs(w[0][0] - 11))[0]
            if not go_warp(tgt, "enter-center"):
                L("!! can't enter the League center — abort")
                return 1
        elif center is None and any(d == INDIGO_EXT for _t, d in warps) \
                and any(t == LEAGUE_DOOR for t, _d in warps):
            center = here
            L(f"   League center = {here}")
        elif here == center:
            if not healed:
                r = camp.heal_nearest()
                L(f"   [heal] heal_nearest -> {r} (lead {lead_frac():.0%})")
                drain()
                healed = lead_frac() > 0.99
                continue
            if not shopped:
                if not stock_up():
                    L("!! shopping failed — proceeding with what's aboard (LOUD)")
                shopped = True
                continue
            if not go_warp(LEAGUE_DOOR, "league-door"):
                L("!! League door failed — abort")
                return 1
            _stage_save("league_door")
        elif here != center and center is not None:
            # inside the door chain: an E4 room, the Champion's room, or the HoF
            if here not in seen_rooms:
                seen_rooms.append(here)
                L(f"   room #{len(seen_rooms)}: map {here} @ {tv.coords(b)} "
                  f"[lead {lead_frac():.0%}, alive {party_alive()}]")
                snap(f"room{len(seen_rooms)}_enter")
                _stage_save(f"room{len(seen_rooms)}")
                bank(BANK, f"room{len(seen_rooms)}")
            if len(seen_rooms) >= 6 and not hof_banked:
                # room #6 past the champion = HALL OF FAME — the credits are rolling
                hof_banked = True
                L("   *** HALL OF FAME — CREDITS INBOUND ***")
                snap("hall_of_fame")
                bank(BANK_CREDITS, "hall_of_fame")
                break
            npcs = live_npc_tiles()
            room_warps = sorted([t for t, _d in warps], key=lambda t: t[1])
            north = room_warps[0] if room_warps else (6, 2)
            if npcs:
                # the E4 member (or Gary): stand SOUTH of them, face UP, talk
                trainer = min(npcs, key=lambda t: t[1])
                stand = (trainer[0], trainer[1] + 1)
                if tuple(tv.coords(b) or ()) != stand:
                    if not walk(lambda c, s=stand: c == s, "trainer-approach"):
                        # entry scene may still own the avatar — settle and retry
                        settle(180)
                        drain()
                        continue
                fought = False
                for _try in range(3):
                    b.press("UP", 8, 8, camp.render, owner="agent")
                    b.press("A", 8, 12, camp.render, owner="agent")
                    for _ in range(120):
                        b.run_frame()
                        if fight_open():
                            break
                    if fight_open():
                        L(f"   E4 battle #{len(seen_rooms)} OPENS "
                          f"[lead {lead_frac():.0%}, alive {party_alive()}, "
                          f"FR x{camp.bag_count(FULL_RESTORE)}]")
                        fight()
                        drain()
                        fought = True
                        break
                    drain(key="A")
                    if not dd_box(b) and not fight_open():
                        break
                # post-battle (or already-beaten): scripted scenes may hold the
                # avatar (champion room: Oak escort) — drain, then try the door
                for _ in range(30):
                    if handle_interrupts():
                        continue
                    if tuple(tv.map_id(b)) != here:
                        break
                    settle(30)
                    if not dd_box(b) and not fight_open():
                        break
                if tuple(tv.map_id(b)) != here:
                    continue
                if fought:
                    rooms_cleared += 1
                    _stage_save(f"cleared{rooms_cleared}")
            if not go_warp(north, "north-door"):
                # door still closed (loss? guard scene?) — re-dispatch
                if party_alive() == 0 or lead_frac() == 0:
                    L("   whiteout state — loop recovers via the center")
                settle(120)
        else:
            L(f"   off-route at {here} — exiting")
            camp.enter_warp(prefer="south")
            settle(80)
    if not hof_banked:
        L("!! deadline without the Hall of Fame — read the log, fix, re-run")
        return 1

    # ── CREDITS: drain the HoF scene + let the credits roll ──────────────────
    L("   draining the Hall of Fame scene + credits (A every ~2s)")
    t_cred = time.time()
    last_map = tuple(tv.map_id(b))
    while time.time() - t_cred < 600:
        for _ in range(120):
            b.run_frame()
        if dd_box(b):
            b.press("A", 8, 12, camp.render, owner="agent")
        m = tuple(tv.map_id(b))
        if m != last_map:
            L(f"   [credits] map {last_map} -> {m} at +{time.time() - t_cred:.0f}s")
            last_map = m
    snap("99_post_credits")
    L(f"CREDITS SEQUENCE DRAINED | battles {n_battles[0]} | money ${camp.money()} | "
      f"FR left x{camp.bag_count(FULL_RESTORE)}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK_CREDITS} hall_of_fame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
