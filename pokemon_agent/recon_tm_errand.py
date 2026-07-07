"""recon_tm_errand.py — THE MOVESET-FAMINE KILLER (night shift #5).

The Route 8 Vileplume trainer whiteout-beat her twice (flute_run7/8): root cause is MOVESET
FAMINE, not levels — Venusaur's ONLY damaging move is Razor Leaf (0.25x into grass/poison) and
Fearow's best is Fury Attack (15 power). The fix at the definition: BUY TM43 Secret Power
(Normal 70/100 — neutral into grass/poison, STAB on Fearow) at the Celadon Dept Store 2F TM
clerk and TEACH it to both. Kills the wall class instead of grinding around every resist.

DISASM GROUND TRUTH (pret/pokefirered map.json + scripts.inc, fetched 2026-07-07):
  Celadon City door (11,14) -> DeptStore_1F (arrive ~(2,14); side door (15,14) -> (10,14))
  1F stairs (4,2) -> 2F (arrive (3,2));  2F back-stairs (3,2) -> 1F;  elevator (6,1) UNUSED
  2F TM clerk at (1,6)  -> player stands (2,6) FACING LEFT (items clerk (1,8); lass (5,10))
  ClerkTMs BUY rows (0-based): TM05, TM15, TM28, TM31, [4]=TM43, TM45
  ITEM_TM43 = 331 (TM_FIRST 289 + 42); MOVE Secret Power = 290

Canonical protection: recon_longrun's staging pattern (stage dir; bank -> %TEMP%/longrun/
banked_TM43 for promote_bank.py). On success the active vileplume wall is retired with an
honest reason (the famine that caused it is cured; a re-loss re-records it).
"""
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
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

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_tm_errand")
BANK = os.path.join(SCRATCH, "banked_TM43")

CELADON = (3, 6)
DEPT_DOOR = (11, 14)          # Celadon overworld -> Dept 1F
DEPT_STAIRS_1F = (4, 2)       # 1F -> 2F
DEPT_STAIRS_2F = (3, 2)       # 2F -> 1F
DEPT_EXIT_1F = (2, 14)        # 1F -> Celadon
CLERK_FRONT_2F = (3, 6)       # customer side of the 0x80 counter col x=2; TM clerk behind at (1,6)
TM43_ROW = 4                  # ClerkTMs BUY list row (disasm scripts.inc order)
ITEM_TM43 = 331
SECRET_POWER = 290
BUY_BUDGET = 3500             # per unit — TM43 lists at 3000; hard sanity ceiling on the money drop
# status tools the engine USES (sleep-then-throw catch, PoisonPowder chip past resists) — never
# auto-forget these, on top of hm_teach's _PRECIOUS.
KEEP_MOVES = ht._PRECIOUS | {77, 79, 92}     # PoisonPowder, Sleep Powder, Toxic
TARGET_SPECIES = (3, 22)      # Venusaur, Fearow — the two famine mons


def tm_case_qty(b, item_id):
    """Quantity of item_id in the TM CASE pocket (qty u16 XOR the low-16 security key — same
    decode as campaign.bag_count, different pocket)."""
    sb1 = ht._sb1(b)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
    for i in range(58):
        slot = sb1 + ht.TM_CASE_OFF + i * 4
        iid = b.rd16(slot)
        if iid == 0:
            break
        if iid == item_id:
            return b.rd16(slot + 2) ^ key
    return 0


def pick_forget(b, slot):
    """forget_idx for the Secret Power teach: a free slot wins; else the first no-power move
    that isn't a status tool we use; else the weakest non-kept damaging move."""
    moves = st.read_party_moves(b, slot)
    if 0 in moves or len([m for m in moves if m]) < 4:
        return None, "free slot"
    scored = []
    for i, m in enumerate(moves):
        if not m or m in KEEP_MOVES:
            continue
        _t, power = st.move_info(b, m)
        scored.append((power or 0, i, m))
    if not scored:
        return 0, "all four kept?! overwriting move 0 (should not happen)"
    scored.sort()
    p, i, m = scored[0]
    return i, f"forgetting move {m} (power {p})"


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=180)

    camp = Campaign(b, battle_runner=runner,
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    os.makedirs(STAGE, exist_ok=True)

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

    def pc():
        return b.rd8(ram.GPLAYER_PARTY_CNT)

    dbg = os.path.join(SCRATCH, "tm_probe")
    os.makedirs(dbg, exist_ok=True)

    def snap(name):
        try:
            b.frame_rgb().resize((480, 320)).save(os.path.join(dbg, name + ".png"))
        except Exception as e:
            L(f"   snap {name} failed: {e}")

    def dump(tag):
        L(f"-- {tag}: map={tv.map_id(b)} coords={tv.coords(b)} money={camp.money()}")
        case = ht.pocket_items(b, ht.TM_CASE_OFF, 58)
        L(f"   TM case: {[(iid, tm_case_qty(b, iid)) for iid in case]}")
        for s in range(pc()):
            sp = st.read_party_species(b, s)
            lv = b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54)
            mv = st.read_party_moves(b, s)
            mi = [(m,) + st.move_info(b, m) for m in mv if m]
            L(f"   slot {s}: species {sp} L{lv} moves {mi}")

    dump("GROUND TRUTH (canonical)")
    if tuple(tv.map_id(b)) != CELADON:
        L(f"!! ABORT: not in Celadon ({tv.map_id(b)}) — this errand is Celadon-anchored")
        return 1

    # which of the two famine mons actually need + can take Secret Power (ROM truth)
    plan = []                          # (slot, species)
    for s in range(pc()):
        sp = st.read_party_species(b, s)
        if sp not in TARGET_SPECIES:
            continue
        if SECRET_POWER in st.read_party_moves(b, s):
            L(f"   slot {s} (species {sp}) already knows Secret Power — skipping")
            continue
        if not ht.tm_compatible(b, 43, sp):
            L(f"!! species {sp} NOT TM43-compatible per ROM — skipping (unexpected)")
            continue
        plan.append((s, sp))
    have = tm_case_qty(b, ITEM_TM43)
    need = max(0, len(plan) - have)
    L(f"teach plan: {plan}; TM43 in case: {have}; to buy: {need}")
    if not plan:
        L("nothing to teach — errand is a no-op")
        return 1
    if camp.money() < need * BUY_BUDGET + C.SHOP_MONEY_FLOOR:
        L(f"!! ABORT: money {camp.money()} can't cover {need}x TM43 + floor")
        return 1

    # 0. heal first (canonical rode out of the gym hurt — bank forward CLEAN)
    if not camp._party_fully_healed():
        L("healing first (party rode out of the Erika fight hurt)...")
        r = camp.heal_nearest()
        L(f"   heal -> {r}")

    if need > 0:
        # 1. into the Dept Store, up to 2F
        for pick_tile, want_inside in ((DEPT_DOOR, True), (DEPT_STAIRS_1F, True)):
            m0 = tuple(tv.map_id(b))
            r = camp.enter_warp(pick=pick_tile)
            L(f"   warp via {pick_tile}: {r} (map {m0} -> {tv.map_id(b)})")
            if tuple(tv.map_id(b)) == m0:
                L("!! warp didn't flip the map — abort LOUD")
                return 1
            for _ in range(80):
                b.run_frame()
        L(f"   on 2F: map={tv.map_id(b)} coords={tv.coords(b)} "
          f"npcs={sorted(camp.trav._npc_tiles())}")
        L(f"   2F door tiles: {sorted(camp._door_tiles())}")
        g2 = tv.Grid(b)
        for yy in range(4, 10):
            row = " ".join(f"{camp._tile_behavior(xx, yy):02x}{'W' if g2.walkable(xx, yy) else '.'}"
                           for xx in range(0, 5))
            L(f"   behav y={yy}: {row}")

        # 2. stand at the customer side of the counter, face LEFT, enter the BUY list.
        if (camp.trav.travel(target_map=None, arrive_coord=CLERK_FRONT_2F,
                             max_steps=60, max_seconds=45) != "arrived"
                and not camp._step_to(CLERK_FRONT_2F)):
            L(f"!! couldn't reach the clerk front {CLERK_FRONT_2F} — abort")
            return 1
        if tuple(tv.coords(b) or ()) != CLERK_FRONT_2F:
            L(f"!! not actually at {CLERK_FRONT_2F} (coords {tv.coords(b)}) — abort LOUD")
            return 1
        L(f"   at clerk front {CLERK_FRONT_2F}")
        def shop_index():
            """TRUE buy-list selection = selectedRow + scrollOffset (sShopData 0x02039934 +0xC/+0xE,
            pret symbols). MART_CURSOR alone LIES on lists deeper than the window (run-11: (row 4,
            scroll 2) = CANCEL, whose A exited the shop and the A-mash re-bought row 0)."""
            return b.rd16(0x02039940) + b.rd16(0x02039942)

        def list_live():
            """POSITIVE buy-list confirm: a DOWN press must move the TRUE index."""
            c0 = shop_index()
            b.press("DOWN", 8, 10, camp.render, owner="agent")
            for _ in range(20):
                b.run_frame()
            if shop_index() != c0:
                return True
            b.press("UP", 8, 10, camp.render, owner="agent")   # undo (a BUY/SELL menu clamps at top)
            for _ in range(12):
                b.run_frame()
            return False

        def shop_goto_index(target, tries=16):
            """Move the buy-list selection to true index `target`, settle-verified (the row byte
            lags the scroll animation — re-read after settling before trusting arrival)."""
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

        b.set_input_owner("agent")
        guard = camp.money()
        opened = False                     # engage: the box opens a beat late across the counter
        for _ in range(8):
            b.press("LEFT", 8, 8, camp.render, owner="agent")
            b.press("A", 8, 10, camp.render, owner="agent")
            for _ in range(40):
                b.run_frame()
                if C.dd_box_open(b):
                    opened = True
                    break
            if opened:
                break
        if not opened:
            snap("no_greeting")
            L(f"!! clerk never opened a dialog (coords {tv.coords(b)}) — abort LOUD")
            return 1
        stable = 0                         # drain the greeting: multi-page text flickers the box
        for _ in range(30):                # closed between pages — need it STABLY closed
            if C.dd_box_open(b):
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
        L(f"   drained greeting: box_open={C.dd_box_open(b)} cursor={b.rd8(C.MART_CURSOR)} "
          f"scroll={b.rd16(0x02039942)} row={b.rd16(0x02039940)} n={b.rd16(0x02039944)}")
        snap("10_drained")
        # dept clerks may drop straight into the BUY list — only pick BUY if the list isn't live
        if not list_live():
            b.press("A", 8, 10, camp.render, owner="agent")  # BUY (top of BUY/SELL)
            for _ in range(120):
                b.run_frame()
            if not list_live():
                snap("entry_fail")
                L(f"!! BUY list didn't confirm (cursor dead; box_open={C.dd_box_open(b)}) — "
                  f"abort LOUD (frame -> {dbg}/entry_fail.png)")
                return 1
        if camp.money() < guard:
            L(f"!! money dropped during entry ({guard}->{camp.money()}) — accidental buy, abort")
            return 1

        # 3. buy `need` x TM43 — verify each unit by money drop + TM-case qty (pocket-true)
        for u in range(need):
            q0, m0 = tm_case_qty(b, ITEM_TM43), camp.money()
            if not shop_goto_index(TM43_ROW):
                L(f"!! couldn't reach index {TM43_ROW} (row={b.rd16(0x02039940)} "
                  f"scroll={b.rd16(0x02039942)}) — abort LOUD")
                return 1
            L(f"   pre-buy: row={b.rd16(0x02039940)} scroll={b.rd16(0x02039942)} "
              f"case={ht.pocket_items(b, ht.TM_CASE_OFF, 58)}")
            snap(f"20_prebuy_u{u}")
            if shop_index() != TM43_ROW:
                L(f"!! selection drifted off {TM43_ROW} — abort LOUD")
                return 1
            price = camp._mart_buy_one()
            q1 = tm_case_qty(b, ITEM_TM43)
            if price <= 0 or price > BUY_BUDGET or q1 != q0 + 1:
                snap("buy_fail")
                L(f"!! buy-verify FAILED (price={price}, case x{q0}->x{q1}) — "
                  f"case now {ht.pocket_items(b, ht.TM_CASE_OFF, 58)} "
                  f"cursor={b.rd8(C.MART_CURSOR)} — abort LOUD")
                return 1
            L(f"   bought TM43 #{u + 1} for {price} (case x{q1}, money {camp.money()})")
        for _ in range(8):
            b.press("B", 6, 12, camp.render, owner="agent")
            for _ in range(14):
                b.run_frame()
        stable = 0                       # the clerk's closing text lingers past the blind B's —
        for _ in range(20):              # drain until the box is STABLY closed (teach needs START)
            if C.dd_box_open(b):
                stable = 0
                b.press("B", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    break
                for _ in range(30):
                    b.run_frame()
        for _ in range(60):
            b.run_frame()
        # step AWAY from the counter before menu work: an eaten START + stray A at the
        # counter RE-ENGAGES the clerk (run-15: teach 2 bought a TM05 through the shop)
        camp._step_to((4, 4))
        L(f"   stepped clear of the counter -> {tv.coords(b)}")

    # 4. teach each famine mon (TeachFlow drives the case; item/move overrides = the TM path)
    class DbgFlow(ht.TeachFlow):
        def _classify(self):
            scr = super()._classify()
            self._dbg_i = getattr(self, "_dbg_i", 0) + 1
            snap(f"teach_{self._dbg_i:02d}_{scr}")
            print(f"   [dbgteach] step {self._dbg_i}: scr={scr}", flush=True)
            return scr

    flow = DbgFlow(camp, log=lambda m: print(m, flush=True))
    for s, sp in plan:
        idx, why = pick_forget(b, s)
        L(f"   teaching Secret Power -> slot {s} (species {sp}); forget_idx={idx} ({why})")
        r = flow.teach("tm43", s, forget_idx=idx, item_override=ITEM_TM43,
                       move_override=SECRET_POWER)
        L(f"   teach slot {s} -> {r}")
        if r != "taught":
            L("!! teach FAILED — abort LOUD (nothing banked)")
            return 1

    # 5. back to the overworld (bank a clean, watch-ready spot)
    if need > 0:
        for pick_tile in (DEPT_STAIRS_2F, None):
            m0 = tuple(tv.map_id(b))
            if pick_tile is not None:
                r = camp.enter_warp(pick=pick_tile)
            else:                          # exit mats want the south-exit ritual (step DOWN)
                r = camp.enter_warp(prefer="south")
            L(f"   warp out via {pick_tile or 'south exit'}: {r} (map {m0} -> {tv.map_id(b)})")
            if tuple(tv.map_id(b)) == m0:
                L("!! exit warp didn't flip the map — abort (state NOT banked)")
                return 1
            for _ in range(80):
                b.run_frame()

    dump("POST-ERRAND")
    # 6. the famine that built the vileplume wall is cured — retire it honestly so the
    #    routing ungates Route 8 and the flute chain resumes (a re-loss re-records it).
    rec = camp.strat.active_wall_rec()
    if rec and "vileplume" in str(camp.strat.active_wall):
        camp.strat.retire_active_wall(
            "moveset famine fixed: Secret Power taught (neutral coverage past the grass/poison resist)")
        L(f"   retired active wall {rec.get('lead')} — famine cured, retry is honest now")

    _stage_save("errand done")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} tm43_secret_power")
    return 0


if __name__ == "__main__":
    sys.exit(main())
