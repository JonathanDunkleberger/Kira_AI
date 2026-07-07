"""recon_hm05.py — THE FLASH ERRAND (Sherpa strike, 2026-07-07 night shift).

From the canonical save (Route 9/10 country, dex >= 10 required): walk the known road back to
Vermilion, east onto Route 11, cross DIGLETT'S CAVE (north mouth -> B1F tunnel -> Route 2 east
strip), enter the aide's gatehouse, talk until FLAG_GOT_HM05 (0x23B) sets, TEACH Flash via the
proven hm_teach flow, then cross back to Route 11 and bank the bundle for promote_bank.py.

Canonical protection: recon_longrun's staging pattern — all persistence redirected to a scratch
stage dir; the bank lands in %TEMP%/longrun/banked_HM05 for the sanctity-gated promotion.
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
import field_moves as fm             # noqa: E402
import hm_teach as ht                # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_hm05")
BANK = os.path.join(SCRATCH, "banked_HM05")
FLAG_GOT_HM05 = 0x23B
FLASH_MOVE = 148
ROUTE11, ROUTE2, VERMILION = (3, 29), (3, 20), (3, 5)
# the westbound road home (reverse of the billed Celadon road), one leg per map
BACK_LEGS = {(3, 28): ("west", (3, 27)), (3, 27): ("west", (3, 3)), (3, 3): ("south", (3, 23)),
             (3, 23): ("pass", (3, 24)), (3, 24): ("south", (3, 5)), (3, 5): ("east", ROUTE11),
             (3, 22): ("east", (3, 3))}   # Route 4 (the dex-11 catch bank) joins at Cerulean


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
    # canonical protection: persistence -> STAGE only
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

    dex = ram.pokedex_owned_count(b)
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} dex={dex}")
    if dex < 10:
        L("!! ABORT: dex < 10 — the aide refuses below ten owned; run the catch stretch first")
        return 1

    def pc():
        return b.rd8(ram.GPLAYER_PARTY_CNT)

    def walk_legs(stop_map, legs, budget_s=900):
        """Leg-walk the known road: one billed hop per iteration until stop_map."""
        t = time.time()
        while tuple(tv.map_id(b)) != stop_map:
            if time.time() - t > budget_s:
                L(f"!! walk_legs TIMEOUT at {tv.map_id(b)}")
                return False
            m = tuple(tv.map_id(b))
            leg = legs.get(m)
            if leg is None:
                L(f"!! walk_legs: off the road at {m} — no billed leg")
                return False
            go, nxt = leg
            L(f"leg: {m} -{go}-> {nxt}")
            if go == "pass":
                r = camp._door_passthrough()
                L(f"   passthrough -> {r} (now {tv.map_id(b)})")
                if r == "need_heal":
                    camp.heal_nearest()
                continue
            r = camp._edge_travel(nxt, go)
            L(f"   edge -> {r} (now {tv.map_id(b)})")
            if r == "need_heal":
                camp.heal_nearest()
            elif r in ("no_path", "stuck", "no_route_hm_blocked", "no_route_npc_blocked"):
                L(f"!! walk_legs: leg failed hard ({r})")
                return False
        return True

    def cross_cave(into_prefer, out_map, budget_s=420):
        """Cross a 2-warps-per-room cave chain (Diglett's): enter via the preferred-side warp on
        this map, then in each interior room walk to the FARTHEST warp tile until we pop out on
        out_map. Wild battles handled by the traveler underneath."""
        m0 = tuple(tv.map_id(b))
        w = camp.enter_warp(prefer=into_prefer)
        L(f"cave: enter_warp({into_prefer}) -> {w} (now {tv.map_id(b)})")
        if tuple(tv.map_id(b)) == m0:
            return False
        t = time.time()
        while tv.map_id(b)[0] != 3:
            if time.time() - t > budget_s:
                L(f"!! cave crossing TIMEOUT at {tv.map_id(b)}")
                return False
            pos = tuple(tv.coords(b))
            warps = [tuple(wxy) for (wxy, _d, _i) in tv.read_warps(b)]
            if not warps:
                L(f"!! cave room {tv.map_id(b)} shows no warps — stuck")
                return False
            far = max(warps, key=lambda w_: abs(w_[0] - pos[0]) + abs(w_[1] - pos[1]))
            L(f"cave room {tv.map_id(b)}: at {pos}, heading to far warp {far} (of {warps})")
            before = tuple(tv.map_id(b))
            camp.trav.travel(target_map=None, arrive_coord=far, max_steps=400)
            if tuple(tv.map_id(b)) == before:
                # standing beside a ladder that needs a step-on: nudge through enter_warp
                camp.enter_warp(pick=far)
            if tuple(tv.map_id(b)) == before:
                L(f"!! cave: warp {far} didn't fire")
                return False
        ok = tuple(tv.map_id(b)) == out_map
        L(f"cave: OUT at {tv.map_id(b)} coords={tv.coords(b)} ({'OK' if ok else 'unexpected map'})")
        return ok

    # ── 1) road home: current map -> Vermilion -> Route 11 ─────────────────────────────
    if not walk_legs(ROUTE11, BACK_LEGS):
        _stage_save("fail"); return 1
    L(f"ROUTE 11 reached: {tv.coords(b)}")

    # ── 2) Diglett's Cave: Route 11 north mouth -> Route 2 east strip ──────────────────
    if not cross_cave("north", ROUTE2):
        _stage_save("fail"); return 1

    # ── 3) the aide's gatehouse: enter the door just south, talk until 0x23B sets ──────
    got = False
    for attempt in range(3):
        if fm.read_flag(b, FLAG_GOT_HM05):
            got = True
            break
        if tv.map_id(b)[0] == 3:                      # outside — enter the gate building south
            w = camp.enter_warp(prefer="south")
            L(f"gatehouse: enter_warp(south) -> {w} (now {tv.map_id(b)})")
            if tv.map_id(b)[0] == 3:
                L("!! gatehouse: no door south — trying passthrough search")
                camp._door_passthrough()
        talks = 0
        while not fm.read_flag(b, FLAG_GOT_HM05) and talks < 8:
            r = camp.talk_npc()
            talks += 1
            L(f"gatehouse talk {talks}: {r} (flag={bool(fm.read_flag(b, FLAG_GOT_HM05))})")
        if fm.read_flag(b, FLAG_GOT_HM05):
            got = True
            break
        L("!! aide not in this room — exiting to look again")
        camp._exit_to_overworld()
    if not got:
        L("!! ABORT: FLAG_GOT_HM05 never set (wrong building? dex gate?) — staged, not banked")
        _stage_save("fail"); return 1
    L("HM05 OBTAINED (flag 0x23B set)")

    # ── 4) teach Flash (the proven hm_teach flow; her judgment picks the mon) ──────────
    if st.party_knows_move(b, FLASH_MOVE, pc()) is None:
        plan = ht.default_plan(b, "flash", pc())
        if plan is None:
            L("!! TEACH: no compatible party mon for flash — banking the HM anyway (LOUD)")
        else:
            slot, forget_idx, reason = plan
            mon = st.SPECIES_NAME.get(st.read_party_species(b, slot), f"slot {slot}")
            L(f"TEACH: flash -> {mon} (slot {slot}, {reason})")
            r = ht.TeachFlow(camp, log=lambda m: print(m, flush=True),
                             on_event=lambda s, **k: L(f"[event] {s}")).teach("flash", slot, forget_idx)
            L(f"TEACH result: {r}")
    knows = st.party_knows_move(b, FLASH_MOVE, pc())
    L(f"flash known by party slot: {knows}")

    # ── 5) back through the cave to Route 11, then bank ────────────────────────────────
    if tv.map_id(b)[0] != 3:
        camp._exit_to_overworld()
    if not cross_cave("north", ROUTE11):
        L("!! return crossing failed — banking WHERE SHE STANDS (still resume-safe)")
    _stage_save("final")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK, ignore_errors=True)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    import sanctity
    ok, issues = sanctity.validate_bundle(BANK, prev_dir=CANON, log=print)
    L(f"sanctity: {'VALID' if ok else f'INVALID {issues}'}")
    L(f"final: map={tv.map_id(b)} coords={tv.coords(b)} dex={ram.pokedex_owned_count(b)} "
      f"hm05={bool(fm.read_flag(b, FLAG_GOT_HM05))} flash_slot={knows}")
    L(f"promote with: python pokemon_agent/promote_bank.py {BANK} hm05_flash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
