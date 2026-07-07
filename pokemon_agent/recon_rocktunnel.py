"""recon_rocktunnel.py — THE ROCK TUNNEL STRIKE (Sherpa, 2026-07-07 night shift #2).

From the post-HM05 canonical (Flash taught to a party mon): road east/north back to Route 10,
heal at the mouth-side Center, enter the NORTH mouth (8,19) -> map (1,81), USE FLASH (verified
via FLAG_SYS_FLASH_ACTIVE 0x806 — the dark-gate law: never walk the tunnel dark), then cross the
1F/B1F ladder maze by DESTINATION-AWARE warp choice to the SOUTH mouth (8,57) -> Lavender-side
Route 10 -> walk SOUTH into LAVENDER TOWN (3,4) and bank.

Rock Tunnel truth: BOTH mouths exit to (3,28), so "dest is overworld" alone can't identify the
far door — the exit rule is "an overworld-dest warp TILE we haven't stood on this crossing".
A warp target travel can't reach (the 1F is partitioned; the south section is only reachable
via B1F) is marked visited and the crossing continues on interior ladders — never a hard fail
while fresh warps remain.

Canonical protection: recon_longrun's staging pattern (stage dir; bank -> %TEMP%/longrun/
banked_ROCKTUNNEL for promote_bank.py).
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
STAGE = os.path.join(SCRATCH, "stage_rocktunnel")
BANK = os.path.join(SCRATCH, "banked_ROCKTUNNEL")
FLASH_MOVE = 148
ROUTE10, LAVENDER = (3, 28), (3, 4)
TUNNEL_1F = (1, 81)
NORTH_MOUTH = (8, 19)
# the eastbound road to Route 10, one billed leg per map (any-start: joins wherever canonical is)
FWD_LEGS = {(3, 29): ("west", (3, 5)), (3, 5): ("north", (3, 24)), (3, 24): ("pass", (3, 23)),
            (3, 23): ("north", (3, 3)), (3, 3): ("east", (3, 27)), (3, 27): ("east", ROUTE10),
            (3, 22): ("east", (3, 3))}



def cross_warp_maze(b, camp, L, m0, budget_s=900, stage_fail=None):
    """Destination-aware warp-maze crossing (Diglett's / Rock Tunnel class): exit = an
    overworld-dest warp TILE we haven't stood on (handles both-mouths-same-dest); else the
    farthest unvisited interior warp; unreachable targets are marked and skipped (partitioned
    floors); door mats get the step-through nudge. Returns True when out on the overworld."""
    import time as _t
    t0 = _t.time()
    visited = set()
    fail_streak = 0
    while tuple(tv.map_id(b))[0] != 3:
        if _t.time() - t0 > budget_s:
            L(f"!! maze crossing TIMEOUT at {tv.map_id(b)}")
            return False
        pos = tuple(tv.coords(b))
        mid = tuple(tv.map_id(b))
        warps = [(tuple(wxy), tuple(d)) for (wxy, d, _i) in tv.read_warps(b)]
        if not warps:
            L(f"!! room {mid} shows no warps — stuck")
            return False
        visited.add((mid, pos))
        dist = lambda w_: abs(w_[0] - pos[0]) + abs(w_[1] - pos[1])
        outs = [w_ for (w_, d) in warps if d[0] == 3 and d != m0 and (mid, w_) not in visited
                and w_ != pos]
        if not outs:                                     # both-mouths-same-dest fallback
            outs = [w_ for (w_, d) in warps if d[0] == 3 and (mid, w_) not in visited and w_ != pos]
        if outs:
            far = min(outs, key=dist)
            L(f"room {mid}: at {pos}, trying EXIT door {far} (dest overworld) of {warps}")
        else:
            fresh = [w_ for (w_, d) in warps if (mid, w_) not in visited and w_ != pos]
            cands = fresh or [w_ for (w_, d) in warps if w_ != pos]
            if not cands:
                L(f"!! room {mid}: no warp candidates left")
                return False
            far = max(cands, key=dist)
            L(f"room {mid}: at {pos}, heading to warp {far} (of {warps}; fresh={fresh})")
        visited.add((mid, far))
        before = mid
        camp.trav.travel(target_map=None, arrive_coord=far, max_steps=600, max_seconds=240)
        if tuple(tv.map_id(b)) == before:
            camp.enter_warp(pick=far)
        if tuple(tv.map_id(b)) == before and tuple(tv.coords(b)) == far:
            for d_ in ("DOWN", "UP", "LEFT", "RIGHT"):   # door mats fire on the crossing step
                b.press(d_, 10, 6, lambda: None, owner="agent")
                for _f in range(40):
                    b.run_frame()
                if tuple(tv.map_id(b)) != before:
                    break
                if tuple(tv.coords(b)) != far:
                    camp.trav.travel(target_map=None, arrive_coord=far, max_steps=20)
        if tuple(tv.map_id(b)) == before:
            visited.add((before, far))
            fail_streak += 1
            L(f"room {before}: warp {far} unreachable/didn't fire — marked, trying another "
              f"(streak {fail_streak})")
            if fail_streak >= 6:
                L("!! six dead warp targets in a row")
                return False
        else:
            fail_streak = 0
    return True


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

    flash_slot = st.party_knows_move(b, FLASH_MOVE, pc())
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} flash_slot={flash_slot}")
    if flash_slot is None:
        L("!! ABORT: no party mon knows FLASH — run the HM05 errand first (dark-gate law)")
        return 1

    def walk_legs(stop_map, legs, budget_s=900):
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

    # ── 0) a Route 2 start (the hm05 bank lands there): cross Diglett's Cave back east ─
    if tuple(tv.map_id(b)) == (3, 20):
        L("boot on Route 2 — crossing Diglett's Cave east to Route 11")
        w = camp.enter_warp(pick=(17, 11))
        L(f"cave door (17,11) -> {w} (now {tv.map_id(b)})")
        if tuple(tv.map_id(b))[0] == 3 or not cross_warp_maze(b, camp, L, (3, 20)):
            L("!! couldn't cross back to Route 11 — staged, not banked")
            _stage_save("fail"); return 1
        L(f"back on the east side: {tv.map_id(b)} {tv.coords(b)}")

    # ── 1) road to Route 10 ──────────────────────────────────────────────────────────
    if tuple(tv.map_id(b)) != ROUTE10 and not walk_legs(ROUTE10, FWD_LEGS):
        _stage_save("fail"); return 1
    L(f"ROUTE 10 reached: {tv.coords(b)}")

    # ── 2) heal at the mouth-side Center (full HP + PP for the gauntlet) ─────────────
    hr = camp.heal_nearest()
    L(f"pre-tunnel heal -> {hr} (map {tv.map_id(b)})")
    if tuple(tv.map_id(b)) != ROUTE10:
        camp._exit_to_overworld()
        if tuple(tv.map_id(b)) != ROUTE10 and not walk_legs(ROUTE10, FWD_LEGS, budget_s=300):
            _stage_save("fail"); return 1

    # ── 3) enter the NORTH mouth, use FLASH (verified), cross to the far door ───────
    def flash_lit():
        return bool(fm.read_flag(b, fm.FLAG_SYS_FLASH_ACTIVE))

    L(f"walking to the north mouth {NORTH_MOUTH}")
    camp.trav.travel(target_map=None, arrive_coord=(NORTH_MOUTH[0], NORTH_MOUTH[1] + 1),
                     max_steps=400, max_seconds=180)
    w = camp.enter_warp(pick=NORTH_MOUTH)
    L(f"enter_warp(north mouth) -> {w} (now {tv.map_id(b)})")
    if tuple(tv.map_id(b))[0] == 3:
        L("!! never entered the tunnel — staged, not banked")
        _stage_save("fail"); return 1

    entry_room = tuple(tv.map_id(b))
    L(f"INSIDE {entry_room} at {tv.coords(b)} — lighting it up")
    r = ht.TeachFlow(camp, log=lambda m: print(m, flush=True)).use_field_move(
        flash_slot, verify=flash_lit, label="flash")
    L(f"use_field_move(flash) -> {r} (flag 0x806={flash_lit()})")
    if r != "used":
        L("!! FLASH did not verify — refusing to walk the tunnel dark (staged, not banked)")
        _stage_save("fail"); return 1
    camp.on_event("and there's light! okay, Rock Tunnel — let's do this properly.", kind="route", tier=2)

    # destination-aware crossing (the shared maze crosser; m0=(3,255) sentinel so BOTH
    # Route-10 mouths count as exits — the visited-set already excludes the one we entered on)
    if not cross_warp_maze(b, camp, L, (3, 255)):
        _stage_save("fail"); return 1

    L(f"OUT of the tunnel at {tv.map_id(b)} coords={tv.coords(b)}")
    if tuple(tv.map_id(b)) != ROUTE10:
        L("(unexpected overworld map — continuing to bank where she stands)")

    # ── 4) south to Lavender Town ─────────────────────────────────────────────────────
    if tuple(tv.map_id(b)) == ROUTE10 and tv.coords(b)[1] > 40:
        r = camp._edge_travel(LAVENDER, "south")
        L(f"south edge -> {r} (now {tv.map_id(b)})")
    if tuple(tv.map_id(b)) == LAVENDER:
        camp.on_event("Lavender Town... the little town with the tower. we made it through the dark.",
                      kind="route", tier=2)
        hr = camp.heal_nearest()
        L(f"Lavender heal -> {hr} (map {tv.map_id(b)})")

    # ── 5) bank ────────────────────────────────────────────────────────────────────────
    _stage_save("final")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK, ignore_errors=True)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    import sanctity
    ok, issues = sanctity.validate_bundle(BANK, prev_dir=CANON, log=print)
    L(f"sanctity: {'VALID' if ok else f'INVALID {issues}'}")
    L(f"final: map={tv.map_id(b)} coords={tv.coords(b)} dex={ram.pokedex_owned_count(b)}")
    L(f"promote with: python pokemon_agent/promote_bank.py {BANK} rocktunnel_lavender")
    return 0


if __name__ == "__main__":
    sys.exit(main())
