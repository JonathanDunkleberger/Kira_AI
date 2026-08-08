"""OFFLINE smoke check for the VICTORY LAP checklist (2026-08-04) — no emulator, no ROM.

Stubs the bridge + state readers and drives the pure sequencing logic:
  1. _victory_lap_next walks the EXPLICIT order (earthquake -> box_bench -> moltres ->
     articuno -> zapdos -> repack) as items complete, and reads CLEAR at the end;
  2. honest skips latch (Bill-gone kills moltres; TM26 missing kills earthquake) and the
     bounded-fail counter skips a wedged item; B4F-current-live does NOT skip articuno
     (ensure_b4f_calm owns the dam — soak 20260807_120648);
  3. _lap_eq_forget_idx never sacrifices a protected move (Surf/Ice Beam/HMs) and prefers
     charge/status junk over real attacks; all-protected refuses with 'no_room';
  4. (box flow, 2026-08-05) _lap_bench_plan sizes the deposit to exactly the owed
     join-items (skips shrink it), never the ace, never below BOX_BENCH_MIN_PARTY;
  5. POKEMON_BOX_FLOW=0 removes box_bench/repack from the checklist cleanly (kill switch);
  6. _lap_box_bench multi-deposits end-to-end (stubbed deposit_mon), re-deriving slots per
     deposit (menu-time order law); _lap_box_withdraw prefers lap birds, repack refills;
  7. _lap_restock_balls buys at Cinnabar's OWN shelf (the 16:19 loop root), marches only to
     RIDEABLE shelves, and a no-path march feeds the bounded fail counter -> honest skip;
  8. (2026-08-05, the Cinnabar door loop) the box_bench ONE-TRIP LATCH: a mid-lap catch
     regrowing the plan must NOT resurrect box_bench (the deposit<->catch shuttle from the
     07:29 session logs); a FAILED deposit does not latch (bounded retry preserved);
  9. (2026-08-05) FERRY-ONLY strike exhaustion: 3 spent Moltres tries surface
     questline_strike_failed (feeding VICTORY_LAP_MAX_FAILS) instead of falling through to
     the compass ("heading NORTH toward One Island" = surfing circles on Route 21); a
     road-reachable strike still falls through; map progress refunds a try (bounded);
 10. (2026-08-05, the One-Island teleport-back) the REGION-PARTITIONED reload law:
     map_region splits every map Kanto|Sevii; a Sevii wedge REFUSES the Kanto recent-good
     (the 08:50 cross-sea teleport) and falls to the newest SAME-REGION disk checkpoint;
     an in-region hatch still fires (bound, don't blind); the gain guard is untouched;
     a reload clears the in-memory strike try-counters (the moltres-outranked desync);
     the deep-wedge ring excludes cross-region seams; NEW-AREA GRACE reads >0 only on a
     recently-entered NAMELESS map (virgin territory = progress, not a wedge).
Run:  python3 recon_lap_check.py   (from pokemon_agent/) — prints PASS/FAIL per check.
"""
import sys
import types

# Mac dev box has no mgba/emulator stack — stub it so campaign imports (logic-only test).
_mgba = types.ModuleType("mgba")
for _sub in ("core", "image", "log", "vfs"):
    _mod = types.ModuleType(f"mgba.{_sub}")
    sys.modules[f"mgba.{_sub}"] = _mod
    setattr(_mgba, _sub, _mod)
_mgba.log.silence = lambda *a, **k: None
sys.modules["mgba"] = _mgba

import campaign as C
import pokemon_state as st
import firered_ram as ram
import field_moves as fm
import hm_teach as ht

PASS = []
WORLD = {}          # the ONE mutable world every patched reader closes over

SURF, ICE_BEAM, EQ, STRENGTH = 57, 58, 89, 70
BITE, WITHDRAW, SKULL_BASH, TACKLE = 44, 110, 130, 33
MOVE_DATA = {SURF: ("water", 95), ICE_BEAM: ("ice", 95), BITE: ("dark", 60),
             WITHDRAW: ("water", 0), SKULL_BASH: ("normal", 100),
             TACKLE: ("normal", 35), STRENGTH: ("normal", 80), EQ: ("ground", 100)}


def check(name, cond):
    PASS.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def set_world(**kw):
    WORLD.clear()
    WORLD.update({"moves": [], "owned": {}, "flags": set(), "tm26_row": 3,
                  "eq_compat": True})
    WORLD.update(kw)


def make_camp():
    camp = C.Campaign.__new__(C.Campaign)
    camp.b = types.SimpleNamespace(rd8=lambda a: 0)
    camp.on_event = lambda *a, **k: None
    camp._lap_skipped = set()
    camp._lap_fails = {}
    camp._lap_ace_slot = lambda: 0
    camp._lap_sevii_stranded = lambda: False
    return camp


def set_party(camp, levels):
    """Wire a live-RAM party into the stub bridge: count + per-slot level reads resolve
    against this (mutable) list, exactly the addresses _lap_bench_plan reads."""

    def rd8(a):
        if a == ram.GPLAYER_PARTY_CNT:
            return len(levels)
        off = a - ram.GPLAYER_PARTY
        if off >= 0 and off % st.PARTY_MON_SIZE == 0x54:
            s = off // st.PARTY_MON_SIZE
            if s < len(levels):
                return levels[s]
        return 0

    camp.b.rd8 = rd8
    return levels


def main():
    # one-time reader patches, all closing over WORLD
    st.read_party_moves = lambda b, slot=0: list(WORLD["moves"])
    st.read_party_species = lambda b, slot=0: 9                # Blastoise
    st.move_info = lambda b, m: MOVE_DATA.get(m, ("?", 50))
    ram.pokedex_owns = lambda b, sp: WORLD["owned"].get(sp, False)
    fm.read_flag = lambda b, flag: flag in WORLD["flags"]
    ht.tm_case_row = lambda b, item: WORLD["tm26_row"]
    ht.tm_compatible = lambda b, tm, sp: WORLD["eq_compat"]

    print("== 1. explicit order walk ==")
    set_world(moves=[SURF, BITE, WITHDRAW, SKULL_BASH])
    camp = make_camp()
    check("EQ first", camp._victory_lap_next() == "earthquake")
    WORLD["moves"] = [SURF, EQ, WITHDRAW, SKULL_BASH]          # taught
    set_party(camp, [61, 19, 19, 18, 25, 3])                   # the full 6/6 canonical party
    check("then box_bench (full party, 3 owed joins — birds only, no Eevee)",
          camp._victory_lap_next() == "box_bench")
    set_party(camp, [61, 25])                                  # passengers deposited
    check("then moltres", camp._victory_lap_next() == "moltres")
    WORLD["flags"].add(0x2BD)                                  # moltres fought
    WORLD["flags"].add(0x2D3)                                  # B4F current stopped (Articuno ok)
    check("then articuno", camp._victory_lap_next() == "articuno")
    WORLD["owned"][144] = True                                 # articuno caught
    check("then zapdos (Eevee is NOT on the credits-first lap)",
          camp._victory_lap_next() == "zapdos")
    WORLD["flags"].add(0x05D)                                  # zapdos hidden (battled away)
    set_party(camp, [61, 25, 50, 50])                          # a hunt skipped -> party of 4
    camp._lap_deposited = 4
    check("repack owed (deposited earlier, party short of six)",
          camp._victory_lap_next() == "repack")
    set_party(camp, [61, 25, 50, 50, 50, 25])                  # party whole again
    check("checklist CLEAR", camp._victory_lap_next() is None)
    check("eevee is not a lap checklist key",
          "eevee" not in C.VICTORY_LAP_ORDER)

    print("== 2. honest skips ==")
    # Bill gone + B4F current live (0x2D3 unset, B4F boulder hides SET = absent).
    # Bill-gone still skips moltres; B4F rip must NOT skip articuno (hunt can calm it).
    set_world(moves=[SURF, EQ, WITHDRAW, SKULL_BASH],
              flags={0x0A2, 0x04C, 0x04D})
    camp2 = make_camp()
    nxt = camp2._victory_lap_next()
    check("Bill-gone skips moltres; B4F-rip keeps articuno pending (not zapdos)",
          nxt == "articuno"
          and "moltres" in camp2._lap_skipped
          and "articuno" not in camp2._lap_skipped
          and camp2._lap_pending("articuno") is True)
    # Standing on Seafoam B4F with the rip live: proximity trump stays on articuno —
    # never the soak 20260807_120648 skip↔unskip → Zapdos divert.
    _orig_map2 = C.tv.map_id
    C.tv.map_id = lambda b: (1, 87)
    camp2b = make_camp()
    camp2b._lap_skipped, camp2b._lap_fails = set(), {}
    camp2b._lap_verdict_logged = None
    nxt_b4f = camp2b._victory_lap_next()
    C.tv.map_id = _orig_map2
    check("on Seafoam B4F with 0x2D3 clear -> articuno NEXT (proximity + no B4F skip)",
          nxt_b4f == "articuno"
          and "articuno" not in camp2b._lap_skipped
          and camp2b._lap_pending("articuno") is True)
    # Gate must ARM while the rip is live so FIRE-FIRST can run ensure_b4f_calm
    # (old gate returned None → victory_lap fail-counted Articuno into a skip).
    _orig_knows = st.party_knows_move
    st.party_knows_move = lambda b, move, cnt=6: 0 if move in (SURF, STRENGTH) else None
    camp2g = make_camp()
    camp2g._hunt_ready = lambda *a, **k: None
    camp2g.b.rd8 = lambda a: 2
    gate2 = camp2g._articuno_gate(None)
    st.party_knows_move = _orig_knows
    check("articuno gate ARMS with Surf+Strength even when 0x2D3 clear",
          gate2 is not None and getattr(gate2, "missing", None) == "articuno")
    for _ in range(C.VICTORY_LAP_MAX_FAILS):
        camp2._lap_note_fail("zapdos", "unit-test wedge")
    check("bounded fails latch the skip",
          "zapdos" in camp2._lap_skipped and not camp2._lap_pending("zapdos"))
    WORLD["moves"] = [SURF, BITE, WITHDRAW, SKULL_BASH]        # EQ not known...
    WORLD["tm26_row"] = None                                   # ...and TM26 not in the case
    check("no TM26 in case -> earthquake honestly skipped",
          camp2._lap_pending("earthquake") is False and "earthquake" in camp2._lap_skipped)

    print("== 3. forget-idx protection ==")
    set_world(moves=[SURF, BITE, WITHDRAW, SKULL_BASH])
    camp3 = make_camp()
    check("charge move (Skull Bash) sacrificed before status/attacks",
          camp3._lap_eq_forget_idx(0) == 3)
    WORLD["moves"] = [SURF, ICE_BEAM, BITE, WITHDRAW]
    check("status junk (Withdraw) over a real attack; Surf/IceBeam untouchable",
          camp3._lap_eq_forget_idx(0) == 3)
    WORLD["moves"] = [SURF, ICE_BEAM, BITE, TACKLE]
    check("weakest chip (Tackle) when only attacks remain",
          camp3._lap_eq_forget_idx(0) == 3)
    WORLD["moves"] = [SURF, ICE_BEAM, STRENGTH, EQ]
    check("all four protected -> no_room (teach refuses)",
          camp3._lap_eq_forget_idx(0) == "no_room")
    WORLD["moves"] = [SURF, ICE_BEAM, BITE]
    check("free slot -> None (no forget needed)", camp3._lap_eq_forget_idx(0) is None)

    print("== 4. bench-plan sizing (box flow) ==")
    set_world(moves=[SURF, EQ, WITHDRAW, SKULL_BASH])          # EQ done -> joins drive the plan
    camp4 = make_camp()
    set_party(camp4, [61, 19, 19, 18, 25, 3])
    check("full party + 3 owed joins -> 3 passengers, lowest level first, ace untouched",
          camp4._lap_bench_plan() == [5, 3, 1])
    camp4._lap_skipped.add("moltres")                          # a skip shrinks the deposit
    check("moltres skipped -> only 2 seats needed", camp4._lap_bench_plan() == [5, 3])
    camp4._lap_skipped.clear()
    set_party(camp4, [61, 10, 12])                             # 3 free seats already
    check("3 free seats + 3 joins -> deposit nothing", camp4._lap_bench_plan() == [])
    set_party(camp4, [61, 10])
    check("min-party floor: a 2-mon party never deposits", camp4._lap_bench_plan() == [])
    set_party(camp4, [61, 50, 48, 3])                          # high-level bench is NOT chaff
    check("only sub-L22 passengers qualify", camp4._lap_bench_plan() == [3])

    print("== 5. POKEMON_BOX_FLOW kill switch ==")
    camp5 = make_camp()
    set_party(camp5, [61, 19, 19, 18, 25, 3])
    camp5._lap_deposited = 2
    _saved = C.BOX_FLOW_ENABLED
    C.BOX_FLOW_ENABLED = False
    check("box_bench off the checklist when disabled",
          not camp5._lap_pending("box_bench") and camp5._victory_lap_next() == "moltres")
    check("repack off the checklist when disabled", not camp5._lap_pending("repack"))
    C.BOX_FLOW_ENABLED = _saved
    check("re-enabled -> box_bench owed again", camp5._lap_pending("box_bench"))

    print("== 6. multi-deposit end-to-end + withdraw preference ==")
    C.tv.map_id = lambda b: WORLD.get("here", (3, 8))          # standing on Cinnabar
    camp6 = make_camp()
    WORLD["here"] = (3, 8)
    levels6 = set_party(camp6, [61, 19, 19, 18, 25, 3])
    camp6.world = types.SimpleNamespace(name=lambda m: str(m))
    camp6._swap_party_slots = lambda i, j: None
    camp6._box_scan = lambda: (0, {})
    deposited = []

    def _fake_deposit(slot, pc_door):
        deposited.append((slot, pc_door))
        levels6.pop(slot)                                      # the party shifts up, like FRLG
        return "deposited"

    camp6.deposit_mon = _fake_deposit
    r6 = camp6._lap_box_bench({"map": (3, 8)})
    check("box_bench deposits for 3 bird seats (live re-derived slots)",
          r6 == "ok" and len(deposited) == 3 and levels6 == [61, 19, 25]
          and camp6._lap_deposited == 3)
    check("box_bench reads DONE after the deposits", not camp6._lap_pending("box_bench"))
    check("every deposit aimed at the Cinnabar Center PC",
          all(d == C.CITY_PC_DOORS[(3, 8)] for _s, d in deposited))
    boxed = {(0, 0): 21, (0, 2): 146}                          # a Spearow + an auto-boxed Moltres
    camp6._box_scan = lambda: (0, dict(boxed))
    pulls = []

    def _fake_withdraw(box, slot, pc_door):
        pulls.append((box, slot))
        boxed.pop((box, slot))
        levels6.append(50)
        return "withdrawn"

    camp6.withdraw_mon = _fake_withdraw
    got = camp6._lap_box_withdraw(C.CITY_PC_DOORS[(3, 8)], targets_only=True)
    check("targets-only withdraw pulls the boxed bird, leaves the chaff",
          got == 1 and pulls == [(0, 2)] and (0, 0) in boxed)
    got2 = camp6._lap_box_withdraw(C.CITY_PC_DOORS[(3, 8)], targets_only=False)
    check("repack withdraw refills from the remaining bench", got2 == 1 and pulls[-1] == (0, 0))

    print("== 7. hunt restock: Cinnabar shelf + rideable march + bounded fails ==")
    check("Cinnabar mart KB landed (pret rows, ball tiers first)",
          C.MART_STOCK.get(C.CINNABAR, [])[:2] == [2, 3]
          and C.CITY_MART_DOORS.get(C.CINNABAR) == C.CINNABAR_MART_DOOR)
    camp7 = make_camp()
    camp7._balls_pocket_count = lambda i: 0
    camp7.money = lambda: 35000
    buys = []
    camp7.buy_at_mart = lambda door, want: buys.append((door, want)) or {want[0][0]: 8}
    WORLD["here"] = (3, 8)
    r7 = camp7._lap_restock_balls({"map": (3, 8)}, "moltres")
    check("standing on Cinnabar -> buys Ultra Balls at HER OWN mart (no march)",
          r7 == "ok" and buys and buys[0][0] == C.CINNABAR_MART_DOOR
          and buys[0][1][0][0] == 2 and not camp7._lap_fails)
    WORLD["here"] = (3, 38)                                    # Route 20 west pocket
    camp8 = make_camp()
    camp8._balls_pocket_count = lambda i: 0
    camp8.money = lambda: 35000
    camp8._next_step_rideable = lambda cur, dst, avoid: ("hop" if dst == C.CINNABAR else None)
    marched = []
    camp8._travel_to_known = (lambda pick, state, hunt_on_arrival=True:
                              marched.append(pick) or "hop_ok")
    r8 = camp8._lap_restock_balls({"map": (3, 38)}, "moltres")
    check("on Route 20 -> marches to the RIDEABLE shelf (Cinnabar), never severed Fuchsia",
          r8 == "hop_ok" and marched == ["travel:3,8"] and not camp8._lap_fails)
    camp9 = make_camp()
    camp9._balls_pocket_count = lambda i: 0
    camp9.money = lambda: 35000
    camp9._next_step_rideable = lambda cur, dst, avoid: "hop"
    camp9._travel_to_known = lambda pick, state, hunt_on_arrival=True: "travel:no_path"
    for _ in range(C.VICTORY_LAP_MAX_FAILS):
        camp9._lap_restock_balls({"map": (3, 38)}, "moltres")
    check("a no-path restock march now feeds the bounded counter -> honest skip "
          "(the 16:19 infinite loop is structurally dead)",
          "moltres" in camp9._lap_skipped)

    print("== 8. the deposit<->catch SHUTTLE latch (2026-08-05 Cinnabar door loop) ==")
    # THE LOGGED FAILURE (tail_supervisor 07-31-11): box_bench completes -> moltres arms ->
    # a misheard catch_now lands a Route-21 tentacool -> the plan REGROWS -> box_bench flips
    # back to 'pending' -> the lap marches her back into the Center. Forever.
    set_world(moves=[SURF, EQ, WITHDRAW, SKULL_BASH], here=(3, 8))
    camp10 = make_camp()
    levels10 = set_party(camp10, [61, 19, 19, 18, 25, 3])
    camp10.world = types.SimpleNamespace(name=lambda m: str(m))
    camp10._swap_party_slots = lambda i, j: None
    camp10._box_scan = lambda: (0, {})

    def _dep10(slot, pc_door):
        levels10.pop(slot)
        return "deposited"

    camp10.deposit_mon = _dep10
    r10 = camp10._lap_box_bench({"map": (3, 8)})
    check("bench trip completes and LATCHES done",
          r10 == "ok" and getattr(camp10, "_lap_bench_done", False)
          and not camp10._lap_pending("box_bench"))
    levels10.append(8)                     # the Route-21 tentacool (a misheard catch_now)
    check("mid-lap catch REGROWS the raw plan (the old pending trigger)",
          bool(camp10._lap_bench_plan()))
    check("...but the LATCH holds: box_bench stays done, next is moltres (shuttle dead)",
          not camp10._lap_pending("box_bench")
          and camp10._victory_lap_next() == "moltres")
    camp11 = make_camp()
    set_party(camp11, [61, 19, 19, 18, 25, 3])
    camp11.world = types.SimpleNamespace(name=lambda m: str(m))
    camp11._swap_party_slots = lambda i, j: None
    camp11._box_scan = lambda: (0, {})
    camp11.deposit_mon = lambda slot, pc_door: "menu_wedge"
    camp11._lap_box_bench({"map": (3, 8)})
    check("a FAILED deposit does NOT latch (bounded retry preserved)",
          not getattr(camp11, "_lap_bench_done", False)
          and camp11._lap_pending("box_bench") and camp11._lap_fails.get("box_bench"))

    print("== 9. FERRY-ONLY strike exhaustion (Moltres) -> bounded lap failure, never compass ==")
    import legendary_strikes as LS
    C.tv.coords = lambda b: (7, 7)                             # strike result-log read
    logs9 = []
    _oldlog, C.log = C.log, lambda s: logs9.append(str(s)) or _oldlog(s)
    try:
        succ_m = ("flag", "FLAG_FOUGHT_MOLTRES")
        stepM = types.SimpleNamespace(success=succ_m, door=None)
        WORLD["here"] = (3, 8)                                 # Cinnabar — a Moltres anchor
        campF = make_camp()
        campF._ql_strike_tries_map = {succ_m: 3}
        rF = campF._questline_strike(stepM)
        check("exhausted FERRY-ONLY strike surfaces questline_strike_failed "
              "(feeds VICTORY_LAP_MAX_FAILS; the compass surf-circles are structurally dead)",
              rF == "questline_strike_failed"
              and any("FERRY-ONLY" in ln for ln in logs9))
        succ_z = ("flag", "FLAG_FOUGHT_ZAPDOS")
        stepZ = types.SimpleNamespace(success=succ_z, door=None)
        WORLD["here"] = next(iter(LS.ZAPDOS_ANCHORS))
        campZ = make_camp()
        campZ._ql_strike_tries_map = {succ_z: 3}
        rZ = campZ._questline_strike(stepZ)
        check("exhausted ROAD-reachable strike still falls through to the general layer",
              rZ is None and any("attempts exhausted" in ln and "FERRY-ONLY" not in ln
                                 for ln in logs9))
        # map-progress refund: a 'failed' hunt that SAILED somewhere keeps its try
        WORLD["here"] = (3, 8)
        campR = make_camp()
        campR._ql_strike_tries_map = {}
        _old_moltres = LS.run_moltres

        def _fake_moltres(camp, log, dbg_dir=None):
            WORLD["here"] = next(m for m in LS.MOLTRES_ANCHORS if m != (3, 8))
            return "failed"

        LS.run_moltres = _fake_moltres
        try:
            rR = campR._questline_strike(stepM)
        finally:
            LS.run_moltres = _old_moltres
        check("a failed hunt with MAP PROGRESS refunds the try (resumes from the new anchor)",
              rR == "questline_strike_failed"
              and campR._ql_strike_tries_map.get(succ_m) == 0
              and campR._ql_strike_refunds.get(succ_m) == 1
              and any("MAP PROGRESS" in ln for ln in logs9))
    finally:
        C.log = _oldlog

    print("== 10. REGION-PARTITIONED reload law + new-area grace (the teleport-back fix) ==")
    import json as _json
    import os as _os
    import tempfile as _tf
    import time as _tm
    check("map_region: partition ground truth (pret map_groups.json)",
          C.map_region((3, 8)) == "kanto" and C.map_region((3, 44)) == "kanto"
          and C.map_region((3, 12)) == "sevii" and C.map_region((3, 45)) == "sevii"
          and C.map_region((1, 95)) == "kanto" and C.map_region((1, 96)) == "sevii"
          and C.map_region((2, 0)) == "sevii" and C.map_region((32, 0)) == "sevii"
          and C.map_region((12, 5)) == "kanto" and C.map_region((34, 1)) == "sevii")

    def make_reload_camp(tmp, here, ckpts=()):
        """A camp wedged at `here`, with a checkpoints/ tree of (name, map, blob) bundles."""
        cw = make_camp()
        WORLD["here"] = here
        loaded = []
        cw.b.save_state = lambda: b"LIVE"
        cw.b.load_state = lambda s: loaded.append(bytes(s))
        cw._gain_sig = lambda: (8, 2, 15, 86)
        cw._save_campaign = lambda reason="t": True
        cw._wait_overworld = lambda *a, **k: True
        cw._map_first_seen = {}
        cw._region_reload_skips = 0
        cw._ql_strike_tries_map = {("flag", "FLAG_FOUGHT_MOLTRES"): 3}
        cw._ql_strike_refunds = {("flag", "FLAG_FOUGHT_MOLTRES"): 2}
        root = _os.path.join(tmp, "checkpoints")
        for name, m, blob in ckpts:
            d = _os.path.join(root, name)
            _os.makedirs(d, exist_ok=True)
            with open(_os.path.join(d, "checkpoint.json"), "w", encoding="utf-8") as f:
                _json.dump({"map": list(m)}, f)
            with open(_os.path.join(d, C.CAMPAIGN_SAVE), "wb") as f:
                f.write(blob)
        return cw, loaded

    _oldlog10, logs10 = C.log, []
    C.log = lambda s: logs10.append(str(s)) or _oldlog10(s)
    _old_states = C.STATES_CAMPAIGN
    try:
        with _tf.TemporaryDirectory() as tmp:
            C.STATES_CAMPAIGN = tmp
            # a) Sevii wedge + Kanto recent-good -> REFUSED; newest SEVII disk bank loads instead
            camp_a, loaded_a = make_reload_camp(
                tmp, (32, 0),
                ckpts=[("20260805_090000_cinnabar_8b_periodic", (3, 8), b"KANTO-NEW"),
                       ("20260805_085019_unfam_8b_periodic", (32, 0), b"SEVII-MID"),
                       ("20260805_084818_unfam_8b_gain", (32, 0), b"SEVII-OLD")])
            camp_a._last_good_state = b"KANTO-GOOD"
            camp_a._last_good_gain = (8, 2, 15, 86)
            camp_a._last_good_map = (3, 8)
            r_a = camp_a._escape_hatch_reload()
            check("Sevii wedge REFUSES the Kanto recent-good (the cross-sea teleport is dead)",
                  b"KANTO-GOOD" not in loaded_a
                  and any("ACROSS THE SEA" in ln for ln in logs10))
            check("...and reloads the NEWEST same-region (Sevii) disk checkpoint instead",
                  r_a is True and loaded_a == [b"SEVII-MID"])
            check("a reload CLEARS the strike try/refund memory (the moltres-outranked desync)",
                  camp_a._ql_strike_tries_map == {} and camp_a._ql_strike_refunds == {})
            check("the reloaded moment re-anchors as the new SAME-REGION recent-good",
                  camp_a._last_good_map == (32, 0))
            # b) walk-back: a re-wedge skips the already-tried newest bank; GREEN re-arms
            r_b = camp_a._reload_same_region_checkpoint("sevii")
            check("a re-wedge walks FURTHER BACK (skip depth) through the same-region banks",
                  r_b is True and loaded_a[-1] == b"SEVII-OLD"
                  and camp_a._region_reload_skips == 2)
            # c) in-region wedge: the hatch still fires normally (bound, don't blind)
            camp_c, loaded_c = make_reload_camp(tmp, (3, 8))
            camp_c._last_good_state = b"KANTO-GOOD"
            camp_c._last_good_gain = (8, 2, 15, 86)
            camp_c._last_good_map = (3, 1)
            check("an IN-REGION wedge still fires the hatch (its real purpose intact)",
                  camp_c._escape_hatch_reload() is True and loaded_c == [b"KANTO-GOOD"]
                  and camp_c._ql_strike_tries_map == {})
            # d) gain guard untouched: a real gain since the bank refuses ANY rewind
            camp_d, loaded_d = make_reload_camp(tmp, (3, 8))
            camp_d._last_good_state = b"KANTO-GOOD"
            camp_d._last_good_gain = (7, 2, 15, 80)     # live gain (8,...) EXCEEDS this
            camp_d._last_good_map = (3, 8)
            check("the GAIN GUARD is untouched (never rewind past a badge/teammate/catch)",
                  camp_d._escape_hatch_reload() is False and loaded_d == [])
            # e) no same-region bank at all -> honest decline (the ladder escalates loudly)
            with _tf.TemporaryDirectory() as tmp2:
                C.STATES_CAMPAIGN = tmp2
                camp_e, loaded_e = make_reload_camp(
                    tmp2, (32, 0),
                    ckpts=[("20260805_090000_cinnabar_8b_periodic", (3, 8), b"KANTO-NEW")])
                camp_e._last_good_state = b"KANTO-GOOD"
                camp_e._last_good_gain = (8, 2, 15, 86)
                camp_e._last_good_map = (3, 8)
                check("no same-region bank -> honest decline, NOTHING cross-region loaded",
                      camp_e._escape_hatch_reload() is False and loaded_e == [])
            C.STATES_CAMPAIGN = tmp
            # f) deep-wedge ring: cross-region gain-seams are excluded from the revert walk
            camp_f, loaded_f = make_reload_camp(tmp, (32, 0), ckpts=[])
            camp_f._deepwedge_reverts = 0
            camp_f._safe_ring = [
                {"state": b"RING-KANTO", "gain": (7, 2, 14, 80), "map": (3, 8), "label": "k"},
                {"state": b"RING-SEVII", "gain": (8, 2, 15, 86), "map": (32, 0), "label": "s"},
                {"state": b"RING-KANTO2", "gain": (8, 2, 15, 86), "map": (3, 3), "label": "k2"}]
            check("deep-wedge revert picks the SAME-REGION seam (cross-region excluded)",
                  camp_f._deep_wedge_revert() is True and loaded_f == [b"RING-SEVII"]
                  and any("cross-region" in ln for ln in logs10))
    finally:
        C.STATES_CAMPAIGN = _old_states
        C.log = _oldlog10

    print("== 11. NEW-AREA GRACE (arriving somewhere never-visited is progress) ==")
    campg = make_camp()
    campg._map_first_seen = {(32, 0): _tm.time()}
    WORLD["here"] = (32, 0)                      # nameless in _PLACE_NAMES -> virgin ground
    check("fresh NAMELESS map -> grace holds (reload rungs stand down)",
          campg._new_area_grace_left() > 0)
    campg._map_first_seen[(32, 0)] = _tm.time() - C.NEW_AREA_GRACE_S - 1
    check("grace EXPIRES (bounded: the breakers re-arm; never blinded)",
          campg._new_area_grace_left() == 0.0)
    campg._map_first_seen[(3, 8)] = _tm.time()
    WORLD["here"] = (3, 8)                       # Cinnabar has a NAME -> charted ground
    check("a NAMED map gets no grace (charted ground keeps full breaker coverage)",
          campg._new_area_grace_left() == 0.0)

    ok = all(PASS)
    print(f"== {'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(PASS)}/{len(PASS)}) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
