"""OFFLINE smoke check for the VICTORY LAP checklist (2026-08-04) — no emulator, no ROM.

Stubs the bridge + state readers and drives the pure sequencing logic:
  1. _victory_lap_next walks the EXPLICIT order (earthquake -> box_bench -> moltres ->
     articuno -> eevee -> zapdos -> repack) as items complete, and reads CLEAR at the end;
  2. honest skips latch (Bill-gone kills moltres; boulder-flag kills articuno; TM26 missing
     kills earthquake) and the bounded-fail counter skips a wedged item;
  3. _lap_eq_forget_idx never sacrifices a protected move (Surf/Ice Beam/HMs) and prefers
     charge/status junk over real attacks; all-protected refuses with 'no_room';
  4. (box flow, 2026-08-05) _lap_bench_plan sizes the deposit to exactly the owed
     join-items (skips shrink it), never the ace, never below BOX_BENCH_MIN_PARTY;
  5. POKEMON_BOX_FLOW=0 removes box_bench/repack from the checklist cleanly (kill switch);
  6. _lap_box_bench multi-deposits end-to-end (stubbed deposit_mon), re-deriving slots per
     deposit (menu-time order law); _lap_box_withdraw prefers lap birds, repack refills;
  7. _lap_restock_balls buys at Cinnabar's OWN shelf (the 16:19 loop root), marches only to
     RIDEABLE shelves, and a no-path march feeds the bounded fail counter -> honest skip.
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
    check("then box_bench (full party, 4 owed joins)",
          camp._victory_lap_next() == "box_bench")
    set_party(camp, [61, 25])                                  # passengers deposited
    check("then moltres", camp._victory_lap_next() == "moltres")
    WORLD["flags"].add(0x2BD)                                  # moltres fought
    check("then articuno", camp._victory_lap_next() == "articuno")
    WORLD["owned"][144] = True                                 # articuno caught
    check("then eevee", camp._victory_lap_next() == "eevee")
    WORLD["flags"].add(0x263)                                  # FLAG_GOT_EEVEE
    check("then zapdos", camp._victory_lap_next() == "zapdos")
    WORLD["flags"].add(0x05D)                                  # zapdos hidden (battled away)
    set_party(camp, [61, 25, 50, 50])                          # a hunt skipped -> party of 4
    camp._lap_deposited = 4
    check("repack owed (deposited earlier, party short of six)",
          camp._victory_lap_next() == "repack")
    set_party(camp, [61, 25, 50, 50, 50, 25])                  # party whole again
    check("checklist CLEAR", camp._victory_lap_next() is None)

    print("== 2. honest skips ==")
    set_world(moves=[SURF, EQ, WITHDRAW, SKULL_BASH],
              flags={0x0A2, 0x046})                            # Bill gone + a Seafoam boulder up
    camp2 = make_camp()
    nxt = camp2._victory_lap_next()
    check("Bill-gone skips moltres, boulder skips articuno -> eevee next",
          nxt == "eevee" and {"moltres", "articuno"} <= camp2._lap_skipped)
    for _ in range(C.VICTORY_LAP_MAX_FAILS):
        camp2._lap_note_fail("eevee", "unit-test wedge")
    check("bounded fails latch the skip",
          "eevee" in camp2._lap_skipped and not camp2._lap_pending("eevee"))
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
    check("full party + 4 owed joins -> 4 passengers, lowest level first, ace untouched",
          camp4._lap_bench_plan() == [5, 3, 1, 2])
    camp4._lap_skipped.add("moltres")                          # a skip shrinks the deposit
    check("moltres skipped -> only 3 seats needed", camp4._lap_bench_plan() == [5, 3, 1])
    camp4._lap_skipped.clear()
    set_party(camp4, [61, 10, 12])                             # 3 free seats already
    check("3 free seats -> deposit just 1 (the weakest)", camp4._lap_bench_plan() == [1])
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
    check("box_bench deposits down to ace+Lapras (4 deposits, live re-derived slots)",
          r6 == "ok" and len(deposited) == 4 and levels6 == [61, 25]
          and camp6._lap_deposited == 4)
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

    ok = all(PASS)
    print(f"== {'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(PASS)}/{len(PASS)}) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
