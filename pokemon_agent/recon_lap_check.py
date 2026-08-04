"""OFFLINE smoke check for the VICTORY LAP checklist (2026-08-04) — no emulator, no ROM.

Stubs the bridge + state readers and drives the pure sequencing logic:
  1. _victory_lap_next walks the EXPLICIT order (earthquake -> moltres -> articuno -> eevee
     -> zapdos) as items complete, and reads CLEAR at the end;
  2. honest skips latch (Bill-gone kills moltres; boulder-flag kills articuno; TM26 missing
     kills earthquake) and the bounded-fail counter skips a wedged item;
  3. _lap_eq_forget_idx never sacrifices a protected move (Surf/Ice Beam/HMs) and prefers
     charge/status junk over real attacks; all-protected refuses with 'no_room'.
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
    check("then moltres", camp._victory_lap_next() == "moltres")
    WORLD["flags"].add(0x2BD)                                  # moltres fought
    check("then articuno", camp._victory_lap_next() == "articuno")
    WORLD["owned"][144] = True                                 # articuno caught
    check("then eevee", camp._victory_lap_next() == "eevee")
    WORLD["flags"].add(0x263)                                  # FLAG_GOT_EEVEE
    check("then zapdos", camp._victory_lap_next() == "zapdos")
    WORLD["flags"].add(0x05D)                                  # zapdos hidden (battled away)
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

    ok = all(PASS)
    print(f"== {'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(PASS)}/{len(PASS)}) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
