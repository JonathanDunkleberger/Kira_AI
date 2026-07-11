"""recon_road_bench_xp_check.py — decision-logic verifier for PASS-3 team-depth NEW#1 (ROAD BENCH XP).

Proves campaign._road_bench_xp_arm / _road_bench_xp_disarm make the right call WITHOUT the emulator, by
binding the real methods to a lightweight fake campaign and driving the inputs (pick / needs_heal /
prep target / party levels / stalled set / flags) across the scenarios that matter:

  1  non-march pick (wander_catch/heal/beat_gym)      -> NOT armed (weak mon must never lead a leader/catch)
  2  forward-march, bench already at milestone (None)  -> NOT armed
  3  forward-march, party hurt (needs_heal)            -> NOT armed (heal path wants the true ace up)
  4  forward-march, a levelable under-target bench mon -> ARMED: weak mon -> slot 0, PROTECT True;
                                                          disarm -> ace restored to slot 0, PROTECT False
  5  forward-march, only far-below box chaff under tgt -> NOT armed (chaff excluded, no faint-thrash)
  6  forward-march, a stall-marked weak mon only       -> NOT armed (can't earn XP on this route)
  7  flag POKEMON_ROAD_BENCH_XP=0                       -> NOT armed (instant revert honoured)
  8  weakest levelable mon IS already the ace          -> NOT armed (nothing to protect)

RUN:  ../.venv/Scripts/python.exe -u recon_road_bench_xp_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import campaign as C          # noqa: E402
import battle_agent as BA     # noqa: E402
import firered_ram as ram     # noqa: E402  (same module campaign uses for GPLAYER_PARTY)
import pokemon_state as st    # noqa: E402  (PARTY_MON_SIZE lives here — same module campaign uses)


class FakeBridge:
    """Just enough to serve the per-slot PID reads _road_bench_xp_arm does (addr -> slot index)."""
    def __init__(self, levels):
        self.n = len(levels)

    def rd32(self, addr):
        # campaign reads ram.GPLAYER_PARTY + slot * st.PARTY_MON_SIZE for each slot's personality value.
        slot = (addr - ram.GPLAYER_PARTY) // st.PARTY_MON_SIZE
        return 1000 + slot                          # unique, stable per slot


class FakeCamp:
    """Binds the REAL arm/disarm methods; stubs everything they call so the DECISION is under test."""
    _road_bench_xp_arm = C.Campaign._road_bench_xp_arm
    _road_bench_xp_disarm = C.Campaign._road_bench_xp_disarm

    def __init__(self, levels, hurt=False, prep_t=None, stalled=None):
        self._levels = list(levels)
        self._hurt = hurt
        self._prep_t = prep_t
        self._grind_stalled = set(stalled or ())
        self.b = FakeBridge(levels)
        self.swaps = []           # record of _swap_party_slots calls
        self.restored = 0         # count of _restore_ace calls

    # ---- stubs standing in for the real campaign helpers ----
    def needs_heal(self):
        return ("hurt", "…") if self._hurt else None

    def _prep_team_target(self, state):
        return self._prep_t

    def _party_levels(self):
        return list(self._levels)

    def _swap_party_slots(self, i, j):
        self.swaps.append((i, j))
        self._levels[i], self._levels[j] = self._levels[j], self._levels[i]   # reflect the swap

    def _restore_ace(self):
        self.restored += 1
        ace = max(range(len(self._levels)), key=lambda s: self._levels[s])
        if ace != 0:
            self._levels[0], self._levels[ace] = self._levels[ace], self._levels[0]


def _run(name, camp, pick, expect_arm, want_lead_level=None, expect_flag=None):
    BA.PROTECT_LEAD_GRIND = False
    slot0_before = camp._levels[0]
    armed = camp._road_bench_xp_arm(pick, {"party": [{"level": l} for l in camp._levels]})
    fails = []
    if armed != expect_arm:
        fails.append(f"armed={armed} expected {expect_arm}")
    if expect_arm:
        if BA.PROTECT_LEAD_GRIND is not True:
            fails.append("PROTECT_LEAD_GRIND not set True on arm")
        if want_lead_level is not None and camp._levels[0] != want_lead_level:
            fails.append(f"lead is L{camp._levels[0]}, expected the weak L{want_lead_level} to lead")
        # disarm must restore the ace to slot 0 and clear the flag
        camp._road_bench_xp_disarm()
        if BA.PROTECT_LEAD_GRIND is not False:
            fails.append("PROTECT_LEAD_GRIND not cleared on disarm")
        if camp._levels[0] != max(camp._levels):
            fails.append(f"disarm did not restore the ace to slot 0 (slot0=L{camp._levels[0]})")
    else:
        if BA.PROTECT_LEAD_GRIND is not False:
            fails.append("PROTECT_LEAD_GRIND set True when it should NOT have armed")
        if camp._levels[0] != slot0_before:
            fails.append("party order changed on a no-arm path")
    print(f"  [{'PASS' if not fails else 'FAIL'}] {name}: armed={armed} slot0=L{camp._levels[0]} "
          f"PROTECT={BA.PROTECT_LEAD_GRIND} swaps={camp.swaps} restored={camp.restored}")
    return fails


def main():
    # ensure the mechanism gates are open for the logic under test
    C.STRATEGIC_GRIND_ENABLED = True
    BA.GRIND_SWITCH_ENABLED = True
    C.ROAD_BENCH_XP_ENABLED = True
    C.E4_PREP_BAND = 25

    all_fails = []

    # 1 — non-march picks never arm
    for p in ("wander_catch", "heal", "beat_gym", "stock_up", "talk_npc"):
        all_fails += _run(f"non-march pick {p!r}", FakeCamp([32, 14, 14], prep_t=20), p, expect_arm=False)

    # 2 — bench at milestone (prep None)
    all_fails += _run("prep None (bench at milestone)", FakeCamp([32, 30, 30], prep_t=None),
                      "head_to_gym", expect_arm=False)

    # 3 — hurt party
    all_fails += _run("hurt party", FakeCamp([32, 14, 14], hurt=True, prep_t=20),
                      "head_to_gym", expect_arm=False)

    # 4 — the happy path: weak L14 leads, ace restored on disarm
    all_fails += _run("levelable bench under target", FakeCamp([32, 14, 18], prep_t=20),
                      "head_to_gym", expect_arm=True, want_lead_level=14)
    all_fails += _run("travel: leg arms too", FakeCamp([32, 14, 18], prep_t=20),
                      "travel:3,25", expect_arm=True, want_lead_level=14)

    # 5 — only far-below box chaff under target (L8 with target 55, band 25 -> floor 30) -> excluded
    all_fails += _run("only far-below chaff under target", FakeCamp([60, 58, 8], prep_t=55),
                      "head_to_gym", expect_arm=False)

    # 6 — the sole under-target weak mon is stall-marked (pid for slot 1 == 1001)
    c6 = FakeCamp([32, 14, 30], prep_t=20, stalled={1001})
    all_fails += _run("only-weak-mon stall-marked", c6, "head_to_gym", expect_arm=False)

    # 7 — flag off
    C.ROAD_BENCH_XP_ENABLED = False
    all_fails += _run("flag POKEMON_ROAD_BENCH_XP=0", FakeCamp([32, 14, 14], prep_t=20),
                      "head_to_gym", expect_arm=False)
    C.ROAD_BENCH_XP_ENABLED = True

    # 8 — weakest levelable IS the ace (single strong mon + only-chaff bench) -> nothing to protect
    all_fails += _run("weakest levelable is the ace", FakeCamp([40], prep_t=20),
                      "head_to_gym", expect_arm=False)

    print("\n" + ("ALL PASS" if not all_fails else "FAIL:\n  " + "\n  ".join(all_fails)))
    return 0 if not all_fails else 1


if __name__ == "__main__":
    sys.exit(main())
