"""recon_grind_pp_heal_check.py — decision-logic verifier for PASS-3 NS#25 team-depth lever:
the PP-FAMINE HEAL in campaign.Campaign.grind(). Proves, WITHOUT a live emulator, that the new inline
block (grind() ~campaign.py:6313) fires ONLY on the intended shape and is bounded, by binding the REAL
grind() to a lightweight fake bridge/campaign that stubs every helper grind() touches BEFORE my block,
and driving the trigger inputs across the cases that matter:

  1  fragile + flag ON + whole party PP-DRY, floor rising each call  -> 'ace_healed' every call (heal, re-grind)
  2  flag OFF (POKEMON_GRIND_PP_HEAL=0)                              -> NOT 'ace_healed' (byte-inert; falls through)
  3  NON-fragile (ace grind)                                         -> NOT 'ace_healed' (scoped to bench grinds)
  4  party STILL has damaging PP                                     -> NOT 'ace_healed' (nobody-can-attack only)
  5  whole party dry, floor FLAT across CAP+1 calls                  -> 'no_safe_grass' on the (CAP+1)th (stand-down)
  6  ACE-DOWN guard co-exists (ace dinged fires first, PP second)    -> ACE-DOWN wins when both true

In cases 1/5/6 my block is REACHED (ACE-DOWN skipped via _ace_hp_frac=1.0); a 'grass'-sentinel or 'ok' means
the block did NOT fire and control fell through to the grass loop (which the stubs short-circuit to 'ok').

RUN:  ../.venv/Scripts/python.exe -u recon_grind_pp_heal_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import campaign as C          # noqa: E402
import travel as tv           # noqa: E402


class FakeBridge:
    def __init__(self, level):
        self._level = level

    def rd8(self, addr):
        return self._level          # lvl() reads slot-0 level; keep it < target so grind() enters the loop


class FakeGrid:
    grass = [(5, 5), (6, 6)]        # non-empty so grind() never bails "no grass" before my block
    walkable = None                 # only touched on the fall-through grass path (stubbed via tv.bfs)


class FakeTrav:
    def travel(self, *a, **k):
        return "no_path"            # fall-through cases: bounded no_path stand-down (GRIND_NOPATH_CAP=1)


class FakeCamp:
    """Binds the REAL grind(); stubs everything it calls up to (and including the escape in) my block."""
    grind = C.Campaign.grind

    def __init__(self, level=30, party_pp=False, ace_frac=1.0, floors=None):
        self.b = FakeBridge(level)
        self._party_pp = party_pp        # what _party_damaging_pp() returns
        self._ace_frac = ace_frac
        self._floors = list(floors or [level, level, level, level, level, level])
        self._floor_calls = 0
        self.heals = 0
        self.restores = 0
        self.trav = FakeTrav()

    # ---- stubs for the helpers grind() calls before/at my block ----
    def _ensure_move_room(self):
        return None

    def _ace_hp_frac(self):
        return self._ace_frac

    def _party_damaging_pp(self):
        return self._party_pp

    def _party_levels(self):
        # each call returns the next scripted floor list (drives the streak reset / cap)
        f = self._floors[min(self._floor_calls, len(self._floors) - 1)]
        self._floor_calls += 1
        return [f, f + 20, f + 1, f, f + 1, f]

    def _restore_ace(self):
        self.restores += 1

    def heal_nearest(self):
        self.heals += 1

    def _door_tiles(self):
        return []

    def _grind_is_cave(self, m):
        return False


def _patch_tv():
    """Neutralize the map/grid reads grind() does before the loop so control reaches the while body."""
    tv.MAP_OFFSET = getattr(tv, "MAP_OFFSET", 7)
    tv.map_id = lambda b: (3, 28)          # a real non-PEWTER route
    tv.coords = lambda b: (1, 1)
    tv.Grid = lambda b: FakeGrid()
    tv.bfs = lambda grid, start, goal, walkable=None: True   # fall-through grass path: pretend reachable
    C.GRIND_NOPATH_CAP = 1                  # so a non-firing grind stands down fast (no infinite loop)


def run():
    _patch_tv()
    C.ACE_BAIL_ON = True
    C.GRIND_PP_HEAL_CAP = 4
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
        if not ok:
            fails.append(name)

    # 1 — fragile + flag ON + whole party dry, floor RISING each call -> heals + re-grinds ('ace_healed')
    C.GRIND_PP_HEAL_ON = True
    c1 = FakeCamp(level=30, party_pp=False, floors=[30, 31, 32, 33, 34, 35])
    r1 = c1.grind(40, fragile=True)
    check("1 whole-party-dry fragile -> heal", r1, "ace_healed")
    check("1 heal_nearest called", c1.heals, 1)
    check("1 ace restored", c1.restores, 1)

    # 2 — flag OFF -> block byte-inert, falls through to the (stubbed) grass loop -> 'ok'
    C.GRIND_PP_HEAL_ON = False
    c2 = FakeCamp(level=30, party_pp=False)
    r2 = c2.grind(40, fragile=True)
    check("2 flag OFF -> not healed", r2 != "ace_healed", True)
    check("2 flag OFF -> heal NOT called", c2.heals, 0)

    # 3 — NON-fragile (ace grind) -> scoped out even with flag ON + party dry
    C.GRIND_PP_HEAL_ON = True
    c3 = FakeCamp(level=30, party_pp=False)
    r3 = c3.grind(40, fragile=False)
    check("3 non-fragile -> not healed", r3 != "ace_healed", True)

    # 4 — party STILL has damaging PP -> block does not fire
    c4 = FakeCamp(level=30, party_pp=True)
    r4 = c4.grind(40, fragile=True)
    check("4 party has PP -> not healed", r4 != "ace_healed", True)
    check("4 party has PP -> heal NOT called", c4.heals, 0)

    # 5 — whole party dry, floor FLAT across CAP+1 calls -> stand-down 'no_safe_grass' on the (CAP+1)th
    #     Each grind() call fires once; streak lives on self across calls. Floor never rises -> streak climbs.
    c5 = FakeCamp(level=30, party_pp=False, floors=[30])   # _party_levels always returns floor 30 (flat)
    rr = None
    for i in range(C.GRIND_PP_HEAL_CAP + 1):
        rr = c5.grind(40, fragile=True)
    check("5 flat floor -> stand down at cap", rr, "no_safe_grass")
    check("5 stand-down healed exactly CAP times", c5.heals, C.GRIND_PP_HEAL_CAP)

    # 6 — ACE-DOWN guard PRECEDENCE: ace dinged AND party dry -> ACE-DOWN fires first (its own 'ace_healed'
    #     path), proving the PP block is placed AFTER it and never steals a genuine HP heal.
    c6 = FakeCamp(level=30, party_pp=False, ace_frac=0.10)   # ace at 10% -> below GRIND_ACE_BAIL_FRAC
    r6 = c6.grind(40, fragile=True)
    check("6 ace dinged -> ACE-DOWN heals (still 'ace_healed')", r6, "ace_healed")

    print()
    if fails:
        print(f"FAIL: {len(fails)} case(s): {fails}")
        sys.exit(1)
    print("ALL PASS — PP-famine heal fires on whole-party-dry fragile grinds, byte-inert OFF, "
          "scoped to bench grinds, bounded by the streak cap, and yields to the ACE-DOWN HP guard.")


if __name__ == "__main__":
    run()
