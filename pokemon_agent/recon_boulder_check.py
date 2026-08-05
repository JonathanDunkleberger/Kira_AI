"""OFFLINE smoke check for the BOULDER-PUZZLE chain engine + climb durability (2026-08-05 #3,
the Mt. Ember loop: 'she does one or two strength moves... and then just leaves the thing') —
no emulator, no ROM.

Drives boulder_puzzle.solve_room over a scripted boulder world and the campaign helpers:
  1. GEOMETRY: chain_path unrolls multi-segment chains tile-by-tile; the shipped Mt. Ember
     rooms land every boulder on its intended target;
  2. VICTORY ROAD PRE-DERIVATION: room_from_ops over the live victory_road VRnF_PUZZLE op
     tables chains EXACTLY (each push starts where the last one landed) and terminates on the
     switch tiles ((20,16) / (2,19) / (7,7)); a mis-chained table refuses at BUILD time;
  3. FRESH SOLVE: template board -> every chain pushed to target with live readback, ZERO map
     exits (fail-in-place law), a checkpoint after every push (ckpt_every=1);
  4. IDEMPOTENT RESUME — THE over-push regression: a half-pushed board resumes mid-chain and
     pushes ONLY the remainder; a fully-solved board verifies with ZERO presses (the old plan
     re-ran push rows through the radius-8 fuzz and shoved solved boulders further);
  5. THE CAMERA LAW: an off-camera boulder ('absent' in the scan) is never believed — the
     solver walks near the chain's path and re-scans before deciding anything;
  6. HONEST FAILURE LADDER: an UNVERIFIABLE look (approach failed) fails the chain LOUD; a
     verified-GONE boulder (off its whole path) reads as board divergence; both retry IN
     PLACE first, take the door-reset round-trip LAST (exactly one), and a template respawn
     after the reset solves clean;
  7. vanish_ok rows (the VR 3F hole-drop shape) verify-absent as DONE, not diverged;
  8. SUMMIT TURN: the (8,10) UP-then-RIGHTx2 chained segments execute in order, and the
     board's push ORDER vacates each stand tile before it is needed;
  9. _bank_milestone: one call banks save+continuity+auto-checkpoint AND refreshes the
     in-memory recent-good (state/gain/map/clock); mid-battle stands down; a checkpoint flake
     never raises (returns True, logs LOUD);
 10. strike_checkpoint: banks through the campaign AND refunds the lap's bounded-attempt
     counter for THIS quarry only (milestone == progress, the hunt can't honest-skip while
     genuinely advancing);
 11. _release_wedge_marks_on: drops every banked mark on the strike's map set regardless of
     age (menu-frozen phantoms), leaves other maps untouched, saves once, no-ops idempotently;
 12. _reload_same_region_checkpoint: the NEWEST same-region checkpoint wins (strike labels
     like 'pre-moltres' are never filtered), skip depth walks progressively further back, and
     wrong-region banks are never touched.
Run:  python3 recon_boulder_check.py   (from pokemon_agent/) — prints PASS/FAIL per check.
"""
import os
import sys
import tempfile
import time
import types

# Mac dev box has no mgba/emulator stack — stub it so campaign imports (logic-only test).
_mgba = types.ModuleType("mgba")
for _sub in ("core", "image", "log", "vfs"):
    _mod = types.ModuleType(f"mgba.{_sub}")
    sys.modules[f"mgba.{_sub}"] = _mod
    setattr(_mgba, _sub, _mod)
_mgba.log.silence = lambda *a, **k: None
sys.modules["mgba"] = _mgba

import boulder_puzzle as bpz
import campaign as C
import legendary_strikes as LS
import pokemon_state as st
import victory_road as VR

PASS = []
LOGS = []


def check(name, cond):
    PASS.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class FakeRig:
    """A scripted boulder world: pushes obey walls/collisions, sea_walk teleports to the first
    tile satisfying the goal predicate (or refuses for labels in walk_fail), live_boulders
    honors a per-axis visibility radius (THE CAMERA LAW's off-screen unload)."""

    def __init__(self, boulders, player=(28, 48), vis=6, walls=(), walk_fail=()):
        self.boulders = {tuple(t) for t in boulders}
        self.player = tuple(player)
        self.vis = vis
        self.walls = {tuple(t) for t in walls}
        self.walk_fail = tuple(walk_fail)
        self.pushes, self.exits, self.ckpts, self.allows = [], [], [], []
        self.strength_ok = True
        self.reset_board = None      # template respawned on the '-reset-back' door leg
        self.log = LOGS.append

    def live_boulders(self):
        px, py = self.player
        return [t for t in self.boulders
                if abs(t[0] - px) <= self.vis and abs(t[1] - py) <= self.vis]

    def sea_walk(self, pred, label, *a, **kw):
        if any(label.startswith(p) for p in self.walk_fail):
            return False
        if pred(self.player):
            return True
        for y in range(64):
            for x in range(64):
                if (x, y) not in self.boulders and (x, y) not in self.walls and pred((x, y)):
                    self.player = (x, y)
                    return True
        return False

    def push(self, approx, key, n, allow=()):
        if allow:
            self.allows.append(tuple(allow))
        d = bpz.DELTA[key]
        cur = tuple(approx)
        for _ in range(n):
            if cur not in self.boulders:
                return False
            stand = (cur[0] - d[0], cur[1] - d[1])
            if not self.sea_walk(lambda c, s=stand: c == s, "push-approach"):
                return False
            dest = (cur[0] + d[0], cur[1] + d[1])
            if dest in self.walls or dest in self.boulders:
                return False
            self.boulders.discard(cur)
            self.boulders.add(dest)
            self.player = cur
            self.pushes.append((cur, key))
            cur = dest
        return True

    def ensure_strength(self, approx):
        return self.strength_ok

    def handle_interrupts(self):
        return False

    def enter_step(self, tile, dest, label):
        self.exits.append(label)
        if label.endswith("-reset-back") and self.reset_board is not None:
            self.boulders = set(self.reset_board)
            self.player = tuple(tile)
        return True


EMBER_TEMPLATE = {(22, 45), (17, 46)}
EMBER_SOLVED = {(19, 45), (14, 46)}
SUMMIT_TEMPLATE = {(10, 12), (9, 12), (8, 11), (8, 10)}
SUMMIT_SOLVED = {(10, 11), (8, 12), (7, 11), (10, 9)}


def main():
    print("== 1. geometry: chain_path + the shipped Mt. Ember rooms ==")
    check("chain_path unrolls a turning chain tile-by-tile",
          bpz.chain_path((8, 10), [("UP", 1), ("RIGHT", 2)])
          == [(8, 10), (8, 9), (9, 9), (10, 9)])
    check("EMBER_ASCENT chains end on the corridor-clearing targets (19,45)/(14,46)",
          {ch["path"][-1] for ch in bpz.EMBER_ASCENT["chains"]} == EMBER_SOLVED)
    check("EMBER_SUMMIT_BOARD chains end parked clear of the Moltres corridor",
          {ch["path"][-1] for ch in bpz.EMBER_SUMMIT_BOARD["chains"]} == SUMMIT_SOLVED)
    check("EMBER_DESCENT works the FRESH template (re-entry reset by design)",
          {ch["start"] for ch in bpz.EMBER_DESCENT["chains"]} == EMBER_TEMPLATE)

    print("== 2. VICTORY ROAD pre-derivation (room_from_ops over the live op tables) ==")
    r1 = bpz.room_from_ops(VR.VR1F, "1f-switch", VR.VR1F_PUZZLE)
    r2 = bpz.room_from_ops(VR.VR2F, "2f-switch1", VR.VR2F_PUZZLE1)
    r3 = bpz.room_from_ops(VR.VR3F, "3f-switch", VR.VR3F_SWITCH_PUZZLE)
    check("VR1F chain lands the boulder ON the switch (20,16), 19 pushes",
          r1["chains"][0]["path"][-1] == (20, 16) and len(r1["chains"][0]["keys"]) == 19)
    check("VR2F chain lands ON switch 1 (2,19)", r2["chains"][0]["path"][-1] == (2, 19))
    check("VR3F chain lands ON the 3F switch (7,7), 33 pushes",
          r3["chains"][0]["path"][-1] == (7, 7) and len(r3["chains"][0]["keys"]) == 33)
    check("the (11,20) entrance-arrow allow tile rides its VR1F step",
          ((11, 20),) in r1["chains"][0]["allows"])
    try:
        bpz.room_from_ops((1, 39), "bad", [("push", (7, 18), "DOWN", 1),
                                           ("push", (9, 9), "LEFT", 2)])
        mis = False
    except ValueError:
        mis = True
    check("a MIS-CHAINED op table refuses at build time, never mid-climb", mis)

    print("== 3. fresh solve: template board -> targets, zero exits, ckpt-per-push ==")
    rig = FakeRig(EMBER_TEMPLATE | {(36, 14), (35, 14), (35, 17)}, player=(28, 48))
    ok = bpz.solve_room(rig, bpz.EMBER_ASCENT, checkpoint=rig.ckpts.append, log=LOGS.append)
    check("ascent board solved from template (6 verified pushes)",
          ok is True and len(rig.pushes) == 6
          and EMBER_SOLVED <= rig.boulders and not (EMBER_TEMPLATE & rig.boulders))
    check("fail-in-place law: a clean solve NEVER exits the map", rig.exits == [])
    check("checkpoint fired for every push + each chain completion (durability layer)",
          len(rig.ckpts) >= 6 and any("push" in c for c in rig.ckpts)
          and any("chain" in c for c in rig.ckpts))
    check("the far trio (other side of the mountain) was never touched",
          {(36, 14), (35, 14), (35, 17)} <= rig.boulders)

    print("== 4. IDEMPOTENT RESUME — the over-push regression ==")
    rig4 = FakeRig({(21, 45), (17, 46)}, player=(22, 45))     # chain 0 already 1 push in
    ok4 = bpz.solve_room(rig4, bpz.EMBER_ASCENT, log=LOGS.append)
    check("half-pushed board: resume mid-chain, push ONLY the remainder (2+3, not 3+3)",
          ok4 is True and len(rig4.pushes) == 5 and EMBER_SOLVED <= rig4.boulders)
    check("the mid-chain RESUME is logged (idempotent, never over-push)",
          any("RESUME mid-chain" in ln for ln in LOGS))
    rig4b = FakeRig(set(EMBER_SOLVED), player=(19, 44))
    ok4b = bpz.solve_room(rig4b, bpz.EMBER_ASCENT, log=LOGS.append)
    check("ALREADY-SOLVED board verifies with ZERO presses (the old plan shoved it further)",
          ok4b is True and rig4b.pushes == [] and rig4b.boulders == EMBER_SOLVED)

    print("== 5. THE CAMERA LAW: off-screen 'absent' is walked-to, never believed ==")
    rig5 = FakeRig(EMBER_TEMPLATE, player=(2, 2), vis=6)      # both boulders off-camera
    ok5 = bpz.solve_room(rig5, bpz.EMBER_ASCENT, log=LOGS.append)
    check("invisible template board: the look-walk finds it and the solve completes",
          ok5 is True and EMBER_SOLVED <= rig5.boulders)

    print("== 6. honest failure ladder: unverifiable -> in-place -> ONE door reset ==")
    LOGS.clear()
    rig6 = FakeRig(EMBER_TEMPLATE, player=(2, 2), vis=6,
                   walk_fail=("ext-ascent#0-look",))          # every look-approach refused
    ok6 = bpz.solve_room(rig6, bpz.EMBER_ASCENT, log=LOGS.append)
    check("UNVERIFIABLE boulder fails the chain LOUD (never 'assume pushed')",
          ok6 is False and any("UNVERIFIABLE" in ln for ln in LOGS))
    check("exactly ONE door-reset round-trip, taken LAST (round order: in-place first)",
          rig6.exits == ["ext-ascent-reset-out", "ext-ascent-reset-back"]
          and rig6.pushes == [])
    rig6b = FakeRig({(23, 45), (17, 46)}, player=(22, 44))    # chain 0 boulder OFF its path
    rig6b.reset_board = EMBER_TEMPLATE
    ok6b = bpz.solve_room(rig6b, bpz.EMBER_ASCENT, log=LOGS.append)
    check("verified-GONE (off-path) reads as divergence; the reset respawns the template "
          "and the clean solve lands", ok6b is True and len(rig6b.exits) == 2
          and EMBER_SOLVED <= rig6b.boulders)

    print("== 7. vanish_ok rows (the VR 3F hole-drop shape) ==")
    hole_room = bpz.room(
        (1, 41), "hole-row",
        [{"start": (33, 18), "segs": [("RIGHT", 1)], "vanish_ok": True}])
    rig7 = FakeRig(set(), player=(33, 18))
    ok7 = bpz.solve_room(rig7, hole_room, log=LOGS.append)
    check("a verified-absent vanish_ok boulder is DONE (down the hole), not diverged",
          ok7 is True and rig7.pushes == [])

    print("== 8. the summit turn: chained segments + stand-tile vacation order ==")
    rig8 = FakeRig(SUMMIT_TEMPLATE, player=(9, 15))
    ok8 = bpz.solve_room(rig8, bpz.EMBER_SUMMIT_BOARD, log=LOGS.append)
    check("summit board solved: 6 pushes, all four boulders parked on target",
          ok8 is True and len(rig8.pushes) == 6 and rig8.boulders == SUMMIT_SOLVED)
    check("the turning chain executed UP then RIGHTx2 in order",
          rig8.pushes[-3:] == [((8, 10), "UP"), ((8, 9), "RIGHT"), ((9, 9), "RIGHT")])

    print("== 9. _bank_milestone (campaign): one call, both reload rungs refreshed ==")
    C.log = lambda s: LOGS.append(str(s))
    calls = []
    camp = C.Campaign.__new__(C.Campaign)
    camp.b = types.SimpleNamespace(save_state=lambda: b"MILESTONE")
    camp._save_campaign = lambda r: calls.append(("save", r))
    camp._continuity_save = lambda: calls.append(("cont",))
    camp._auto_checkpoint = lambda r: calls.append(("ckpt", r))
    camp._gain_sig = lambda: ("gain", 7)
    st.in_battle = lambda b: False
    C.st.in_battle = st.in_battle
    C.tv.map_id = lambda b: (1, 101)
    t0 = time.time()
    r9 = camp._bank_milestone("moltres-leg")
    check("banks save + continuity + auto-checkpoint in order, returns True",
          r9 is True and calls == [("save", "moltres-leg"), ("cont",),
                                   ("ckpt", "moltres-leg")])
    check("refreshes the in-memory recent-good (escape hatch resumes HERE)",
          camp._last_good_state == b"MILESTONE" and camp._last_good_map == (1, 101)
          and camp._last_good_gain == ("gain", 7) and camp._last_ckpt_t >= t0)
    calls.clear()
    C.st.in_battle = lambda b: True
    check("mid-battle stands down (not a resumable overworld moment), still True",
          camp._bank_milestone("x") is True and calls == [])
    C.st.in_battle = lambda b: False
    camp._auto_checkpoint = lambda r: (_ for _ in ()).throw(RuntimeError("disk flake"))
    LOGS.clear()
    check("a checkpoint flake never raises — LOUD log, True back (composable)",
          camp._bank_milestone("y") is True
          and any("skipped" in ln and "LOUD" in ln for ln in LOGS))

    print("== 10. strike_checkpoint: milestone + attempt-counter refund ==")
    hunt = LS.MoltresHunt.__new__(LS.MoltresHunt)
    hunt.log = lambda s: LOGS.append(str(s))
    banked = []
    hunt.camp = types.SimpleNamespace(
        _bank_milestone=lambda r: banked.append(r) or True,
        _lap_fails={"moltres": 4, "zapdos": 2})
    r10 = hunt.strike_checkpoint()
    check("default label is '<quarry>-leg'; the moltres attempt counter is REFUNDED "
          "(milestone == progress, no honest-skip while advancing)",
          r10 is True and banked == ["moltres-leg"]
          and hunt.camp._lap_fails == {"zapdos": 2})
    hunt.strike_checkpoint("pre-moltres")
    check("explicit labels ('pre-moltres') ride through to the checkpoint inventory",
          banked == ["moltres-leg", "pre-moltres"])
    hunt.camp = types.SimpleNamespace()                       # no campaign machinery at all
    check("a bare camp can never wedge the climb (swallow + True)",
          hunt.strike_checkpoint() is True)

    print("== 11. _release_wedge_marks_on: scoped hygiene, any age ==")
    campW = C.Campaign.__new__(C.Campaign)
    campW._blocked_npcs = {((1, 97), (15, 32)), ((3, 1), (4, 4))}
    campW._looped_spots = {((1, 98), (2, 2))}
    campW._wedge_mem_ts = {("blocked_npcs", (1, 97), (15, 32)): 1.0}
    saves = []
    campW._save_wedge_memory = lambda: saves.append(1)
    campW._release_wedge_marks_on([(1, 97), (1, 98), (1, 101)], "moltres strike start")
    check("every mark on the strike map set dropped (any age), other maps kept",
          campW._blocked_npcs == {((3, 1), (4, 4))} and campW._looped_spots == set()
          and campW._wedge_mem_ts == {} and saves == [1])
    campW._release_wedge_marks_on([(1, 97)], "again")
    check("idempotent no-op: nothing left on those maps -> no re-save", saves == [1])

    print("== 12. _reload_same_region_checkpoint: newest same-region wins, labels unfiltered ==")
    tmp = tempfile.mkdtemp(prefix="bldrck_")
    old_root = C.STATES_CAMPAIGN
    C.STATES_CAMPAIGN = tmp
    try:
        import json as _json
        fixtures = [("20260805_120000_viridian_8b_kanto-roam", (3, 1)),
                    ("20260805_150000_mtember_8b_moltres-leg", (1, 98)),
                    ("20260805_153000_summit_8b_pre-moltres", (1, 101))]
        for name, m in fixtures:
            d = os.path.join(tmp, "checkpoints", name)
            os.makedirs(d)
            with open(os.path.join(d, "checkpoint.json"), "w", encoding="utf-8") as f:
                _json.dump({"map": list(m)}, f)
            with open(os.path.join(d, C.CAMPAIGN_SAVE), "wb") as f:
                f.write(name.encode())
        loads = []
        campR = C.Campaign.__new__(C.Campaign)
        campR.b = types.SimpleNamespace(load_state=loads.append, save_state=lambda: b"S")
        campR._gain_sig = lambda: 0
        campR._reset_strike_memory = lambda why: None
        campR._wait_overworld = lambda: None
        campR._save_campaign = lambda r: None
        campR.on_event = lambda *a, **k: None
        campR._region_reload_skips = 0
        C.tv.map_id = lambda b: (1, 101)
        region = C.map_region((1, 101))
        check("sanity: the summit and Kanto partition into different regions",
              region != C.map_region((3, 1)))
        r12 = campR._reload_same_region_checkpoint(region)
        check("first reload = the NEWEST same-region bank — the seconds-old 'pre-moltres' "
              "strike checkpoint (labels never filtered)",
              r12 is True and loads == [b"20260805_153000_summit_8b_pre-moltres"])
        r12b = campR._reload_same_region_checkpoint(region)
        check("re-wedge walks ONE further back ('moltres-leg'), still never the Kanto bank",
              r12b is True
              and loads[-1] == b"20260805_150000_mtember_8b_moltres-leg")
        LOGS.clear()
        check("past the region's history it DECLINES honestly (no cross-sea teleport)",
              campR._reload_same_region_checkpoint(region) is False)
    finally:
        C.STATES_CAMPAIGN = old_root

    n = len(PASS)
    good = sum(PASS)
    print(f"\n{'ALL GREEN' if good == n else 'FAILURES'}: {good}/{n}")
    return 0 if good == n else 1


if __name__ == "__main__":
    sys.exit(main())
