"""test_stuckwatch.py — unit proof for the LAYER B universal wall-clock watchdog (StuckWatch).

No emulator/bridge: we hand it synthetic WorldFingerprints + a fake clock and assert the trip
behavior on the exact cases the live Slowbro watch exposed. Run:
    .venv\\Scripts\\python.exe pokemon_agent\\test_stuckwatch.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import world_fingerprint as wf  # noqa: E402


def _fp(x=5, y=5, facing=1, box=False, battle=False, party=((1, 7, 20),), badges=2,
        money=3000, bag=((4, 5),), map_id=(3, 4)):
    return wf.WorldFingerprint(map_id=map_id, x=x, y=y, facing=facing, menu_or_dialogue=box,
                               battle_active=battle, party=party, badges=badges, money=money, bag=bag)


def test_progress_never_trips():
    """A normal walk: a NEW tile every step -> the clock restarts each feed -> never trips."""
    w = wf.StuckWatch(stuck_s=30)
    for i in range(200):                      # 200 steps over 200s, each a new (x) -> always progressing
        assert not w.feed(_fp(x=i), now=float(i)), f"false trip while walking at step {i}"
    print("PASS progress_never_trips")


def test_frozen_world_trips():
    """Pinned at one tile (no box): trips once the same key has sat for stuck_s wall-clock seconds."""
    w = wf.StuckWatch(stuck_s=30)
    assert not w.feed(_fp(), now=0.0)
    assert not w.feed(_fp(), now=29.0), "tripped too early"
    assert w.feed(_fp(), now=30.0), "should trip at exactly stuck_s"
    assert w.reason == "frozen_world"
    print("PASS frozen_world_trips")


def test_slowbro_toggling_box_trips():
    """THE LIVE BUG: travel bumps a plain NPC -> its line re-shows (box up) / closes (box down) while
    she never moves. A 2-state toggle the per-tick ledger SKIPPED. The wall-clock watch must still trip."""
    w = wf.StuckWatch(stuck_s=30)
    tripped_at = None
    for i in range(120):                      # 0.5s cadence, box toggles every feed, same tile + line
        t = i * 0.5
        box = (i % 2 == 0)
        text = "SLOWBRO: ...took a snooze." if box else ""
        if w.feed(_fp(box=box), now=t, text=text):
            tripped_at = t
            break
    assert tripped_at is not None, "never tripped on the toggling frozen box (the Slowbro wedge)"
    assert tripped_at <= 31.5, f"tripped too late ({tripped_at}s) — should be ~30s after the last new state"
    assert w.reason in ("frozen_box", "frozen_world")
    print(f"PASS slowbro_toggling_box_trips (tripped at {tripped_at}s)")


def test_facing_sweep_trips():
    """travel's blocker sweep turns her N/S/E/W on one tile (4 facings). A bounded 4-state cycle must
    still trip (each facing is new ONCE, then repeats; clock runs from the last-new facing)."""
    w = wf.StuckWatch(stuck_s=30)
    faces = [1, 2, 3, 4]
    tripped = False
    for i in range(200):
        t = i * 0.5
        if w.feed(_fp(facing=faces[i % 4]), now=t):
            tripped = True
            break
    assert tripped, "a bounded 4-facing sweep on one tile never tripped"
    print("PASS facing_sweep_trips")


def test_legit_long_dialogue_never_trips():
    """A real conversation: the box stays up but the TEXT advances page by page. Each page is a new key
    -> the clock restarts -> a long legit read (even > stuck_s total) never false-trips."""
    w = wf.StuckWatch(stuck_s=30)
    for page in range(40):                     # 40 pages, one every 2s = 80s total, all legit
        t = page * 2.0
        assert not w.feed(_fp(box=True), now=t, text=f"page {page} of the story..."), \
            f"false trip on legit page {page}"
    print("PASS legit_long_dialogue_never_trips")


def test_battle_never_trips():
    """Battle is battle_agent's domain (the flee floor owns it): a battle_active fp resets the watch."""
    w = wf.StuckWatch(stuck_s=30)
    for i in range(200):
        assert not w.feed(_fp(battle=True), now=float(i)), f"tripped mid-battle at {i}"
    print("PASS battle_never_trips")


def test_none_fp_resets():
    """An unreadable fingerprint is 'no judgement' — it must not accumulate toward a trip."""
    w = wf.StuckWatch(stuck_s=30)
    assert not w.feed(_fp(), now=0.0)
    assert not w.feed(None, now=29.0)          # unreadable -> reset
    assert not w.feed(_fp(), now=30.0)         # clock restarted at the None -> not yet stuck
    assert not w.feed(_fp(), now=59.0)
    assert w.feed(_fp(), now=60.0), "should trip 30s after the reset"
    print("PASS none_fp_resets")


def test_recovery_after_reset():
    """After a trip + reset (the roam loop's disengage), a genuine move clears the trip cleanly."""
    w = wf.StuckWatch(stuck_s=30)
    for t in range(31):                         # monotonic clock: pinned 0..30s -> trips at 30
        w.feed(_fp(), now=float(t))
    assert w.tripped, "should have tripped after 30s pinned"
    w.reset()
    assert not w.tripped
    assert not w.feed(_fp(x=99), now=100.0)     # moved -> progressing again, clean
    print("PASS recovery_after_reset")


if __name__ == "__main__":
    test_progress_never_trips()
    test_frozen_world_trips()
    test_slowbro_toggling_box_trips()
    test_facing_sweep_trips()
    test_legit_long_dialogue_never_trips()
    test_battle_never_trips()
    test_none_fp_resets()
    test_recovery_after_reset()
    print("\nALL STUCKWATCH TESTS PASSED")
