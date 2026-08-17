"""recon_revive_frozenframe.py — reproduce the LIVE six-unused-Revives wall, then prove the fix.

LIVE RECEIPT (logs/debug/supervisor/playlive_2026-08-15_07-29-05.log, Agatha's room):
the item-use target screen was verifiably UP — the dialogue reader captured
"REVIVE is selected." and "Use on which POKEMON?" — the walk asked for row 2, and
`_item_slot_id()` reported **pos=0 for 36 consecutive DOWN taps** (3 aims x 12 laps):

    use_item: revive pos=0 want=2 lap=0/12 (DOWN-only ring, len 5)
    ... x12 ...
    use_item: revive ring walk exhausted (pos=0 want=2) - no A (fail-safe)

6 Revives in the bag, THREE corpses on the floor (including the L77 ace Blastoise),
zero consumed, battle-wide REVIVE-FAIL LATCH, whiteout. Every attempt of every E4 run.

WHY the fixture missed it: `recon_revive_land` PASSES on a fresh core (pos walks 0 -> 1 -> 2),
so the ring walk itself is correct. The difference live is the **FROZEN FRAMEBUFFER** — the
long-running-core disease this file has fought for months (see "frozen-frame" notes all over
battle_agent). `_item_slot_id()` voted PIXELS FIRST, so one stale frame pins the vote on
whatever was last drawn (the home/lead panel) FOREVER, and the MEASURED slotId byte
(ITEM_PARTY_CURSOR, derived by recon_revive_cursor2) is never consulted at all.

This recon freezes `b.frame_rgb()` the instant the target screen takes input — exactly what the
live core does — and then runs the REAL `use_item_in_battle`. Pre-fix it reproduces the live
"pos=0 x12 -> no A" wall. Post-fix (RAM-first slotId) the Revive lands on the corpse.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_revive_frozenframe.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                 # noqa: E402
import firered_ram as ram                 # noqa: E402
import pokemon_state as st                # noqa: E402
import battle_agent as ba                 # noqa: E402
from battle_agent import BattleAgent      # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
# SRC defaults to the healthy staged fixture. REVIVE_STATE points it at ANY savestate — the
# reason this exists (2026-08-15): the healthy fixture PASSES every case while the live 55-hour
# core fails all of them, because a savestate is a full RAM snapshot and carries its heap with
# it. To debug the live wedge offline at 14x, copy the sick campaign state and point here.
SRC = os.environ.get("REVIVE_STATE") or os.path.join(_HERE, "states", "workshop",
                                                     "e4_revive_stage.state")
OUT = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "revive_frozenframe")
REVIVE = 24
RAW = None


def log(m):
    print(m, flush=True)


def hp(b, slot):
    return b.rd16(ram.GPLAYER_PARTY + slot * 100 + 0x56)


def party(b):
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    return [(i, st.SPECIES_NAME.get(st.read_party_species(b, i), "?"), hp(b, i))
            for i in range(min(cnt, 6))]


def boot(quiet=False):
    global RAW
    if RAW is None:
        with open(SRC, "rb") as f:
            RAW = f.read()
    b = Bridge(ROM)
    b.load_state(RAW)
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")
    render = lambda: None
    if os.environ.get("LIVE_WRAP") == "1":
        # LIVE_WRAP=1 reproduces the two things the LIVE process does to every frame that this
        # fixture never did (2026-08-15): play_live installs a wall-clock frame pacer around
        # b.core.run_frame, and `_wait` calls a REAL render callback on every single frame. The
        # fixture passes with render=no-op and no pacer while the identical aim fails live, so
        # this is the prime suspect for the live-only wedge: if presses become unreliable under
        # the wrapper, paths that verify-and-retry (the battle move menu, via STREAM COMMIT)
        # survive while single-shot menu presses fail silently — exactly the observed split.
        _orig_rf = b.core.run_frame
        _clock = [time.perf_counter()]
        _dt = 1.0 / max(1.0, float(os.environ.get("POKEMON_FPS_CAP", "60")))

        def _paced():
            _orig_rf()
            _clock[0] += _dt
            slack = _clock[0] - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.25:
                _clock[0] = time.perf_counter()
        b.core.run_frame = _paced
        _rendered = [0]

        def render():                       # noqa: F811 — the live per-frame render callback
            _rendered[0] += 1
            try:
                b.frame_rgb()               # what a real windowed run does every frame
            except Exception:
                pass
        b._recon_render_count = _rendered
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                     log=(lambda m: None) if quiet else log)
    ag.owner = "agent"
    return b, ag


def reach_target_screen(b, ag):
    """Bag -> Items pocket -> REVIVE row -> A until the 'Use on which POKEMON?' screen owns input."""
    ids = [i for i, _ in ag._items_pocket()]
    if REVIVE not in ids:
        return False
    row = ids.index(REVIVE)
    if not ag._settle_action_menu() or not ag._open_bag():
        return False
    for _ in range(6):
        if b.rd8(ram.GBAG_POCKET) == 0:
            break
        ag._tap("LEFT")
        ag._wait(12)

    def sel():
        return b.rd8(ba.BAG_CURSOR) + b.rd16(ba.BAG_SCROLL)
    for _ in range(14):
        if sel() == row:
            break
        ag._tap("DOWN" if sel() < row else "UP")
        ag._wait(10)
    for _ in range(6):
        b.press("A", ag.hold, ag.hold, lambda: None, owner=ag.owner)
        ag._wait(30)
        if ag._party_screen() or ag._party_menu_cb2():
            return True
    return False


def freeze_frames_on_target(b):
    """Make frame_rgb() FREEZE (return one stale image forever) the moment the item-use
    target screen owns input — the live long-core symptom, made deterministic."""
    real = b.frame_rgb
    cache = {"real": real}

    def frozen():
        if "img" in cache:
            return cache["img"]
        try:
            cb2 = b.rd32(ram.GMAIN_CB2)
        except Exception:
            cb2 = 0
        img = real()
        if cb2 in ba._CB2_PARTY_MENU:
            cache["img"] = img          # snapshot the target screen, then never update again
        return img
    b.frame_rgb = frozen
    return cache


# ── CASE 2b: instrumented walk under a frozen frame — engine votes vs GROUND TRUTH ──
def case_frozen_walk_probe(dead_slot=2):
    log("")
    log("=" * 78)
    log(f"CASE 2b: instrumented DOWN walk under a FROZEN frame (corpse row {dead_slot})")
    log("        engine votes vs GROUND TRUTH (the live, unfrozen framebuffer)")
    log("=" * 78)
    b, ag = boot(quiet=True)
    for s in range(b.rd8(ram.GPLAYER_PARTY_CNT)):
        want = 0 if s == dead_slot else b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
        b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + s * 100 + 0x56, want)
    if not reach_target_screen(b, ag):
        log("  STAGE FAIL")
        return False
    cache = freeze_frames_on_target(b)
    ag._item_party_settle()
    real = cache["real"]

    def truth():
        """The position a NON-frozen framebuffer would report."""
        frozen = b.frame_rgb
        b.frame_rgb = real
        try:
            s = ag._party_cursor_slot()
            if s is not None:
                return s
            return 0 if ag._party_cursor_on_lead() else "CANCEL/?"
        finally:
            b.frame_rgb = frozen
    log(f"  frozen? {'yes' if 'img' in cache else 'NO'}  menu_rows="
        f"{[(r['row'], r['hp']) for r in ag._menu_rows()]}")
    log("  lap | TRUTH | engine _item_slot_id | pix_slot | on_lead | ITEM_PARTY_CURSOR | blink")
    seen_truth, seen_engine = [], []
    for lap in range(8):
        t = truth()
        e = ag._item_slot_id()
        seen_truth.append(t)
        seen_engine.append(e)
        log(f"  {lap:>3} | {str(t):>5} | {str(e):>19} | "
            f"{str(ag._party_cursor_slot()):>8} | {str(ag._party_cursor_on_lead()):>7} | "
            f"{b.rd8(ba.ITEM_PARTY_CURSOR):>17} | {b.rd8(ba.PARTY_CURSOR):>5}")
        ag._tap("DOWN")
        ag._wait(24)
    truth_moved = len(set(map(str, seen_truth))) > 1
    log(f"  -> DOWN really moves the highlight under a frozen frame: {truth_moved}")
    log(f"  -> the engine's reader tracked it: {seen_engine == seen_truth}")
    del b
    return truth_moved


# ── CASE 1: on a LIVE frame, does the measured slotId byte track the pixel truth? ──
def case_ring_readers():
    log("")
    log("=" * 78)
    log("CASE 1: walk the ring on a LIVE frame — pixels vs ITEM_PARTY_CURSOR vs PARTY_CURSOR")
    log("=" * 78)
    b, ag = boot(quiet=True)
    if not reach_target_screen(b, ag):
        log("  STAGE FAIL")
        return False
    rows = ag._menu_rows()
    n = len(rows)
    log(f"  target screen up. menu_rows={[(r['row'], r['hp']) for r in rows]}")
    ag._item_party_settle()
    seq_pix, seq_item, seq_blink = [], [], []
    for i in range(n + 2):
        pixslot = ag._party_cursor_slot()
        onlead = ag._party_cursor_on_lead()
        pix = pixslot if pixslot is not None else (0 if onlead else None)
        seq_pix.append(pix)
        seq_item.append(b.rd8(ba.ITEM_PARTY_CURSOR))
        seq_blink.append(b.rd8(ba.PARTY_CURSOR))
        log(f"  stop {i}: pixels={pix!r:>6}  ITEM_PARTY_CURSOR={seq_item[-1]:>3}  "
            f"PARTY_CURSOR(blink)={seq_blink[-1]:>3}")
        ag._tap("DOWN")
        ag._wait(24)
    # The byte must MOVE with our taps and agree with the pixel vote wherever pixels can see it.
    moved = len(set(seq_item)) >= min(3, n)
    agree = all(p is None or p == (ba.ITEM_PARTY_CANCEL if v in (6, 7) else v)
                for p, v in zip(seq_pix, seq_item))
    log(f"  -> byte moves with the taps: {moved} | agrees with pixels where readable: {agree}")
    ok = moved and agree
    log(f"  -> {'PASS' if ok else 'FAIL'}")
    del b
    return ok


# ── CASE 2: THE LIVE BUG — frozen framebuffer, real lander ──
def case_frozen_frame_land(dead_slot=2):
    log("")
    log("=" * 78)
    log(f"CASE 2: FROZEN FRAME + real use_item_in_battle (corpse at row {dead_slot})")
    log("       (this is the live Agatha wall: pos=0 x12, no A, 6 Revives unused)")
    log("=" * 78)
    b, ag = boot()
    for s in range(b.rd8(ram.GPLAYER_PARTY_CNT)):
        want = 0 if s == dead_slot else b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
        b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + s * 100 + 0x56, want)
    cache = freeze_frames_on_target(b)
    n0 = ag._items_count(REVIVE)
    log(f"  before: party={party(b)} revives={n0}")
    res = ag.use_item_in_battle(REVIVE, target="fainted")
    n1 = ag._items_count(REVIVE)
    log(f"  after:  party={party(b)} revives={n1} result={res!r} "
        f"frame_frozen={'yes' if 'img' in cache else 'NO (never froze!)'}")
    ok = bool("img" in cache and res == "used" and n1 == n0 - 1 and hp(b, dead_slot) > 0)
    log(f"  -> {'PASS' if ok else 'FAIL'} (want a frozen frame AND result='used' AND the corpse alive)")
    del b
    return ok


# ── CASE 3: frozen frame must NOT make her A a LIVING mon ──
def case_frozen_frame_no_burn():
    log("")
    log("=" * 78)
    log("CASE 3: FROZEN FRAME, nobody down — the Revive must NOT be spent")
    log("=" * 78)
    b, ag = boot()
    for s in range(b.rd8(ram.GPLAYER_PARTY_CNT)):
        b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + s * 100 + 0x56,
                                    b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58))
    freeze_frames_on_target(b)
    n0 = ag._items_count(REVIVE)
    res = ag.use_item_in_battle(REVIVE, target="fainted")
    n1 = ag._items_count(REVIVE)
    log(f"  result={res!r} revives {n0}->{n1}")
    ok = (n1 == n0 and res != "used")
    log(f"  -> {'PASS' if ok else 'FAIL'} (want the count UNCHANGED)")
    del b
    return ok


# ── CASE 4: the LIVE core's real failure — cursor UNREADABLE, blind DOWN-COUNT sweep must land ──
def case_mute_cursor_sweep(dead_slot=2):
    """Live 2026-08-15 (Lance's Room): the walk logged `readings=[0]` — the slotId byte sat
    frozen for all twelve laps (heap-moved struct on a 55-hour core) and the pixel readers were
    stale too. Simulate exactly that: latch the cursor PROVEN-MUTE so the walk is skipped, and
    demand the blind DOWN-count sweep resurrect the corpse anyway."""
    log("")
    log("=" * 78)
    log(f"CASE 4: cursor PROVEN MUTE -> blind DOWN-COUNT sweep (corpse at row {dead_slot})")
    log("=" * 78)
    b, ag = boot()
    for s in range(b.rd8(ram.GPLAYER_PARTY_CNT)):
        want = 0 if s == dead_slot else b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
        b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + s * 100 + 0x56, want)
    b._item_cursor_mute_seen = True          # the live latch: no readable cursor on this core
    freeze_frames_on_target(b)               # ...and stale frames on top, like the real thing
    n0 = ag._items_count(REVIVE)
    log(f"  before: party={party(b)} revives={n0}")
    res = ag.use_item_in_battle(REVIVE, target="fainted")
    n1 = ag._items_count(REVIVE)
    log(f"  after:  party={party(b)} revives={n1} result={res!r}")
    ok = bool(res == "used" and n1 == n0 - 1 and hp(b, dead_slot) > 0)
    log(f"  -> {'PASS' if ok else 'FAIL'} (want result='used', one Revive spent, the corpse alive)")
    del b
    return ok


def case_mute_cursor_no_burn():
    """Anti-regression for the blind sweep: cursor mute AND nobody down -> it must spend nothing."""
    log("")
    log("=" * 78)
    log("CASE 5: cursor PROVEN MUTE, nobody down — the blind sweep must spend NOTHING")
    log("=" * 78)
    b, ag = boot()
    for s in range(b.rd8(ram.GPLAYER_PARTY_CNT)):
        b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + s * 100 + 0x56,
                                    b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58))
    b._item_cursor_mute_seen = True
    n0 = ag._items_count(REVIVE)
    res = ag.use_item_in_battle(REVIVE, target="fainted")
    n1 = ag._items_count(REVIVE)
    log(f"  result={res!r} revives {n0}->{n1}")
    ok = (n1 == n0 and res != "used")
    log(f"  -> {'PASS' if ok else 'FAIL'} (want the count UNCHANGED)")
    del b
    return ok


# ── CASE 6: THE LIVE BRUNO WALL — mute cursor + OPENING FADE + a lying gBattlerPartyIndexes ──
def case_mute_cursor_fresh_fade(dead_slot=3, lie_home=3):
    """The 2026-08-15 Bruno/Hitmonlee wall: 14 straight blind attempts, ZERO Revives consumed,
    with THREE corpses on the floor — and this fixture PASSED the whole time (case 4).

    Why case 4 missed it: `reach_target_screen` drives A with `_wait(30)` between presses, so by
    the time the blind aim runs the target screen has been up for ~dozens of frames and the
    OPENING FADE is already over. Live, every bag trip opens a FRESH screen and the aim taps
    straight into that fade, where the d-pad is EATEN (measured in recon_revive_cursor and
    written into `_item_party_settle`). With the taps swallowed the cursor never leaves row 0 —
    the living lead — so every A said "It won't have any effect." regardless of k.

    This case reproduces BOTH live conditions honestly:
      * the aim runs with the screen only just having taken input (no settling grace), and
      * gBattlerPartyIndexes[0] is poked to LIE (live it read 3 while a 157-HP bird was out),
        which is what made the old informed guess compute k=0 and press A on the standing body.
    Post-fix the aim waits out the fade and counts from display row 0, so the Revive lands.
    """
    log("")
    log("=" * 78)
    log(f"CASE 6: mute cursor + FRESH OPENING FADE + lying gBattlerPartyIndexes (corpse {dead_slot})")
    log("        (the live Bruno wall: 14 attempts, 3 corpses, 0 Revives consumed)")
    log("=" * 78)
    b, ag = boot()
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    # Live shape: only the lead is standing, everything behind it is a corpse.
    for s in range(cnt):
        want = b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58) if s == 0 else 0
        b.core.memory.u16.raw_write(ram.GPLAYER_PARTY + s * 100 + 0x56, want)
    b._item_cursor_mute_seen = True
    freeze_frames_on_target(b)
    # gBattlerPartyIndexes[0] LIES — this was the old code's "home" (live it read 3 while a
    # 157-HP bird was the active battler, so the informed guess was k=0: A on the living lead).
    try:
        b.core.memory.u16.raw_write(ram.GBATTLER_PARTY_IDX, lie_home)
    except Exception:
        log("  (could not poke GBATTLER_PARTY_IDX — the fix must not read it anyway)")

    # THE FADE, made deterministic: a d-pad tap fired within FADE_FRAMES of the target screen
    # taking input is EATEN (measured in recon_revive_cursor). Any code that waits the fade out
    # before tapping is unaffected; code that taps straight into it can never move the cursor.
    FADE_FRAMES = 40
    fade = {"waited": 0, "eaten": 0, "passed": 0}
    real_wait, real_tap = ag._wait, ag._tap

    def _screen_up():
        try:
            return bool(ag._party_menu_cb2() or ag._party_screen())
        except Exception:
            return False

    def fade_wait(n):
        if _screen_up():
            fade["waited"] += n
        else:
            fade["waited"] = 0            # a fresh screen open restarts the fade
        return real_wait(n)

    def fade_tap(btn):
        if btn == "DOWN" and _screen_up() and fade["waited"] < FADE_FRAMES:
            fade["eaten"] += 1
            return real_wait(6)           # swallowed, exactly like the live fade
        if btn == "DOWN":
            fade["passed"] += 1
        return real_tap(btn)

    ag._wait, ag._tap = fade_wait, fade_tap
    n0 = ag._items_count(REVIVE)
    log(f"  before: party={party(b)} revives={n0} (lying home={lie_home}, fade={FADE_FRAMES}f)")
    # ONE BLIND AIM PER BAG TRIP is a hard rule (a failed Revive unwinds to the bag list, so a
    # second A would scroll the BAG — the Gary wipe). So a single `use_item_in_battle` call gets
    # ONE offset. Live she is offered the revive again every turn and `_revive_blind_try` persists
    # on the bridge, so the plan advances across turns. Model that: the bar is "lands within `ring`
    # turns", not "lands on the first guess" — nobody can guess right when the repo disagrees with
    # itself about where the highlight starts (see `_revive_blind_downcount`).
    ring = len(ag._menu_rows()) + 1
    res, turns = None, 0
    for turns in range(1, ring + 1):
        res = ag.use_item_in_battle(REVIVE, target="fainted")
        if res == "used":
            break
        # BETWEEN TURNS: live, the failed bag trip unwinds, the foe attacks, and the action menu
        # comes back up before the next revive offer. Without putting the battle back into that
        # state the retries all die in `_use_item`'s entry checks and the plan never advances —
        # which is what made a 5-turn loop fire exactly ONE aim.
        for _ in range(6):
            real_tap("B")
            real_wait(14)
        ag._settle_action_menu()
    n1 = ag._items_count(REVIVE)
    ag._wait, ag._tap = real_wait, real_tap
    log(f"  after:  party={party(b)} revives={n1} result={res!r} after {turns} turn(s) "
        f"| DOWN taps: {fade['passed']} landed / {fade['eaten']} eaten by the fade")
    ok = bool(res == "used" and n1 == n0 - 1)
    if ok and fade["eaten"] == 0 and fade["passed"] == 0:
        log("  !! NOTE: no DOWN tap was fired at all — this run did not exercise the fade")
    log(f"  -> {'PASS' if ok else 'FAIL'} (want result='used' within {ring} turns, one Revive spent)")
    del b
    return ok


def main():
    if not os.path.exists(SRC):
        log(f"!! fixture missing: {SRC}\n   run: python -u pokemon_agent/recon_revive_stage.py")
        return 1
    os.makedirs(OUT, exist_ok=True)
    only = (os.environ.get("CASES") or "").strip()
    results = {}
    if not only or "1" in only:
        results["ring_readers"] = case_ring_readers()
    if not only or "2b" in only:
        results["frozen_walk_probe"] = case_frozen_walk_probe(2)
    if not only or "2" in only.replace("2b", ""):
        for s in (1, 2, 3):
            results[f"frozen_frame_row{s}"] = case_frozen_frame_land(s)
        results["frozen_frame_no_burn"] = case_frozen_frame_no_burn()
    if not only or "4" in only:
        for s in (1, 2, 3):
            results[f"mute_cursor_row{s}"] = case_mute_cursor_sweep(s)
        results["mute_cursor_no_burn"] = case_mute_cursor_no_burn()
    if not only or "6" in only:
        results["mute_fresh_fade_row3"] = case_mute_cursor_fresh_fade(3, lie_home=3)
        results["mute_fresh_fade_row1"] = case_mute_cursor_fresh_fade(1, lie_home=2)
    log("")
    log("=" * 78)
    for k, v in results.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    log(f"VERDICT: {'ALL PASS' if ok else 'FAIL'}")
    log("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
