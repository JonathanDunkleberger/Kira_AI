"""starter.py - the starter-pick HANDS. Reach a ball, select it, confirm. Isolated.

Table/ball tile coords are NOT assumed - they come from a capture (states/
starter_waypoints.json, written by m1_starter.py --capture). The left/mid/right ->
species mapping below is the FireRed convention but must be VERIFIED against the
real screen during capture (don't fly blind).
"""
import navigate as nav

# FireRed convention (VERIFY in capture): the three balls on Oak's table, left->right.
BALL_ORDER = ["bulbasaur", "charmander", "squirtle"]

# ⚠ The first capture's marked tiles (8,3)/(9,3)/(10,3) were WRONG — verified
# empirically against starter.state: at those tiles, A opens no prompt. The real
# interaction is the FRONT row (face UP + A) near x=11..13. Pick is UNVERIFIED
# pending a clean "prompt-ready" capture (see m1_starter.py / the report).


def advance_dialogue(bridge, until_fn, render=lambda: None, max_presses=50, log=print):
    """Press A like a HUMAN: only when the dialogue BOX has settled (typewriter done).
    Mashing during the typewriter re-skips and hangs (the bug). We watch ONLY the
    bottom text-box region (rows 112:160) - the rest of the screen (a static starter
    sprite) was keeping the whole-frame diff low and firing A early. Stops when
    until_fn() is true (e.g. party_count grew). Logs each press for diagnosis."""
    import numpy as np

    def box():
        return np.asarray(bridge.frame_rgb(), dtype=np.int16)[112:, :, :]

    def settle(maxf=100):
        prev = box(); stable = 0
        for _ in range(maxf):
            bridge.run_frame(); render()
            cur = box()
            stable = stable + 1 if np.abs(cur - prev).mean() < 0.5 else 0
            prev = cur
            if stable >= 10:       # ~10 stable box-frames = line printed / box ready
                return
    for i in range(max_presses):
        if until_fn():
            log(f"   [advance] done after {i} presses (until_fn true)")
            return True
        settle()
        if until_fn():
            log(f"   [advance] done after {i} presses (settled into target)")
            return True
        bridge.set_keys("A")
        for _ in range(3):
            bridge.run_frame(); render()
        bridge.release()
        log(f"   [advance] press {i}: box-settled then A")
    log(f"   [advance] EXHAUSTED {max_presses} presses without until_fn")
    return until_fn()


def clear_dialogue(bridge, taps=8, hold=8, render=lambda: None):
    """Mash A to advance text boxes / confirm. Movement is locked during dialogue;
    this clears it. Returns when done (fixed taps - cheap)."""
    for _ in range(taps):
        bridge.press("A", hold, hold, render)


def select_starter(bridge, choice, ball_xy, render=lambda: None, hold=8, on_event=None):
    """Walk to the chosen ball's tile, press A, confirm YES. ball_xy is the captured
    (x,y) of that starter's ball tile. Emits NEUTRAL events via on_event."""
    emit = on_event or (lambda s, **k: print(f"   [EVENT] {s}"))
    emit("looking at the three starter Pokemon")
    reached, final, steps = nav.walk_to(bridge, ball_xy, hold=hold, render=render)
    if not reached:
        emit(f"could not reach the {choice} ball (stuck at {final})")
        return False
    bridge.press("A", hold, hold, render)        # interact with the ball
    clear_dialogue(bridge, taps=4, hold=hold, render=render)  # "This is X. Do you want it?"
    bridge.press("A", hold, hold, render)         # confirm YES (default cursor)
    clear_dialogue(bridge, taps=6, hold=hold, render=render)  # name prompt / Oak text
    emit(f"chose {choice}")
    emit(f"{choice} is on the team now")
    return True
