"""starter.py - the starter-pick HANDS. Reach a ball, select it, confirm. Isolated.

Table/ball tile coords are NOT assumed - they come from a capture (states/
starter_waypoints.json, written by m1_starter.py --capture). The left/mid/right ->
species mapping below is the FireRed convention but must be VERIFIED against the
real screen during capture (don't fly blind).
"""
import navigate as nav

# FireRed convention (VERIFY in capture): the three balls on Oak's table, left->right.
BALL_ORDER = ["bulbasaur", "charmander", "squirtle"]


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
