"""starter_session.py - the full starter-pick HANDS sequence. SYNC, zero Kira imports.

Callbacks let the LIVE bot drive the choice + reactions through her real self:
  choose_fn() -> 'bulbasaur'|'charmander'|'squirtle'   (her self picks)
  emit_fn(summary)                                       (neutral event -> _pokemon_react)
Standalone tests pass plain stubs. Dialogue-clear and the pick are confirmed via REAL
reads (coord-unlock, party_count 0->1), never blind fixed counts.
"""
import navigate as nav
import firered_ram as ram
import pokemon_state as st

# Verified ball standing-tiles (from the capture); balls are one tile UP (face up + A).
BALL_TILES = {"bulbasaur": (8, 3), "charmander": (9, 3), "squirtle": (10, 3)}


def _coords(b):
    return nav.coords(b)


def clear_until_movable(b, hold, render, max_taps=16, log=print):
    """Advance the open dialogue with A until the player can MOVE (measured: probe a
    step and see coords change), then restore position facing UP toward the table.
    Returns True when movement is confirmed unlocked."""
    for i in range(max_taps):
        before = _coords(b)
        b.press("DOWN", hold, hold, render)        # probe: south of the table is open floor
        after = _coords(b)
        if after is not None and before is not None and after != before:
            b.press("UP", hold, hold, render)       # step back -> restores tile AND faces UP
            log(f"   [Starter] dialogue cleared after {i} A-taps (movement unlocked at {before})")
            return True
        b.press("A", hold, hold, render)            # still locked -> advance the message
    log("   [Starter] FAIL - dialogue never cleared (movement stayed locked)")
    return False


def select_at(b, ball_xy, hold, render, log=print):
    """From anywhere, walk to the ball tile, face UP, press A, then confirm YES by
    pressing A until party_count goes 0->1 (the measured success signal)."""
    reached, final, _ = nav.walk_to(b, ball_xy, hold=hold, render=render, log=log)
    if not reached:
        log(f"   [Starter] FAIL - couldn't reach ball tile {ball_xy} (stuck {final})")
        return False
    b.press("UP", hold, hold, render)               # face the ball (table above is solid)
    b.press("A", hold, hold, render)                # "This is X. Do you want it?"
    for _ in range(14):                             # advance text + confirm YES (default)
        b.press("A", hold, hold, render)
        if ram.read_party_count(b) == 1:
            for _ in range(4):                      # clear "RED received X!" text
                b.press("A", hold, hold, render)
            return True
    return False


def run(bridge, choose_fn, emit_fn, render=lambda: None, hold=8, log=print):
    """The milestone. Returns a result dict. choose_fn/emit_fn are the self seam."""
    r = dict(dialogue_cleared=False, chose=None, selected=False,
             party_before=ram.read_party_count(bridge), party_after=None,
             species_id=None, species_name="(none)", match=False)

    r["dialogue_cleared"] = clear_until_movable(bridge, hold, render, log=log)
    if not r["dialogue_cleared"]:
        return r

    emit_fn("standing at the table looking at the three starter Pokemon")

    choice = (choose_fn() or "bulbasaur").lower()
    if choice not in BALL_TILES:
        log(f"   [Starter] choose_fn returned {choice!r} - not a starter; FAIL")
        return r
    r["chose"] = choice
    log(f"   [Starter] her self chose: {choice.upper()}")

    r["selected"] = select_at(bridge, BALL_TILES[choice], hold, render, log=log)
    if not r["selected"]:
        log("   [Starter] FAIL - selection did not take (party stayed 0)")
        return r

    # verify the pick against REAL party reads
    r["party_after"] = ram.read_party_count(bridge)
    sid = st.read_party_species(bridge, 0)
    r["species_id"] = sid
    r["species_name"] = st.SPECIES_NAME.get(sid, f"id{sid}")
    r["match"] = (st.STARTER_SPECIES.get(choice) == sid)

    emit_fn(f"chose {choice}")
    emit_fn(f"{choice} joined the team")
    return r
