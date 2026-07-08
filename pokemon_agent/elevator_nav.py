"""elevator_nav.py — THE ELEVATOR PRIMITIVE (general engine asset, ported from
recon_hideout.py's ride() after the Giovanni strike proved it 2026-07-07; ported into
the engine 2026-07-08 night shift 5 — the banked_SCOPE 412-wedge diagnosis).

Ground truth (hideout B4F boss corridor + pret):
  An ELEVATOR CAR is a small room whose EVERY warp event dests to the DYNAMIC map
  (127,127). Stepping on those warps just walks back out onto the BOARDING floor —
  which is why exit machinery ping-pongs car -> floor -> car forever (the SCOPE
  loop: she entered the car, "exited the building", and was back where she started).
  RIDING: the floor PANEL is a BG EVENT on the car map — stand on an adjacent tile,
  face it, A until a box opens, A again ("Which floor?" -> the multichoice), DOWN x
  row, A, drain the confirmation, wait out the ride/shake, then exit via the nearest
  (dynamic) door warp — the same door now lands on the SELECTED floor.
  Floor rows are NOT billed per-building: callers pick row k, verify the LANDING,
  and try the next row if it didn't help (the strike's self-correct-on-landing law).
  The class repeats at Silph Co, Celadon Dept. Store, and the Rocket Hideout.
"""
import travel as tv
from dialogue_drive import box_open as _box_open

DYNAMIC = (127, 127)


def is_car(b):
    """True iff the current map reads as an elevator car: >=1 warp, ALL dynamic. Pure read."""
    ws = tv.read_warps(b)
    return bool(ws) and all(tuple(w[1]) == DYNAMIC for w in ws)


def ride(b, camp, row, log=lambda m: print(m, flush=True)):
    """One panel ride to floor-row `row` + step-out. True = we left the car onto SOME
    floor (the caller verifies WHICH and self-corrects); False = the panel never
    opened / the exit never fired."""
    m0 = tuple(tv.map_id(b))
    g = tv.Grid(b)
    warp_tiles = {tuple(w[0]) for w in tv.read_warps(b)}
    opened = False
    for pt, _kind in tv.read_bg_events(b):
        pt = tuple(pt)
        for nb, face in (((pt[0] + 1, pt[1]), "LEFT"), ((pt[0] - 1, pt[1]), "RIGHT"),
                         ((pt[0], pt[1] + 1), "UP"), ((pt[0], pt[1] - 1), "DOWN")):
            if nb in warp_tiles or not g.walkable(*nb):
                continue
            if tuple(tv.coords(b) or ()) != nb and not camp._step_to(nb):
                continue
            for _ in range(4):
                b.press(face, 8, 10, camp.render, owner="agent")
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(40):
                    b.run_frame()
                    if _box_open(b):
                        opened = True
                        break
                if opened:
                    break
            if opened:
                break
        if opened:
            break
    if not opened:
        log(f"   [elev] no panel opened on car {m0} (bg events: "
            f"{[tuple(p) for p, _k in tv.read_bg_events(b)]})")
        return False
    b.press("A", 8, 12, camp.render, owner="agent")       # "Which floor?" -> the menu
    for _ in range(30):
        b.run_frame()
    for _ in range(row):
        b.press("DOWN", 8, 10, camp.render, owner="agent")
        for _ in range(16):
            b.run_frame()
    b.press("A", 8, 12, camp.render, owner="agent")
    camp._adv_dialogue(12)                                # confirmation text, if any
    for _ in range(300):                                  # the ride/shake
        b.run_frame()
    r = camp.enter_warp(prefer="nearest")                 # the car door IS the nearest warp
    for _ in range(80):
        b.run_frame()
    out = tuple(tv.map_id(b)) != m0
    log(f"   [elev] ride(row={row}) -> "
        f"{('landed ' + str(tuple(tv.map_id(b)))) if out else 'still in the car (' + str(r) + ')'}"
        f" @ {tv.coords(b)}")
    return out
