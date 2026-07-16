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
import firered_ram as ram
from dialogue_drive import box_open as _box_open

DYNAMIC = (127, 127)
_SB1_DYNWARP = 0x14      # SaveBlock1.dynamicWarp (WarpData): +0x14 mapGroup, +0x15 mapNum —
#                          the elevator script's SetDynamicWarp target; pos(0x00)+location(0x04)
#                          +continueGameWarp(0x0C) precede it (offsets cross-checked vs the
#                          proven SB1_MAP_GROUP/NUM = 0x04/0x05 location reads).


def dynamic_dest(b):
    """Where the car's dynamic door currently leads — THE LANDING ORACLE (shift 5): the
    floor multichoice rows are unknowable blind, but the selection immediately calls
    SetDynamicWarp, so this read tells us the landing BEFORE stepping out. Pure read."""
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    return (b.rd8(sb1 + _SB1_DYNWARP), b.rd8(sb1 + _SB1_DYNWARP + 1))


def is_car(b):
    """True iff the current map reads as an elevator car: >=1 warp, ALL dynamic. Pure read."""
    ws = tv.read_warps(b)
    return bool(ws) and all(tuple(w[1]) == DYNAMIC for w in ws)


def ride(b, camp, row, avoid=(), log=lambda m: print(m, flush=True)):
    """One panel ride to floor-row `row` + step-out. True = we left the car onto a NEW
    floor; False = the pick didn't take / it targeted an `avoid` floor (already explored
    by the caller) — in both False cases we STAY ABOARD so the next row rides instantly."""
    m0 = tuple(tv.map_id(b))
    dest0 = dynamic_dest(b)
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
    # MENU SETTLE (round-2 regrade lesson, the battle-menu doctrine: never race the
    # emulator): D-pad presses before the multichoice is input-ready are EATEN, so the
    # select-A lands on the DEFAULT row every time and only the oracle's "unchanged"
    # guard saved us. Give the menu a beat, then space the downs generously.
    for _ in range(90):
        b.run_frame()
    for _ in range(row):
        b.press("DOWN", 8, 16, camp.render, owner="agent")
        for _ in range(24):
            b.run_frame()
    b.press("A", 8, 12, camp.render, owner="agent")
    camp._adv_dialogue(12)                                # confirmation text, if any
    for _ in range(300):                                  # the ride/shake
        b.run_frame()
    # THE LANDING ORACLE (shift-5 regrade lesson: rows 0-2 flaked, self-correct looped
    # B4F<->B1F, never B2F): the selection retargets SaveBlock1.dynamicWarp instantly.
    # Unchanged = the pick didn't take (menu flake / current-floor row) -> STAY ABOARD
    # so the caller's next row rides immediately instead of a wasted step-out cycle.
    dest1 = dynamic_dest(b)
    if dest1 == dest0:
        log(f"   [elev] ride(row={row}): dynamic dest unchanged ({dest1}) — "
            f"selection didn't take, staying aboard")
        return False
    if tuple(dest1) in {tuple(a) for a in (avoid or ())}:
        log(f"   [elev] ride(row={row}): retargeted to already-explored floor {dest1} — "
            f"staying aboard, next row")
        return False
    r = camp.enter_warp(prefer="nearest")                 # the car door IS the nearest warp
    for _ in range(80):
        b.run_frame()
    out = tuple(tv.map_id(b)) != m0
    log(f"   [elev] ride(row={row}) retargeted {dest0} -> {dest1}; "
        f"{('landed ' + str(tuple(tv.map_id(b)))) if out else 'step-out FAILED (' + str(r) + ')'}"
        f" @ {tv.coords(b)}")
    return out
