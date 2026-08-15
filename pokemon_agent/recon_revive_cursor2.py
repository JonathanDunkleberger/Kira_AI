"""recon_revive_cursor2.py — derive the TRUE item-use party cursor byte, and prove the pixel reader.

Phase 1 (recon_revive_cursor.py) established:
  * the D-pad DOES work on "Use on which POKEMON?" — DOWN walks lead -> 1 -> 2 -> .. -> CANCEL -> wrap
  * PARTY_CURSOR 0x02020777 (gPartyMenu.slotId, derived on the SWITCH screen) is NOT the cursor here:
    it alternated 1,2,1,2,... in lockstep with the highlight BLINK, i.e. it is an animation counter
  * _party_cursor_slot()/_party_cursor_on_lead() read the position correctly at every stop

This phase nails it down. It walks the whole ring taking a WIDE EWRAM snapshot at every stop, then
reports every address whose value sequence equals the true position sequence — the real slotId, if
one is visible at all. It also records what RIGHT does from home, and how long the opening fade
keeps the highlight unreadable (the eaten-tap window).

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_revive_cursor2.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                 # noqa: E402
import firered_ram as ram                 # noqa: E402
import battle_agent as ba                 # noqa: E402
from battle_agent import BattleAgent      # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(_HERE, "states", "workshop", "e4_revive_stage.state")
OUT = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "revive_cursor2")
REVIVE = 24
# Scan ALL of EWRAM *and* IWRAM — phase 1 found no cursor in EWRAM, so the byte (if any)
# must live in IWRAM or nowhere readable.
REGIONS = ((0x02000000, 0x02040000), (0x03000000, 0x03008000))


def log(m):
    print(m, flush=True)


def pos(ag):
    """True cursor position from pixels: 0..5 = a party row, 'CANCEL', or None (unreadable)."""
    s = ag._party_cursor_slot()
    if s is not None:
        return s
    if ag._party_cursor_on_lead():
        return 0
    return "CANCEL"


def readable(ag):
    return ag._party_cursor_slot() is not None or ag._party_cursor_on_lead()


def settle_highlight(b, ag, need=4, gap=8, max_frames=400):
    """Wait for the highlight to READ THE SAME `need` times in a row. The opening fade
    paints half-drawn palettes that the orange test misreads (phase 1 read 'slot 2' 5
    frames in, then one DOWN landed on the LEAD — i.e. the early read was junk and the
    early tap was eaten)."""
    seen, last, f = 0, object(), 0
    while f < max_frames:
        p = pos(ag) if readable(ag) else None
        if p is not None and p == last:
            seen += 1
            if seen >= need:
                return p, f
        else:
            seen = 1 if p is not None else 0
        last = p
        for _ in range(gap):
            b.run_frame()
            f += 1
    return None, f


def addrs():
    for lo, hi in REGIONS:
        for a in range(lo, hi):
            yield a


def snapshot(b):
    u8 = b.core.memory.u8
    return bytearray(u8[a] for a in addrs())


ADDR_LIST = None


def reach_target_screen(b, ag):
    ids = [i for i, _ in ag._items_pocket()]
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
    # A#1 selects the item (USE/CANCEL sub-box), A#2 confirms USE -> target screen. Never B.
    for _ in range(6):
        b.press("A", ag.hold, ag.hold, lambda: None, owner=ag.owner)
        ag._wait(30)
        if ag._party_screen() or ag._party_menu_cb2():
            return True
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    b = Bridge(ROM)
    with open(SRC, "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
    ag.owner = "agent"
    if not reach_target_screen(b, ag):
        log("STAGE FAIL")
        return 1
    rows = ag._menu_rows()
    log(f"target screen up. menu_rows={[(r['row'], r['hp']) for r in rows]}")

    # ── (1) THE OPENING FADE: how long until the highlight is STABLY readable? ──
    log("")
    log("--- opening fade ---")
    p0, f0 = settle_highlight(b, ag)
    log(f"first readable-at-all: see phase 1 (5 frames, and it LIED)")
    log(f"STABLE highlight = {p0} after {f0} frames  (this is HOME)")
    home = p0

    # ── (2) what does RIGHT do from home? ──
    log("")
    log("--- RIGHT from home ---")
    ag._tap("RIGHT")
    ag._wait(24)
    log(f"after RIGHT: pos={pos(ag)} (home was {home})  moved={pos(ag) != home}")
    right_moved = pos(ag) != home
    ag._tap("LEFT")
    ag._wait(20)
    log(f"after LEFT (undo): pos={pos(ag)}  [LEFT is CANCEL-ish on this screen: watch for a screen exit]")
    log(f"still on target screen? party_cb2={ag._party_menu_cb2()} party_pix={ag._party_screen()}")

    # re-stage clean (LEFT may have cancelled out)
    b.load_state(open(SRC, "rb").read())
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")
    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None)
    ag.owner = "agent"
    if not reach_target_screen(b, ag):
        log("RESTAGE FAIL")
        return 1
    settle_highlight(b, ag)

    # ── (3) walk the whole ring, wide-snapshot at every stop ──
    log("")
    log("--- walking the ring with DOWN, EWRAM+IWRAM snapshot at every stop ---")
    stops, snaps = [], []
    p = pos(ag)
    stops.append(p)
    snaps.append(snapshot(b))
    log(f"stop 0: pos={p}  (HOME)")
    for i in range(1, 10):
        ag._tap("DOWN")
        ag._wait(24)
        p = pos(ag)
        stops.append(p)
        snaps.append(snapshot(b))
        log(f"stop {i}: pos={p}  (after DOWN #{i})")
        b.frame_rgb().save(os.path.join(OUT, f"ring_{i}_{p}.png"))

    # ── (4) which address tracks the true position? ──
    log("")
    log("--- addresses whose byte sequence == the true position sequence ---")
    n_rows = len(rows)
    want = [n_rows if s == "CANCEL" else s for s in stops]     # CANCEL = index n_rows
    log(f"true sequence (CANCEL={n_rows}): {want}")
    global ADDR_LIST
    if ADDR_LIST is None:
        ADDR_LIST = list(addrs())
    first = snaps[0]
    hits = [ADDR_LIST[i] for i in range(len(first))
            if first[i] == want[0] and all(snaps[k][i] == want[k] for k in range(1, len(snaps)))]
    log(f"EXACT matches ({len(hits)}): {[hex(a) for a in hits[:40]]}")
    loose = []
    for i in range(len(first)):
        vals = [snaps[k][i] for k in range(len(snaps))]
        if max(vals) > 7 or len(set(vals)) < 3:
            continue
        if all((vals[k] == vals[k - 1]) == (want[k] == want[k - 1])
               for k in range(1, len(vals))):
            loose.append((hex(ADDR_LIST[i]), vals))
    log(f"MOVE-tracking candidates ({len(loose)}): {loose[:30]}")
    log("")
    pc_i = ADDR_LIST.index(ba.PARTY_CURSOR)
    log(f"PARTY_CURSOR {hex(ba.PARTY_CURSOR)} across the ring: "
        f"{[snaps[k][pc_i] for k in range(len(snaps))]}  (want {want})")
    log(f"RIGHT-from-home moved the cursor: {right_moved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
