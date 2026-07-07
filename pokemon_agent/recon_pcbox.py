"""recon_pcbox.py — PC BOX DEPOSIT primitive recon (Tier-2 block #15's first slice).

Goal tonight: from the canonical save (Route 10, party 6/6 with a dupe Spearow in slot 4
0-indexed), enter the Route 10 Pokémon Center, drive the PC: BILL'S PC -> DEPOSIT -> pick the
slot -> confirm box -> verify party count 6 -> 5 by RAM. Screenshots at every stage so the menu
sequence can be calibrated by eye (the grab-and-look arsenal).

Usage: python recon_pcbox.py [probe|deposit] [slot]
  probe   — enter the Center, walk to the PC, press A once, screenshot. No commitment.
  deposit — full flow for party slot N (default 4, the dupe spearow).
Nothing here touches canonical; this runs on a RAM copy and saves nothing unless DEPOSIT
succeeds, in which case it banks stage_pcbox/ for promote_bank.
"""
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import firered_ram as ram            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SHOTS = r"G:\temp\pcbox"
PC_STAND = (13, 4)                    # in front of the PC console (counter right end); calibrate
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_pcbox")
BANK = os.path.join(SCRATCH, "banked_PCBOX")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    slot = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    global PC_STAND
    if len(sys.argv) > 3:
        PC_STAND = tuple(int(v) for v in sys.argv[3].split(","))
    os.makedirs(SHOTS, exist_ok=True)
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def shot(name):
        p = os.path.join(SHOTS, f"{name}.png")
        b.frame_rgb().save(p)
        L(f"shot -> {p}")

    def press(key, settle=30, hold=8, rel=10):
        b.press(key, hold, rel, owner="agent")
        for _ in range(settle):
            b.run_frame()

    def party_count():
        return b.rd8(ram.GPLAYER_PARTY_CNT)

    def party_species():
        return [st.SPECIES_NAME.get(st.read_party_species(b, s), "?")
                for s in range(party_count())]

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=180)

    camp = Campaign(b, battle_runner=runner, on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True     # no canonical writes from this recon
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} party={party_species()}")

    # ── 1) find + enter the Route 10 Center (dest group 9 = a PC interior) ──────────────
    # PC interiors get their own dest group per area (Vermilion 9, Route 10 21); the cave/
    # connector class is group 1 (Rock Tunnel = (1,81) here) — pick the non-group-1 door.
    pcdoor = None
    for (wxy, wdest, _wid) in tv.read_warps(b):
        if tuple(wdest)[0] not in (1,):
            pcdoor = tuple(wxy)
            break
    if pcdoor is None:
        L("!! no PC-interior door on this map — wrong start?")
        return 1
    L(f"PC door on {tv.map_id(b)}: {pcdoor}")
    r = camp.trav.travel(target_map=None, arrive_coord=(pcdoor[0], pcdoor[1] + 1),
                         max_steps=400, max_seconds=120)
    L(f"to door+1 -> {r} (at {tv.coords(b)})")
    before = tuple(tv.map_id(b))
    for _ in range(6):
        press("UP", settle=20)
        if tuple(tv.map_id(b)) != before:
            break
    for _ in range(90):
        b.run_frame()
    b.set_input_owner("agent")
    L(f"inside: map={tv.map_id(b)} coords={tv.coords(b)}")
    shot("00_inside")

    # ── 2) to the PC console ────────────────────────────────────────────────────────────
    camp._step_to(PC_STAND)
    L(f"at PC stand: {tv.coords(b)}")
    press("UP", settle=16)                          # face the console
    shot("01_at_pc")
    press("A", settle=60)                           # boot the PC
    shot("02_pc_menu")
    if mode == "probe":
        L("probe done — look at the shots")
        return 0

    # ── 3) drain the boot text -> "Which PC?" menu -> BILL'S PC -> box system ───────────
    # screen-state helpers (drive by what's actually on screen, not by frame-count guesses —
    # FRLG text boxes EAT presses during scroll/animation)
    import numpy as _np

    def _px():
        return _np.asarray(b.frame_rgb(), dtype=_np.int32)

    def _menu_open(img):
        # the "Which PC?" list draws a white box in the TOP-LEFT quadrant
        reg = img[8:56, 8:110]
        return (reg > 235).all(axis=2).mean() > 0.5

    def _wait(pred, frames=420):
        for _ in range(frames // 6):
            for _ in range(6):
                b.run_frame()
            if pred():
                return True
        return False

    ok = False
    for _ in range(6):                              # drain boot text until the PC-choice list shows
        press("A", settle=45)
        if _menu_open(_px()):
            ok = True
            break
    shot("03_which_pc")
    if not ok:
        L("!! PC-choice menu never appeared — aborting (see shots)")
        camp._exit_to_overworld()
        return 1
    # TOP-RIGHT corner probe: dialog boxes and the PC-choice menu never cover it — only the
    # full-screen storage applet repaints it. (The top-LEFT diff false-fired when the menu CLOSED.)
    base_tr = _px()[4:28, 188:236]

    def _applet():
        return _np.abs(_px()[4:28, 188:236] - base_tr).mean() > 30

    press("A", settle=45)                           # BILL'S PC (cursor boots on top)
    # drain "Accessed BILL's PC." / "Storage System opened." until the STORAGE list menu
    # (WITHDRAW/DEPOSIT/MOVE POKeMON/MOVE ITEMS/SEE YA — a plain menu over the room, NOT a
    # full-screen applet; that comes after picking DEPOSIT) re-opens top-left
    storage = False
    for i in range(8):
        if _menu_open(_px()):
            storage = True
            break
        press("A", settle=70)
    shot("04_storage_menu")
    if not storage:
        L("!! storage menu never appeared — aborting (see shots)")
        camp._exit_to_overworld()
        return 1
    press("DOWN", settle=30)                        # -> DEPOSIT POKeMON
    shot("05_cursor_deposit")
    base_tr = _px()[4:28, 188:236]                  # room corner visible NOW; the deposit GUI kills it
    press("A", settle=45)
    if not _wait(_applet, frames=700):
        L("!! deposit GUI never opened — aborting (see shots)")
        shot("06_no_gui")
        camp._exit_to_overworld()
        return 1
    for _ in range(160):                            # let the GUI finish drawing
        b.run_frame()
    shot("06_deposit_gui")
    for _ in range(slot):                           # cursor to the target party slot
        press("DOWN", settle=24)
    shot("07_cursor_slot")
    n0 = party_count()
    press("A", settle=90)                           # pick the mon -> action submenu
    shot("08_minimenu")
    press("A", settle=180)                          # DEPOSIT -> box select / deposit animation
    shot("09_boxselect")
    press("A", settle=260)                          # confirm
    shot("10_after_deposit")
    # ── 4) leave the box system + the Center, whatever state we're in ───────────────────
    for i in range(12):
        if tuple(tv.map_id(b))[0] == 3:
            break
        press("B", settle=40)
        if i == 6:
            shot("11_escaping")
    camp._exit_to_overworld()
    n1 = party_count()                              # verify AFTER settle/exit — the mid-animation
    #                                                 read showed a compacting 6-with-ghost party
    L(f"party count {n0} -> {n1} | party now {party_species()}")
    L(f"out: map={tv.map_id(b)} coords={tv.coords(b)}")
    shot("12_out")

    if n1 != n0 - 1:
        L("!! DEPOSIT FAILED (party count unchanged) — calibrate with the shots")
        return 1
    L("DEPOSIT VERIFIED (party count dropped by 1)")
    # bank the post-deposit state for chaining (voltorb catch rides the next script)
    os.makedirs(STAGE, exist_ok=True)
    with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
        f.write(b.save_state())
    for f_ in ("world_model.json", "strat_memory.json", "soul.json", "journey_core.json"):
        src = os.path.join(CANON, f_)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(STAGE, f_))
    if os.path.isdir(BANK):
        shutil.rmtree(BANK, ignore_errors=True)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK} (sidecars carried from canonical; deposit is game-state only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
