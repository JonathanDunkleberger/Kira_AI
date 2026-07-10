"""recon_lapras.py — PC-BOX WITHDRAW (competency #15's second half): LAPRAS joins the party.

The Surf chain's first link (badge 7 = Blaine needs Surf; Venusaur can't learn it; LAPRAS —
the Silph gift, banked to Bill's PC party-full — is the natural carrier). This strike:
  1. RAM-locates Lapras in storage: gPokemonStoragePtr = 0x03005010 (pret pokefirered — the
     third consecutive DMA-protected pointer after SB1/SB2), currentBox u8 @+0, boxes @+4,
     BoxPokemon = 80 bytes with the SAME PID^OTID substruct encryption as party mons (species
     in Growth @+32). Cross-check: the 2026-07-06 deposited Spearow must also be found.
  2. Deposits the dead-weight bench tail (Mankey L10, party slot 5) — withdrawal needs a free
     party slot (recon_pcbox's calibrated deposit flow verbatim).
  3. WITHDRAWs Lapras (storage menu TOP item; box applet grid = 6 cols x 5 rows; submenu top
     item = WITHDRAW), verifies party count 5->6 and slot-5 species == 131 by RAM.
  4. Banks -> %TEMP%/longrun/banked_LAPRAS for promote_bank (label lapras_party).
Boot = canonical (sabrina_badge6: Saffron (3,10)@(46,13); Center door (24,38) -> (14,6)).
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_lapras.py [probe|run]
"""
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

import numpy as np                   # noqa: E402
from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import firered_ram as ram            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
import json                          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_lapras")
BANK = os.path.join(SCRATCH, "banked_LAPRAS")
SHOTS = os.path.join(SCRATCH, "lapras_probe")


def _resolve_state(name):
    """Resolve a state BASENAME or path to (state_path, sidecar_dir, sidecar_prefix).
    Kit fixtures live in states/workshop as <name>.state + <name>.<sidecar>.json; the credits
    canonical is states/campaign/kira_campaign.state with bare-named sidecars."""
    if not name:
        return (os.path.join(CANON, "kira_campaign.state"), CANON, "kira_campaign")
    for cand in (name, os.path.join(_HERE, "states", name),
                 os.path.join(_HERE, "states", "workshop", name),
                 os.path.join(_HERE, "states", name + ".state"),
                 os.path.join(_HERE, "states", "workshop", name + ".state")):
        if os.path.exists(cand):
            d = os.path.dirname(cand)
            base = os.path.basename(cand)
            pref = base[:-6] if base.endswith(".state") else base
            return (cand, d, pref)
    return (name, os.path.dirname(name) or CANON, "kira_campaign")

GSTORAGE_PTR = 0x03005010            # gPokemonStoragePtr (pret pokefirered)
BOX_MON_SIZE = 80
BOXES, PER_BOX = 14, 30
SAFFRON = (3, 10)
PC_INTERIOR = (14, 6)                # Saffron Center (campaign disasm bill 2026-07-07)
MB_PC = 0x83                         # the PC console METATILE BEHAVIOR (behavior-scan truth
#                                      2026-07-07: Saffron's console is at (11,1) — NOT the
#                                      recon_pcbox (13,4) stand; 'shared PC layout' is false
#                                      for stand tiles. Scanning 0x83 generalizes to ANY room.)
LAPRAS, SPEAROW, MANKEY = 131, 21, 56
DEPOSIT_SLOT = 5                     # party slot (0-idx) = Mankey L10, the bench tail


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    os.makedirs(SHOTS, exist_ok=True)
    os.makedirs(STAGE, exist_ok=True)
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("LAPRAS_STATE", ""))
    b = Bridge(ROM)
    with open(state_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def shot(name):
        try:
            b.frame_rgb().save(os.path.join(SHOTS, f"{name}.png"))
        except Exception as e:
            L(f"   shot {name} failed: {e}")

    def press(key, settle=30, hold=8, rel=10):
        b.press(key, hold, rel, owner="agent")
        for _ in range(settle):
            b.run_frame()

    def party_count():
        return b.rd8(ram.GPLAYER_PARTY_CNT)

    def party_species():
        return [st.SPECIES_NAME.get(st.read_party_species(b, s), "?")
                for s in range(party_count())]

    def box_mon(box, slot):
        """Decrypt a BoxPokemon's species (same substruct scheme as the party read)."""
        base = b.rd32(GSTORAGE_PTR) + 4 + (box * PER_BOX + slot) * BOX_MON_SIZE
        pid = b.rd32(base + 0)
        if pid == 0 and b.rd32(base + 4) == 0:
            return 0
        key = pid ^ b.rd32(base + 4)
        order = st._SUBSTRUCT_ORDER[pid % 24]
        return (b.rd32(base + 32 + order.index("G") * 12) ^ key) & 0xFFFF

    def storage_scan():
        cur_box = b.rd8(b.rd32(GSTORAGE_PTR))
        found = {}
        for bx in range(BOXES):
            for sl in range(PER_BOX):
                sp = box_mon(bx, sl)
                if sp:
                    found[(bx, sl)] = sp
        return cur_box, found

    def runner():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=180)

    camp = Campaign(b, battle_runner=runner, on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    # sidecars: prefer the resolved state's own sidecars (kit fixtures), else the canonical paths
    _w_side = os.path.join(sc_dir, sc_pref + ".world_model.json")
    _s_side = os.path.join(sc_dir, sc_pref + ".strat_memory.json")
    _soul_side = os.path.join(sc_dir, sc_pref + ".soul.json")
    for loader, path, fallback in (
            (camp.world.load, _w_side, C.WORLD_JSON),
            (camp.strat.load, _s_side, C.STRAT_JSON)):
        try:
            loader(path if os.path.exists(path) else fallback)
        except Exception:
            pass
    try:
        if camp.soul is not None:
            camp.soul.load(_soul_side if os.path.exists(_soul_side)
                           else os.path.join(CANON, "soul.json"))
    except Exception:
        pass

    cur_box, found = storage_scan()
    named = {k: st.SPECIES_NAME.get(v, f"#{v}") for k, v in found.items()}
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} party={party_species()}")
    L(f"storage: current_box={cur_box} occupied={named}")
    lap = next((k for k, v in found.items() if v == LAPRAS), None)
    if lap is None:
        L("!! LAPRAS not found in any box — aborting")
        return 1
    # STORAGE-READ SANITY: the decrypt scheme is trustworthy only if it also decodes a SECOND
    # occupant to a valid species (credits-line = the deposited Spearow; kit line = the boxed
    # bug-catch bench). A lone Lapras with everything else zero would mean a bad read.
    others = {k: v for k, v in found.items() if v != LAPRAS and 1 <= v <= 411}
    if not others:
        L("!! cross-check FAILED: no second valid box occupant decoded — the storage read is "
          "not trustworthy, aborting before any menu drive")
        return 1
    L(f"   LAPRAS at box {lap[0]} slot {lap[1]} | cross-check occupants "
      f"{ {k: st.SPECIES_NAME.get(v, v) for k, v in list(others.items())[:4]} } | "
      f"current box {cur_box}")
    if mode == "probe":
        return 0
    if lap[0] != cur_box:
        L(f"!! LAPRAS is in box {lap[0]} but the applet opens on box {cur_box} — box "
          f"switching is unbuilt; aborting LOUD (build the title-bar hop)")
        return 1

    # ── to the Saffron Center PC ──────────────────────────────────────────────────────
    if tuple(tv.map_id(b)) == SAFFRON:
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == PC_INTERIOR]
        if not cands:
            L("!! no Center door found on Saffron — abort")
            return 1
        r = camp.enter_warp(pick=cands[0])
        if tuple(tv.map_id(b)) != PC_INTERIOR:
            L(f"!! Center entry failed ({r}, at {tv.map_id(b)}) — abort")
            return 1
    if tuple(tv.map_id(b)) != PC_INTERIOR:
        L(f"!! not in the Center (at {tv.map_id(b)}) — abort")
        return 1
    # find the console by METATILE BEHAVIOR (0x83 = MB_PC) — layout-agnostic
    w_ = b.rd32(tv.BACKUP_LAYOUT)
    mp_ = b.rd32(tv.BACKUP_LAYOUT + 8)
    ml_ = b.rd32(tv.GMAPHEADER)
    attr_ = (b.rd32(b.rd32(ml_ + 0x10) + 0x14), b.rd32(b.rd32(ml_ + 0x14) + 0x14))
    g0 = tv.Grid(b)
    pc_tile = None
    for sy in range(0, g0.sy_hi + 1):
        for sx in range(0, g0.sx_hi + 1):
            e = b.rd16(mp_ + ((sy + tv.MAP_OFFSET) * w_ + (sx + tv.MAP_OFFSET)) * 2)
            mid = e & 0x3FF
            bs, ix = (attr_[0], mid) if mid < tv.NUM_PRIMARY else (attr_[1], mid - tv.NUM_PRIMARY)
            if (b.rd32(bs + ix * 4) & 0xFF) == MB_PC:
                pc_tile = (sx, sy)
                break
        if pc_tile:
            break
    if pc_tile is None:
        L("!! no MB_PC (0x83) tile in this room — abort")
        return 1
    stand = (pc_tile[0], pc_tile[1] + 1)
    L(f"   PC console at {pc_tile} (behavior scan) — standing {stand}")
    camp._step_to(stand)
    camp._step_to(stand)                           # tap-turn: the second call finishes a turn
    if tuple(tv.coords(b) or ()) != stand:
        if not camp.trav.travel(target_map=None, arrive_coord=stand,
                                max_steps=60, max_seconds=30) == "arrived":
            L(f"!! can't reach the PC stand {stand} (at {tv.coords(b)}) — abort")
            shot("00_no_stand")
            return 1
    press("UP", settle=16)
    if tuple(tv.coords(b) or ()) != stand:         # the face press WALKED (already facing UP)
        camp._step_to(stand)
        camp._step_to(stand)
        press("UP", settle=16)
    press("A", settle=60)                          # boot the PC
    shot("01_pc_boot")

    def _px():
        return np.asarray(b.frame_rgb(), dtype=np.int32)

    def _menu_open(img):
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
    for _ in range(6):
        press("A", settle=45)
        if _menu_open(_px()):
            ok = True
            break
    if not ok:
        L("!! PC-choice menu never appeared — abort (see shots)")
        shot("02_no_pcmenu")
        camp._exit_to_overworld()
        return 1
    base_tr = _px()[4:28, 188:236]

    def _applet():
        return np.abs(_px()[4:28, 188:236] - base_tr).mean() > 30

    press("A", settle=45)                          # BILL'S PC
    storage = False
    for _ in range(8):
        if _menu_open(_px()):
            storage = True
            break
        press("A", settle=70)
    shot("03_storage_menu")
    if not storage:
        L("!! storage menu never appeared — abort")
        camp._exit_to_overworld()
        return 1

    # ── deposit Mankey (slot 5) — the calibrated recon_pcbox flow ─────────────────────
    n0 = party_count()
    press("DOWN", settle=30)                       # WITHDRAW -> DEPOSIT
    base_tr = _px()[4:28, 188:236]
    press("A", settle=45)
    if not _wait(_applet, frames=700):
        L("!! deposit GUI never opened — abort")
        shot("04_no_deposit_gui")
        camp._exit_to_overworld()
        return 1
    for _ in range(160):
        b.run_frame()
    for _ in range(DEPOSIT_SLOT):
        press("DOWN", settle=24)
    shot("05_cursor_mankey")
    press("A", settle=90)                          # -> action submenu
    press("A", settle=180)                         # DEPOSIT -> box select
    press("A", settle=260)                         # confirm
    shot("06_after_deposit")
    # back out of the deposit GUI to the storage menu
    esc = 0
    while not _menu_open(_px()) and esc < 10:
        press("B", settle=50)
        esc += 1
    for _ in range(60):
        b.run_frame()
    nd = party_count()
    L(f"   deposit: party {n0} -> {nd} ({party_species()})")
    if nd != n0 - 1:
        L("!! deposit did not land — abort (see shots)")
        shot("07_deposit_fail")
        camp._exit_to_overworld()
        return 1

    # ── WITHDRAW Lapras ───────────────────────────────────────────────────────────────
    if not _menu_open(_px()):
        L("!! storage menu didn't re-open after the deposit — abort")
        shot("08_no_menu_back")
        camp._exit_to_overworld()
        return 1
    # re-read Lapras' slot — the deposit may have shifted nothing, but truth is cheap
    _cb2, found2 = storage_scan()
    lap = next((k for k, v in found2.items() if v == LAPRAS), None)
    L(f"   post-deposit storage: lapras at {lap}, current box {_cb2}")
    if lap is None or lap[0] != _cb2:
        L("!! lapras not in the open box after deposit — abort")
        camp._exit_to_overworld()
        return 1
    base_tr = _px()[4:28, 188:236]
    # the storage list REMEMBERS its cursor (still on DEPOSIT after the deposit leg —
    # the strike's first run reopened the deposit GUI and nearly STORE'd Persian);
    # one UP puts it on WITHDRAW.
    press("UP", settle=30)
    press("A", settle=45)                          # WITHDRAW
    if not _wait(_applet, frames=700):
        L("!! withdraw GUI never opened — abort")
        shot("09_no_withdraw_gui")
        camp._exit_to_overworld()
        return 1
    for _ in range(160):
        b.run_frame()
    shot("10_withdraw_gui")
    row, col = lap[1] // 6, lap[1] % 6
    for _ in range(row):
        press("DOWN", settle=24)
    for _ in range(col):
        press("RIGHT", settle=24)
    shot("11_cursor_lapras")
    press("A", settle=90)                          # -> submenu (WITHDRAW on top)
    shot("12_submenu")
    press("A", settle=260)                         # WITHDRAW -> fly-to-party animation
    shot("13_after_withdraw")
    esc = 0
    while not _menu_open(_px()) and esc < 10:
        press("B", settle=50)
        esc += 1
    n1 = party_count()
    got = [st.read_party_species(b, s) for s in range(n1)]
    L(f"   withdraw: party {nd} -> {n1} | species {got}")
    if n1 != nd + 1 or LAPRAS not in got:
        L("!! WITHDRAW FAILED — abort (see shots)")
        shot("14_withdraw_fail")
        camp._exit_to_overworld()
        return 1

    # ── out + bank ────────────────────────────────────────────────────────────────────
    for i in range(12):
        if tuple(tv.map_id(b)) == SAFFRON:
            break
        press("B", settle=40)
        if i >= 6:
            camp._exit_to_overworld()
    L(f"   out: map={tv.map_id(b)} coords={tv.coords(b)} party={party_species()}")
    shot("15_out")

    with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
        f.write(b.save_state())
    try:
        camp.world.save(os.path.join(STAGE, "world_model.json"))
        camp.strat.save(os.path.join(STAGE, "strat_memory.json"))
        if camp.soul is not None:
            camp.soul.save(os.path.join(STAGE, "soul.json"))
        with open(os.path.join(STAGE, "journey_core.json"), "w", encoding="utf-8") as jf:
            json.dump(camp._journey_narrative(), jf, ensure_ascii=False, indent=2)
    except Exception as e:
        L(f"!! stage continuity failed: {e}")
    if os.path.isdir(BANK):
        shutil.rmtree(BANK, ignore_errors=True)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} lapras_party")
    return 0


if __name__ == "__main__":
    sys.exit(main())
