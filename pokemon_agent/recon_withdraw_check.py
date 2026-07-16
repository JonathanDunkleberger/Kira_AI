"""recon_withdraw_check.py — VERIFY the PC/BOX WITHDRAW slice (Tier-1 #15, NS#39):
  PART A (instant, RAM-truth): _box_scan decodes the boxed occupants of erika_done_kit (weedle/pidgey/
    caterpie in box 0) + the current open box — the read the swap-in hook relies on.
  PART B (live actuation, ~3min): erika_done_kit is a FULL party-6 at Celadon (a mapped-Center city) with
    3 mons in box 0. Drive a full ROUND-TRIP through one Center visit: deposit_mon(worst_chaff) 6->5, then
    withdraw_mon(box 0, slot 0) 5->6 — assert the party count round-trips by RAM AND the withdrawn species
    (Weedle) lands in the party. Also checks the guards (full-party -> 'full', wrong box -> 'wrong_box').
Read-only w.r.t. canonical (RAM copy of a workshop fixture; banks nothing).
Run: POKEMON_PCBOX=1 ../.venv/Scripts/python.exe recon_withdraw_check.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_PCBOX", "1")
os.environ.setdefault("POKEMON_TEAM_PLANNER", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge
import travel as tv
import firered_ram as ram
import pokemon_state as st
import campaign as C
from campaign import Campaign, resolve_state

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WEEDLE = 13
_fails = []


def _ck(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        _fails.append(name)


def _mk_camp(boot):
    b = Bridge(ROM)
    sp = resolve_state(boot)
    with open(sp, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "win",
                    on_event=lambda s, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    camp._continuity_load = lambda *a, **k: None
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    # prefer the fixture's own world sidecar (nav needs it) else canonical
    side = os.path.join(os.path.dirname(sp), boot.replace(".state", ".world_model.json"))
    try:
        camp.world.load(side if os.path.exists(side) else C.WORLD_JSON)
    except Exception:
        pass
    return b, camp


def part_a():
    print("== PART A: _box_scan RAM-truth ==")
    b, camp = _mk_camp("erika_done_kit.state")
    cb, occ = camp._box_scan()
    named = {k: st.SPECIES_NAME.get(v, v) for k, v in sorted(occ.items())}
    print(f"  boot map={tv.map_id(b)} current_box={cb} occupants={named}")
    _ck("A _box_scan finds the 3 boxed occupants", len(occ) == 3)
    _ck("A current open box is 0", cb == 0)
    _ck("A Weedle decoded at box0 slot0", occ.get((0, 0)) == WEEDLE)
    # guard: withdraw on a FULL party -> 'full' (no menu opened)
    r_full = camp.withdraw_mon(0, 0, C.CELADON_PC_DOOR)
    _ck("A withdraw on a full party returns 'full'", r_full == "full")


def part_b():
    print("== PART B: live deposit->withdraw round-trip (erika_done_kit @ Celadon) ==")
    b, camp = _mk_camp("erika_done_kit.state")
    ls = camp.read_live_state()
    party = ls.get("party") or []
    camp.team_planner.assess(party, ls.get("badge_count", 0), bag=ls.get("bag"),
                             dex=ls.get("dex_caught"), post_game=bool(ls.get("post_game")))
    n0 = b.rd8(ram.GPLAYER_PARTY_CNT)
    slot = camp._worst_chaff_slot(party)
    print(f"  boot map={tv.map_id(b)} party_count={n0} worst_chaff_slot={slot}")
    if n0 != 6 or not isinstance(slot, int):
        _ck("B fixture is a full party-6 with boxable chaff", False)
        return
    # 1) deposit the worst chaff (6 -> 5)
    rd = camp.deposit_mon(slot, C.CELADON_PC_DOOR)
    n1 = b.rd8(ram.GPLAYER_PARTY_CNT)
    print(f"  deposit_mon -> {rd} | party {n0} -> {n1}")
    _ck("B deposit_mon returns 'deposited'", rd == "deposited")
    _ck("B party dropped to 5", n1 == 5)
    if n1 != 5:
        return
    # 2) withdraw the boxed Weedle (box 0 slot 0) (5 -> 6)
    rw = camp.withdraw_mon(0, 0, C.CELADON_PC_DOOR)
    n2 = b.rd8(ram.GPLAYER_PARTY_CNT)
    got = [st.read_party_species(b, s) for s in range(n2)]
    print(f"  withdraw_mon -> {rw} | party {n1} -> {n2} | species {got}")
    _ck("B withdraw_mon returns 'withdrawn'", rw == "withdrawn")
    _ck("B party rose to 6", n2 == 6)
    _ck("B the withdrawn Weedle is in the party", WEEDLE in got)


def part_c():
    print("== PART C: swap_keeper gate + errand (boxed keeper -> fielded) ==")
    # erika_done_kit's box holds only chaff, so no fixture has a boxed KEEPER. Exercise the MACHINERY by
    # wrapping _is_target_line so Weedle (box0 slot0) counts on-plan (real keeper lines preserved) — the
    # gate should then see a boxed keeper + a party chaff to deposit, and the errand should swap them.
    b, camp = _mk_camp("erika_done_kit.state")
    ls = camp.read_live_state()
    party = ls.get("party") or []
    camp.team_planner.assess(party, ls.get("badge_count", 0), bag=ls.get("bag"),
                             dex=ls.get("dex_caught"), post_game=bool(ls.get("post_game")))
    _orig = camp.team_planner._is_target_line
    camp.team_planner._is_target_line = lambda s: (s or "").lower() == "weedle" or _orig(s)
    # gate: boxed Weedle is "on-plan" + party full + a boxable chaff -> (box0, slot0, chaff_slot, door)
    tgt = camp._box_keeper_swap_target(ls)
    ok_gate = (isinstance(tgt, tuple) and len(tgt) == 4 and tgt[0] == 0 and tgt[1] == 0
               and isinstance(tgt[2], int) and tgt[2] != 0 and tgt[3] == C.CELADON_PC_DOOR)
    _ck("C gate returns (box0, slot0, chaff_slot, door) when a keeper is boxed at full party", ok_gate)
    # negative: with no on-plan boxed mon (default plan), the gate is silent
    camp.team_planner._is_target_line = _orig
    _ck("C gate None when nothing on-plan is boxed", camp._box_keeper_swap_target(ls) is None)
    # end-to-end errand: swap the boxed Weedle in for the worst chaff (party stays 6, composition improves)
    camp.team_planner._is_target_line = lambda s: (s or "").lower() == "weedle" or _orig(s)
    n0 = b.rd8(ram.GPLAYER_PARTY_CNT)
    chaff_slot = tgt[2] if ok_gate else camp._worst_chaff_slot(party)
    chaff_sp = st.read_party_species(b, chaff_slot)
    r = camp._swap_keeper_errand(ls)
    n1 = b.rd8(ram.GPLAYER_PARTY_CNT)
    got = [st.read_party_species(b, s) for s in range(n1)]
    print(f"  swap_keeper_errand -> {r} | party {n0}->{n1} | chaff sp{chaff_sp} out | species {got}")
    _ck("C errand returns 'swapped'", r == "swapped")
    _ck("C party count stays 6 (chaff-for-keeper)", n1 == 6)
    _ck("C the boxed Weedle is now on the active team", WEEDLE in got)
    _ck("C the deposited chaff left the party", chaff_sp not in got)


def main():
    part_a()
    try:
        part_b()
    except Exception as e:
        import traceback
        traceback.print_exc()
        _ck(f"B round-trip raised {type(e).__name__}: {e}", False)
    try:
        part_c()
    except Exception as e:
        import traceback
        traceback.print_exc()
        _ck(f"C swap raised {type(e).__name__}: {e}", False)
    print()
    if _fails:
        print(f"FAILURES ({len(_fails)}): " + "; ".join(_fails))
        return 1
    print("ALL WITHDRAW CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
