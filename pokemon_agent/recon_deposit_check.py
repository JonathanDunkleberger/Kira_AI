"""recon_deposit_check.py — VERIFY the PC/BOX slice (Tier-1 #15, NS#38):
  PART A (decision logic, instant): _worst_chaff_slot picks the lowest-level OFF-PLAN non-lead mon,
    protects the lead + on-plan lines; _chaff_swap_target gates on flag / party-full / catch_keeper-due /
    boxable-chaff / mapped-Center-city. Run against a REAL party-6 fixture (erika_done) + synthetic cases.
  PART B (live actuation, ~90s): boot surge_done (party 4 at Vermilion, a mapped-Center city) and drive
    deposit_mon(slot 3, VERMILION_PC_DOOR) for real -> assert the party count drops 4 -> 3 by RAM.
Read-only w.r.t. canonical (RAM copies + workshop fixtures only; banks nothing).
Run: POKEMON_PCBOX=1 ../.venv/Scripts/python.exe recon_deposit_check.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("POKEMON_PCBOX", "1")            # arm the gate for the test
os.environ.setdefault("POKEMON_TEAM_PLANNER", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge
import travel as tv
import firered_ram as ram
import campaign as C
from campaign import Campaign, resolve_state

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
_fails = []


def _ck(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        _fails.append(name)


def _mk_camp(boot):
    b = Bridge(ROM)
    with open(resolve_state(boot), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "win",
                    on_event=lambda s, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    camp._continuity_load = lambda *a, **k: None
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    try:
        camp.world.load(C.WORLD_JSON)
    except Exception:
        pass
    return b, camp


def part_a():
    print("== PART A: decision + selection logic ==")
    b, camp = _mk_camp("erika_done.state")
    ls = camp.read_live_state()
    party = ls.get("party") or []
    names = [(m.get("species"), m.get("level")) for m in party]
    print(f"  erika_done party: {names}")
    # init the persistent plan exactly as the real flow does (_chaff_swap_target calls assess() first)
    camp.team_planner.assess(party, ls.get("badge_count", 0), bag=ls.get("bag"),
                             dex=ls.get("dex_caught"), post_game=bool(ls.get("post_game")))

    # _worst_chaff_slot: a real, off-plan, non-lead, lowest-level slot; never slot 0
    slot = camp._worst_chaff_slot(party)
    _ck("A worst_chaff_slot returns a slot (party-6 chaff exists)", isinstance(slot, int))
    _ck("A worst_chaff_slot is NOT the lead (slot 0)", slot != 0)
    if isinstance(slot, int):
        picked = party[slot]
        # it is the lowest level among off-plan non-lead mons
        offplan = []
        for i, m in enumerate(party):
            if i == 0:
                continue
            sp = (m.get("species") or "").lower()
            try:
                on = bool(camp.team_planner._is_target_line(sp))
            except Exception:
                on = False
            if not on:
                offplan.append((m.get("level", 0), i))
        offplan.sort()
        _ck("A worst_chaff picks the LOWEST-level off-plan mon",
            bool(offplan) and slot == offplan[0][1])
        _ck("A picked mon is OFF-PLAN (not a planned keeper line)",
            not camp.team_planner._is_target_line((picked.get("species") or "").lower()))

    # synthetic: all-on-plan party -> None (nothing to box). Fake the planner to claim every species on-plan.
    _orig = camp.team_planner._is_target_line
    camp.team_planner._is_target_line = lambda s: True
    _ck("A all-on-plan -> None (never box a keeper)", camp._worst_chaff_slot(party) is None)
    camp.team_planner._is_target_line = _orig
    # synthetic: single-mon party -> None
    _ck("A single-mon party -> None", camp._worst_chaff_slot(party[:1]) is None)

    # _chaff_swap_target gate. erika_done is a group-10 interior (no mapped PC door) — assert the
    # gate refuses THERE (can't deposit), then assert the positive path from a mapped-Center city.
    tgt_here = camp._chaff_swap_target(ls)
    _ck("A gate None when not in a mapped-Center city (erika_done interior)", tgt_here is None)

    # flag OFF -> None regardless
    C.PCBOX_ENABLED = False
    _ck("A gate None when flag OFF", camp._chaff_swap_target(ls) is None)
    C.PCBOX_ENABLED = True

    # positive gate: from surge_done's Vermilion, force a full off-plan party + a catch_keeper due.
    b2, camp2 = _mk_camp("surge_done.state")
    ls2 = camp2.read_live_state()
    # synthesize a party-6 of off-plan chaff so assess wants a keeper and chaff exists
    fake_party = [{"species": "venusaur", "level": 31}, {"species": "rattata", "level": 15},
                  {"species": "spearow", "level": 15}, {"species": "ekans", "level": 9},
                  {"species": "meowth", "level": 10}, {"species": "pidgey", "level": 13}]
    ls2 = dict(ls2); ls2["party"] = fake_party; ls2["party_count"] = 6
    # ROUTABILITY GATE (NS#39): the gate now also requires the due keeper to be fetchable. Drive that
    # deterministically (independent of the loaded world graph) by stubbing _reachable_keeper_host /
    # _species_on_map, so the positive/negative cases prove the GATE logic, not world state.
    _orig_reach = camp2._reachable_keeper_host
    _orig_onmap = camp2._species_on_map
    camp2._reachable_keeper_host = lambda *a, **k: (3, 20)     # a reachable hosting map
    camp2._species_on_map = lambda *a, **k: False             # keeper NOT on this map -> router path
    tgt = camp2._chaff_swap_target(ls2)
    _ck("A gate FIRES when keeper is off-map but ROUTABLE (returns (slot,door))",
        isinstance(tgt, tuple) and len(tgt) == 2 and tgt[1] == C.VERMILION_PC_DOOR)
    # NS#39 core: keeper off-map AND un-routable -> refuse to box (no box-for-nothing that thins the team)
    camp2._reachable_keeper_host = lambda *a, **k: None
    _ck("A gate None when the due keeper is off-map AND un-routable (NS#39 refusal)",
        camp2._chaff_swap_target(ls2) is None)
    # keeper already ON this map -> box (the on-map un-gate catches it once there's room), routability moot
    camp2._species_on_map = lambda *a, **k: True
    _ck("A gate FIRES when the keeper is already on THIS map (on-map catch after room)",
        isinstance(camp2._chaff_swap_target(ls2), tuple))
    camp2._reachable_keeper_host = _orig_reach
    camp2._species_on_map = _orig_onmap
    # party with room (5) -> None (router catches directly, no swap needed)
    ls3 = dict(ls2); ls3["party"] = fake_party[:5]; ls3["party_count"] = 5
    _ck("A gate None when party has room (<6)", camp2._chaff_swap_target(ls3) is None)


def part_b():
    print("== PART B: live deposit_mon actuation (surge_done @ Vermilion) ==")
    b, camp = _mk_camp("surge_done.state")
    n0 = b.rd8(ram.GPLAYER_PARTY_CNT)
    print(f"  boot map={tv.map_id(b)} coords={tv.coords(b)} party_count={n0}")
    if n0 < 4:
        _ck("B fixture has a deposit-able bench (party>=4)", False)
        return
    r = camp.deposit_mon(3, C.VERMILION_PC_DOOR)      # box the slot-3 bench mon
    n1 = b.rd8(ram.GPLAYER_PARTY_CNT)
    print(f"  deposit_mon -> {r} | party {n0} -> {n1}")
    _ck("B deposit_mon returns 'deposited'", r == "deposited")
    _ck("B party count dropped by exactly 1", n1 == n0 - 1)


def main():
    part_a()
    try:
        part_b()
    except Exception as e:
        _ck(f"B live deposit raised {type(e).__name__}: {e}", False)
    print()
    if _fails:
        print(f"FAILURES ({len(_fails)}): " + "; ".join(_fails))
        return 1
    print("ALL PC/BOX CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
