"""recon_dest_interact.py — verify the general DESTINATION-INTERACTION layer in ISOLATION (no gauntlet).

The full Bill chain is blocked by the Nugget-Bridge gauntlet (underlevel — needs a levelled save). But the
destination-interaction LAYER (enter building -> talk NPC -> re-check flag -> exit-if-wrong) is general and
can be exercised at ANY building. This drives her to Cerulean (an overworld map with buildings) and invokes
`_questline_interact` directly, confirming the composition:

  1. OVERWORLD: picks an un-entered building door + enters it (group flips to an interior).  -> questline_entered
  2. INTERIOR: talks the occupant (nurse/clerk).                                              -> questline_talked
  3. FLAG-CHECK: FLAG_GOT_SS_TICKET stays False (it's not Bill) -> the questline does NOT false-complete.
  4. WRONG-BUILDING: once everyone's talked and the flag's unset, releases `_ql_inside_target` + EXITS to the
     overworld (cooperating with the blackout-recovery) so she'd move on to the next candidate building.

This proves the layer is wired + safe (it can't false-trigger a flag). The Bill-specific flag SET is the only
bit that needs the real Bill (a levelled save to cross the gauntlet). Read-only; never banks a canonical save.
RUN: python pokemon_agent\\recon_dest_interact.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                 # noqa: E402
import firered_ram as ram                                 # noqa: E402
import field_moves as fm                                  # noqa: E402
import travel as tv                                       # noqa: E402
from campaign import Campaign, resolve_state, WORLD_JSON  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SS_TICKET = 0x234


def _heal(b):
    for s in range(ram.read_party_count(b)):
        base = ram.GPLAYER_PARTY + s * 100
        b.core.memory.u16.raw_write(base + 0x56, b.rd16(base + 0x58))


def main():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    _heal(b)
    camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                    on_event=lambda s, **k: print(f"   [event] {s}", flush=True),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp.world.load(WORLD_JSON)
    camp.trav.battle_runner = camp._flee_runner

    # get onto a CITY overworld map with buildings (the live save is on Route 4 → east into Cerulean)
    if tv.map_id(b) != (3, 3):
        print(f"   driving {tv.map_id(b)} -> Cerulean (3,3) for the test…", flush=True)
        camp.trav.travel(target_map=(3, 3), edge="east")
        _heal(b)
    if tv.map_id(b)[0] != 3:                      # landed inside a building (e.g. healed in the Center)
        print(f"   landed in interior {tv.map_id(b)} — exiting to the overworld for the test…", flush=True)
        camp._exit_to_overworld()
        camp._wait_overworld()
        _heal(b)
    print(f"==== START map={tv.map_id(b)} coords={tv.coords(b)} group={tv.map_id(b)[0]} "
          f"(3 = overworld) FLAG={fm.read_flag(b, SS_TICKET)} ====\n", flush=True)
    if tv.map_id(b)[0] != 3:
        print("   (not on an overworld map — can't run the building test from here) INSPECT", flush=True)
        return

    # open the questline so there's an actionable talk_npc step (Bill / S.S. Ticket)
    gate = camp._gate_recognizer.recognize(tuple(tv.map_id(b)), blocked_dir="south")
    camp._active_questline = None
    if gate:
        camp._open_questline(gate, camp.read_live_state())
    q = camp._active_questline
    step = q.actionable if q else None
    print(f"   questline open={q is not None}  step.via={getattr(step,'via',None)}  "
          f"npc={getattr(step,'npc',None)}\n", flush=True)
    if step is None or step.via != "talk_npc":
        print("   (no talk_npc step) INSPECT", flush=True)
        return

    doors_here = camp._door_tiles()
    print(f"   buildings (door tiles) reachable on this map: {len(doors_here)}", flush=True)

    entered = talked = exited = False
    false_complete = False
    for i in range(1, 7):
        state = camp.read_live_state()
        grp = tv.map_id(b)[0]
        r = camp._questline_interact(state, step)
        grp2 = tv.map_id(b)[0]
        print(f"   step {i}: group {grp}->{grp2}  interact -> {r}  inside_target={camp._ql_inside_target}  "
              f"FLAG={fm.read_flag(b, SS_TICKET)}", flush=True)
        if r == "questline_entered" and grp2 != 3:
            entered = True
        if r == "questline_talked":
            talked = True
        if r == "questline_wrong_building" and tv.map_id(b)[0] == 3:
            exited = True
        # the questline must NOT self-complete on a non-Bill NPC
        camp._active_questline = camp._derive_questline(q.gate)
        if camp._active_questline.complete:
            false_complete = True
        if exited:
            break

    print("\n---- checks ----", flush=True)
    print(f"  entered a building (overworld -> interior):           {entered}", flush=True)
    print(f"  talked an NPC inside (questline_talked):              {talked}", flush=True)
    print(f"  recognised WRONG building + exited to overworld:      {exited}", flush=True)
    print(f"  did NOT false-complete the questline on a non-Bill:   {not false_complete}", flush=True)
    print(f"  FLAG_GOT_SS_TICKET still correctly False:             {not fm.read_flag(b, SS_TICKET)}", flush=True)
    ok = entered and talked and (not false_complete) and (not fm.read_flag(b, SS_TICKET))
    print(f"\n==== destination-interaction layer (mechanics): {'PASS' if ok else 'INSPECT'} ====", flush=True)
    print("   NOTE: Bill-specific FLAG SET needs the real Bill (a levelled save to cross Nugget Bridge).",
          flush=True)


if __name__ == "__main__":
    main()
