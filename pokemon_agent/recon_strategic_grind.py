"""recon_strategic_grind.py — ISOLATED checks for the STRATEGIC UNDERLEVEL-GRIND capability (Task B).

Boots a FRESH core from the live campaign save and verifies the new pieces WITHOUT needing real
battles (the loop control + reorder mechanics are exercised in RAM; real leveling happens via real
battles in a live run). Six checks:

  C1 PARTY-REORDER SAVE-SAFETY — _swap_party_slots moves two 100-byte structs INTACT: after a swap the
     slots are byte-for-byte exchanged, and a swap-back restores the party byte-for-byte. (THE save-safety
     proof — an intact struct move changes no checksum, exactly like the in-menu 'switch order'.)
  C2 RECOGNITION — with an injected active wall (foe Lv13) and an under-levelled team (floor < 14),
     _prep_team_target returns 14; with a LOW foe (Lv5, target 6 <= floor) it returns None (that's a
     type/strategy loss, NOT an underlevel problem — distinguished, not mis-fired).
  C3 GRIND-WEAK LOOP — with grind() stubbed to 'level' slot 0 to target, grind_weak_members fields the
     WEAKEST under-target member each pass, exits when the team FLOOR crosses, and RESTORES the ace
     (highest-level mon) to slot 0. Species follow their structs through every swap.
  C4 ACTION-SET FRAMING — when prep fires, 'battle' is reframed to the STRENGTHEN-the-weak-ones plan and
     forward-drive STANDS DOWN (battle stays available at the wall — the smart middle, not the
     stubborn-charge prune).
  C5 REVERSIBILITY — POKEMON_STRATEGIC_GRIND=0 -> _prep_team_target None, no reframe, the executor falls
     back to the ordinary grind(lead+2). Strictly conditional on the flag.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                                 # noqa: E402
import firered_ram as ram                                 # noqa: E402
import pokemon_state as st                                # noqa: E402
import travel as tv                                       # noqa: E402
import campaign as C                                      # noqa: E402
import pokemon_strategy as ps                              # noqa: E402
from campaign import Campaign, resolve_state, WORLD_JSON  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
PMON = st.PARTY_MON_SIZE
LVL_OFF = 0x54


def _heal(b):
    for s in range(ram.read_party_count(b)):
        base = ram.GPLAYER_PARTY + s * PMON
        b.core.memory.u16.raw_write(base + 0x56, b.rd16(base + 0x58))


def _fresh():
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    _heal(b)
    camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp.world.load(WORLD_JSON)
    return b, camp


def _slot_bytes(b, s):
    base = ram.GPLAYER_PARTY + s * PMON
    return bytes(b.rd8(base + i) for i in range(PMON))


def _species_levels(b, camp):
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    sp = [st.SPECIES_NAME.get(st.read_party_species(b, s), f"#{s}") for s in range(min(cnt, 6))]
    return list(zip(sp, camp._party_levels()))


def _inject_wall(camp, foe_level, place="the Nugget Bridge", map_id=(3, 43)):
    key = f"trainer:{place}:sig"
    camp.strat.losses[key] = {
        "count": 1, "name": "a trainer", "lead": "Onix", "is_trainer": True,
        "lead_level": foe_level, "place": place, "types": ["rock", "ground"],
        "size": 2, "my_party": 3, "my_level": 24, "map_id": tuple(map_id), "coords": map_id,
        "roster": ["Onix", "Geodude"],
    }
    camp.strat.active_wall = key


def main():
    out = []
    ok = {}

    # ── C1 — PARTY-REORDER SAVE-SAFETY (byte-for-byte intact move) ────────────────────────────────
    b, camp = _fresh()
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    out.append(f"START map={tv.map_id(b)} party_count={cnt}")
    out.append(f"      party: {_species_levels(b, camp)}")
    if cnt >= 2:
        s0_before, s1_before = _slot_bytes(b, 0), _slot_bytes(b, 1)
        camp._swap_party_slots(0, 1)
        s0_swap, s1_swap = _slot_bytes(b, 0), _slot_bytes(b, 1)
        swapped_ok = (s0_swap == s1_before and s1_swap == s0_before)
        camp._swap_party_slots(0, 1)                       # swap back
        s0_back, s1_back = _slot_bytes(b, 0), _slot_bytes(b, 1)
        restored_ok = (s0_back == s0_before and s1_back == s1_before)
        ok["C1"] = swapped_ok and restored_ok
        out.append("")
        out.append(f"C1  swap 0<->1 exchanged structs byte-for-byte: {swapped_ok}")
        out.append(f"    swap-back restored party byte-for-byte:    {restored_ok}")
    else:
        ok["C1"] = None
        out.append("C1  SKIPPED — need >=2 party members in the save")

    # ── C2 — RECOGNITION (fires on real underlevel; ignores a higher-level/strategy loss) ──────────
    b, camp = _fresh()
    state = camp.read_live_state()
    floor = min(m["level"] for m in state["party"])
    _inject_wall(camp, foe_level=floor + 5)               # foe 5 above the floor -> underlevelled
    t_under = camp._prep_team_target(state)
    _inject_wall(camp, foe_level=max(1, floor - 3))       # foe BELOW the floor -> not a level problem
    t_strat = camp._prep_team_target(state)
    expect_under = floor + 5 + ps.UNDERLEVEL_MARGIN
    ok["C2"] = (t_under == expect_under) and (t_strat is None)
    out.append("")
    out.append(f"C2  team floor=L{floor}; foe L{floor+5} -> prep target={t_under} (expect {expect_under}); "
               f"foe L{max(1,floor-3)} (>= floor) -> {t_strat} (expect None)")

    # ── C3 — GRIND-WEAK LOOP (fields weakest, exits on floor, restores ace) ────────────────────────
    b, camp = _fresh()
    state = camp.read_live_state()
    floor = min(m["level"] for m in state["party"])
    target = floor + 4
    _inject_wall(camp, foe_level=target - ps.UNDERLEVEL_MARGIN)   # -> underlevel_target == target

    fielded = []                                          # species fielded as lead each grind pass

    def _stub_grind(tg):
        # simulate "leveled the lead to target": stamp slot-0 level byte (loop-control test only — real
        # XP needs real battles). Record who got fielded.
        base = ram.GPLAYER_PARTY
        fielded.append(st.SPECIES_NAME.get(st.read_party_species(b, 0), "?"))
        b.core.memory.u8.raw_write(base + LVL_OFF, tg)
        return "ok"

    camp.grind = _stub_grind
    ace_before = max(_species_levels(b, camp), key=lambda x: x[1])[0]
    res = camp.grind_weak_members(target)
    after = _species_levels(b, camp)
    floor_after = min(l for _, l in after)
    ace_slot0 = after[0][0]
    only_weak_fielded = all(f != ace_before for f in fielded)   # the ace was never fielded for grinding
    ok["C3"] = (res == "ready" and floor_after >= target and ace_slot0 == ace_before and only_weak_fielded)
    out.append("")
    out.append(f"C3  target=L{target}; fielded (lead each pass)={fielded}; ace={ace_before!r}")
    out.append(f"    result={res!r} floor_after=L{floor_after} (>= {target}: {floor_after>=target})")
    out.append(f"    ace restored to slot 0: {ace_slot0==ace_before} ({ace_slot0!r}); "
               f"ace never grind-fielded: {only_weak_fielded}")
    out.append(f"    party after: {after}")

    # ── C4 — ACTION-SET FRAMING + forward-drive stand-down ────────────────────────────────────────
    # The live save sits at a no-grass interior, so 'battle' wouldn't be offered there (honest action
    # set). Force grass reachable to isolate the framing/stand-down logic from the save's geography, and
    # open the gate questline + go off-branch (the exact situation forward-drive would normally prune the
    # grind in) — the smart-middle requirement is that the weak-grind SURVIVES that prune at the wall.
    b, camp = _fresh()
    state = camp.read_live_state()
    floor = min(m["level"] for m in state["party"])
    _inject_wall(camp, foe_level=floor + 6)
    camp._grass_target = lambda _s: ("here", (10, 10))    # grass reachable -> 'battle' is offered
    C.FORWARD_DRIVE_ENABLED = True
    camp._active_questline = None
    camp._ensure_forward_questline(state)                 # open/attempt the gate questline (would prune grind)
    acts = camp._available_actions(state)
    battle_desc = acts.get("battle", "")
    reframed = "STRENGTHEN FIRST" in battle_desc and "L" in battle_desc
    stood_down = "battle" in acts                         # forward-drive did NOT prune the grind at the wall
    ok["C4"] = reframed and stood_down
    out.append("")
    out.append(f"C4  forced-grass; 'battle' present (stood down)={('battle' in acts)}; "
               f"questline open={camp._active_questline is not None}")
    out.append(f"    reframed to weak-grind: {reframed}")
    out.append(f"    battle: {battle_desc[:140]}")

    # ── C5 — REVERSIBILITY (flag OFF restores ordinary behaviour) ─────────────────────────────────
    b, camp = _fresh()
    state = camp.read_live_state()
    floor = min(m["level"] for m in state["party"])
    _inject_wall(camp, foe_level=floor + 6)
    C.STRATEGIC_GRIND_ENABLED = False
    t_off = camp._prep_team_target(state)
    acts_off = camp._available_actions(state)
    C.STRATEGIC_GRIND_ENABLED = True
    ok["C5"] = (t_off is None) and ("STRENGTHEN FIRST" not in acts_off.get("battle", ""))
    out.append("")
    out.append(f"C5  flag OFF -> prep target={t_off} (expect None); "
               f"battle reframed={'STRENGTHEN FIRST' in acts_off.get('battle','')} (expect False)")

    print("\n".join(out), flush=True)
    print("\n---- checks ----", flush=True)
    for k in ("C1", "C2", "C3", "C4", "C5"):
        print(f"  {k}: {ok[k]}", flush=True)
    hard = [v for kk, v in ok.items() if v is not None]
    print(f"\n==== strategic-grind checks: {'PASS' if all(hard) else 'INSPECT'} ====", flush=True)


if __name__ == "__main__":
    main()
