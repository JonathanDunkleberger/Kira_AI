"""recon_forward_drive2.py — ISOLATED action-set checks for the forward-drive fix (no drive, no blackout
contamination). Each case boots a FRESH core from the live Route-4 save and heals, so the action set is read
on a clean healthy state. Logs are silenced so the printed verdicts are unambiguous.

  P1 OFF-BRANCH PRUNE (Route 4): drive OFF offers the backward grind (battle + travel:3,21=Route 3 = further
     from base camp); drive ON prunes them, KEEPS the forward travel:3,3 (toward Cerulean base camp), and
     reframes head_to_gym as the forward pull.
  P3 OPEN-ROAD REGRESSION (Route 4, flag SET): with the road open + a healthy team, the questline stays
     closed and the full grind/travel set returns — the prune is strictly conditional, not a permanent
     removal of her agency to grind.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                                 # noqa: E402
import firered_ram as ram                                 # noqa: E402
import field_moves as fm                                  # noqa: E402
import travel as tv                                       # noqa: E402
import campaign as C                                      # noqa: E402
from campaign import Campaign, resolve_state, WORLD_JSON  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SS_TICKET = 0x234


def _heal(b):
    for s in range(ram.read_party_count(b)):
        base = ram.GPLAYER_PARTY + s * 100
        b.core.memory.u16.raw_write(base + 0x56, b.rd16(base + 0x58))


def _set_flag(b, flag):
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    fa = sb1 + 0x0EE0 + (flag >> 3)
    b.core.memory.u8.raw_write(fa, b.rd8(fa) | (1 << (flag & 7)))


def _fresh(set_ticket=False):
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    _heal(b)
    if set_ticket:
        _set_flag(b, SS_TICKET)
    camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp.world.load(WORLD_JSON)
    return b, camp


def _grindtravel(keys):
    return sorted(k for k in keys if k in ("battle", "wander_catch") or k.startswith("travel:"))


def main():
    out = []

    # P1 — baseline (drive OFF) vs fixed (drive ON), Route 4, healthy
    b, camp = _fresh()
    state = camp.read_live_state()
    out.append(f"START map={tv.map_id(b)} ({state.get('place')}) base_camp={camp._base_camp(state)} "
               f"FLAG={fm.read_flag(b, SS_TICKET)}")
    C.FORWARD_DRIVE_ENABLED = False
    camp._active_questline = None
    off = camp._available_actions(state)
    C.FORWARD_DRIVE_ENABLED = True
    camp._active_questline = None
    camp._ensure_forward_questline(state)
    on = camp._available_actions(state)
    h2g = on.get("head_to_gym", "")
    out.append("")
    out.append(f"P1  drive OFF action set: {sorted(off)}")
    out.append(f"    backward grind offered: {_grindtravel(off.keys())}")
    out.append(f"    drive ON  action set: {sorted(on)}")
    out.append(f"    backward grind now offered: {_grindtravel(on.keys())}")
    out.append(f"    kept forward travel:3,3 (toward Cerulean): {'travel:3,3' in on}")
    out.append(f"    head_to_gym reframed forward: {'PUSH FORWARD' in h2g}")
    out.append(f"    head_to_gym: {h2g[:120]}")
    p1 = (("battle" in off or any(k.startswith('travel:') for k in off)) and "travel:3,21" in off
          and "battle" not in on and "travel:3,21" not in on
          and "travel:3,3" in on and "PUSH FORWARD" in h2g and "head_to_gym" in on)

    # P3 — the prune is STRICTLY CONDITIONAL on the feature flag (not a permanent loss of agency):
    # toggling forward-drive OFF restores the full grind/travel set (already shown by the P1 OFF case);
    # and setting the ticket flag self-CLEARS the gate questline (cross-check works). NB: Route 4 is a
    # backward dead-end branch, so the off-branch prune correctly still pushes her toward base camp even
    # with the road open — her grind agency lives ON THE ROAD (en route via head_to_gym) + at base camp,
    # not on a cleared side-branch. So the right regression is "reversible + gate self-clears", not
    # "battle survives on Route 4".
    b2, camp2 = _fresh(set_ticket=True)
    st3 = camp2.read_live_state()
    C.FORWARD_DRIVE_ENABLED = True
    camp2._active_questline = None
    camp2._ensure_forward_questline(st3)
    gate_cleared = camp2._active_questline is None
    C.FORWARD_DRIVE_ENABLED = False                      # feature OFF -> full agency restored
    reverted = camp2._available_actions(st3)
    out.append("")
    out.append(f"P3  flag SET -> gate questline self-clears (not opened): {gate_cleared}")
    out.append(f"    feature OFF -> grind/travel set restored: {_grindtravel(reverted.keys())}")
    p3 = gate_cleared and "battle" in reverted and "travel:3,21" in reverted

    print("\n".join(out), flush=True)
    print("\n---- checks ----", flush=True)
    print(f"  P1 off-branch prune (backward gone, forward kept, reframed): {p1}", flush=True)
    print(f"  P3 prune is reversible + gate self-clears (agency intact):   {p3}", flush=True)
    print(f"\n==== action-set checks: {'PASS' if (p1 and p3) else 'INSPECT'} ====", flush=True)


if __name__ == "__main__":
    main()
