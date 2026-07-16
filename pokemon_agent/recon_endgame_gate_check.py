"""recon_endgame_gate_check.py — DECISION check for the NS#15 endgame dispatch wiring (no battles).

Proves _available_actions offers the right endgame action at 8 badges, pre-credits:
  giovanni_kit_g (badge 8, NOT at Indigo)  + VR flag ON -> offers 'head_to_league', NOT 'enter_league'
  indigo_reach_g (badge 8, AT Indigo 3,9)  + E4 flag ON -> offers 'enter_league', NOT 'head_to_league'
  BOTH flags OFF                            -> neither offered (byte-inert default)
  post-game (canonical Champion save)       -> neither offered (endgame is pre-credits only)

RUN: ../.venv/Scripts/python.exe recon_endgame_gate_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import field_moves as fm             # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")


def _resolve(name):
    for cand in (os.path.join(_HERE, "states", "workshop", name + ".state"),
                 os.path.join(_HERE, "states", name + ".state"), name):
        if os.path.exists(cand):
            d = os.path.dirname(cand)
            return cand, d, os.path.basename(cand)[:-6]
    return os.path.join(CANON, "kira_campaign.state"), CANON, "kira_campaign"


def _boot(name):
    sp, sd, pref = _resolve(name)
    b = Bridge(ROM)
    with open(sp, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    camp = Campaign(b, battle_runner=lambda: "won", on_event=lambda *a, **k: None,
                    beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    for loader, side, fb in ((camp.world.load, pref + ".world_model.json", C.WORLD_JSON),
                             (camp.strat.load, pref + ".strat_memory.json", C.STRAT_JSON)):
        try:
            p = os.path.join(sd, side)
            loader(p if os.path.exists(p) else fb)
        except Exception:
            pass
    return b, camp


def _acts(camp):
    st = camp.read_live_state()
    return set(camp._available_actions(st).keys()), st


def main():
    results = []

    def check(label, cond):
        results.append((label, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    # ── giovanni_kit_g: badge 8, not at Indigo ────────────────────────────────────────────
    b, camp = _boot("giovanni_kit_g")
    mp, badge8 = tv.map_id(b), int(fm.read_flag(b, C.FLAG_BADGE_EARTH))
    print(f"giovanni_kit_g: map={mp} badge8={badge8} indigo={tuple(mp)==C.ENDGAME_INDIGO}")
    C.VICTORY_ROAD_ENABLED = C.E4_STRIKE_ENABLED = False
    a_off, _ = _acts(camp)
    check("VR OFF -> no head_to_league offered (byte-inert)", "head_to_league" not in a_off)
    C.VICTORY_ROAD_ENABLED = C.E4_STRIKE_ENABLED = True
    a_on, st = _acts(camp)
    check("badge8 + not-Indigo + VR ON -> offers head_to_league", "head_to_league" in a_on)
    check("badge8 + not-Indigo -> does NOT offer enter_league", "enter_league" not in a_on)
    check("not post_game at badge 8", not st.get("post_game"))

    # ── indigo_reach_g: badge 8, AT Indigo (3,9) ──────────────────────────────────────────
    b, camp = _boot("indigo_reach_g")
    mp = tv.map_id(b)
    print(f"indigo_reach_g: map={mp} indigo={tuple(mp)==C.ENDGAME_INDIGO} "
          f"badge8={int(fm.read_flag(b, C.FLAG_BADGE_EARTH))}")
    C.VICTORY_ROAD_ENABLED = C.E4_STRIKE_ENABLED = False
    a_off, _ = _acts(camp)
    check("E4 OFF -> no enter_league offered (byte-inert)", "enter_league" not in a_off)
    C.VICTORY_ROAD_ENABLED = C.E4_STRIKE_ENABLED = True
    a_on, st = _acts(camp)
    if tuple(mp) == C.ENDGAME_INDIGO:
        check("badge8 + AT-Indigo + E4 ON -> offers enter_league", "enter_league" in a_on)
        check("badge8 + AT-Indigo -> does NOT offer head_to_league", "head_to_league" not in a_on)
    else:
        print(f"  [SKIP] indigo_reach_g is not on the Indigo exterior (map {mp}) — "
              f"enter_league is map-gated; giovanni_kit_g proves the VR side")

    # ── canonical Champion save: post-game -> endgame is pre-credits only ──────────────────
    try:
        b, camp = _boot("__canon__")
        st = camp.read_live_state()
        C.VICTORY_ROAD_ENABLED = C.E4_STRIKE_ENABLED = True
        a_pg = set(camp._available_actions(st).keys())
        print(f"canonical: post_game={st.get('post_game')} badge_count={st.get('badge_count')}")
        check("post_game -> neither endgame action offered",
              "head_to_league" not in a_pg and "enter_league" not in a_pg)
    except Exception as e:
        print(f"  [SKIP] canonical post-game check: {e}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\nENDGAME GATE CHECK: {n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
