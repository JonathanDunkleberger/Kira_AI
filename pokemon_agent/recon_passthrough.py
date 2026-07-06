"""recon_passthrough.py — verify the door pass-through primitive on the real south gate (2026-07-06).

Boots the staged post-ticket state (Cerulean, ticket flag set, tree at (26,32) blocking the direct
gap), then calls the REAL `_edge_travel((3,23), 'south')`. PASS = she ends standing on Route 5.
Expected route: no-route on the fence -> _door_passthrough -> burgled house (30,11) -> back door ->
garden -> east corridor -> fence crossing (39-40,32) -> strip -> south border -> (3,23).
RUN: python pokemon_agent/recon_passthrough.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import field_moves as fm               # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import Campaign          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STAGE_STATE = r"G:\temp\longrun\stage\kira_campaign.state"


def main():
    b = Bridge(ROM)
    with open(STAGE_STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    b.set_input_owner("agent")

    def runner():
        out = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                          log=lambda m: print(m, flush=True)).run(120)
        b.set_input_owner("agent")
        return out

    camp = Campaign(b, battle_runner=runner, render=lambda: None,
                    on_event=lambda s, **k: print(f"   [voice] {s[:100]}", flush=True))
    print(f"start: map={tv.map_id(b)} pos={tv.coords(b)} ticket={fm.read_flag(b, 0x234)}", flush=True)
    out = camp._edge_travel((3, 23), "south", budget_s=420)
    m, c = tv.map_id(b), tv.coords(b)
    print(f"\n_edge_travel -> {out}; now map={m} pos={c}", flush=True)
    print("PASS — she stands on Route 5" if tuple(m) == (3, 23) else "FAIL — not on Route 5", flush=True)


if __name__ == "__main__":
    main()
