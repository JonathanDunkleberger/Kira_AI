"""recon_gauntlet_fight.py — watch ONE Route-6 gauntlet engagement end-to-end (arsenal #4).

Loads the live stage state (mid-gauntlet at (8,16)-ish), walks into the blocker trainer's line,
runs BattleAgent with full logging, and saves a frame every ~3s of battle so we SEE where the
engagement's time goes (the 120s budget dies somewhere between a Peck KO and the next mon).
RUN: python pokemon_agent/recon_gauntlet_fight.py
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge              # noqa: E402
import travel as tv                    # noqa: E402
import pokemon_state as st             # noqa: E402
from battle_agent import BattleAgent   # noqa: E402
from campaign import Campaign          # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OUT = r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\661370f2-1025-435c-8cf5-d2593621c432\scratchpad"


def main():
    b = Bridge(ROM)
    with open(r"G:\temp\longrun\stage\kira_campaign.state", "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    b.set_input_owner("agent")
    print(f"boot {tv.map_id(b)}@{tv.coords(b)} party0={st.read_party_species(b,0)}", flush=True)
    camp = Campaign(b, battle_runner=lambda: "ok", render=lambda: None)
    camp._suppress_heal = True
    # walk toward the gauntlet chokepoint to trigger the trainer (same target the leg used)
    camp.trav.travel(target_map=(3, 5), edge="south", max_seconds=60)
    if not st.in_battle(b):
        print("no battle engaged — walking again", flush=True)
        camp.trav.travel(target_map=(3, 5), edge="south", max_seconds=60)
    print(f"in_battle={st.in_battle(b)}", flush=True)
    if not st.in_battle(b):
        return
    shots = [0]
    t0 = time.time()

    def render():
        el = time.time() - t0
        if el > shots[0] * 3:
            shots[0] += 1
            try:
                b.frame_rgb().resize((480, 320)).save(os.path.join(OUT, f"fight_{shots[0]:02d}.png"))
            except Exception:
                pass

    agent = BattleAgent(b, on_event=lambda s, **k: print(f"   [voice] {s[:80]}", flush=True),
                        render=render, log=lambda m: print(m, flush=True))
    out = agent.run(240)
    print(f"\nengagement -> {out} after {time.time()-t0:.0f}s; in_battle={st.in_battle(b)}", flush=True)
    rb = st.read_battle(b)
    if rb:
        print(f"enemy: {rb['enemy']}", flush=True)


if __name__ == "__main__":
    main()
