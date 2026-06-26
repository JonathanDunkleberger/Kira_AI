"""recon_battletest.py - battle-logic regression: run BattleAgent on a battle fixture and report
outcome + whether the move-verify FALSE-benched a working move (the bug: slow trainer turns timed
out the verify -> benched a working Tackle -> lost). PASS = win with no spurious benching.
RUN: STATE=brock_battle .venv\\Scripts\\python.exe -u pokemon_agent\\recon_battletest.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge              # noqa: E402
import pokemon_state as st            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")


def main():
    state = os.getenv("STATE", "brock_battle")
    b = Bridge(ROM)
    with open(os.path.join(STATES, state + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    logs = []
    agent = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                        log=lambda m: (logs.append(m), print(f"   {m}", flush=True)))
    out = agent.run(max_seconds=120)
    benched = [m for m in logs if "benching" in m]
    print(f"\n   [test] state={state} OUTCOME={out} in_battle_now={st.in_battle(b)}")
    print(f"   [test] false-bench events: {len(benched)}")
    for m in benched:
        print(f"   [test]   {m}")
    print(f"   [test] {'PASS' if out in ('win', 'done') and not benched else 'CHECK'}")


if __name__ == "__main__":
    main()
