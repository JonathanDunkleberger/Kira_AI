"""recon_grindstall_test.py — VERIFY the GRIND-WEAK stall fix (night-train shift 3).

Case A (the Rattata bug): when grind() makes NO level progress, GRIND-WEAK must mark the mon stalled
and STOP re-picking it (terminate), not spin for the whole budget. We monkeypatch camp.grind to return
'ok' without leveling anything (the exact stall condition) and assert grind_weak_members returns quickly
with a BOUNDED number of grind() calls (~party size), not hundreds.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent/recon_grindstall_test.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["POKEMON_GRIND_WEAK_BUDGET_S"] = "20"   # low cap so a BROKEN fix still ends the test

from bridge import Bridge          # noqa: E402
import firered_ram as ram          # noqa: E402
import travel as tv                # noqa: E402
from campaign import Campaign      # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = os.getenv("STATE", r"G:\temp\rattata_grind.state")


def main():
    b = Bridge(ROM)
    with open(STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    b.set_input_owner("agent")
    camp = Campaign(b, battle_runner=lambda *a, **k: "win",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)

    calls = {"n": 0}
    real_swap = camp._swap_party_slots

    def fake_grind(target, fragile=False, budget_s=480):
        calls["n"] += 1
        # simulate a mon that CANNOT gain XP here: return 'ok' with NO level change
        return "ok"
    camp.grind = fake_grind

    print(f"START map={tv.map_id(b)} party_levels={camp._party_levels()}")
    import time
    t0 = time.time()
    res = camp.grind_weak_members(14)
    dt = round(time.time() - t0, 1)
    n = calls["n"]
    stalled = getattr(camp, "_grind_stalled", set())
    print(f"grind_weak_members(14) -> {res!r} in {dt}s | grind() calls={n} | stalled pids={len(stalled)}")
    # PASS: bounded calls (one per weak mon, maybe +1), terminated well under the 20s budget
    party = b.rd8(ram.GPLAYER_PARTY_CNT)
    ok = (res == "ready") and (n <= party + 1) and (len(stalled) >= 1)
    print("RESULT:", "PASS — GRIND-WEAK stalls out cleanly, no spin" if ok
          else f"FAIL — n={n} party={party} res={res} (a spin would show n in the hundreds)")


if __name__ == "__main__":
    main()
