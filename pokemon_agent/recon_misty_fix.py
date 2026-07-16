"""recon_misty_fix.py — VERIFY the Misty pool-gym fix (night-train shift 3).

Loads the pre-gym Cerulean state (workshop/seg_cerulean.state: at Cerulean, badge 1, solo Ivysaur)
and runs the REAL beat_gym('Misty'). PASS = the swimmer at (10,12) is actually fought (no false
'beaten'), Misty un-gates, and the Cascade Badge flag sets. Headless, no audio.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_misty_fix.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge          # noqa: E402
import firered_ram as ram          # noqa: E402
import travel as tv                # noqa: E402
import pokemon_state as st         # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign      # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATE = os.getenv("STATE", os.path.join(_HERE, "states", "workshop", "seg_cerulean.state"))
CASCADE_FLAG = 0x821


def main():
    b = Bridge(ROM)
    with open(STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()
    b.set_input_owner("agent")

    def render():
        pass

    def fr():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=render,
                           log=lambda m: print(f"      {m}", flush=True)).run(120)

    def has_cascade():
        sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
        fa = sb1 + 0x0EE0 + (CASCADE_FLAG >> 3)
        return bool(b.rd8(fa) & (1 << (CASCADE_FLAG & 7)))

    camp = Campaign(b, battle_runner=fr, on_event=lambda *a, **k: print(f"   [kira] {a[0] if a else ''}", flush=True),
                    beat=lambda *a, **k: None, render=render)

    print(f"START map={tv.map_id(b)} coords={tv.coords(b)} party={b.rd8(ram.GPLAYER_PARTY_CNT)} "
          f"cascade={has_cascade()}", flush=True)
    res = camp.beat_gym("Misty")
    print(f"\nbeat_gym('Misty') -> {res}", flush=True)
    print(f"END map={tv.map_id(b)} coords={tv.coords(b)} cascade={has_cascade()}", flush=True)
    print("RESULT:", "PASS — Cascade Badge won" if has_cascade() else "FAIL — no badge", flush=True)


if __name__ == "__main__":
    main()
