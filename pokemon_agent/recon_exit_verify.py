"""recon_exit_verify.py — night shift 6 verify: the full hideout exit chain off
banked_SCOPE with tonight's three fixes (sealed-door skip, persistent elevator rows,
sealed-by-a-guard engagement). Expect: B4F -> car -> B1F -> guard fight (Grunt 12,
the pret barrier) -> barrier opens -> Game Corner -> street (group 3). READ-ONLY."""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                  # noqa: E402
import travel as tv                                        # noqa: E402
import pokemon_state as st                                 # noqa: E402
from battle_agent import BattleAgent                       # noqa: E402
from campaign import Campaign                              # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BANK = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_SCOPE")

b = Bridge(ROM)
with open(os.path.join(BANK, "kira_campaign.state"), "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()

def runner():
    return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                       log=lambda m: None, choose=lambda *a, **k: None).run(max_seconds=120)

camp = Campaign(b, battle_runner=runner, on_event=lambda s, **k: None,
                beat=lambda *a, **k: None, render=lambda: None)
camp._save_campaign = lambda *a, **k: True
camp._continuity_save = lambda *a, **k: None

print(f"spawn: map={tv.map_id(b)} coords={tv.coords(b)}", flush=True)
t0 = time.time()
# the real loop invokes _exit_to_overworld once per leave_building tick — allow a few
for tick in range(6):
    ok = camp._exit_to_overworld()
    print(f"-- leave_building tick {tick}: exited={ok} map={tv.map_id(b)} "
          f"coords={tv.coords(b)} ({time.time() - t0:.0f}s)", flush=True)
    if ok:
        break
print(f"RESULT: {'PASS — on the overworld' if tv.map_id(b)[0] == 3 else 'FAIL — still inside'} "
      f"map={tv.map_id(b)} coords={tv.coords(b)} elapsed={time.time() - t0:.0f}s", flush=True)
