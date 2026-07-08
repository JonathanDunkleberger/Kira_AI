"""recon_hof_exit.py — why does the Hall-of-Fame exit fail off banked_CREDITS in the harness?
Loads the bank, snaps the boot screen, tries _exit_to_overworld with travel debug, snaps after.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_hof_exit.py"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ["POKEMON_CAMPAIGN_DIR"] = os.path.join(
    os.environ.get("TEMP", _HERE), "longrun", "voidcore_probe", "campaign_sandbox")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                    # noqa: E402
import firered_ram as ram                    # noqa: E402
import travel as tv                          # noqa: E402
from dialogue_drive import box_open          # noqa: E402
from battle_agent import BattleAgent         # noqa: E402
from campaign import Campaign                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
BOOT = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "banked_CREDITS", "kira_campaign.state")
OUT = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "voidcore_probe")
os.makedirs(OUT, exist_ok=True)

b = Bridge(ROM)
with open(BOOT, "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()
print(f"boot: map={tv.map_id(b)} coords={tv.coords(b)} box_open={box_open(b)}", flush=True)
b.frame_rgb().resize((480, 320)).save(os.path.join(OUT, "hof_boot.png"))

# what warps does the HoF map expose live?
try:
    warps = tv.read_warp_events(b)
    print(f"live warp events: {warps}", flush=True)
except Exception as e:
    print(f"warp read failed: {e}", flush=True)

camp = Campaign(b, battle_runner=lambda: BattleAgent(
                    b, on_event=lambda *a, **k: None, render=lambda: None,
                    log=lambda m: None).run(max_seconds=60),
                on_event=lambda t, **k: print(f"   [voice] {t}", flush=True),
                beat=lambda *a, **k: None, render=lambda: None)
camp._save_campaign = lambda reason="tick": True
camp._continuity_save = lambda *a, **k: None
camp._continuity_load = lambda *a, **k: None

ok = camp._exit_to_overworld()
print(f"_exit_to_overworld -> {ok!r} | now map={tv.map_id(b)} coords={tv.coords(b)} "
      f"box_open={box_open(b)}", flush=True)
b.frame_rgb().resize((480, 320)).save(os.path.join(OUT, "hof_after_exit.png"))
