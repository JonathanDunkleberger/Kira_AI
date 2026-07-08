"""recon_voidguard_verify.py — fault-injection verify for the VOID-CORE class killers.
Boots banked_CREDITS (real world), primes _last_good_state, then INJECTS the actual dead-world
state the QW-4 summit run banked (title screen) and runs free_roam. PASS requires, in order:
  1. _world_lost() reads True on the injected state, False on the real one.
  2. _save_campaign REFUSES while void (poison guard).
  3. free_roam's tick-top tripwire fires, does NOT offer actions/burn a pick, RECOVERS via the
     last-good snapshot, and the next tick plays the REAL world again.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_voidguard_verify.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ["POKEMON_CAMPAIGN_DIR"] = os.path.join(  # sandbox ALL campaign-dir writes
    os.environ.get("TEMP", _HERE), "longrun", "voidcore_probe", "campaign_sandbox")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                    # noqa: E402
import firered_ram as ram                    # noqa: E402
import travel as tv                          # noqa: E402
from battle_agent import BattleAgent         # noqa: E402
from campaign import Campaign                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
REAL = os.path.join(SCRATCH, "banked_CREDITS", "kira_campaign.state")
DEAD = os.path.join(os.environ.get("TEMP", _HERE), "kira_watch",
                    "sandbox_canonical_20260707_232814", "kira_campaign.state")
STAGE = os.path.join(SCRATCH, "voidcore_probe", "verify_stage")
os.makedirs(STAGE, exist_ok=True)

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


b = Bridge(ROM)
with open(REAL, "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()

camp = Campaign(b, battle_runner=lambda: BattleAgent(
                    b, on_event=lambda *a, **k: None, render=lambda: None,
                    log=lambda m: None).run(max_seconds=60),
                on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)


def _stage_save(reason="tick"):
    with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
        f.write(b.save_state())
    return True


_real_save = camp._save_campaign            # keep the REAL method (poison guard under test)
camp._continuity_save = lambda *a, **k: None
camp._continuity_load = lambda *a, **k: None

# 1) detector truth table
check("world_lost=False on the real world", not camp._world_lost(),
      f"map={tv.map_id(b)} party={b.rd8(ram.GPLAYER_PARTY_CNT)}")
good = b.save_state()
camp._last_good_state = good
camp._last_good_gain = camp._gain_sig()
with open(DEAD, "rb") as f:
    b.load_state(f.read())
for _ in range(30):
    b.run_frame()
check("world_lost=True on the injected dead world", camp._world_lost(),
      f"map={tv.map_id(b)} party={b.rd8(ram.GPLAYER_PARTY_CNT)}")

# 2) poison guard: the REAL _save_campaign must refuse while void
check("_save_campaign REFUSES while void", _real_save("verify") is False)

# 3) tick-top tripwire + recovery inside the REAL free_roam loop
camp._save_campaign = _stage_save            # roam's own banking goes to the stage
picks = []


def chooser(kind, options, ctx):
    opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
    picks.append((kind, opts))
    return opts[0] if opts else None


camp._oracle_choose = chooser
out = camp.free_roam(max_ticks=2, max_seconds=120, want_every=999)
check("free_roam recovered (not abandoned)", out != "abandoned", f"roam={out}")
check("recovered world is REAL again", not camp._world_lost(),
      f"map={tv.map_id(b)} party={b.rd8(ram.GPLAYER_PARTY_CNT)}")
acts = [p for p in picks if p[0] == "action"]
check("no action pick was burned on the dead world",
      all("head_to_gym" not in o or tv.map_id(b) != (0, 0) for _, o in acts),
      f"action picks={len(acts)}")

fails = [n for n, ok in results if not ok]
print(("\nALL PASS (" + str(len(results)) + ")") if not fails else f"\nFAILURES: {fails}", flush=True)
sys.exit(1 if fails else 0)
