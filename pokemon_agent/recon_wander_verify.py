"""recon_wander_verify.py — verify the F-1 WANDER TRIPWIRE in the REAL free_roam loop.
Boots banked_SNORLAX (Route 12 overworld, rich learned graph), scripts the oracle to commit to
the FIRST travel:* objective on tick 1 and then DITHER (never re-pick it) every tick after,
with POKEMON_NAV_TRIPWIRE_S=1. PASS = the tripwire fires, the harness executes HER objective
deterministically on a tick with NO action-oracle call, and the run keeps going (no crash).
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_wander_verify.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ["POKEMON_NAV_TRIPWIRE_S"] = "1"          # trip fast for the verify
os.environ["POKEMON_CAMPAIGN_DIR"] = os.path.join(  # sandbox ALL campaign-dir writes
    os.environ.get("TEMP", _HERE), "longrun", "voidcore_probe", "campaign_sandbox")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                    # noqa: E402
from battle_agent import BattleAgent         # noqa: E402
import campaign as C                         # noqa: E402
from campaign import Campaign                # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
BOOT = os.path.join(SCRATCH, "banked_SNORLAX")
STAGE = os.path.join(SCRATCH, "voidcore_probe", "wander_stage")
os.makedirs(STAGE, exist_ok=True)

b = Bridge(ROM)
with open(os.path.join(BOOT, "kira_campaign.state"), "rb") as f:
    b.load_state(f.read())
for _ in range(60):
    b.run_frame()

camp = Campaign(b, battle_runner=lambda: BattleAgent(
                    b, on_event=lambda *a, **k: None, render=lambda: None,
                    log=lambda m: None).run(max_seconds=120),
                on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
camp._save_campaign = lambda reason="tick": True     # no banking in a verify
camp._continuity_save = lambda *a, **k: None
camp._continuity_load = lambda *a, **k: None
try:
    camp.world.load(os.path.join(BOOT, "world_model.json"))
except Exception as e:
    print(f"!! world load failed: {e}", flush=True)

state = {"objective": None, "action_calls": 0, "forced_seen": 0}


def chooser(kind, options, ctx):
    opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
    if kind != "action":
        return opts[0] if opts else None
    state["action_calls"] += 1
    trav = [o for o in opts if o.startswith("travel:")]
    if state["objective"] is None and trav:
        state["objective"] = trav[0]                 # tick 1: SHE commits to a destination
        print(f"[verify] objective committed: {trav[0]}", flush=True)
        return trav[0]
    # every later tick: DITHER with NON-NAV picks only (nav picks would REPLACE the standing
    # objective by design — RECALIBRATION semantics — and re-arm the watch on the new target)
    for o in ("talk_npc", "battle", "stock_up", "heal"):    # wander_catch/head_to_gym are nav picks
        if o in opts and o != state["objective"]:
            return o
    return state["objective"] if state["objective"] in opts else (opts[0] if opts else None)


camp._oracle_choose = chooser
out = camp.free_roam(max_ticks=8, max_seconds=600, want_every=999)
fired = getattr(camp, "_nav_tripwire_total", 0)
print(f"\n[verify] roam={out} action_oracle_calls={state['action_calls']} "
      f"tripwire_fired={fired} objective={state['objective']}", flush=True)
ok = fired >= 1
print("PASS: tripwire fired and took the wheel" if ok else
      "FAIL: tripwire never fired (check the log above for why)", flush=True)
sys.exit(0 if ok else 1)
