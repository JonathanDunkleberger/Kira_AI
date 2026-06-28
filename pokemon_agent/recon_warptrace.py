"""recon_warptrace.py — HEADLESS forward-chain trace (option b).

Proves (or disproves) autonomous Cerulean->Vermilion traversal OFFLINE, with NO live oracle and NO
watch. A STUB oracle picks `head_to_gym` every tick (so the GOAL is fixed forward) while the REAL
free_roam recovery machinery runs underneath — Layer-A blocked-NPC route-around, the universal
watchdog disengage, the no-move guard, off-spine recovery, warp-aware head_to_gym. So this faithfully
tests whether she clears the Cerulean SOUTH-EXIT NPC wedge (the hop-1 gate) and walks the warp chain.

Each map's warps are read LIVE (free_roam._learn_map -> world.note_visit), so the chain graph is built
from live RAM truth, never disasm-blind.

Safety for a trace (never touches canonical saves): _save_campaign / _continuity_* are no-op'd; HP is
re-patched to full at each tick (via the stub) so a wild can't black her out and end the trace early.

RUN: .venv\\Scripts\\python.exe pokemon_agent\\recon_warptrace.py [boot_state] [max_ticks]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
# the decision-ctx logs contain Unicode (e.g. '→'); a cp1252 console crashes on them -> force utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge              # noqa: E402
import firered_ram as ram             # noqa: E402
import travel as tv                   # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def _heal_patch(b):
    """Write each party mon's current HP = max (test-only — keeps the trace alive through wild fights)."""
    try:
        w16 = b.core.memory.u16.raw_write
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            w16(base + 0x56, b.rd16(base + 0x58))
    except Exception:
        pass


def main():
    boot = sys.argv[1] if len(sys.argv) > 1 else "misty_done"
    max_ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    b = Bridge(ROM)
    from campaign import resolve_state
    with open(resolve_state(boot + ".state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(15):
        b.run_frame()
    _heal_patch(b)

    def fight():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=60)

    events = []
    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: events.append(("event", s)),
                    beat=lambda *a, **k: None, render=lambda: None)
    # FAST trace: flee wild battles (the south connector chain Route 5/6 has ~no trainers) so ticks are
    # nav-bound, not battle-bound — lets the full chain trace complete in one run. (--fight to force real
    # battles if a trainer gauntlet must be cleared.)
    if "--fight" not in sys.argv:
        camp.trav.battle_runner = camp._flee_runner

    # STUB ORACLE: fix the GOAL to head_to_gym (else fall back so the run continues); re-heal each tick.
    def stub_choose(kind, options, ctx):
        if kind != "action":
            return None                          # no want/name surfacing in the trace
        _heal_patch(b)
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        if "head_to_gym" in opts:
            return "head_to_gym"
        return opts[0] if opts else None         # head_to_gym pruned -> take anything so we SEE what happens
    camp._oracle_choose = stub_choose

    # never write canonical saves / touch soul sidecars during a trace
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    print(f"==== WARP-TRACE start boot={boot} map={tv.map_id(b)} coords={tv.coords(b)} "
          f"max_ticks={max_ticks} ====", flush=True)
    out = camp.free_roam(max_ticks=max_ticks, max_seconds=240, want_every=999)
    print(f"==== WARP-TRACE end: free_roam -> {out} | FINAL map={tv.map_id(b)} "
          f"coords={tv.coords(b)} ====", flush=True)
    # what did the world-model learn (the live-cross-checked chain graph)?
    print("---- learned graph (visited nodes + their warps) ----", flush=True)
    for key, node in camp.world.nodes.items():
        if node.get("visited"):
            print(f"   {key} {node.get('name')}: edges={node.get('edges')} warps={node.get('warps')}",
                  flush=True)
    reached = tv.map_id(b) == (3, 5)
    print(f"\nRESULT: reached Vermilion (3,5) = {reached}", flush=True)


if __name__ == "__main__":
    main()
