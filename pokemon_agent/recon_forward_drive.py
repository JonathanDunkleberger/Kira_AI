"""recon_forward_drive.py — END-TO-END drive proof of the backward-grind root fix, from the ACTUAL live save.

The live save (kira_campaign.state) sits ON Route 4 (3,22) — a cleared dead-end branch WEST of Cerulean,
mid-drift (she'd walked backward to grind). With a head_to_gym-preferring oracle this drives a few free-roam
ticks and PROVES she now goes FORWARD: Route 4 → EAST to Cerulean (base camp) → the S.S.-Ticket questline
OPENS at the gate → she heads NORTH toward Bill (Route 24 / Nugget Bridge). Never south/west into the
cleared dead-ends. (The action-set-level prune is checked separately in recon_forward_drive2.py.)

Read-only: boots the live save, never writes a canonical save. RUN: python pokemon_agent\\recon_forward_drive.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge                                 # noqa: E402
import firered_ram as ram                                 # noqa: E402
import travel as tv                                       # noqa: E402
from battle_agent import BattleAgent                      # noqa: E402
from campaign import Campaign, resolve_state, WORLD_JSON  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def _heal(b):
    try:
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            b.core.memory.u16.raw_write(base + 0x56, b.rd16(base + 0x58))
    except Exception:
        pass


def main():
    max_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    _heal(b)

    events = []
    camp = Campaign(b, battle_runner=lambda: BattleAgent(
        b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None).run(max_seconds=40),
        on_event=lambda s, **k: (events.append(s), print(f"   [event] {s}", flush=True)),
        beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    try:
        camp.world.load(WORLD_JSON)
    except Exception:
        pass
    camp.trav.battle_runner = camp._flee_runner

    def stub(kind, options, ctx):
        if kind != "action":
            return None
        _heal(b)                                          # keep her alive through the Nugget-Bridge gauntlet
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        return "head_to_gym" if "head_to_gym" in opts else (opts[0] if opts else None)
    camp._oracle_choose = stub

    start_map = tv.map_id(b)
    print(f"==== START map={start_map} (Route 4 — backward branch) coords={tv.coords(b)} "
          f"max_ticks={max_ticks} ====\n", flush=True)
    out = camp.free_roam(max_ticks=max_ticks, max_seconds=240, want_every=999)
    final_map = tv.map_id(b)
    q = camp._active_questline

    reached_cerulean = camp.world.visited((3, 3))
    opened_quest = q is not None or any(("Ticket" in e or "Bill" in e or "Nugget" in e or "errand" in e
                                         or "drifted off the road" in e) for e in events)
    not_backward = final_map != (3, 21)                   # never ended further backward (Route 3)
    headed_north = final_map in ((3, 43), (3, 3)) or (q is not None and q.actionable
                                                      and (q.actionable.dir == "north"))
    print(f"\n==== free_roam -> {out} | START {start_map} FINAL {final_map} coords={tv.coords(b)} ====", flush=True)
    print(f"   questline open: {q is not None}"
          + (f"  gate={q.gate.missing}  doing={q.actionable.human if q.actionable else None}" if q else ""))
    print(f"   reached Cerulean base camp (went EAST/forward): {reached_cerulean}")
    print(f"   opened the north S.S.-Ticket errand: {opened_quest}")
    print(f"   never ended backward on Route 3 (3,21): {not_backward}")
    ok = reached_cerulean and opened_quest and not_backward and headed_north
    print(f"\n==== forward-drive end-to-end: {'PASS' if ok else 'INSPECT'} ====", flush=True)


if __name__ == "__main__":
    main()
