"""recon_questline_p3.py — Phase 3 verify: the questline DRIVES her north (the unlock errand), not south
into the Slowbro wall.

Boots the LIVE campaign save (player wedged at the Cerulean south chokepoint, FLAG_GOT_SS_TICKET unset),
stubs the oracle to always pick head_to_gym (her "advance the quest" intent), flees wilds for speed, and
runs a few free-roam ticks. PASS = she OPENS the S.S.-Ticket questline and her map moves NORTH (toward
Route 24 (3,43) / Nugget Bridge), never crossing the gated south edge. Never writes canonical saves.

RUN: .venv\\Scripts\\python.exe pokemon_agent\\recon_questline_p3.py [max_ticks]
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

from bridge import Bridge              # noqa: E402
import firered_ram as ram             # noqa: E402
import travel as tv                   # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign, resolve_state  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")


def _heal(b):
    try:
        w16 = b.core.memory.u16.raw_write
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            w16(base + 0x56, b.rd16(base + 0x58))
    except Exception:
        pass


def main():
    max_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    _heal(b)

    def fight():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: None).run(max_seconds=40)

    events = []
    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: (events.append(s), print(f"   [event] {s}", flush=True)),
                    beat=lambda *a, **k: None, render=lambda: None)
    camp.trav.battle_runner = camp._flee_runner          # nav-bound (flee wilds) for speed
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None

    def stub_choose(kind, options, ctx):
        if kind != "action":
            return None
        _heal(b)
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        return "head_to_gym" if "head_to_gym" in opts else (opts[0] if opts else None)
    camp._oracle_choose = stub_choose

    start_map = tv.map_id(b)
    print(f"==== START map={start_map} coords={tv.coords(b)} ticket={camp.b.rd8(0) and ''}"
          f" (Cerulean, pre-Bill) max_ticks={max_ticks} ====", flush=True)
    import field_moves as fm
    print(f"   FLAG_GOT_SS_TICKET={fm.read_flag(b, 0x234)} (expect False)", flush=True)

    out = camp.free_roam(max_ticks=max_ticks, max_seconds=240, want_every=999)

    final_map = tv.map_id(b)
    q = camp._active_questline
    opened = q is not None
    print(f"\n==== END free_roam -> {out} | START map={start_map} FINAL map={final_map} coords={tv.coords(b)} ====",
          flush=True)
    print(f"   questline active: {opened}"
          + (f"  gate={q.gate.kind}/{q.gate.missing}  doing={q.actionable.human if q.actionable else None}"
             if opened else ""), flush=True)
    # PASS conditions
    moved_north = final_map != start_map and final_map != (3, 23)   # left Cerulean, NOT via the south(Route5)
    reached_r24 = final_map == (3, 43)
    opened_ticket_quest = any("Ticket" in e or "Bill" in e or "Nugget" in e for e in events)
    print(f"\n   left Cerulean: {final_map != start_map} (final {final_map})", flush=True)
    print(f"   went NORTH (not south to Route5 3,23): {moved_north}", flush=True)
    print(f"   reached Route 24 (3,43): {reached_r24}", flush=True)
    print(f"   surfaced the S.S.-Ticket plan in voice: {opened_ticket_quest}", flush=True)
    ok = opened_ticket_quest and (reached_r24 or (moved_north and final_map[0] == 3))
    print(f"\n==== Phase-3 questline drive: {'PASS' if ok else 'INSPECT'} ====", flush=True)


if __name__ == "__main__":
    main()
