"""recon_route24.py — RECON the Route 24 / Nugget Bridge north-traversal blocker (read-only).

Drives NORTH from the live save (head_to_gym oracle, heal each tick to survive the gauntlet) and at EVERY
map dumps the RAW live map-header connection table + warps + dims + player position, so we can see exactly
why the questline executor finds no north edge at (3,43): is the north connection genuinely absent, is it a
sub-map, or is it a walk-to-the-edge situation (she's mid-map, the connection only fires at the boundary)?

Never writes a canonical save. RUN: python pokemon_agent\\recon_route24.py [max_ticks]
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
_DIRN = {1: "S", 2: "N", 3: "W", 4: "E"}


def _heal(b):
    try:
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            b.core.memory.u16.raw_write(base + 0x56, b.rd16(base + 0x58))
    except Exception:
        pass


def _boost(b):
    """GEOGRAPHY RECON ONLY (no save banked): bump the party to crush the Nugget-Bridge gauntlet so she
    reaches (3,43)+ and we can read what's actually north. Level byte + all six stat halfwords."""
    # VERIFICATION boost (geography only, no save banked): a pure HP/stat RAM write gets WIPED the moment a
    # battle recomputes stats from LEVEL — so set the LEVEL high (the recompute then yields strong L-scaled
    # stats that survive the battle) AND write fat HP + lead offence for the pre-recompute moment. Only HP +
    # the two offence slots are written directly (writing def/spd/spdef desyncs the engine into a false 'menu
    # glitched' wedge — learned the hard way). This simulates a properly-LEVELLED team (the real path is she
    # grinds up; the live run uses a levelled save, not this poke).
    try:
        for s in range(ram.read_party_count(b)):
            base = ram.GPLAYER_PARTY + s * 100
            b.core.memory.u8.raw_write(base + 0x54, 50)            # level (drives the stat recompute)
            b.core.memory.u16.raw_write(base + 0x58, 600)          # max HP
            b.core.memory.u16.raw_write(base + 0x56, 600)         # cur HP
        lead = ram.GPLAYER_PARTY
        b.core.memory.u16.raw_write(lead + 0x5A, 350)             # attack
        b.core.memory.u16.raw_write(lead + 0x60, 350)            # sp.atk
    except Exception as e:
        print(f"   (boost skipped: {e})", flush=True)


def raw_connections(b):
    """Raw header connection table: [(dir_letter, (grp,num))] + the raw count, for debugging _map_connections."""
    try:
        ch = b.rd32(0x02036DFC + 0x0C)
        if ch < 0x02000000:
            return ("no conn block", [])
        cnt, arr = b.rd32(ch), b.rd32(ch + 0x04)
        if not (0 <= cnt < 64) or arr < 0x02000000:
            return (f"cnt={cnt} arr={arr:08x} (invalid)", [])
        out = []
        for i in range(cnt):
            c = arr + i * 0xC
            d = b.rd8(c)
            out.append((_DIRN.get(d, f"?{d}"), (b.rd8(c + 0x08), b.rd8(c + 0x09))))
        return (f"cnt={cnt}", out)
    except Exception as e:
        return (f"err {e}", [])


def dump(b, tag=""):
    mid = tv.map_id(b)
    co = tv.coords(b)
    desc, conns = raw_connections(b)
    try:
        g = tv.Grid(b)
        dims = f"{getattr(g, 'w', '?')}x{getattr(g, 'h', '?')}"
    except Exception:
        dims = "?"
    try:
        warps = tv.read_warps(b)
    except Exception as e:
        warps = f"(err {e})"
    print(f"  {tag}MAP {mid} coords={co} dims={dims} | header {desc} conns={conns}", flush=True)
    print(f"       warps={warps}", flush=True)
    return mid


def main():
    max_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    b = Bridge(ROM)
    with open(resolve_state("kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()
    _heal(b)

    camp = Campaign(b, battle_runner=lambda: BattleAgent(
        b, on_event=lambda *a, **k: None, render=lambda: None, log=lambda m: None).run(max_seconds=40),
        on_event=lambda s, **k: print(f"   [event] {s}", flush=True),
        beat=lambda *a, **k: None, render=lambda: None)
    camp._save_campaign = lambda *a, **k: None
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    try:
        camp.world.load(WORLD_JSON)
    except Exception:
        pass
    camp.trav.battle_runner = camp._flee_runner

    seen_maps = []

    def stub(kind, options, ctx):
        if kind != "action":
            return None
        _boost(b); _heal(b)
        m = dump(b, tag="DECISION ")
        if m not in seen_maps:
            seen_maps.append(m)
        opts = list(options.keys()) if isinstance(options, dict) else list(options or [])
        return "head_to_gym" if "head_to_gym" in opts else (opts[0] if opts else None)
    camp._oracle_choose = stub

    print(f"==== START ====", flush=True)
    dump(b, tag="INITIAL ")
    camp.free_roam(max_ticks=max_ticks, max_seconds=300, want_every=999)
    print(f"\n==== FINAL ====", flush=True)
    dump(b, tag="FINAL ")
    import field_moves as fm
    print(f"\n   maps visited (decision points): {seen_maps}", flush=True)
    print(f"   FLAG_GOT_SS_TICKET (0x234) = {fm.read_flag(b, 0x234)}  (True = ticket obtained, gate opens)",
          flush=True)
    # focus: did she reach (3,43) and what is north of it?
    print(f"\n   NOTE: look for a map whose header has NO 'N' connection while the questline wanted north —"
          f" that's the no-edge wedge. If 'N' IS present, the executor bug is elsewhere.", flush=True)


if __name__ == "__main__":
    main()
