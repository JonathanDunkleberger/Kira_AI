"""recon_transplant_party.py — build a fixture = DONOR's party at DEST's map position.

Night-shift #4 (2026-07-12): the multi-gym selective-solo demo needs an EVENED badge-6 bench
at a CLEAN-NAV position. koga_done_kit HAS the evened bench (Venusaur L62 + L28-29 x5) but its
BOOT POSITION is the Route-9 step-freeze pocket (NS#27 — nav-blocked, never reaches Fuchsia).
surf_ready_kit is AT Fuchsia and navs cleanly past it, but its bench is catastrophically thin
(L8-25). Transplant koga's PARTY bytes into surf_ready's Fuchsia RAM => "a fresh-GO player who
just beat Koga, standing at Fuchsia, with an evened team" — game-consistent, clean-nav, right shape.

Party structs (6 x 100 B at GPLAYER_PARTY) are self-contained (own checksum, same OT=Kira => obeys)
so a verbatim copy is valid — same operation _swap_party_slots does. Position stays DEST's (valid).

SIDECARS written for OUT: world_model = DEST's (matches the Fuchsia position/graph);
journey_core/soul/strat_memory = DONOR's (match the transplanted team levels/species).

RUN: DONOR=koga_done_kit DEST=surf_ready_kit OUT=fuchsia_evened_kit python recon_transplant_party.py
"""
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge
import firered_ram as ram
import pokemon_state as st

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WS = os.path.join(_HERE, "states", "workshop")
PARTY_BYTES = 6 * st.PARTY_MON_SIZE   # 600


def state_path(name):
    p = os.path.join(WS, name if name.endswith(".state") else name + ".state")
    if not os.path.exists(p):
        raise SystemExit(f"missing state: {p}")
    return p


def boot(path):
    b = Bridge(ROM)
    with open(path, "rb") as f:
        b.load_state(f.read())
    for _ in range(6):
        b.run_frame()
    return b


def dump_party(name):
    b = boot(state_path(name))
    n = ram.read_party_count(b)
    data = b.read_bytes(ram.GPLAYER_PARTY, PARTY_BYTES)
    levels = [b.rd8(ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE + 0x54) for s in range(n)]
    print(f"   DONOR {name}: count={n} levels={levels}")
    return n, bytes(data)


def main():
    donor = os.environ["DONOR"]
    dest = os.environ["DEST"]
    out = os.environ["OUT"]

    n, party = dump_party(donor)

    b = boot(state_path(dest))
    print(f"   DEST {dest}: pre-transplant count={ram.read_party_count(b)} "
          f"levels={[b.rd8(ram.GPLAYER_PARTY + s*st.PARTY_MON_SIZE + 0x54) for s in range(ram.read_party_count(b))]}")
    # SLOTS: which slots to overwrite with the SAME-INDEX donor slot; others keep DEST's mon.
    # Default all 6. Use e.g. SLOTS=1,2,3,4 to keep DEST's slot0 (ace) + slot5 (HM-user) and only
    # swap the bench — so the fixture stays HM-consistent with DEST's map position.
    _slots = os.environ.get("SLOTS")
    slots = [int(x) for x in _slots.split(",")] if _slots else list(range(6))
    MS = st.PARTY_MON_SIZE
    for s in slots:
        for i in range(s * MS, s * MS + MS, 4):
            word = party[i] | (party[i+1] << 8) | (party[i+2] << 16) | (party[i+3] << 24)
            b.core.memory.u32.raw_write(ram.GPLAYER_PARTY + i, word)
    if _slots is None:
        b.core.memory.u8.raw_write(ram.GPLAYER_PARTY_CNT, n)   # full copy => donor count
    print(f"   copied donor slots {slots}")
    for _ in range(4):
        b.run_frame()
    post = [b.rd8(ram.GPLAYER_PARTY + s*st.PARTY_MON_SIZE + 0x54) for s in range(ram.read_party_count(b))]
    print(f"   POST-transplant: count={ram.read_party_count(b)} levels={post}")

    out_state = os.path.join(WS, out + ".state")
    with open(out_state, "wb") as f:
        f.write(b.save_state())
    print(f"   wrote {out_state}")

    # sidecars: ALL from DEST — the position/world_model/spine-context must be self-consistent for
    # nav (koga's journey/world context wedges surf_ready's Fuchsia graph). The live party RAM is
    # koga's; team_planner/grind/battle read the LIVE party, so the journey roster mismatch is cosmetic
    # (narration only). SIDE_FROM_DONOR=1 to instead pull journey/soul/strat from DONOR.
    _side_src = donor if os.environ.get("SIDE_FROM_DONOR") else dest
    for suf, src in (("world_model.json", dest), ("journey_core.json", _side_src),
                     ("soul.json", _side_src), ("strat_memory.json", _side_src)):
        s = os.path.join(WS, f"{src}.{suf}")
        d = os.path.join(WS, f"{out}.{suf}")
        if os.path.exists(s):
            shutil.copyfile(s, d)
            print(f"   sidecar {suf} <- {src}")
        else:
            print(f"   sidecar {suf} MISSING from {src} (skipped)")
    print("DONE.")


if __name__ == "__main__":
    main()
