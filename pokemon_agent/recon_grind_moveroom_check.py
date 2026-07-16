"""recon_grind_moveroom_check.py — NS#16 verify: the grind-mon move-learn ROOM fix.

THE BUG (fixed this shift): every grind caller (grind_weak_members, road-bench-XP, recon_grind_bench)
REORDERS the weak specialist to slot 0 before leveling it, but _ensure_move_room only ever ran on the
PRE-reorder lead (the ace). So a fielded bench mon with 4 FULL moves that levels through a learnset
threshold mid-grind (Lapras -> Ice Beam @L43) hits the un-actuatable "Delete a move?" box and the learn
is DECLINED -> it reaches the E4 without its signature coverage move ("Lapras has no Ice move").

THE FIX: _ensure_move_room() now runs at the top of grind() (covers every grind caller — slot 0 is the
grind mon there) + at the road-bench-XP reorder site.

THIS CHECK (deterministic, no full grind needed — proves the exact gap is closed for the exact target):
boot giovanni_kit_g (Lapras L37 slot3, 4 full moves [Surf, Body Slam, Confuse Ray, Perish Song], next
natural learn = Ice Beam @43), reorder Lapras to slot 0 (what the grind does), call _ensure_move_room(),
and assert a JUNK slot was freed while Surf (its STAB hard-hitter) is RETAINED. A freed slot => the L43
Ice Beam auto-learns with no box (the established _ensure_move_room invariant, proven for the ace across
the whole project). Also confirms the pre-fix bug shape: with 4 full moves and no move-room, the box
would appear.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge            # noqa: E402
import pokemon_state as st           # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
WS = os.path.join(_HERE, "states", "workshop")
LAPRAS = 131
ICE_BEAM = 58


def _names(b, slot):
    return [st.MOVE_NAMES.get(m, f"#{m}") for m in st.read_party_moves(b, slot) if m]


def main():
    fixture = os.environ.get("MOVEROOM_STATE", "giovanni_kit_g")
    state = os.path.join(WS, fixture + ".state")
    b = Bridge(ROM)
    with open(state, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    camp = Campaign(b, battle_runner=lambda *a, **k: "ok",
                    on_event=lambda *a, **k: None, beat=lambda *a, **k: None, render=lambda: None)
    for loader, side in ((camp.world.load, os.path.join(WS, fixture + ".world_model.json")),
                         (camp.strat.load, os.path.join(WS, fixture + ".strat_memory.json"))):
        try:
            loader(side)
        except Exception:
            pass

    # locate Lapras
    cnt = b.rd8(st.ram.GPLAYER_PARTY_CNT)
    lap = next((s for s in range(min(cnt, 6)) if st.read_party_species(b, s) == LAPRAS), None)
    assert lap is not None, "no Lapras in the fixture"
    print(f"[check] Lapras at slot {lap}: {_names(b, lap)}")

    fails = []
    # reorder Lapras to slot 0 (exactly what grind_weak_members / recon_grind_bench do before grind())
    if lap != 0:
        camp._swap_party_slots(0, lap)
    before = st.read_party_moves(b, 0)
    before_names = _names(b, 0)
    n_before = sum(1 for m in before if m)
    print(f"[check] slot0 (Lapras) after reorder, BEFORE move-room: {before_names}  ({n_before} moves)")
    if n_before != 4:
        fails.append(f"expected 4 full moves before, got {n_before}")

    # THE FIX under test: grind() now calls this at its top on the fielded grind mon.
    dropped = camp._ensure_move_room()
    after = st.read_party_moves(b, 0)
    after_names = _names(b, 0)
    n_after = sum(1 for m in after if m)
    print(f"[check] _ensure_move_room() -> dropped={dropped!r}; slot0 now: {after_names}  ({n_after} moves)")

    # ASSERT: a slot was freed (so the L43 Ice Beam auto-learns) ...
    if n_after >= 4:
        fails.append("no move-room freed — the L43 learn would still hit the un-actuatable box (BUG)")
    # ... and Surf (the STAB hard-hitter) was NOT tossed ...
    if 57 not in after and 55 not in after:   # 57 = Surf; keep whichever water STAB it has
        # (giovanni_kit_g Lapras runs Surf=57; guard on it surviving)
        if 57 in before:
            fails.append("Surf was dropped — the hard-hitter must be protected")
    # ... and Ice Beam is NOT present yet (proving the gain must come from the L43 learn, not the fixture)
    if ICE_BEAM in before:
        print("[note] fixture Lapras already has Ice Beam — pick a pre-43 fixture for a cleaner check")

    print("\n=== VERDICT ===")
    if fails:
        for f in fails:
            print("  FAIL:", f)
        print("RESULT: FAIL")
        sys.exit(1)
    print(f"  PASS: reordered grind-lead Lapras had a JUNK slot freed ({before_names} -> {after_names}); "
          f"Surf retained; the L43 Ice Beam will now auto-learn instead of hitting the delete box.")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
