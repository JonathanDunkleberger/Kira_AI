"""recon_surf_teach.py — teach SURF (HM03) to LAPRAS + STRENGTH (HM04) to a carrier.

The link after the Safari strike (canonical = safari_hms: HM03+HM04 in the TM case).
hm_teach.HMTeach.teach is the standing verified vehicle; this wrapper:
  1. teaches SURF -> Lapras (party slot 5), picking the forget row at runtime from her
     actual moves (never overwrite a damaging move: prefer Mist/Sing/Growl-class chaff);
  2. teaches STRENGTH -> the best ROM-compatible carrier that has a droppable move
     (Victory Road needs it later; teaching now costs nothing);
  3. verifies by read_party_moves (Surf=57, Strength=70), banks -> banked_SURF_TAUGHT.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_surf_teach.py
"""
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import hm_teach as ht                # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_surf_teach")
BANK = os.path.join(SCRATCH, "banked_SURF_TAUGHT")

SURF_MOVE, STRENGTH_MOVE = 57, 70
# moves safe to forget, worst first (status chaff a real player would drop)
CHAFF = [47, 45, 39, 43, 54, 145, 48, 111, 28, 46, 104]   # sing,growl,tailwhip,leer,mist,
#                                                           bubble,supersonic,defcurl,sand-
#                                                           attack,roar,double-team


def pick_forget(moves):
    """Index of the most droppable move, or None if a slot is free."""
    if len([m for m in moves if m]) < 4:
        return None
    for chaff in CHAFF:
        if chaff in moves:
            return moves.index(chaff)
    return 0                        # nothing chaffy — drop slot 0 (oldest)


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    camp = Campaign(b, battle_runner=lambda: "skip",
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    os.makedirs(STAGE, exist_ok=True)
    camp._save_campaign = lambda *a, **k: True
    camp._continuity_save = lambda *a, **k: None
    camp._continuity_load = lambda *a, **k: None
    for loader, path in ((camp.world.load, C.WORLD_JSON), (camp.strat.load, C.STRAT_JSON)):
        try:
            loader(path)
        except Exception:
            pass
    try:
        if camp.soul is not None:
            camp.soul.load(os.path.join(CANON, "soul.json"))
    except Exception:
        pass

    teacher = ht.TeachFlow(camp, log=lambda m: print(m, flush=True))

    def party():
        n = b.rd8(__import__("firered_ram").GPLAYER_PARTY_CNT)
        return [(s, st.read_party_species(b, s), st.read_party_moves(b, s))
                for s in range(n)]

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)}")
    roster = party()
    for s, sp, mv in roster:
        L(f"   slot {s}: {st.SPECIES_NAME.get(sp, sp)} moves={mv}")

    # ── SURF -> Lapras ──
    lap = next((s for s, sp, _m in roster if sp == 131), None)
    if lap is None:
        L("!! no Lapras in party — abort")
        return 1
    mv = st.read_party_moves(b, lap)
    if SURF_MOVE not in mv:
        r = teacher.teach("surf", lap, forget_idx=pick_forget(mv))
        L(f"   SURF -> lapras slot {lap}: {r} (moves now {st.read_party_moves(b, lap)})")
        if SURF_MOVE not in st.read_party_moves(b, lap):
            L("!! Surf not on Lapras — abort LOUD")
            return 1
    else:
        L("   Surf already known")

    # ── STRENGTH -> the first ROM-compatible carrier with chaff to drop ──
    taught_str = False
    for s, sp, _m in party():
        if sp == 131 and len([x for x in st.read_party_moves(b, s) if x]) >= 4 \
                and pick_forget(st.read_party_moves(b, s)) is None:
            continue
        try:
            compat = ht.hm_compatible(b, "strength", sp)
        except Exception:
            compat = None
        if compat is False:
            continue
        mv2 = st.read_party_moves(b, s)
        if STRENGTH_MOVE in mv2:
            taught_str = True
            L(f"   Strength already on slot {s}")
            break
        r = teacher.teach("strength", s, forget_idx=pick_forget(mv2))
        L(f"   STRENGTH -> slot {s} ({st.SPECIES_NAME.get(sp, sp)}): {r} "
          f"(moves now {st.read_party_moves(b, s)})")
        if STRENGTH_MOVE in st.read_party_moves(b, s):
            taught_str = True
            break
    if not taught_str:
        L("!! Strength found no carrier — CONTINUING (Surf is the mission; Strength owed)")

    # ── bank ──
    with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
        f.write(b.save_state())
    try:
        camp.world.save(os.path.join(STAGE, "world_model.json"))
        camp.strat.save(os.path.join(STAGE, "strat_memory.json"))
        if camp.soul is not None:
            camp.soul.save(os.path.join(STAGE, "soul.json"))
        with open(os.path.join(STAGE, "journey_core.json"), "w", encoding="utf-8") as jf:
            json.dump(camp._journey_narrative(), jf, ensure_ascii=False, indent=2)
    except Exception as e:
        L(f"!! stage continuity failed: {e}")
    if os.path.isdir(BANK):
        shutil.rmtree(BANK, ignore_errors=True)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} surf_taught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
