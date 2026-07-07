"""recon_snorlax.py — WAKE THE ROUTE 12 SNORLAX (night shift #6): the flute's payoff.

koga_run1 truth: the questline walks her to the Snorlax's face ((3,30)@(12,0)) but nothing
wires "press A on the sleeping blocker" — head_to_gym just re-routes into the body x14.
The ritual (KB frlg_gates + Bulbapedia): face the SNORLAX object, A → "...play the POKE
FLUTE?" YES → it wakes and ATTACKS (wild L30) → beat it → FLAG_WOKE_UP_ROUTE_12_SNORLAX
(0x253) sets and the body leaves the road. Success = flag truth + the template gone.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_snorlax.py
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
import field_moves as fm             # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_snorlax")
BANK = os.path.join(SCRATCH, "banked_SNORLAX")
DBG = os.path.join(SCRATCH, "tower_probe")

ROUTE12 = (3, 30)
FLAG_WOKE = 0x253
GFX_SNORLAX = None      # discovered live from the template blocking the road


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    def fight():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None,
                           log=lambda m: print(m, flush=True)).run(max_seconds=300)

    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=lambda: None)
    os.makedirs(STAGE, exist_ok=True)
    os.makedirs(DBG, exist_ok=True)

    def _stage_save(reason="tick"):
        try:
            with open(os.path.join(STAGE, "kira_campaign.state"), "wb") as f:
                f.write(b.save_state())
            return True
        except Exception as e:
            L(f"!! STAGE SAVE FAILED [{reason}]: {e}")
            return False

    def _stage_continuity():
        try:
            camp.world.save(os.path.join(STAGE, "world_model.json"))
            camp.strat.save(os.path.join(STAGE, "strat_memory.json"))
            if camp.soul is not None:
                camp.soul.save(os.path.join(STAGE, "soul.json"))
            with open(os.path.join(STAGE, "journey_core.json"), "w", encoding="utf-8") as jf:
                json.dump(camp._journey_narrative(), jf, ensure_ascii=False, indent=2)
        except Exception as e:
            L(f"!! stage continuity failed: {e}")

    camp._save_campaign = _stage_save
    camp._continuity_save = _stage_continuity
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

    def drain(max_a=40):
        stable = 0
        for _ in range(max_a):
            if st.in_battle(b):
                return
            if dd_box(b):
                stable = 0
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(30):
                    b.run_frame()

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} woke_flag={fm.read_flag(b, FLAG_WOKE)}")
    if fm.read_flag(b, FLAG_WOKE):
        L("Snorlax already woken — nothing to do")
        return 0
    if tuple(tv.map_id(b)) != ROUTE12:
        L("!! not on Route 12 — this strike boots from the snorlax-front canonical; abort")
        return 1

    # 1. the NORTH GATE (live warp truth: north doors (14,15)/(15,15) -> (23,0), south door
    #    (14,21)): enter from the north (prefer='south' = approach y-1, step DOWN), cross the
    #    interior, exit its south door. The Snorlax itself sleeps at (14,70) — the koga_run1
    #    (12,0) stall was head_to_gym failing to route past it, not a blocker at the top.
    if tuple(tv.coords(b))[1] < 15:
        m0 = tuple(tv.map_id(b))
        for door in ((14, 15), (15, 15)):
            if camp.enter_warp(prefer="south", pick=door) == "warped" \
                    and tuple(tv.map_id(b)) != m0:
                break
        if tuple(tv.map_id(b)) != m0:
            for _ in range(60):
                b.run_frame()
            camp.enter_warp(prefer="south")
            for _ in range(60):
                b.run_frame()
        L(f"   past the gate: {tv.map_id(b)}@{tv.coords(b)}")
        if tuple(tv.map_id(b)) != ROUTE12 or tuple(tv.coords(b))[1] < 16:
            L("!! gate pass-through didn't land on the south road — abort")
            return 1

    # 2. the Snorlax body — disasm truth (14,70), verified against the live template list
    SNORLAX = (14, 70)
    cur = tuple(tv.coords(b))
    present = {t for t, _g, p in tv.read_object_templates(b) if p}
    body = SNORLAX if SNORLAX in present else next(
        (t for t in sorted(present, key=lambda t: abs(t[0] - SNORLAX[0]) + abs(t[1] - SNORLAX[1]))
         if abs(t[0] - SNORLAX[0]) + abs(t[1] - SNORLAX[1]) <= 4), None)
    if body is None:
        L(f"!! no present template near {SNORLAX} — already woken? flag={fm.read_flag(b, FLAG_WOKE)}")
        return 1
    L(f"   SNORLAX body at {body} (she is at {cur})")

    # face + A ritual from each adjacent tile (nearest first); YES is the default cursor
    def adj_key(f):
        return {(1, 0): "LEFT", (-1, 0): "RIGHT", (0, -1): "DOWN", (0, 1): "UP"}[
            (f[0] - body[0], f[1] - body[1])]

    fronts = sorted(((body[0] + dx, body[1] + dy) for dx, dy in
                     ((0, -1), (0, 1), (1, 0), (-1, 0))),
                    key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
    woke_battle = False
    for front in fronts:
        if tuple(tv.coords(b)) != front:
            r = camp.trav.travel(target_map=None, arrive_coord=front, max_steps=300,
                                 max_seconds=240)
            if st.in_battle(b):                    # fisherman gauntlet en route
                L(f"   battle en route -> {camp.battle_runner()}")
                drain()
                r = camp.trav.travel(target_map=None, arrive_coord=front, max_steps=300,
                                     max_seconds=240)
            if r != "arrived" and not camp._step_to(front):
                continue
        key = adj_key(front)
        for _ in range(6):
            b.press(key, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if dd_box(b):
                drain()                    # "...play the POKE FLUTE?" -> A = YES -> music
                for _ in range(400):       # the wake cutscene
                    b.run_frame()
                    if st.in_battle(b):
                        break
                drain()
            if st.in_battle(b):
                L("   SNORLAX WOKE — battle!")
                out = camp.battle_runner()
                L(f"   snorlax battle -> {out}")
                drain()
                woke_battle = True
                break
        if woke_battle:
            break

    for _ in range(120):
        b.run_frame()
    gone = body not in {t for t, _g, present in tv.read_object_templates(b) if present}
    flag = fm.read_flag(b, FLAG_WOKE)
    L(f"   woke flag={flag} body_gone={gone} (battled={woke_battle})")
    try:
        b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, "snorlax_final.png"))
    except Exception:
        pass
    if not (flag or gone):
        L("!! Snorlax still asleep — NOT banking; read the frame")
        return 1

    _stage_save("snorlax")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} snorlax_woken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
