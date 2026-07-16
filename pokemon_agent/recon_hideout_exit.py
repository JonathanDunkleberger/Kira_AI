"""recon_hideout_exit.py — walk her OUT of the hideout to Celadon street + heal + bank.

The scope strike banked mid-dungeon (B4F boss corridor). That canonical violates
watch-readiness: generic navigation (longrun/play_live) cannot cross the B3F/B2F spin
mazes (the crosser lives in spin_nav, not travel yet), so a resume there risks a wedge.
Exit route (all ground-truthed tonight): boss corridor -> elevator (20-21,23) -> ride UP
to B2F (28-29,16) -> spin-cross to the top corridor -> up-stairs (28,2) -> B1F north
(17,2 arrival) -> GC door (12,2) -> Game Corner -> street mats (9-11,13) -> Celadon
(3,6) -> heal at the registered Center -> bank -> promote as silph_scope_out.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_hideout_exit.py
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
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402
from spin_nav import SpinNav         # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_hideout_exit")
BANK = os.path.join(SCRATCH, "banked_SCOPE_OUT")
DBG = os.path.join(SCRATCH, "hideout_probe")

B4F, B2F, B1F, GC, ELEV, CELADON = (1, 45), (1, 43), (1, 42), (10, 14), (1, 46), (3, 6)


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
                           log=lambda m: print(m, flush=True)).run(max_seconds=240)

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

    def drain(max_a=30, key="A"):
        stable = 0
        for _ in range(max_a):
            if st.in_battle(b):
                return
            if dd_box(b):
                stable = 0
                b.press(key, 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(30):
                    b.run_frame()

    nav = SpinNav(b, camp, fight, drain, log=L)

    def snap(name):
        try:
            b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, name + ".png"))
        except Exception as e:
            L(f"   snap {name} failed: {e}")

    def ride(presses, key):
        """Board assumed done. Panel bg (0,2); floor multichoice; exit nearest door."""
        opened = False
        for front, face in (((1, 2), "LEFT"), ((0, 3), "UP"), ((0, 2), "UP")):
            r = camp.trav.travel(target_map=None, arrive_coord=front, max_steps=60, max_seconds=30)
            if r != "arrived" and not camp._step_to(front):
                continue
            for _ in range(4):
                b.press(face, 8, 10, camp.render, owner="agent")
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(40):
                    b.run_frame()
                    if dd_box(b):
                        opened = True
                        break
                if opened:
                    break
            if opened:
                break
        if not opened:
            return False
        b.press("A", 8, 12, camp.render, owner="agent")
        for _ in range(30):
            b.run_frame()
        for _ in range(presses):
            b.press(key, 8, 10, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
        b.press("A", 8, 12, camp.render, owner="agent")
        drain()
        for _ in range(300):
            b.run_frame()
        m0 = tuple(tv.map_id(b))
        camp.enter_warp(prefer="nearest")
        for _ in range(80):
            b.run_frame()
        return tuple(tv.map_id(b)) != m0

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)}")
    if tuple(tv.map_id(b)) == B4F:
        # 1. board the elevator from the boss corridor (doors 20-21,23; Lift Key in bag)
        boarded = False
        for door in ((20, 23), (21, 23)):
            m0 = tuple(tv.map_id(b))
            if camp.enter_warp(pick=door) == "warped" and tuple(tv.map_id(b)) != m0:
                boarded = True
                break
        if not boarded:
            snap("exit_10_no_board")
            L("!! couldn't board the B4F elevator — abort")
            return 1
        L(f"   in the elevator: {tv.map_id(b)}@{tv.coords(b)}")
        # 2. ride UP to B2F (from B4F's row, 1 UP = B2F; self-correct on the landing)
        landed_b2f = False
        for ups in (1, 2, 0, 3):
            if not ride(ups, "UP"):
                L(f"   ride(ups={ups}): no exit — retrying")
                continue
            L(f"   ride(ups={ups}) landed: map={tv.map_id(b)} coords={tv.coords(b)}")
            if tuple(tv.map_id(b)) == B2F:
                landed_b2f = True
                break
            camp.enter_warp(prefer="nearest")      # wrong floor: re-board (nearest = elevator)
            for _ in range(80):
                b.run_frame()
        if not landed_b2f:
            snap("exit_20_wrong_floor")
            L(f"!! never landed on B2F (at {tv.map_id(b)}@{tv.coords(b)}) — abort")
            return 1
    if tuple(tv.map_id(b)) == B2F:
        # 3. spin-cross to the top corridor, then the up-stairs (28,2) -> B1F north
        if not nav.cross(lambda t: abs(t[0] - 27) + abs(t[1] - 3) <= 1, "exit-b2f-top"):
            snap("exit_30_no_top")
            L("!! couldn't cross B2F back to the top corridor — abort")
            return 1
        m0 = tuple(tv.map_id(b))
        if camp.enter_warp(pick=(28, 2)) != "warped" or tuple(tv.map_id(b)) == m0:
            L("!! B2F up-stairs (28,2) didn't warp — abort")
            return 1
        for _ in range(80):
            b.run_frame()
        L(f"   B1F: {tv.map_id(b)}@{tv.coords(b)}")
    if tuple(tv.map_id(b)) == B1F:
        m0 = tuple(tv.map_id(b))
        if camp.enter_warp(pick=(12, 2)) != "warped" or tuple(tv.map_id(b)) == m0:
            L("!! B1F -> Game Corner stairs (12,2) didn't warp — abort")
            return 1
        for _ in range(80):
            b.run_frame()
        L(f"   Game Corner: {tv.map_id(b)}@{tv.coords(b)}")
    if tuple(tv.map_id(b)) == GC:
        m0 = tuple(tv.map_id(b))
        r = camp.enter_warp(prefer="south")
        if r != "warped" or tuple(tv.map_id(b)) == m0:
            for door in ((10, 13), (9, 13), (11, 13)):
                if camp.enter_warp(pick=door) == "warped" and tuple(tv.map_id(b)) != m0:
                    break
        for _ in range(80):
            b.run_frame()
        L(f"   out: {tv.map_id(b)}@{tv.coords(b)}")
    if tuple(tv.map_id(b)) != CELADON:
        snap("exit_40_not_out")
        L(f"!! not on Celadon street (at {tv.map_id(b)}@{tv.coords(b)}) — NOT banking")
        return 1

    # 4. heal (watch-readiness: canonical never sits hurt) + bank
    hr = camp.heal_nearest()
    L(f"   heal -> {hr}; now {tv.map_id(b)}@{tv.coords(b)}")
    snap("exit_50_celadon")
    _stage_save("exit")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} silph_scope_out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
