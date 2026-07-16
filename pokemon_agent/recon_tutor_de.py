"""recon_tutor_de.py - the DOUBLE-EDGE ERRAND (shift-18): walk from the Indigo center to the
Victory Road 2F move tutor, teach Venusaur DOUBLE-EDGE over EARTHQUAKE, walk back, heal, bank.

WHY (run21-23 evidence): three consecutive laps reached GARY THE CHAMPION and died PP-dry.
~50 damaging PP vs 26 gauntlet mons is the binding constraint; no Ether-class exists in bag or
shops (recon_bagdump VERIFIED). DE = 15 PP of 120 neutral replacing EQ's 10 PP that reads x0
into half the gauntlet (Golbat/Gyarados/Aerodactyl by chart, Gengar/Haunter by Levitate).
Strength CANNOT be the forget target (HM moves refuse the forget screen) - EQ is forced anyway.

GROUND TRUTH (pret, G:/temp/longrun/pret):
  - tutor NPC obj 12 in VictoryRoad_2F.json: (40,9), MOVEMENT_TYPE_FACE_DOWN, static ->
    approach (40,10), face UP, A. Script VictoryRoad_2F_EventScript_DoubleEdgeTutor,
    one-shot flag FLAG_TUTOR_DOUBLE_EDGE=0x2C0.
  - route: center exit -> Indigo ext (3,9) -> south edge connection -> Route 23 top ->
    door (18,28) -> VR 2F @ (48,12) -> tutor. All badge gates are SOUTH of the VR door.
  - VR 2F exit back: warp tiles (47,13)/(48,12)/(49,13) -> Route 23 (18,28).

RUN (staging rehearsal - never touches the real bank):
  .venv\\Scripts\\python.exe -u pokemon_agent\\recon_tutor_de.py
RUN (apply to the real ratchet bank, ONLY between runs, at a fresh whiteout-center bank):
  TUTOR_APPLY=1 -> banks the taught+healed state back to banked_E4.
Boot bank override: TUTOR_BOOT (default banked_E4). Success signal = move 38 in Venusaur's
decrypted move list + banked; every failure path B-cascades out and reports LOUD.
"""
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.environ.get("WATCH") != "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
os.environ.setdefault("BATTLE_DEBUG_DIR", os.path.join(
    os.environ.get("TEMP", _HERE), "longrun", "tutor_probe"))

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import firered_ram as ram            # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402
from hm_teach import TeachFlow       # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
BANK_E4 = os.path.join(SCRATCH, "banked_E4")
STAGE = os.path.join(SCRATCH, "stage_tutor")
BANK_OUT = BANK_E4 if os.environ.get("TUTOR_APPLY") == "1" else os.path.join(SCRATCH, "banked_TUTOR")
DBG = os.path.join(SCRATCH, "tutor_probe")

MOVE_DE, MOVE_EQ = 38, 89
CENTER_EXIT = (11, 16)               # inside the Indigo center -> exterior
CENTER_DOOR_EXT = (11, 6)            # exterior tile of the center door
VR_DOOR_R23 = (18, 28)               # Route 23 -> VR 2F (lands (48,12))
TUTOR = (40, 9)                      # static, faces DOWN
TUTOR_STAND = (40, 10)               # stand below, face UP
VR_EXIT_TILE = (48, 12)              # VR 2F -> Route 23 (the landing tile IS a warp back)
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    boot = os.environ.get("TUTOR_BOOT", BANK_E4)
    b = Bridge(ROM)
    with open(os.path.join(boot, "kira_campaign.state"), "rb") as f:
        b.load_state(f.read())
    print(f"BOOT from bank: {boot}  (apply={os.environ.get('TUTOR_APPLY') == '1'})", flush=True)
    for _ in range(40):
        b.run_frame()

    render_fn = lambda: None                       # noqa: E731

    def _choose(ptype, offers, ctx):
        for k in ("use_potion", "use_cure", "use_ether", "use_revive"):
            if k in offers:
                return k
        return "keep_fighting"

    def fight():
        return BattleAgent(b, on_event=lambda *a, **k: None, render=render_fn,
                           log=lambda m: print(m, flush=True),
                           choose=_choose).run(max_seconds=600)

    camp = Campaign(b, battle_runner=fight,
                    on_event=lambda s, **k: L(f"[event] {s}"),
                    beat=lambda *a, **k: None, render=render_fn)
    os.makedirs(STAGE, exist_ok=True)
    os.makedirs(DBG, exist_ok=True)

    def snap(name):
        try:
            b.frame_rgb().resize((480, 320)).save(os.path.join(DBG, name + ".png"))
        except Exception as e:
            L(f"   snap {name} failed: {e}")

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
            import json as _json
            camp.world.save(os.path.join(STAGE, "world_model.json"))
            camp.strat.save(os.path.join(STAGE, "strat_memory.json"))
            if camp.soul is not None:
                camp.soul.save(os.path.join(STAGE, "soul.json"))
            with open(os.path.join(STAGE, "journey_core.json"), "w", encoding="utf-8") as jf:
                _json.dump(camp._journey_narrative(), jf, ensure_ascii=False, indent=2)
        except Exception as e:
            L(f"!! stage continuity failed: {e}")

    camp._save_campaign = _stage_save
    camp._continuity_save = _stage_continuity
    camp._continuity_load = lambda *a, **k: None
    for name, loader in (("world_model.json", camp.world.load), ("strat_memory.json", camp.strat.load)):
        try:
            loader(os.path.join(boot, name))
        except Exception:
            pass
    try:
        if camp.soul is not None:
            camp.soul.load(os.path.join(boot, "soul.json"))
    except Exception:
        pass

    def bank(label):
        _stage_save(label)
        _stage_continuity()
        if os.path.isdir(BANK_OUT):
            shutil.rmtree(BANK_OUT)
        shutil.copytree(STAGE, BANK_OUT)
        L(f"BANKED [{label}] -> {BANK_OUT}")

    def fight_open():
        return (ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))
                and not ram.battle_cb2_dead(b))

    def drain(max_n=60, key="B"):
        stable = 0
        for _ in range(max_n):
            if fight_open():
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

    def handle_interrupts():
        if fight_open():
            fight()
            drain()
            return True
        if dd_box(b):
            drain()
            return True
        return False

    def settle(n=90):
        for _ in range(n):
            b.run_frame()

    def live_npc_tiles():
        OB, SZ = 0x02036E38, 0x24
        out = []
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.append((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                        b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    def walk(goal_test, label, tries=14, allow=()):
        budget = tries
        frozen, last_pos = 0, None
        map_start = tuple(tv.map_id(b))
        while budget > 0:
            budget -= 1
            if handle_interrupts():
                budget += 1
                continue
            if tuple(tv.map_id(b)) != map_start:
                L(f"   [{label}] map changed {map_start} -> {tuple(tv.map_id(b))} — bail")
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            if cur == last_pos:
                frozen += 1
                if frozen >= 3:
                    L(f"!! [{label}] FROZEN at {cur} x{frozen} replans — blind B/A drain + frame")
                    snap(f"frozen_{label[:12]}")
                    for k in ("B", "B", "A", "B"):
                        b.press(k, 8, 12, camp.render, owner="agent")
                        for _ in range(30):
                            b.run_frame()
                    drain()
                    frozen = 0
                    budget += 1
            else:
                frozen = 0
            last_pos = cur
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - set(allow)
            npcs = set(live_npc_tiles())
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs)
            L(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                L(f"   [{label}] no path from {cur}")
                snap(f"nopath_{label[:12]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                if handle_interrupts():
                    budget += 1
                    break
                if not camp._step_to(tuple(t)):
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
        return goal_test(tuple(tv.coords(b) or ()))

    def go_warp(tile, label):
        m0 = tuple(tv.map_id(b))
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in
               ((0, 1), (0, -1), (1, 0), (-1, 0))]
        for _attempt in range(4):
            cur = tuple(tv.coords(b) or ())
            if cur != tile and cur not in nbs:
                if not walk(lambda c, s=set(nbs): c in s, f"{label}-approach", allow=(tile,)):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            key = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            for _press in range(4):
                if key:
                    b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    break
            if handle_interrupts():
                continue
            if tuple(tv.map_id(b)) != m0:
                settle(180)
                L(f"   [{label}] {m0} -> {tuple(tv.map_id(b))} @ {tv.coords(b)}")
                return True
        return False

    def edge_cross(direction_key, edge_test, label, tries=10):
        """Cross a MAP CONNECTION (no warp tile): walk to the edge, keep stepping `direction_key`
        until the map id changes. edge_test(c) = are we standing on the crossable edge row."""
        m0 = tuple(tv.map_id(b))
        if not walk(edge_test, f"{label}-edge"):
            L(f"   [{label}] never reached the edge row")
            return False
        for _ in range(tries):
            if handle_interrupts():
                continue
            b.press(direction_key, 26, 10, camp.render, owner="agent")
            for _ in range(90):
                b.run_frame()
                if tuple(tv.map_id(b)) != m0:
                    break
            if tuple(tv.map_id(b)) != m0:
                settle(120)
                L(f"   [{label}] crossed {m0} -> {tuple(tv.map_id(b))} @ {tv.coords(b)}")
                return True
        L(f"   [{label}] edge never crossed (at {tv.coords(b)})")
        return False

    b.set_input_owner("agent")
    mid = tuple(tv.map_id(b))
    L(f"boot map={mid} coords={tv.coords(b)} money=${camp.money()}")
    moves0 = st.read_party_moves(b, 0)
    L(f"Venusaur moves BEFORE: {moves0}")
    if MOVE_DE in moves0:
        L("Double-Edge ALREADY known — nothing to do; banking as-is")
        bank("tutor_already")
        return
    if MOVE_EQ not in moves0:
        L("!! EQ (89) not in the move list — forget target missing, ABORT (needs eyes)")
        return

    # ── LEG 1: center -> exterior ────────────────────────────────────────────────
    if mid == (13, 0):
        if not go_warp(CENTER_EXIT, "center-exit"):
            L("!! stuck exiting the center"); snap("stuck_center"); return
    if tuple(tv.map_id(b)) != (3, 9):
        L(f"!! not on Indigo exterior (at {tuple(tv.map_id(b))}) — ABORT")
        return

    # ── LEG 2: exterior -> Route 23 (south edge connection) ─────────────────────
    g = tv.Grid(b)
    south_y = g.height - 1 if hasattr(g, "height") else 19
    if not edge_cross("DOWN", lambda c: c[1] >= south_y - 1, "indigo-south"):
        # fallback: the exterior is tiny — just try walking straight down from wherever
        L("!! indigo south edge cross failed"); snap("stuck_indigo_edge"); return
    r23 = tuple(tv.map_id(b))
    L(f"on Route 23 as map {r23} @ {tv.coords(b)}")

    # ── LEG 3: Route 23 -> VR 2F door at (18,28) ────────────────────────────────
    if not go_warp(VR_DOOR_R23, "vr-door"):
        L("!! couldn't enter Victory Road"); snap("stuck_vr_door"); return
    vr2f = tuple(tv.map_id(b))
    L(f"inside VR 2F as map {vr2f} @ {tv.coords(b)}")

    # ── LEG 4: stand below the tutor, face UP, TALK ──────────────────────────────
    if not walk(lambda c: c == TUTOR_STAND, "tutor-approach", allow=(TUTOR_STAND,)):
        L("!! couldn't reach the tutor stand tile"); snap("stuck_tutor_path"); return
    tf = TeachFlow(camp, log=lambda m: L(m))
    b.press("UP", 8, 10, camp.render, owner="agent")     # face, don't step (tap-turn)
    settle(30)

    # ── LEG 5: the tutor teach state machine (party/forget screens = hm_teach's) ─
    forget_idx = moves0.index(MOVE_EQ)
    L(f"teach: forget row {forget_idx} (EQ) -> Double-Edge")
    party_navved = forgot = False
    t_teach = time.time()
    b.press("A", 8, 12, camp.render, owner="agent")      # open the tutor dialogue
    settle(40)
    for _ in range(80):
        if MOVE_DE in st.read_party_moves(b, 0):
            break
        if time.time() - t_teach > 120:
            L("!! teach loop timed out"); snap("teach_timeout"); break
        scr = tf._classify()
        if scr == "party":
            if not party_navved:
                if not tf._party_goto(0):
                    L("!! party cursor never reached slot 0"); snap("teach_party"); break
                party_navved = True
                tf._press("A", settle=90)
            else:
                tf._press("A", settle=60)
        elif scr == "forget" and not forgot:
            if not tf._forget_goto(forget_idx):
                L(f"!! forget cursor never reached row {forget_idx}"); snap("teach_forget"); break
            tf._press("A", settle=90)
            forgot = True
        else:                                            # dialogue / YES-default boxes
            tf._press("A", settle=50)
    tf._b_cascade()
    moves1 = st.read_party_moves(b, 0)
    L(f"Venusaur moves AFTER: {moves1}")
    if MOVE_DE not in moves1:
        snap("teach_failed")
        L("!! TEACH FAILED — Double-Edge not in the move list (LOUD, nothing banked)")
        return
    if MOVE_EQ in moves1:
        L("   note: EQ still present too (tutor filled an empty slot?) — fine, strictly better")
    L("*** DOUBLE-EDGE TAUGHT (ground-truthed) ***")
    snap("taught")

    # ── LEG 6: walk back — VR 2F -> Route 23 -> north edge -> Indigo -> center ──
    if not go_warp(VR_EXIT_TILE, "vr-exit"):
        L("!! couldn't exit VR (banking taught state IN PLACE as fallback)")
        bank("tutor_taught_in_vr")
        return
    if not edge_cross("UP", lambda c: c[1] <= 1, "route23-north"):
        L("!! couldn't cross back to Indigo (banking taught state on Route 23)")
        bank("tutor_taught_r23")
        return
    if not go_warp(CENTER_DOOR_EXT, "center-door"):
        L("!! couldn't re-enter the center (banking on the exterior)")
        bank("tutor_taught_ext")
        return
    r = camp.heal_nearest()
    L(f"[heal] heal_nearest -> {r}")
    drain()
    bank("tutor_done")
    L(f"DONE in {time.time() - t0:.0f}s — Double-Edge aboard, healed, banked to {BANK_OUT}")


if __name__ == "__main__":
    main()
