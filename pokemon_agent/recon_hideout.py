"""recon_hideout.py — THE ROCKET HIDEOUT STRIKE (night shift #5): Silph Scope end-to-end.

run-12 truth: the questline lands her INSIDE the Game Corner (door hint (34,21) works) but the
generic room tour talks gamblers, never the grunt/poster gate, then wanders the Celadon mansion.
The gate is a fixed interior RITUAL (disasm ground truth, STATE §0 shift-4 bill):
  Game Corner (10,14): grunt (11,2) guards poster (11,1) — beat him, press the poster -> the
  stairs (15,2) unlock -> B1F. Descent: B1F down (17,2) -> B2F (arrive 28,2) down (21,2) ->
  B3F (arrive 18,2) down (15,18) -> B4F (arrive 11,15): GIOVANNI (19,4), Silph Scope ball
  (20,5), door grunts (16,14)/(19,14). Success = ('item', 359) in Key Items (bag truth; the
  0x037 flag LIES).
Canonical protection: staging pattern; bank -> %TEMP%/longrun/banked_SCOPE for promote_bank.py.
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
import firered_ram as ram            # noqa: E402
import hm_teach as ht                # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_hideout")
BANK = os.path.join(SCRATCH, "banked_SCOPE")
DBG = os.path.join(SCRATCH, "hideout_probe")

CELADON = (3, 6)
GC_DOOR = (34, 21)                 # Celadon -> Game Corner
GRUNT_FRONT, POSTER_FRONT = (11, 3), (11, 2)   # face UP at each
GC_STAIRS = (15, 2)                # unlocked by the poster -> B1F
SILPH_SCOPE = 359
# per-floor down-stairs (ridden in order); arrive coords are logged, never assumed
DESCENT = [(17, 2), (21, 2), (15, 18)]
GIOVANNI_FRONT = (19, 5)           # boss at (19,4), face UP
SCOPE_BALL = (20, 5)               # item ball he leaves; press from below (20,6) or beside


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

    def key_items():
        return ht.pocket_items(b, ht.KEY_ITEMS_OFF, 30)

    def drain(max_a=30, key="A"):
        """Advance/close any open box; stop when stably closed."""
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

    def goto(tile, label):
        r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=200, max_seconds=90)
        if st.in_battle(b):                       # LoS trainer triggered en route
            L(f"   [{label}] battle en route -> {fight()}")
            drain()
            r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=200, max_seconds=90)
        if r != "arrived" and not camp._step_to(tile):
            return False
        return tuple(tv.coords(b) or ()) == tile

    def engage(front, face, label, expect_battle):
        """Stand at `front`, face `face`, press A; run the battle if one starts; drain text.
        Returns 'battled' | 'talked' | 'nothing'."""
        if not goto(front, label):
            L(f"!! [{label}] couldn't reach {front} (at {tv.coords(b)})")
            return "nothing"
        out = "nothing"
        for _ in range(8):
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if st.in_battle(b):
                L(f"   [{label}] battle -> {fight()}")
                drain()
                return "battled"
            if dd_box(b):
                out = "talked"
                drain()
                if st.in_battle(b):               # the talk escalated into the fight
                    L(f"   [{label}] battle -> {fight()}")
                    drain()
                    return "battled"
                break
        return out

    # ── SPIN-TILE SLIDE CROSSER (B3F class; pret metatile_behaviors.h ground truth) ────────────
    # 0x54-0x57 = MB_SPIN_RIGHT/LEFT/UP/DOWN (forced glide, redirected by each spinner crossed),
    # 0x58 = MB_STOP_SPINNING (rest dot). Travel's BFS treats spinners as plain floor, so its
    # plans diverge the instant she touches one (the position-loop). Here: simulate every glide
    # deterministically, BFS over REST tiles with glide edges, execute press-by-press.
    SPIN = {0x54: (1, 0), 0x55: (-1, 0), 0x56: (0, -1), 0x57: (0, 1)}

    def spin_cross(target_pred, label, rounds=3, node_ok=None):
        for _round in range(rounds):
            if _spin_cross_once(target_pred, f"{label}#{_round + 1}", node_ok=node_ok):
                return True
            if tuple(tv.coords(b) or ()) and target_pred(tuple(tv.coords(b))):
                return True
        return False

    def _spin_cross_once(target_pred, label, max_hops=60, node_ok=None):
        g = tv.Grid(b)
        npc = set(camp.trav._npc_tiles())      # live NPCs block glides (a grunt IS a wall here)
        # playable dims (BACKUP_LAYOUT includes the +14 border) — Grid reads WRAP at the edges,
        # which planned a LEFT glide off x=0 (hideout7)
        w_play = b.rd32(tv.BACKUP_LAYOUT) - 14
        h_play = b.rd32(tv.BACKUP_LAYOUT + 4) - 14

        def in_bounds(x, y):
            return 0 <= x < w_play and 0 <= y < h_play

        def bh(t):
            return camp._tile_behavior(*t)

        def glide(frm, d):
            """One press from `frm` moving `d`. Mechanics (hideout3 ground truth): stepping onto
            a spinner starts a SLIDE; while sliding, spinners redirect, 0x58 stop-dots stop, and
            PLAIN floor does NOT stop (momentum carries) — only a wall or a dot ends it. A press
            while standing ON a spinner resumes that spinner's own direction."""
            x, y = frm
            dx, dy = d
            sliding = False
            v0 = bh(frm)
            if v0 in SPIN:
                dx, dy = SPIN[v0]
                sliding = True
            for _ in range(300):
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny) or not g.walkable(nx, ny) or (nx, ny) in npc:
                    return (x, y) if (x, y) != frm else None
                x, y = nx, ny
                v = bh((x, y))
                if v in SPIN:
                    dx, dy = SPIN[v]
                    sliding = True
                    continue
                if v == 0x58 or not sliding:
                    return (x, y)
                # plain floor mid-slide: momentum carries — keep going
            return (x, y)

        last = tuple(tv.coords(b))         # settle any in-flight slide before planning
        still = 0
        for _ in range(300):
            b.run_frame()
            cur = tuple(tv.coords(b))
            if cur == last:
                still += 1
                if still >= 40:
                    break
            else:
                last, still = cur, 0
        start = tuple(tv.coords(b))
        from collections import deque
        prev = {start: None}
        q = deque([start])
        goal = None
        while q:
            cur = q.popleft()
            if target_pred(cur):
                goal = cur
                break
            for key, d in (("RIGHT", (1, 0)), ("LEFT", (-1, 0)), ("UP", (0, -1)), ("DOWN", (0, 1))):
                dst = glide(cur, d)
                if dst and dst not in prev and (node_ok is None or node_ok(dst)):
                    prev[dst] = (cur, key)
                    q.append(dst)
        if goal is None:
            L(f"!! [{label}] spin-BFS found no route from {start}")
            return False
        plan = []
        n = goal
        while prev[n] is not None:
            p, k = prev[n]
            plan.append((k, n))
            n = p
        plan.reverse()
        L(f"   [{label}] spin plan: {plan}")
        for key, expect in plan[:max_hops]:
            ok = False
            for _ in range(4):                     # first press can be an eaten TURN
                c0 = tuple(tv.coords(b))
                b.press(key, 8, 10, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
                # wait for the glide to SETTLE (coords stable across a beat)
                last = tuple(tv.coords(b))
                still = 0
                for _ in range(240):
                    b.run_frame()
                    cur = tuple(tv.coords(b))
                    if cur == last:
                        still += 1
                        if still >= 30:
                            break
                    else:
                        last, still = cur, 0
                if st.in_battle(b):                # LoS trainer mid-route: fight, then REPLAN
                    L(f"   [{label}] battle mid-cross -> {fight()}")
                    drain()
                    return False
                if last == expect:
                    ok = True
                    break
                if last != c0:                     # moved but NOT where simulated — replan
                    L(f"!! [{label}] glide diverged: pressed {key} at {c0}, expected {expect}, got {last}")
                    return False
            if not ok:
                L(f"!! [{label}] press {key} never moved off {tuple(tv.coords(b))} — replanning")
                return False
        return True

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} key_items={key_items()}")
    if SILPH_SCOPE in key_items():
        L("already holding the Silph Scope — nothing to do")
        return 0
    if tuple(tv.map_id(b)) != CELADON:
        L("!! not in Celadon — this strike is Celadon-anchored; abort")
        return 1

    # 1. into the Game Corner
    if camp.enter_warp(pick=GC_DOOR) != "warped":
        L("!! couldn't enter the Game Corner — abort")
        return 1
    for _ in range(80):
        b.run_frame()
    L(f"   inside: map={tv.map_id(b)} coords={tv.coords(b)} npcs={sorted(camp.trav._npc_tiles())}")

    # 2. the grunt guarding the poster
    r = engage(GRUNT_FRONT, "UP", "poster-grunt", expect_battle=True)
    L(f"   grunt engage -> {r}")
    snap("10_post_grunt")
    # 3. the poster (the grunt walks off after losing; his old tile is the poster front)
    for attempt in range(3):
        r2 = engage(POSTER_FRONT, "UP", "poster", expect_battle=False)
        L(f"   poster press -> {r2}")
        drain()
        if r2 in ("talked",):
            break
        for _ in range(60):
            b.run_frame()
    snap("20_post_poster")

    # 4. the revealed stairs -> B1F
    m0 = tuple(tv.map_id(b))
    if camp.enter_warp(pick=GC_STAIRS) != "warped" or tuple(tv.map_id(b)) == m0:
        snap("30_stairs_fail")
        L(f"!! stairs {GC_STAIRS} didn't warp (poster gate still shut?) — abort "
          f"(frame -> {DBG}/30_stairs_fail.png)")
        return 1
    for _ in range(80):
        b.run_frame()
    L(f"   B1F: map={tv.map_id(b)} coords={tv.coords(b)}")

    # 5. descend B1F -> B4F on the billed stairs
    for i, stairs in enumerate(DESCENT, 1):
        m0 = tuple(tv.map_id(b))
        r = camp.enter_warp(pick=stairs)
        if r == "need_heal":
            L("   heal interrupt mid-floor — healing")
            camp.heal_nearest()
            r = camp.enter_warp(pick=stairs)
        if r != "warped" or tuple(tv.map_id(b)) == m0:
            L(f"   floor {i}: plain route to {stairs} failed — engaging the spin-tile crosser")
            if not spin_cross(lambda t: abs(t[0] - stairs[0]) + abs(t[1] - stairs[1]) <= 1,
                              f"floor{i}-spin"):
                snap(f"40_floor{i}_fail")
                L(f"!! floor {i}: spin crossing failed from {tv.coords(b)} — abort "
                  f"(frame -> {DBG}/40_floor{i}_fail.png)")
                return 1
            r = camp.enter_warp(pick=stairs)
        if r != "warped" or tuple(tv.map_id(b)) == m0:
            snap(f"40_floor{i}_fail")
            L(f"!! floor {i}: stairs {stairs} didn't warp from {m0}@{tv.coords(b)} — abort "
              f"(frame -> {DBG}/40_floor{i}_fail.png)")
            return 1
        for _ in range(80):
            b.run_frame()
        L(f"   floor {i + 1} down: map={tv.map_id(b)} coords={tv.coords(b)}")

    # 6. LIFT KEY (grunt (4,2), ball (3,2)) — the boss corridor is walled off on foot;
    #    the ONLY way in is the elevator (disasm B4F: elevator door (20-21,23) sits inside it)
    snap("50_b4f")
    LIFT_KEY = 356
    if LIFT_KEY not in key_items():
        rg = engage((4, 3), "UP", "liftkey-grunt", expect_battle=True)
        L(f"   lift-key grunt -> {rg}")
        for front, face in (((3, 3), "UP"), ((4, 2), "LEFT"), ((2, 2), "RIGHT")):
            engage(front, face, "liftkey-ball", expect_battle=False)
            if LIFT_KEY in key_items():
                break
        if LIFT_KEY not in key_items():
            snap("55_liftkey_fail")
            L(f"!! no Lift Key (key_items={key_items()}) — abort")
            return 1
    L(f"   LIFT KEY in bag: {key_items()}")

    # 7. back up to B2F's elevator: B4F stairs (11,15) -> B3F, spin-cross to up-stairs (18,2)
    #    -> B2F (arrive 21,2), spin-cross to elevator door (28-29,16)
    for stairs_up, label in (((11, 15), "B4F->B3F"), ((18, 2), "B3F->B2F")):
        m0 = tuple(tv.map_id(b))
        r = camp.enter_warp(pick=stairs_up)
        if r != "warped" or tuple(tv.map_id(b)) == m0:
            if not spin_cross(lambda t: abs(t[0] - stairs_up[0]) + abs(t[1] - stairs_up[1]) <= 1,
                              f"{label}-spin"):
                snap(f"56_{label.replace('>', '')}_fail")
                L(f"!! {label}: couldn't reach up-stairs {stairs_up} — abort")
                return 1
            r = camp.enter_warp(pick=stairs_up)
        if r != "warped" or tuple(tv.map_id(b)) == m0:
            L(f"!! {label}: stairs {stairs_up} didn't warp — abort")
            return 1
        for _ in range(80):
            b.run_frame()
        L(f"   {label}: map={tv.map_id(b)} coords={tv.coords(b)}")

    def board_elevator(door_pick):
        m0 = tuple(tv.map_id(b))
        r = camp.enter_warp(pick=door_pick)
        if r != "warped" or tuple(tv.map_id(b)) == m0:
            # bias EAST: the west half of B2F is a one-way spinner pocket (hideout7/8)
            if not spin_cross(lambda t: abs(t[0] - door_pick[0]) + abs(t[1] - door_pick[1]) <= 1,
                              "to-elevator", node_ok=lambda t: t[0] >= 12):
                return False
            r = camp.enter_warp(pick=door_pick)
        for _ in range(80):
            b.run_frame()
        return tuple(tv.map_id(b)) != m0

    if not board_elevator((28, 16)) and not board_elevator((29, 16)):
        snap("57_no_elevator")
        L("!! couldn't board the B2F elevator — abort")
        return 1
    L(f"   in the elevator: map={tv.map_id(b)} coords={tv.coords(b)}")

    # 8. ride to B4F: panel = bg event (0,2) (face from (1,2) LEFT or (0,3) UP); the floor
    #    multichoice rows aren't billed — self-correct on the LANDING (B4F = arrive (20-21,23))
    def ride(downs):
        opened = False
        for front, face in (((1, 2), "LEFT"), ((0, 3), "UP"), ((0, 2), "UP")):
            if not goto(front, "panel"):
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
        b.press("A", 8, 12, camp.render, owner="agent")     # advance "Which floor?" -> menu
        for _ in range(30):
            b.run_frame()
        for _ in range(downs):
            b.press("DOWN", 8, 10, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
        b.press("A", 8, 12, camp.render, owner="agent")
        drain()
        for _ in range(300):                                # the ride/shake
            b.run_frame()
        m0 = tuple(tv.map_id(b))
        camp.enter_warp(prefer="nearest")
        for _ in range(80):
            b.run_frame()
        return tuple(tv.map_id(b)) != m0

    landed_b4f = False
    for downs in (2, 1, 0, 3):
        if not ride(downs):
            L(f"   ride(downs={downs}): no exit — retrying")
            continue
        L(f"   ride(downs={downs}) landed: map={tv.map_id(b)} coords={tv.coords(b)}")
        cx, cy = tuple(tv.coords(b) or (0, 0))
        if abs(cy - 23) <= 3 and 18 <= cx <= 23:            # B4F elevator alcove
            landed_b4f = True
            break
        camp.enter_warp(prefer="nearest")                   # wrong floor: re-board (nearest door
        for _ in range(80):                                 #  IS the elevator we just left)
            b.run_frame()
    if not landed_b4f:
        snap("58_wrong_floor")
        L(f"!! elevator never landed in the B4F corridor (at {tv.map_id(b)}@{tv.coords(b)}) — abort")
        return 1

    # 9. the boss-door grunts, then GIOVANNI
    snap("59_corridor")
    for gx, gy in ((19, 14), (16, 14)):
        rg = engage((gx, gy + 1), "UP", f"door-grunt{gx}", expect_battle=True)
        L(f"   door grunt ({gx},{gy}) -> {rg}")
    r = engage(GIOVANNI_FRONT, "UP", "giovanni", expect_battle=True)
    L(f"   giovanni engage -> {r}")
    drain()
    snap("60_post_giovanni")

    # 7. the Silph Scope (his parting gift — ball at (20,5), or the beat-script hands it over)
    if SILPH_SCOPE not in key_items():
        for front, face in (((20, 6), "UP"), ((19, 5), "RIGHT"), ((20, 5), "UP")):
            engage(front, face, "scope-ball", expect_battle=False)
            if SILPH_SCOPE in key_items():
                break
    got = SILPH_SCOPE in key_items()
    L(f"   SILPH SCOPE in Key Items: {got} (key_items={key_items()})")
    snap("70_final")
    if not got:
        L("!! scope NOT in bag — NOT banking; read the frames")
        return 1

    # 8. bank (mid-dungeon is fine — resume-safe savestate; the longrun exits via GO-DEEPER/travel)
    _stage_save("scope")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} silph_scope")
    return 0


if __name__ == "__main__":
    sys.exit(main())
