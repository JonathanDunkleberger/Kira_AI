"""recon_silph.py — THE SILPH CO STRIKE (night shift #8): free Saffron, unblock Sabrina's gym.

sabrina_run7 truth: she reached Saffron through the TEA gate and STALLED at the gym door —
RocketGrunt3 (46,13) blocks it until FLAG_HIDE_SAFFRON_ROCKETS (0x3E) is set, which ONLY
Giovanni's defeat on Silph 11F sets (SilphCo_11F scripts.inc). Scripted spec, disasm ground
truth (pret, fetched 2026-07-07):
  Maps: SilphCo_1F..11F = (1,46)..(1,56); elevator (1,57). Street door on Saffron (33,30);
  the door guard (34,31) is ASLEEP once FLAG_RESCUED_MR_FUJI (0x23C) is set — she has it.
  Stairs: each floor's NW pair links floor±1 — enter_to(dest) needs no stair coords.
  5F (1,50): CARD KEY item ball at (22,21) (ITEM 355; pickup flag 0x192). Doors all over the
  tower are card-locked BG 'sign' events ON the barrier tiles; with the key-pickup flag set,
  an A-press opens them for good (silphco_doors.inc; per-door flags 0x27C..0x28D).
  3F (1,48): doors x9-10/x20-21 rows 11-13 flank the middle room; its PAD (13,14) -> 7F (5,4).
  7F (1,52): GARY triggers at (2,4)/(2,5) (auto rival battle; his ace counters her starter);
  LAPRAS guy at (0,7) (free L25, flag 0x246; party full -> PC); PAD (5,8) -> 11F (2,5);
  doors (11-12,8-10)/(24-25,7-9)/(25-26,13-15).
  11F (1,56): GIOVANNI at (6,11) — beating him sets FLAG_HIDE_SAFFRON_ROCKETS 0x3E and clears
  the tower; PRESIDENT (9,9) then hands the MASTER BALL; door (5-6,16-18); pad (2,5) -> 7F.
Success = flag 0x3E. Exit via reverse pads + stairs -> street -> heal -> bank.
Canonical protection: staging pattern; bank -> %TEMP%/longrun/banked_SILPH for promote_bank.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_silph.py
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
import field_moves as fm             # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_silph")
BANK = os.path.join(SCRATCH, "banked_SILPH")
DBG = os.path.join(SCRATCH, "silph_probe")

SAFFRON = (3, 10)
SILPH = [(1, 47 + i) for i in range(11)]          # [0]=1F .. [10]=11F (LIVE: (33,30)->(1,47))
F1, F3, F5, F7, F11 = SILPH[0], SILPH[2], SILPH[4], SILPH[6], SILPH[10]
CARD_KEY_ITEM = 355
FLAG_SAFFRON_FREE = 0x3E                          # FLAG_HIDE_SAFFRON_ROCKETS — the strike's win
FLAG_CARD_KEY = 0x192                             # 5F ball picked up (doors check THIS)
CARD_BALL_5F = (22, 21)
DOORS_3F = [(20, 12), (20, 13), (21, 12), (21, 13),   # Door2 (east, nearer the stairs) first
            (9, 12), (9, 13), (10, 12), (10, 13)]     # Door1 (west)
DOORS_7F = [(11, 9), (12, 9), (11, 8), (12, 8), (11, 10), (12, 10)]
DOORS_11F = [(5, 17), (6, 17), (5, 16), (6, 16), (5, 18), (6, 18)]
GARY_TRIGGER_7F = (2, 5)
LAPRAS_FRONT_7F = (0, 8)                          # guy at (0,7), face UP
PAD_3F_TO_7F = (13, 14)
PAD_7F_TO_11F = (5, 8)
GIOVANNI_FRONT = (6, 12)                          # Giovanni (6,11), face UP
PRESIDENT_FRONT = (9, 10)                         # president (9,9), face UP


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

    def saffron_free():
        return fm.read_flag(b, FLAG_SAFFRON_FREE)

    def have_key():
        return CARD_KEY_ITEM in key_items() or fm.read_flag(b, FLAG_CARD_KEY)

    def drain(max_a=40, key="A"):
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
        r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=250, max_seconds=120)
        if st.in_battle(b):
            L(f"   [{label}] battle en route -> {camp.battle_runner()}")
            drain()
            r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=250, max_seconds=120)
        if r != "arrived" and not camp._step_to(tile):
            return False
        return tuple(tv.coords(b) or ()) == tile

    def engage(front, face, label, drains=1):
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
                L(f"   [{label}] battle -> {camp.battle_runner()}")
                drain()
                return "battled"
            if dd_box(b):
                out = "talked"
                for _k in range(drains):
                    drain()
                    if st.in_battle(b):
                        L(f"   [{label}] battle -> {camp.battle_runner()}")
                        drain()
                        return "battled"
                    for _ in range(40):
                        b.run_frame()
                break
        return out

    def enter_to(dest, label):
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == dest]
        if not cands:
            L(f"!! [{label}] no warp on {m0} leads to {dest}")
            return False
        cur = tuple(tv.coords(b) or (0, 0))
        cands.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        for wt in cands:
            r = camp.enter_warp(pick=wt)
            if r == "need_heal":
                L(f"   [{label}] heal interrupt — healing, then retrying")
                camp.heal_nearest()
                r = camp.enter_warp(pick=wt)
            if st.in_battle(b):
                L(f"   [{label}] battle on approach -> {camp.battle_runner()}")
                drain()
                r = camp.enter_warp(pick=wt)
            if tuple(tv.map_id(b)) == dest:
                for _ in range(80):
                    b.run_frame()
                L(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)}")
                return True
        L(f"!! [{label}] no candidate warp fired (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    def open_doors(tiles, label):
        """A-press the card-locked door barrier tiles (BG 'sign' events sit ON them). With the
        key-pickup flag set the script opens the door permanently. Stand at any reachable
        orthogonal neighbor, face the tile, press. Success = a dialogue fired (fanfare box) or
        the tile became walkable."""
        if not have_key():
            L(f"!! [{label}] no Card Key yet — doors won't open")
            return False
        opened = False
        grid = tv.Grid(b)
        cur = tuple(tv.coords(b) or (0, 0))
        for dt in tiles:
            for (nb, face) in (((dt[0] + 1, dt[1]), "LEFT"), ((dt[0] - 1, dt[1]), "RIGHT"),
                               ((dt[0], dt[1] + 1), "UP"), ((dt[0], dt[1] - 1), "DOWN")):
                if not tv.bfs(grid, cur, lambda t, a=nb: t == a, walkable=grid.walkable):
                    continue
                if not goto(nb, f"{label}-stand"):
                    continue
                b.press(face, 8, 10, camp.render, owner="agent")
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(40):
                    b.run_frame()
                if dd_box(b):
                    drain()
                    L(f"   [{label}] door script fired at {dt}")
                    opened = True
                    grid = tv.Grid(b)                     # metatiles changed — re-read
                    cur = tuple(tv.coords(b) or (0, 0))
                    break
            if opened:
                break
        return opened

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} key_items={key_items()} "
      f"saffron_free={saffron_free()} card_key={have_key()} fuji={fm.read_flag(b, 0x23C)}")
    if saffron_free():
        L("Saffron already free — nothing to strike")
        return 0

    # ── 1. street → Silph 1F (the door guard is asleep post-Fuji; the door itself is open)
    here = tuple(tv.map_id(b))
    if here == SAFFRON:
        if not enter_to(F1, "silph-door"):
            snap("10_no_silph")
            return 1
    if tuple(tv.map_id(b)) not in SILPH:
        L(f"!! not in Silph (at {tv.map_id(b)}) — abort")
        return 1

    # ── 2. THE CLIMB + ERRANDS — a state machine on the current floor. Stairs pair adjacent
    #      floors, so enter_to(next) walks the tower without one hardcoded stair coord. Heal
    #      bounces re-dispatch (beaten trainers stay beaten). Milestones ratchet via flags.
    def lead_frac():
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    gary_done = [False]
    lapras_done = [False]
    wedges = {}
    deadline = time.time() + 1800
    while time.time() < deadline and not saffron_free():
        here = tuple(tv.map_id(b))
        if lead_frac() < 0.5:
            L(f"   lead at {lead_frac():.0%} — descending to heal (from {here})")
            if here in SILPH and here != F1:
                enter_to(SILPH[SILPH.index(here) - 1], "heal-descent")
            elif here == F1:
                enter_to(SAFFRON, "silph-exit")
            elif here == SAFFRON:
                camp.heal_nearest()
            else:
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
            continue
        if here == SAFFRON:
            if not enter_to(F1, "silph-reenter"):
                snap("25_reenter_fail")
                return 1
            continue
        if here not in SILPH:
            L(f"   off-route at {here} (heal interior?) — exiting to the overworld")
            camp.enter_warp(prefer="south")
            for _ in range(80):
                b.run_frame()
            if tuple(tv.map_id(b)) == here:
                L(f"!! stuck off-route at {here}@{tv.coords(b)} — abort")
                snap("26_offroute")
                return 1
            continue
        idx = SILPH.index(here)

        # standing ON the arrival stair re-fires it mid-pathfind (strike2: she silently dropped
        # back to 4F and A-pressed empty tiles) — step off to a non-warp neighbor first
        cur = tuple(tv.coords(b) or (0, 0))
        warp_tiles = {tuple(w[0]) for w in tv.read_warps(b)}
        if cur in warp_tiles:
            g5 = tv.Grid(b)
            for nb in ((cur[0], cur[1] + 1), (cur[0] + 1, cur[1]),
                       (cur[0] - 1, cur[1]), (cur[0], cur[1] - 1)):
                if nb not in warp_tiles and g5.walkable(nb[0] + tv.MAP_OFFSET, nb[1] + tv.MAP_OFFSET):
                    camp._step_to(nb)
                    break

        # PHASE A — no Card Key yet: climb the stairs to 5F and take the ball.
        if not have_key():
            if here == F5:
                for face, front in (("RIGHT", (CARD_BALL_5F[0] - 1, CARD_BALL_5F[1])),
                                    ("LEFT", (CARD_BALL_5F[0] + 1, CARD_BALL_5F[1])),
                                    ("DOWN", (CARD_BALL_5F[0], CARD_BALL_5F[1] - 1)),
                                    ("UP", (CARD_BALL_5F[0], CARD_BALL_5F[1] + 1))):
                    if tuple(tv.map_id(b)) != F5:      # an accidental stair re-fire mid-approach
                        enter_to(F5, "back-to-5f")
                        continue
                    engage(front, face, "card-key-ball")
                    if have_key():
                        break
                L(f"   CARD KEY: item={CARD_KEY_ITEM in key_items()} "
                  f"flag={fm.read_flag(b, FLAG_CARD_KEY)}")
                if not have_key():
                    if tuple(tv.map_id(b)) != F5:
                        continue                       # bounced floors — re-dispatch, don't abort
                    snap("30_no_key")
                    L("!! Card Key not obtained on 5F — abort LOUD")
                    return 1
                _stage_save("card_key")
                continue
            nxt = SILPH[idx + 1] if idx + 1 < len(SILPH) else None
            if nxt and not enter_to(nxt, f"floor{idx + 2}-up"):
                if tuple(tv.map_id(b)) == here:
                    wedges[here] = wedges.get(here, 0) + 1
                    if wedges[here] >= 3:
                        snap(f"31_climb_wedge_{idx + 1}F")
                        L(f"!! climb wedged x3 on {idx + 1}F — abort")
                        return 1
                    drain()
            else:
                wedges.pop(here, None)
            continue

        # PHASE B — key in hand: 3F pad -> 7F (Gary + Lapras) -> pad -> 11F (Giovanni).
        if here == F7:
            if not gary_done[0]:
                L("   7F: walking the rival trigger row (Gary auto-engages)")
                goto(GARY_TRIGGER_7F, "gary-trigger")
                if st.in_battle(b):
                    L(f"   GARY -> {camp.battle_runner()}")
                    drain()
                else:
                    drain()
                gary_done[0] = True
                _stage_save("post_gary")
                snap("40_post_gary")
                continue
            if not lapras_done[0]:
                r = engage(LAPRAS_FRONT_7F, "UP", "lapras", drains=6)
                drain(key="B")                       # nickname/PC boxes resolve on B
                lapras_done[0] = True
                L(f"   LAPRAS guy -> {r} (flag={fm.read_flag(b, 0x246)})")
                _stage_save("post_lapras")
                continue
            if not enter_to(F11, "pad-11f"):
                open_doors(DOORS_7F, "7f-doors")
                if not enter_to(F11, "pad-11f-2"):
                    wedges[here] = wedges.get(here, 0) + 1
                    if wedges[here] >= 3:
                        snap("50_no_11f")
                        L("!! can't reach the 11F pad — abort")
                        return 1
                    drain()
            continue
        if here == F11:
            open_doors(DOORS_11F, "11f-door")        # the barrier before Giovanni's room
            r = engage(GIOVANNI_FRONT, "UP", "giovanni", drains=3)
            L(f"   GIOVANNI -> {r}; saffron_free={saffron_free()}")
            drain()
            for _ in range(240):                     # his removeobject cutscene
                b.run_frame()
                if dd_box(b):
                    b.press("A", 8, 12, camp.render, owner="agent")
            if saffron_free():
                _stage_save("giovanni_down")
                snap("60_giovanni_down")
                r2 = engage(PRESIDENT_FRONT, "UP", "president", drains=8)
                drain()
                L(f"   PRESIDENT -> {r2} (master ball flag={fm.read_flag(b, 0x250)})")
                break
            wedges[here] = wedges.get(here, 0) + 1
            if wedges[here] >= 3:
                snap("55_giovanni_wedge")
                L("!! Giovanni not falling / not reachable x3 — abort")
                return 1
            continue
        if here == F3:
            if not enter_to(F7, "pad-7f"):
                open_doors(DOORS_3F, "3f-doors")
                if not enter_to(F7, "pad-7f-2"):
                    wedges[here] = wedges.get(here, 0) + 1
                    if wedges[here] >= 3:
                        snap("35_no_7f")
                        L("!! can't reach the 7F pad from 3F — abort")
                        return 1
                    drain()
            continue
        # any other floor with the key: route to 3F (stairs pair adjacent floors)
        step = -1 if idx > 2 else 1
        if not enter_to(SILPH[idx + step], f"to3f-via{idx + 1 + step}F"):
            wedges[here] = wedges.get(here, 0) + 1
            if wedges[here] >= 3:
                snap(f"36_route3f_wedge_{idx + 1}F")
                L(f"!! routing to 3F wedged on {idx + 1}F — abort")
                return 1
            drain()
        else:
            wedges.pop(here, None)

    if not saffron_free():
        L(f"!! Saffron NOT freed (at {tv.map_id(b)}@{tv.coords(b)}) — NOT banking")
        snap("70_fail")
        return 1

    # ── 3. walk out (reverse pads + stairs), heal, bank
    L("   walking out of the tower")
    out_deadline = time.time() + 420
    while tuple(tv.map_id(b)) != SAFFRON and time.time() < out_deadline:
        here = tuple(tv.map_id(b))
        if here == F11:
            enter_to(F7, "out-7f")
        elif here == F7:
            enter_to(F3, "out-3f")
        elif here in SILPH:
            i = SILPH.index(here)
            if i == 0:
                enter_to(SAFFRON, "out-street")
            else:
                enter_to(SILPH[i - 1], "out-down")
        else:
            camp.enter_warp(prefer="south")
            for _ in range(80):
                b.run_frame()
    if tuple(tv.map_id(b)) == SAFFRON:
        camp.heal_nearest()
    else:
        L(f"!! walk-out incomplete (at {tv.map_id(b)}) — banking anyway (flag holds; the "
          f"longrun's recovery owns the exit)")

    L(f"   SAFFRON FREE: flag={saffron_free()} | key_items={key_items()} | "
      f"pos {tv.map_id(b)}@{tv.coords(b)}")
    snap("80_final")
    _stage_save("silph_cleared")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} silph_cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
