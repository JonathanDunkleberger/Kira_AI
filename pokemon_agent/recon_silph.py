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
F1, F3, F5, F7, F9, F11 = SILPH[0], SILPH[2], SILPH[4], SILPH[6], SILPH[8], SILPH[10]
CARD_KEY_ITEM = 355
FLAG_SAFFRON_FREE = 0x3E                          # FLAG_HIDE_SAFFRON_ROCKETS — the strike's win
FLAG_CARD_KEY = 0x192                             # 5F ball picked up (doors check THIS)
CARD_BALL_5F = (22, 21)
# ROUTE TRUTH (pret map.json 3F/7F/9F/11F, cross-checked vs strike12 live-learned warps):
# 7F's Gary pocket (triggers (2,4)/(2,5), Lapras (0,7), 11F pad (5,8)) is PAD-ONLY — no card
# door reaches x<=5; entrances are the 3F pad landing (5,4) and the 11F pad. The pad chain:
# 9F (9,4) <-> 3F (2,14), 3F (13,14) <-> 7F (5,4), 7F (5,8) <-> 11F (2,5). And 9F holds a
# FREE FULL HEAL: the hostage woman (2,16), no hide flag (SilphCo_9F_EventScript_HealWoman).
DOORS_3F_WEST = [(9, 12), (10, 12), (9, 13), (10, 13)]     # flanks the pad room, west side
DOORS_3F_EAST = [(20, 12), (21, 12), (20, 13), (21, 13)]   # east side (stairs approach)
DOORS_9F_WEST = [(2, 10), (3, 10), (2, 11), (3, 11)]       # west corridor -> the heal woman
DOORS_11F = [(5, 16), (6, 16), (5, 17), (6, 17)]           # south entrance (unused from the pad)
GARY_TRIGGER_7F = (2, 5)
LAPRAS_FRONT_7F = (0, 8)                          # guy at (0,7), face UP
LAPRAS_FLAG = 0x246
PAD_3F_TO_7F = (13, 14)
PAD_7F_TO_11F = (5, 8)
PAD_9F_TO_3F = (9, 4)
PAD_3F_TO_9F = (2, 14)
PAD_7F_TO_3F = (5, 4)                             # the pocket's exit back down
PAD_11F_TO_7F = (2, 5)
HEAL_9F_FRONT = (2, 17)                           # hostage woman (2,16), face UP
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

    n_battles = [0]

    def fight():
        n_battles[0] += 1
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

    def walk_path_to(tile, label, tries=6):
        """Deterministic same-map mover (the strike pattern): static BFS with WARPS and
        template-NPC BODIES masked (strike6 truth: the shortest 5F path threads the beaten
        hypno's tile (35,7) — grid BFS is NPC-blind, and travel TTL-thrashes between two
        NPC-sealed plans forever), then step it tile-by-tile. Battles/wanderers recompute;
        a step that fails OUTSIDE battle is a collision-walkable-but-game-blocked tile
        (strike9: the 5F stair alcove (28,3) — elevation quirk) → DEAD for this call."""
        dead = set()
        for _ in range(tries):
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile:
                return True
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
            npcs = ({tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
                    | dead) - {tile}
            p = tv.bfs(g, cur, lambda t: t == tile,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs)
            if not p:
                L(f"   [{label}] no NPC-free static path {cur} -> {tile} "
                  f"(dead={sorted(dead)})")
                return False
            for t in p[1:]:
                ok = camp._step_to(tuple(t))
                # battle check BEFORE judging the step: a trainer's SIGHT LINE fires mid-walk
                # and freezes coords — _step_to then reads as failed with the fight unfought
                # (strike8: Grunt2 saw her on row 6, walk_path_to burned its tries in-battle).
                if st.in_battle(b):
                    L(f"   [{label}] battle mid-path -> {camp.battle_runner()}")
                    drain()
                    break
                if not ok:
                    # an engagement CUTSCENE (trainer "!" + walk-up) freezes inputs BEFORE
                    # in_battle reads true (strike10: (36,6) beside Grunt2 got dead-marked
                    # mid-spotting) — wait it out before judging the tile static-blocked.
                    # A COORD-TRIGGER scene (Gary's 7F walk-up) opens DIALOGUE first, battle
                    # after — drain the box before the battle check, never dead-mark it.
                    for _ in range(120):
                        b.run_frame()
                    if dd_box(b):
                        drain()
                    if st.in_battle(b):
                        L(f"   [{label}] step was a trainer spotting -> "
                          f"{camp.battle_runner()}")
                        drain()
                        break
                    dead.add(tuple(t))
                    L(f"   [{label}] step into {tuple(t)} failed — dead-marked, recompute")
                    break
            if tuple(tv.coords(b) or ()) == tile:
                return True
        return tuple(tv.coords(b) or ()) == tile

    def goto(tile, label):
        if walk_path_to(tile, label):
            return True
        # fallback: travel's NPC-aware walker, warp tiles masked (the Mt Moon avoid=
        # mechanism — strike5: the ball approach BFS'd across the 5F pad, rode it to 9F).
        av = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
        r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=250, max_seconds=120,
                             avoid=av)
        if st.in_battle(b):
            L(f"   [{label}] battle en route -> {camp.battle_runner()}")
            drain()
            r = camp.trav.travel(target_map=None, arrive_coord=tile, max_steps=250,
                                 max_seconds=120, avoid=av)
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
        if cur in cands:
            # standing ON the pad/warp we need — it won't re-fire underfoot (strike5's
            # back-to-5f wedge on the 9F return pad). Step off to a safe neighbor first;
            # enter_warp then walks back on and it fires on contact.
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            g6 = tv.Grid(b)
            for nb in ((cur[0], cur[1] + 1), (cur[0] + 1, cur[1]),
                       (cur[0] - 1, cur[1]), (cur[0], cur[1] - 1)):
                if nb not in wts and g6.walkable(nb[0], nb[1]):
                    camp._step_to(nb)
                    cur = tuple(tv.coords(b) or (0, 0))
                    break
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

    def ride_pad(pad, label):
        """Deliberately ride a teleport pad. Pad mechanics (strike11 truth): fires on STEP-ON
        contact; landing on its pair does NOT fire; but a tap EATEN AS A TURN while standing
        on one counts as a movement ending on the tile and RE-fires it. So: approach a free
        neighbor, then one LONG-HOLD press onto the pad (turn+walk in one continuous input).
        The loop-top guard long-hold-steps off the landing pad next dispatch."""
        m0 = tuple(tv.map_id(b))
        g = tv.Grid(b)
        wts = {tuple(w[0]) for w in tv.read_warps(b)}
        npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
        cur0 = tuple(tv.coords(b) or (0, 0))
        # order neighbors by REAL path length from here — the fixed R/L/D/U order sent
        # strike13 on a 40-step trap route to the far-side neighbor when the near one
        # was 11 steps away (the 5F pocket exit wander).
        cands = []
        for nb, kin in (((pad[0] - 1, pad[1]), "RIGHT"), ((pad[0] + 1, pad[1]), "LEFT"),
                        ((pad[0], pad[1] - 1), "DOWN"), ((pad[0], pad[1] + 1), "UP")):
            if nb in wts or not g.walkable(nb[0], nb[1]):
                continue
            p = tv.bfs(g, cur0, lambda t, a=nb: t == a,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs) \
                if cur0 != nb else [cur0]
            if p:
                cands.append((len(p), nb, kin))
        for _len, nb, kin in sorted(cands):
            if tuple(tv.coords(b) or ()) != nb and not walk_path_to(nb, f"{label}-approach"):
                continue
            for _try in range(3):
                b.press(kin, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    for _ in range(60):
                        b.run_frame()
                    L(f"   [{label}] rode pad {pad}: {m0} -> {tuple(tv.map_id(b))} "
                      f"@ {tv.coords(b)}")
                    return True
            break
        L(f"!! [{label}] pad ride {pad} failed (at {tv.map_id(b)}@{tv.coords(b)})")
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
    heal_mode = [False]
    wedges = {}

    def heal_9f():
        """The 9F hostage (2,16) fully heals the party — free, no flag, stays post-strike
        (disasm SilphCo_9F_EventScript_HealWoman). Kills the tower-descent heal entirely."""
        open_doors(DOORS_9F_WEST, "9f-heal-door")     # west corridor door; no-op if open
        r = engage(HEAL_9F_FRONT, "UP", "9f-heal", drains=4)
        for _ in range(240):                          # the heal fade/jingle
            b.run_frame()
        drain()
        L(f"   9F hostage heal -> {r}; lead {lead_frac():.0%}")
        return lead_frac() > 0.9

    deadline = time.time() + 1800
    while time.time() < deadline and not saffron_free():
        here = tuple(tv.map_id(b))
        if lead_frac() < 0.5:
            heal_mode[0] = True
        if heal_mode[0] and lead_frac() > 0.9:
            heal_mode[0] = False
        if heal_mode[0]:
            cur = tuple(tv.coords(b) or (0, 0))
            if have_key() and here in SILPH:
                # route to the 9F hostage over the pad chain — never leave the tower
                if here == F9:
                    heal_9f()
                elif here == F3:
                    open_doors(DOORS_3F_WEST if cur[0] < 15 else DOORS_3F_EAST, "heal-3f-door")
                    ride_pad(PAD_3F_TO_9F, "heal-to-9f") or drain()
                elif here == F7 and cur[0] <= 6:
                    ride_pad(PAD_7F_TO_3F, "heal-to-3f") or drain()
                elif here == F11:
                    ride_pad(PAD_11F_TO_7F, "heal-to-7f") or drain()
                elif here == F5 and cur[1] >= 19:
                    ride_pad((10, 20), "heal-pocket-exit") or drain()
                else:
                    i9 = SILPH.index(here)
                    enter_to(SILPH[i9 + (1 if i9 < 8 else -1)], "heal-stairs")
                continue
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

        # standing ON any warp tile is a live grenade: directional stairs re-fire mid-pathfind
        # (strike2), and on a PAD even a tap EATEN AS A TURN counts as a movement ending on
        # the tile and re-fires it (strike11: the (10,21) bounce-back to 9F). Step off with a
        # LONG-HOLD press (turn+walk in one continuous input), never in a directional tile's
        # own fire direction (strike3/4: the 1F mat is 0x65 DOWN-arrow = out the front door).
        cur = tuple(tv.coords(b) or (0, 0))
        warp_tiles = {tuple(w[0]) for w in tv.read_warps(b)}
        if cur in warp_tiles:
            ent = camp._WARP_ENTRY.get(camp._tile_behavior(*cur))
            fire = ent[1] if ent else None
            g5 = tv.Grid(b)
            for nb, k in (((cur[0], cur[1] + 1), "DOWN"), ((cur[0] + 1, cur[1]), "RIGHT"),
                          ((cur[0] - 1, cur[1]), "LEFT"), ((cur[0], cur[1] - 1), "UP")):
                if fire is not None and (nb[0] - cur[0], nb[1] - cur[1]) == fire:
                    continue
                if nb not in warp_tiles and g5.walkable(nb[0], nb[1]):
                    b.press(k, 26, 10, camp.render, owner="agent")
                    for _ in range(20):
                        b.run_frame()
                    break
            if tuple(tv.map_id(b)) != here:
                continue                     # the step-off itself warped us — re-dispatch

        # PHASE A — no Card Key yet. THE POCKET TRUTH (strike10 + probe3/4 + live NPC reads):
        # the ball (22,21) lies in a SEALED south pocket — the hall room is walled off at row
        # 18, the east column + stair alcoves are ELEVATION-sealed (collision-open, unwalkable
        # — Grid reads collision only), the west corridor is Grunt1's permanent body. The only
        # non-circular entrance is the 9F pad (22,18) -> landing (10,20) INSIDE the pocket.
        # Route: stairs UP to 9F -> ride the pad -> row 21 east -> ball from its WEST front.
        if not have_key():
            if here == F5 and cur[1] >= 19:            # inside the pocket (pad landed us)
                for face, front in (("RIGHT", (CARD_BALL_5F[0] - 1, CARD_BALL_5F[1])),
                                    ("LEFT", (CARD_BALL_5F[0] + 1, CARD_BALL_5F[1])),
                                    ("DOWN", (CARD_BALL_5F[0], CARD_BALL_5F[1] - 1)),
                                    ("UP", (CARD_BALL_5F[0], CARD_BALL_5F[1] + 1))):
                    if tuple(tv.map_id(b)) != F5:      # an accidental pad re-fire mid-approach
                        break
                    engage(front, face, "card-key-ball")
                    if have_key():
                        break
                L(f"   CARD KEY: item={CARD_KEY_ITEM in key_items()} "
                  f"flag={fm.read_flag(b, FLAG_CARD_KEY)}")
                if not have_key():
                    if tuple(tv.map_id(b)) != F5 or tuple(tv.coords(b) or (0, 0))[1] < 19:
                        continue                       # bounced out — re-dispatch, don't abort
                    snap("30_no_key")
                    L("!! Card Key not obtained in the 5F pocket — abort LOUD")
                    return 1
                _stage_save("card_key")
                continue
            if here == SILPH[8]:                       # 9F: ride the pad down into the pocket
                if not ride_pad((22, 18), "pad-to-pocket"):
                    wedges[here] = wedges.get(here, 0) + 1
                    if wedges[here] >= 3:
                        snap("29_no_pad_ride")
                        L("!! can't ride the 9F pad x3 — abort")
                        return 1
                    drain()
                continue
            # otherwise: climb toward 9F (descend if somehow above it)
            step_a = 1 if idx < 8 else -1
            nxt = SILPH[idx + step_a] if 0 <= idx + step_a < len(SILPH) else None
            if nxt and not enter_to(nxt, f"floor{idx + 1 + step_a}-{'up' if step_a > 0 else 'down'}"):
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

        # PHASE B — key in hand. THE PAD CHAIN (route truth in the header): stairs -> 9F
        # (heal if hurt) -> pad (9,4) -> 3F west -> open the west door -> pad (13,14) ->
        # 7F pocket (5,4): Gary trigger row, Lapras, then pad (5,8) -> 11F -> Giovanni.
        if here == F7 and cur[0] <= 6:
            # the west pocket — the mission floor (PAD-ONLY; strike12 proved the stairs
            # side can never reach it: gary/lapras/pad all failed from (27,4))
            if not gary_done[0]:
                L("   7F pocket: walking the rival trigger row (Gary auto-engages)")
                nb0 = n_battles[0]
                ok = goto(GARY_TRIGGER_7F, "gary-trigger")
                if st.in_battle(b):
                    L(f"   GARY -> {camp.battle_runner()}")
                drain()
                if n_battles[0] > nb0 or ok:
                    # a battle fired on the trigger walk, or the row was reached clean
                    # (trigger already spent) — only THEN latch (strike12 latched on failure)
                    gary_done[0] = True
                    _stage_save("post_gary")
                    snap("40_post_gary")
                else:
                    wedges["gary"] = wedges.get("gary", 0) + 1
                    if wedges["gary"] >= 3:
                        L("!! Gary trigger row unreachable x3 — proceeding to 11F LOUD")
                        gary_done[0] = True
                continue
            if not lapras_done[0]:
                r = engage(LAPRAS_FRONT_7F, "UP", "lapras", drains=6)
                drain(key="B")                       # nickname/PC boxes resolve on B
                got = fm.read_flag(b, LAPRAS_FLAG)
                if r != "nothing" or got:
                    lapras_done[0] = True
                    L(f"   LAPRAS guy -> {r} (flag={got})")
                    _stage_save("post_lapras")
                else:
                    wedges["lapras"] = wedges.get("lapras", 0) + 1
                    if wedges["lapras"] >= 3:
                        L("!! Lapras guy unreachable x3 — proceeding LOUD (bonus, not mission)")
                        lapras_done[0] = True
                continue
            if not ride_pad(PAD_7F_TO_11F, "pad-11f"):
                wedges[here] = wedges.get(here, 0) + 1
                if wedges[here] >= 3:
                    snap("50_no_11f")
                    L("!! can't ride the 11F pad from the pocket x3 — abort")
                    return 1
                drain()
            continue
        if here == F11:
            # from the pad landing (2,5) Giovanni's office is OPEN — the (5-6,16-17) door
            # guards the south entrance we don't use. Giovanni first; doors only as fallback.
            r = engage(GIOVANNI_FRONT, "UP", "giovanni", drains=3)
            if r == "nothing":
                r = engage((7, 11), "LEFT", "giovanni-side", drains=3)
            if r == "nothing":
                open_doors(DOORS_11F, "11f-door")
                r = engage(GIOVANNI_FRONT, "UP", "giovanni-2", drains=3)
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
        if here == F9:
            if lead_frac() < 0.85:
                heal_9f()                            # she's right there — top up pre-Gary
            if not ride_pad(PAD_9F_TO_3F, "pad-3f"):
                open_doors(DOORS_9F_WEST, "9f-door")
                if not ride_pad(PAD_9F_TO_3F, "pad-3f-2"):
                    wedges[here] = wedges.get(here, 0) + 1
                    if wedges[here] >= 3:
                        snap("33_no_3f_pad")
                        L("!! can't ride the 9F->3F pad x3 — abort")
                        return 1
                    drain()
            continue
        if here == F3:
            doors = DOORS_3F_WEST if cur[0] < 15 else DOORS_3F_EAST
            if not ride_pad(PAD_3F_TO_7F, "pad-7f"):
                open_doors(doors, "3f-doors")
                if not ride_pad(PAD_3F_TO_7F, "pad-7f-2"):
                    wedges[here] = wedges.get(here, 0) + 1
                    if wedges[here] >= 3:
                        snap("35_no_7f")
                        L("!! can't reach the 7F pad from 3F — abort")
                        return 1
                    drain()
            continue
        if here == F5 and cur[1] >= 19:
            # key in hand but still in the pad pocket — the stairs are unreachable from
            # here; ride the pad (10,20) back to 9F and let the pad chain take over.
            if not ride_pad((10, 20), "pocket-exit") and tuple(tv.map_id(b)) == F5:
                wedges[here] = wedges.get(here, 0) + 1
                if wedges[here] >= 3:
                    snap("32_pocket_stuck")
                    L("!! can't exit the 5F pocket x3 — abort")
                    return 1
                drain()
            continue
        # any other floor (incl. 7F stairs side — the pocket is PAD-ONLY from there):
        # walk the stairs toward 9F, the pad-chain junction (stairs pair adjacent floors)
        step = 1 if idx < 8 else -1
        if not enter_to(SILPH[idx + step], f"to9f-via{idx + 1 + step}F"):
            wedges[here] = wedges.get(here, 0) + 1
            if wedges[here] >= 3:
                snap(f"36_route9f_wedge_{idx + 1}F")
                L(f"!! routing to 9F wedged on {idx + 1}F — abort")
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
            if here == F3:
                open_doors(DOORS_3F_EAST, "out-3f-door")   # stairs live east; no-op if open
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
