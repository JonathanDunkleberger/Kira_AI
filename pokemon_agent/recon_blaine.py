"""recon_blaine.py — THE CINNABAR GYM STRIKE (badge 7): Blaine through the quiz doors.

Ground truth (pret CinnabarGym_scripts.inc + CinnabarGym.json, cached G:\\temp\\longrun\\pret\\):
the gym is SIX quiz machines (FACING_NORTH bg-event pairs); each door opens on EITHER the
correct YES/NO answer OR beating that room's trainer (a wrong answer walks the trainer to
you, and winning fires the same QuizNComplete) — fail-safe both ways, a botched press only
costs a battle we win. Correct answers: Q1 YES, Q2 NO, Q3 NO, Q4 NO, Q5 YES, Q6 NO.
B advances plain msgboxes AND selects NO on a YES/NO box, so each station drains on ONE key.

Island side: the locked-door coord event (20,5) is dead once OnTransition saw the Secret
Key flag (0x1A8) — canonical banked AFTER that transition, VAR_TEMP_1=1 in the state.
⚠️ THE BILL AMBUSH: beating Blaine sets VAR_MAP_SCENE_CINNABAR=1; the first transition
back onto the island fires a FORCED scene — Bill (spawn (20,7), the gym doorstep) runs up
with a YES/NO "sail to One Island?" where A/YES ships her to the Sevii Islands. The
post-badge island drain is B-ONLY until stable, then heal + bank.

Blaine (5,4) face DOWN -> front (5,5) face UP; Arcanine L47 tops his roster vs Venusaur
L59 (Sleep Powder + Strength). Post-win flags: 0x4B6 defeated + 0x826 BADGE 7; TM38 gift
A-drains (no Y/N). Success = flag 0x826. Bank -> %TEMP%/longrun/banked_BLAINE.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_blaine.py     (WATCH=1 = live window)
"""
import json
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

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import pokemon_state as st           # noqa: E402
import firered_ram as ram            # noqa: E402
import field_moves as fm             # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_blaine")
BANK = os.path.join(SCRATCH, "banked_BLAINE")
DBG = os.path.join(SCRATCH, "blaine_probe")

ISLAND = (3, 8)
GYM = (12, 0)
BLAINE_FRONT = (5, 5)                # Blaine (5,4), face UP
FLAG_BADGE_VOLCANO = 0x826
FLAG_DEFEATED_BLAINE = 0x4B6
GYM_ENTRY_MATS = {(24, 23), (25, 23), (26, 23)}

# (label, quiz flag, [front tiles: stand here + face UP], drain key for the CORRECT answer)
QUIZ_CHAIN = [
    ("quiz1", 0x265, [(22, 11), (23, 11)], "A"),   # YES
    ("quiz2", 0x267, [(15, 3), (16, 3)], "B"),     # NO
    ("quiz3", 0x268, [(13, 11), (14, 11)], "B"),   # NO
    ("quiz4", 0x269, [(13, 18), (14, 18)], "B"),   # NO
    ("quiz5", 0x26A, [(1, 19), (2, 19)], "A"),     # YES
    ("quiz6", 0x26B, [(1, 11), (2, 11)], "B"),     # NO
]


def _resolve_state(name):
    """Resolve a state BASENAME/path -> (state_path, sidecar_dir, sidecar_prefix)."""
    if not name:
        return (os.path.join(CANON, "kira_campaign.state"), CANON, "kira_campaign")
    for cand in (name, os.path.join(_HERE, "states", name),
                 os.path.join(_HERE, "states", "workshop", name),
                 os.path.join(_HERE, "states", name + ".state"),
                 os.path.join(_HERE, "states", "workshop", name + ".state")):
        if os.path.exists(cand):
            d = os.path.dirname(cand)
            base = os.path.basename(cand)
            pref = base[:-6] if base.endswith(".state") else base
            return (cand, d, pref)
    return (name, os.path.dirname(name) or CANON, "kira_campaign")


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("BLAINE_STATE", ""))
    b = Bridge(ROM)
    with open(state_path, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()

    render_fn = lambda: None                       # noqa: E731
    if os.environ.get("WATCH") == "1":
        import pygame
        pygame.init()
        _scale = 3
        _win = (b.width * _scale, b.height * _scale)
        _screen = pygame.display.set_mode(_win)
        pygame.display.set_caption("Kira — CINNABAR GYM STRIKE (live watch)")

        def render_fn():
            for _ev in pygame.event.get():
                if _ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
            _surf = pygame.image.fromstring(b.frame_rgb().tobytes(),
                                            (b.width, b.height), "RGB")
            _screen.blit(pygame.transform.scale(_surf, _win), (0, 0))
            pygame.display.flip()

        _fc = [0]
        _orig_rf = b.run_frame

        def _rf_watch():
            _orig_rf()
            _fc[0] += 1
            if _fc[0] % 4 == 0:
                render_fn()
        b.run_frame = _rf_watch

    n_battles = [0]

    def fight():
        n_battles[0] += 1
        return BattleAgent(b, on_event=lambda *a, **k: None, render=render_fn,
                           log=lambda m: print(m, flush=True)).run(max_seconds=300)

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
    _w_side = os.path.join(sc_dir, sc_pref + ".world_model.json")
    _s_side = os.path.join(sc_dir, sc_pref + ".strat_memory.json")
    _soul_side = os.path.join(sc_dir, sc_pref + ".soul.json")
    for loader, path, fallback in (
            (camp.world.load, _w_side, C.WORLD_JSON),
            (camp.strat.load, _s_side, C.STRAT_JSON)):
        try:
            loader(path if os.path.exists(path) else fallback)
        except Exception:
            pass
    try:
        if camp.soul is not None:
            camp.soul.load(_soul_side if os.path.exists(_soul_side)
                           else os.path.join(CANON, "soul.json"))
    except Exception:
        pass

    def badge():
        return fm.read_flag(b, FLAG_BADGE_VOLCANO)

    def lead_frac():
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

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
        """The Silph/Sabrina deterministic same-map mover: static BFS, warps + template-NPC
        bodies masked; battles recompute; a step that fails outside battle after the
        spotting-wait is DEAD for this call. NO per-tile elevation law here — the opened
        quiz doorways are setmetatile'd to collision-0 ELEVATION-0 (run1 truth: (26,9)/
        (27,9) read elev 0 beside elev-3 floor), and elev 0 is the game's wildcard;
        Grid.edge_open already enforces the real per-EDGE elevation rule inside bfs."""
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
                if st.in_battle(b):
                    L(f"   [{label}] battle mid-path -> {camp.battle_runner()}")
                    drain()
                    break
                if not ok:
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

    def engage(front, face, label, drains=1, key="A"):
        if not walk_path_to(front, label):
            L(f"!! [{label}] couldn't reach {front} (at {tv.coords(b)})")
            return "unreached"
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
                    drain(key=key)
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

    def island_b_drain(label, seconds=25.0):
        """THE BILL-AMBUSH DRAIN: on the island the on-frame scene locks her, walks Bill
        up, and asks a YES/NO where YES sails to One Island. B advances every box and
        answers NO. Patience windows cover the applymovement walks between boxes."""
        end = time.time() + seconds
        quiet = 0
        while time.time() < end:
            if dd_box(b):
                quiet = 0
                b.press("B", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                quiet += 1
                for _ in range(45):
                    b.run_frame()
                if quiet >= 8:                      # ~6s of silence = the scene is over
                    break
        L(f"   [{label}] island B-drain done (quiet={quiet}, at {tv.coords(b)})")

    def gym_exit_and_bank():
        L("   badge in hand — walking out (B-ONLY on the island: the Bill ambush)")
        out_deadline = time.time() + 300
        while tuple(tv.map_id(b)) != ISLAND and time.time() < out_deadline:
            here = tuple(tv.map_id(b))
            if here == GYM:
                mats = sorted(GYM_ENTRY_MATS,
                              key=lambda t: abs(t[0] - (tv.coords(b) or (25, 12))[0]))
                tgt = (mats[0][0], mats[0][1] - 1)   # tile just above the mat row
                walk_path_to(tgt, "walk-out")
                if not enter_to(ISLAND, "out-door"):
                    drain(key="B")
            else:
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
        if tuple(tv.map_id(b)) != ISLAND:
            L(f"!! walk-out incomplete (at {tv.map_id(b)}) — banking anyway (the badge "
              f"holds; recovery owns the exit)")
        else:
            island_b_drain("bill-ambush")
            snap("60_post_bill")
            camp.heal_nearest()
            island_b_drain("post-heal", seconds=8.0)

        L(f"   VOLCANO BADGE: flag={badge()} | pos {tv.map_id(b)}@{tv.coords(b)} | "
          f"lead {lead_frac():.0%} | battles {n_battles[0]}")
        snap("80_final")
        _stage_save("blaine_badge7")
        _stage_continuity()
        if os.path.isdir(BANK):
            shutil.rmtree(BANK)
        shutil.copytree(STAGE, BANK)
        L(f"BANKED -> {BANK}")
        L(f"promote: python pokemon_agent/promote_bank.py {BANK} blaine_badge7")

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} badge7={badge()} "
      f"lead={lead_frac():.0%} key_flag={fm.read_flag(b, 0x1A8)} "
      f"blaine_done={fm.read_flag(b, FLAG_DEFEATED_BLAINE)}")
    if badge():
        L("Volcano Badge already held — nothing to strike")
        return 0

    wedges = {}
    deadline = time.time() + 1800
    while time.time() < deadline and not badge():
        here = tuple(tv.map_id(b))
        if here == ISLAND:
            if lead_frac() < 0.6:
                L(f"   lead at {lead_frac():.0%} — healing at the Cinnabar Center first")
                camp.heal_nearest()
                continue
            if not enter_to(GYM, "gym-door"):
                wedges["door"] = wedges.get("door", 0) + 1
                if wedges["door"] >= 2:
                    # the locked-door coord event only dies when OnTransition ran with
                    # the key flag — re-fire it via a Center round-trip, then retry
                    L("   gym door bounced — re-firing OnTransition via the Center")
                    camp.heal_nearest()
                if wedges["door"] >= 4:
                    snap("10_no_gym_door")
                    L("!! can't enter the gym x4 — abort LOUD")
                    return 1
                drain(key="B")
            continue
        if here != GYM:
            L(f"   off-route at {here} (whiteout/heal interior?) — exiting to the overworld")
            camp.enter_warp(prefer="south")
            for _ in range(80):
                b.run_frame()
            if tuple(tv.map_id(b)) == here:
                L(f"!! stuck off-route at {here}@{tv.coords(b)} — abort")
                snap("11_offroute")
                return 1
            continue

        # hurt mid-gym: leave, heal, come back (flags/doors persist; trainers stay beaten)
        if lead_frac() < 0.5:
            L(f"   lead at {lead_frac():.0%} — leaving to heal (doors stay open)")
            walk_path_to((25, 22), "heal-exit")
            enter_to(ISLAND, "exit-to-heal")
            continue

        # ── THE QUIZ CHAIN: open every door not already open ──
        progressed = False
        blocked = False
        for label, flag, fronts, key in QUIZ_CHAIN:
            if fm.read_flag(b, flag):
                continue
            L(f"   [{label}] door closed — engaging (answer key {key})")
            done = False
            for front in fronts:
                r = engage(front, "UP", label, drains=4, key=key)
                drain(key=key)
                for _ in range(60):
                    b.run_frame()
                if fm.read_flag(b, flag):
                    L(f"   [{label}] DOOR OPEN ({r}) [lead {lead_frac():.0%}]")
                    _stage_save(label)
                    done = True
                    break
                L(f"   [{label}] engaged from {front} -> {r}, flag still unset")
            if not done:
                wedges[label] = wedges.get(label, 0) + 1
                if wedges[label] >= 3:
                    snap(f"20_{label}_wedge")
                    L(f"!! [{label}] door never opened x3 — abort LOUD")
                    return 1
                blocked = True
            progressed = True
            break                                   # re-evaluate the chain from the top
        if progressed:
            if blocked:
                drain(key="B")
            continue

        # all six doors open — Blaine
        L(f"   quiz doors ALL OPEN — engaging Blaine [lead {lead_frac():.0%}]")
        nb0 = n_battles[0]
        r = engage(BLAINE_FRONT, "UP", "blaine", drains=6)
        drain()
        for _ in range(240):                        # badge fanfare + TM38 gift pacing
            b.run_frame()
            if dd_box(b):
                b.press("A", 8, 12, camp.render, owner="agent")
        drain()
        L(f"   BLAINE -> {r} (battles {nb0}->{n_battles[0]}) badge7={badge()} "
          f"lead={lead_frac():.0%}")
        if badge():
            _stage_save("badge7")
            snap("40_badge7")
            break
        if tuple(tv.map_id(b)) != GYM:
            continue                                # whiteout — the loop heals + re-enters
        wedges["blaine"] = wedges.get("blaine", 0) + 1
        if wedges["blaine"] >= 3:
            snap("50_blaine_wedge")
            L("!! Blaine engaged x3 without a badge — abort LOUD")
            return 1

    if not badge():
        L(f"!! Volcano Badge NOT won (at {tv.map_id(b)}@{tv.coords(b)}) — NOT banking")
        snap("70_fail")
        return 1

    gym_exit_and_bank()
    return 0


if __name__ == "__main__":
    sys.exit(main())
