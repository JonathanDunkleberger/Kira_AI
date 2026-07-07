"""recon_sabrina.py — THE SAFFRON GYM STRIKE (badge 6): Sabrina through the teleport-pad maze.

sabrina_run8 truth (night shift #1, 2026-07-07): the gym interior (14,3) is warp-partitioned —
travel BFS reads no_route from the entrance pocket to ANYTHING (the billed teleport-pad maze),
campaign's gym handler then false-latches "juniors cleared" off the one reachable obj and
A-mashes Sabrina from 11 tiles away, forever. The gym needs PAD ROUTING.

THE NEW GENERAL CAPABILITY — pad_plan(): the pad graph is computed AT RUNTIME, zero hardcoded
room sequence. read_warps gives every warp event in id order; a warp whose dest is the CURRENT
map is a teleport pad, and its dest_warp_id indexes the landing tile. Flood-fill walk-regions
(warps + NPC bodies masked), meta-BFS over regions with pad rides as edges. Ports to any
teleport maze (Sabrina's gym here; the class repeats at Cinnabar Mansion-style layouts).

Disasm ground truth (campaign.py billed 2026-07-07, pret SaffronCity_Gym/map.json):
  gym = (14,3); entrance warps (13-15,23) -> Saffron door (46,12); Sabrina object (14,11) ->
  front (14,12), face UP; badge flag 0x825 (Marsh); interior = 32 warp events.
  Sabrina: Psychic L37-43 (Alakazam L43) — Venusaur L57; psychic hits her poison x2, the
  NUKE-SLEEP/battle agent owns the fight.
Success = flag 0x825. Exit via pad route back to the entrance, heal, bank sabrina_badge6.
Canonical protection: staging pattern; bank -> %TEMP%/longrun/banked_SABRINA.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_sabrina.py     (WATCH=1 = live window)
"""
import json
import os
import shutil
import sys
import time
from collections import deque

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
STAGE = os.path.join(SCRATCH, "stage_sabrina")
BANK = os.path.join(SCRATCH, "banked_SABRINA")
DBG = os.path.join(SCRATCH, "sabrina_probe")

SAFFRON = (3, 10)
GYM = (14, 3)
SABRINA_FRONT = (14, 12)             # Sabrina (14,11), face UP
FLAG_BADGE_MARSH = 0x825
GYM_ENTRY = (14, 23)                 # the middle entrance mat (a door warp, NOT a pad)


def main():
    t0 = time.time()

    def L(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    b = Bridge(ROM)
    with open(os.path.join(CANON, "kira_campaign.state"), "rb") as f:
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
        pygame.display.set_caption("Kira — SAFFRON GYM STRIKE (live watch)")

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

    def badge():
        return fm.read_flag(b, FLAG_BADGE_MARSH)

    def lead_frac():
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def elev_fn():
        """Per-tile ELEVATION reader (map-grid u16 bits 12-15). strike2 truth: the void
        strip beside the rooms (col x29) reads collision-0 elevation-0 while the floor is
        elevation 3 — the game blocks 3<->0 steps, so an elevation-blind flood welds every
        room together (the Silph strike9 'elevation-sealed' class, now solved generally
        for the strike's planners: a tile is only walkable at the START tile's elevation,
        0xF = multi-level always allowed)."""
        w = b.rd32(tv.BACKUP_LAYOUT)
        h = b.rd32(tv.BACKUP_LAYOUT + 4)
        mp = b.rd32(tv.BACKUP_LAYOUT + 8)
        off = tv.MAP_OFFSET

        def elev(sx, sy):
            bx, by = sx + off, sy + off
            if not (0 <= bx < w and 0 <= by < h):
                return -1
            return (b.rd16(mp + (by * w + bx) * 2) >> 12) & 0xF
        return elev

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
        """Deterministic same-map mover (the Silph strike pattern verbatim): static BFS,
        WARPS + template-NPC BODIES masked, stepped tile-by-tile; battles recompute; a
        step that fails outside battle after the spotting-wait is DEAD for this call."""
        dead = set()
        for _ in range(tries):
            cur = tuple(tv.coords(b) or (0, 0))
            if cur == tile:
                return True
            g = tv.Grid(b)
            el = elev_fn()
            e0 = el(*cur)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - {tile}
            npcs = ({tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
                    | dead) - {tile}
            p = tv.bfs(g, cur, lambda t: t == tile,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs
                       and el(sx, sy) in (e0, 0xF))
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

    def ride_pad(pad, label):
        """Silph strike verbatim: approach a free pad-neighbor (nearest by real BFS), then
        one LONG-HOLD press onto the pad (turn+walk in one continuous input)."""
        m0 = tuple(tv.map_id(b))
        c0 = tuple(tv.coords(b) or (0, 0))
        g = tv.Grid(b)
        el = elev_fn()
        e0 = el(*c0)
        wts = {tuple(w[0]) for w in tv.read_warps(b)}
        npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
        cur0 = tuple(tv.coords(b) or (0, 0))
        cands = []
        for nb, kin in (((pad[0] - 1, pad[1]), "RIGHT"), ((pad[0] + 1, pad[1]), "LEFT"),
                        ((pad[0], pad[1] - 1), "DOWN"), ((pad[0], pad[1] + 1), "UP")):
            if nb in wts or not g.walkable(nb[0], nb[1]) or el(*nb) not in (e0, 0xF):
                continue
            p = tv.bfs(g, cur0, lambda t, a=nb: t == a,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs
                       and el(sx, sy) in (e0, 0xF)) \
                if cur0 != nb else [cur0]
            if p:
                cands.append((len(p), nb, kin))
        for _len, nb, kin in sorted(cands):
            if tuple(tv.coords(b) or ()) != nb and not walk_path_to(nb, f"{label}-approach"):
                continue
            for _try in range(3):
                b.press(kin, 26, 10, camp.render, owner="agent")
                # SAME-MAP pads never change map_id — arrival = coords JUMPED to a tile
                # that is none of {start, approach square, the pad itself} (the teleport
                # fade takes a beat after she lands on the pad; don't judge early).
                for _ in range(180):
                    b.run_frame()
                    cnow = tuple(tv.coords(b) or (0, 0))
                    if tuple(tv.map_id(b)) != m0 or cnow not in (c0, nb, pad):
                        break
                cnow = tuple(tv.coords(b) or (0, 0))
                if tuple(tv.map_id(b)) != m0 or cnow not in (c0, nb, pad):
                    for _ in range(60):
                        b.run_frame()
                    L(f"   [{label}] rode pad {pad}: {c0} -> {tuple(tv.coords(b) or ())} "
                      f"(map {tuple(tv.map_id(b))})")
                    return True
            break
        L(f"!! [{label}] pad ride {pad} failed (at {tv.map_id(b)}@{tv.coords(b)})")
        return False

    # ── THE PAD-GRAPH ROUTER (the general capability this strike banks) ──────────────────
    def pad_plan(goal_tile, label="pad-plan"):
        """Plan the pad-ride sequence from HERE to goal_tile's walk-region on the current
        map. Warp events whose dest is THIS map are teleport pads; dest_warp_id indexes the
        landing warp's tile (read_warps returns events in id order). Regions = flood-fill
        with all warp tiles + NPC bodies masked; edges = pads whose neighbor is in-region.
        Returns [] (already there), [pad, ...] to ride in order, or None (no route)."""
        g = tv.Grid(b)
        el = elev_fn()
        here = tuple(tv.map_id(b))
        warps = tv.read_warps(b)
        npcs = {tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
        wtiles = {tuple(w[0]) for w in warps}
        pads = {}
        for xy, dest, wid in warps:
            if tuple(dest) == here and 0 <= wid < len(warps):
                pads[tuple(xy)] = tuple(warps[wid][0])

        def region(seed):
            # BOUND to the playable rectangle (border tiles read collision-0 — strike1's
            # leak) AND to the seed's ELEVATION (the x29 void strip reads collision-0
            # elevation-0 beside elevation-3 floor — strike2's leak; the game blocks
            # cross-elevation steps, 0xF = multi-level pass).
            e0 = el(*seed)
            seen, q = {seed}, [seed]
            while q:
                x, y = q.pop()
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    n = (x + dx, y + dy)
                    if n in seen or n in wtiles or n in npcs:
                        continue
                    if not (g.sx_lo <= n[0] <= g.sx_hi and g.sy_lo <= n[1] <= g.sy_hi):
                        continue
                    if el(n[0], n[1]) not in (e0, 0xF):
                        continue
                    if not g.walkable(n[0], n[1]) or not g.edge_open(x, y, dx, dy):
                        continue
                    seen.add(n)
                    q.append(n)
            return frozenset(seen)

        def pads_of(reg):
            out = []
            for p in pads:
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    if (p[0] + dx, p[1] + dy) in reg:
                        out.append(p)
                        break
            return out

        cur = tuple(tv.coords(b) or (0, 0))
        start = region(cur)
        if goal_tile in start:
            return []
        seen_regions = {start}
        q = deque([(start, [])])
        while q:
            reg, path = q.popleft()
            for p in pads_of(reg):
                land = pads[p]
                lr = None
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    n = (land[0] + dx, land[1] + dy)
                    if n not in wtiles and n not in npcs and g.walkable(n[0], n[1]):
                        lr = region(n)
                        break
                if lr is None or lr in seen_regions:
                    continue
                if goal_tile in lr:
                    L(f"   [{label}] route: {' -> '.join(str(x) for x in path + [p])} "
                      f"then walk to {goal_tile}")
                    return path + [p]
                seen_regions.add(lr)
                q.append((lr, path + [p]))
        L(f"!! [{label}] NO pad route from {cur} to {goal_tile} "
          f"({len(pads)} pads, {len(seen_regions)} regions swept)")
        return None

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} badge6={badge()} "
      f"lead={lead_frac():.0%} saffron_free={fm.read_flag(b, 0x3E)}")
    if badge():
        L("Marsh Badge already held — nothing to strike")
        return 0

    wedges = {}
    deadline = time.time() + 1800
    while time.time() < deadline and not badge():
        here = tuple(tv.map_id(b))
        if here == SAFFRON:
            if lead_frac() < 0.6:
                L(f"   lead at {lead_frac():.0%} — healing at the Saffron Center first")
                camp.heal_nearest()
                continue
            if not enter_to(GYM, "gym-door"):
                wedges["door"] = wedges.get("door", 0) + 1
                if wedges["door"] >= 3:
                    snap("10_no_gym_door")
                    L("!! can't enter the gym x3 — abort")
                    return 1
                drain()
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

        # inside the gym — standing ON a warp tile (a pad landing) is a live grenade:
        # step off with a LONG-HOLD press before anything else (Silph loop-top guard).
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
                continue

        # hurt mid-gym: exit to the Center, heal, come back (beaten trainers stay beaten)
        if lead_frac() < 0.5:
            L(f"   lead at {lead_frac():.0%} — leaving to heal (beaten trainers stay beaten)")
            plan_out = pad_plan((14, 22), "exit-route")     # the tile above the entrance mats
            if plan_out:
                ride_pad(plan_out[0], "exit-ride") or drain()
                continue
            if plan_out == []:
                enter_to(SAFFRON, "exit-to-heal")
            continue

        plan = pad_plan(SABRINA_FRONT, "to-sabrina")
        if plan is None:
            wedges["plan"] = wedges.get("plan", 0) + 1
            if wedges["plan"] >= 3:
                snap("20_no_route")
                L("!! pad router found no route to Sabrina x3 — abort LOUD")
                return 1
            drain()
            continue
        if plan:
            if not ride_pad(plan[0], "ride"):
                wedges["ride"] = wedges.get("ride", 0) + 1
                if wedges["ride"] >= 4:
                    snap("30_ride_wedge")
                    L(f"!! pad ride wedged x4 at {tv.coords(b)} — abort")
                    return 1
                drain()
            else:
                wedges.pop("ride", None)
                _stage_save("pad_hop")
            continue

        # in Sabrina's region — engage her
        L(f"   Sabrina's room reached (at {cur}) — engaging [lead {lead_frac():.0%}]")
        nb0 = n_battles[0]
        r = engage(SABRINA_FRONT, "UP", "sabrina", drains=6)
        drain()
        for _ in range(240):                     # badge fanfare + TM gift pacing
            b.run_frame()
            if dd_box(b):
                b.press("A", 8, 12, camp.render, owner="agent")
        drain()
        L(f"   SABRINA -> {r} (battles {nb0}->{n_battles[0]}) badge6={badge()} "
          f"lead={lead_frac():.0%}")
        if badge():
            _stage_save("badge6")
            snap("40_badge6")
            break
        if tuple(tv.map_id(b)) != GYM:
            continue                             # lost/whiteout — the loop heals + re-enters
        wedges["sabrina"] = wedges.get("sabrina", 0) + 1
        if wedges["sabrina"] >= 3:
            snap("50_sabrina_wedge")
            L("!! Sabrina engaged x3 without a badge — abort LOUD")
            return 1

    if not badge():
        L(f"!! Marsh Badge NOT won (at {tv.map_id(b)}@{tv.coords(b)}) — NOT banking")
        snap("70_fail")
        return 1

    # ── walk out (pad route to the entrance region), heal, bank ──
    L("   badge in hand — walking out")
    out_deadline = time.time() + 300
    while tuple(tv.map_id(b)) != SAFFRON and time.time() < out_deadline:
        here = tuple(tv.map_id(b))
        if here == GYM:
            cur = tuple(tv.coords(b) or (0, 0))
            if cur in {tuple(w[0]) for w in tv.read_warps(b)}:
                g7 = tv.Grid(b)
                for nb, k in (((cur[0], cur[1] + 1), "DOWN"), ((cur[0] + 1, cur[1]), "RIGHT"),
                              ((cur[0] - 1, cur[1]), "LEFT"), ((cur[0], cur[1] - 1), "UP")):
                    if g7.walkable(nb[0], nb[1]):
                        b.press(k, 26, 10, camp.render, owner="agent")
                        for _ in range(20):
                            b.run_frame()
                        break
                if tuple(tv.map_id(b)) != GYM:
                    continue
            plan_out = pad_plan((14, 22), "out-route")
            if plan_out:
                ride_pad(plan_out[0], "out-ride") or drain()
            elif plan_out == []:
                enter_to(SAFFRON, "out-door")
            else:
                drain()
        else:
            camp.enter_warp(prefer="south")
            for _ in range(80):
                b.run_frame()
    if tuple(tv.map_id(b)) == SAFFRON:
        camp.heal_nearest()
    else:
        L(f"!! walk-out incomplete (at {tv.map_id(b)}) — banking anyway (the badge holds; "
          f"the longrun's recovery owns the exit)")

    L(f"   MARSH BADGE: flag={badge()} | pos {tv.map_id(b)}@{tv.coords(b)} | "
      f"lead {lead_frac():.0%}")
    snap("80_final")
    _stage_save("sabrina_badge6")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} sabrina_badge6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
