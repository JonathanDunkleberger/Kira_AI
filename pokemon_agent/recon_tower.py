"""recon_tower.py — THE POKÉMON TOWER STRIKE (night shift #6): Poké Flute end-to-end.

flute_run13 truth: the questline door-hint lands her INSIDE the Tower and even go-deepers to
2F, but GARY (rival object (16,5), no script on the object — the battle needs the multi-box
talk escalation) wedges the generic tour at (16,6), and the climb ritual (7 floors, ghost
gate, grunt gauntlet, Fuji two-stage give) isn't a "destination". Scripted spec, disasm
ground truth (pret PokemonTower_1F..7F map.json, fetched 2026-07-07):
  1F (1,88): up (18,9). 2F (1,89): GARY (16,5); up (4,10). 3F (1,90): up (18,10).
  4F (1,91): up (4,10). 5F (1,92): up (18,10). 6F (1,93): up (11,16) — MAROWAK GHOST coord
  triggers (11,15)/(12,16) flank the stairs (Scope in bag = a normal wild fight, KO it).
  7F (1,94): grunts (9,10)/(13,8)/(9,6) sight 4; MR. FUJI (11,4) — talk → auto-warp to his
  house → he hands the POKE FLUTE (item 350; flag FLAG_GOT_POKE_FLUTE 0x23D).
Approach: Celadon → R7 (east edge) → UGP#2 (warp-dest routing) → R8 → Lavender (east edge)
→ Tower door (18,6). Success = item 350 in Key Items (item truth first, flag second).
Canonical protection: staging pattern; bank -> %TEMP%/longrun/banked_FLUTE for promote_bank.
RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_tower.py
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
STAGE = os.path.join(SCRATCH, "stage_tower")
BANK = os.path.join(SCRATCH, "banked_FLUTE")
DBG = os.path.join(SCRATCH, "tower_probe")

CELADON, ROUTE7, ROUTE8, LAVENDER = (3, 6), (3, 25), (3, 26), (3, 4)
UGP_IN, UGP_TUNNEL, UGP_OUT = (1, 33), (1, 34), (1, 35)
TOWER = [(1, 88), (1, 89), (1, 90), (1, 91), (1, 92), (1, 93), (1, 94)]
GARY_FRONT = (16, 6)               # rival at (16,5) on 2F, face UP
FUJI_FRONT = (11, 5)               # Fuji at (11,4) on 7F, face UP
POKE_FLUTE, FLAG_FLUTE = 350, 0x23D


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

    def got_flute():
        return POKE_FLUTE in key_items() or fm.read_flag(b, FLAG_FLUTE)

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
        """Stand at `front`, face `face`, press A; run any battle via the OBSERVED runner
        (strat/rival/evolution recording — the run-5 lost-Gary-win lesson)."""
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
                    if st.in_battle(b):        # the talk escalated into the fight (rival class)
                        L(f"   [{label}] battle -> {camp.battle_runner()}")
                        drain()
                        return "battled"
                    for _ in range(40):
                        b.run_frame()
                break
        return out

    def enter_to(dest, label):
        """Warp to `dest` map via ANY warp on the current map whose table dest matches —
        read_warps live truth, enter_warp's full ritual stack (directional/mat-row)."""
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        cands = [xy for xy, d, _w in tv.read_warps(b) if tuple(d) == dest]
        if not cands:
            L(f"!! [{label}] no warp on {m0} leads to {dest} (warps={tv.read_warps(b)})")
            return False
        cur = tuple(tv.coords(b) or (0, 0))
        cands.sort(key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
        for wt in cands:
            r = camp.enter_warp(pick=wt)
            if r == "need_heal":
                L(f"   [{label}] heal interrupt — healing, then retrying")
                camp.heal_nearest()
                r = camp.enter_warp(pick=wt)
            if st.in_battle(b):                    # LoS trainer on the approach
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

    def edge_to(dest, edge, label):
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        r = camp.trav.travel(target_map=dest, edge=edge, max_seconds=240)
        if st.in_battle(b):
            L(f"   [{label}] battle at the edge -> {camp.battle_runner()}")
            drain()
            r = camp.trav.travel(target_map=dest, edge=edge, max_seconds=240)
        ok = tuple(tv.map_id(b)) == dest
        L(f"   [{label}] edge {edge}: {m0} -> {tv.map_id(b)}@{tv.coords(b)} ({r})")
        return ok

    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} key_items={key_items()} "
      f"flute_flag={fm.read_flag(b, FLAG_FLUTE)}")
    if got_flute():
        L("already holding the Poké Flute — nothing to do")
        return 0

    # ── 1. the approach: Celadon → R7 → UGP#2 → R8 → Lavender (skip legs already behind us)
    here = tuple(tv.map_id(b))
    if here == CELADON and not edge_to(ROUTE7, "east", "celadon-r7"):
        return 1
    if tuple(tv.map_id(b)) == ROUTE7 and not enter_to(UGP_IN, "r7-ugp"):
        return 1
    if tuple(tv.map_id(b)) == UGP_IN and not enter_to(UGP_TUNNEL, "ugp-down"):
        return 1
    if tuple(tv.map_id(b)) == UGP_TUNNEL and not enter_to(UGP_OUT, "ugp-cross"):
        return 1
    if tuple(tv.map_id(b)) == UGP_OUT and not enter_to(ROUTE8, "ugp-r8"):
        return 1
    if tuple(tv.map_id(b)) == ROUTE8 and not edge_to(LAVENDER, "east", "r8-lavender"):
        return 1
    if tuple(tv.map_id(b)) == LAVENDER and not enter_to(TOWER[0], "tower-door"):
        snap("10_no_tower")
        return 1
    if tuple(tv.map_id(b)) not in TOWER:
        L(f"!! not in the Tower (at {tv.map_id(b)}) — abort")
        return 1

    # ── 2. the climb — a STATE MACHINE on the current map, not a linear chain: a mid-climb
    #      heal interrupt (tower2: Venusaur 55% after Gary + channelers) legitimately bounces
    #      her to the Lavender Center; the loop just re-dispatches (re-enter, re-climb; beaten
    #      channelers stay beaten). 2F: GARY first (his talk escalates into rival battle;
    #      his body wedges the generic floor otherwise). 6F: the Marowak coord-trigger flanks
    #      the up-stairs — the approach trips it; the observed runner fights the unmasked ghost.
    def lead_frac():
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    gary_done = False
    wedges = {}
    deadline = time.time() + 900
    while time.time() < deadline:
        here = tuple(tv.map_id(b))
        if here == TOWER[6]:
            break
        # ATTRITION VALVE (tower3: wild ghosts chip the lead every floor; travel enters a
        # heal-when-low pause that starves the stair ritual) — descend + heal + re-climb.
        if lead_frac() < 0.5:
            L(f"   lead at {lead_frac():.0%} — descending to heal (from {here})")
            if here in TOWER and here != TOWER[0]:
                enter_to(TOWER[TOWER.index(here) - 1], "heal-descent")
            elif here == TOWER[0]:
                enter_to(LAVENDER, "tower-exit")
            elif here == LAVENDER:
                camp.heal_nearest()
            else:
                camp.enter_warp(prefer="south")
                for _ in range(80):
                    b.run_frame()
            continue
        if here == LAVENDER:
            if not enter_to(TOWER[0], "tower-reenter"):
                snap("25_reenter_fail")
                return 1
            continue
        if here not in TOWER:
            L(f"   off-route at {here} (heal interior?) — exiting to the overworld")
            camp.enter_warp(prefer="south")
            for _ in range(80):
                b.run_frame()
            if tuple(tv.map_id(b)) == here:
                L(f"!! stuck off-route at {here}@{tv.coords(b)} — abort")
                snap("26_offroute")
                return 1
            continue
        idx = TOWER.index(here)
        if here == (1, 89) and not gary_done:
            rg = engage(GARY_FRONT, "UP", "gary", drains=3)
            L(f"   GARY on 2F -> {rg}")
            gary_done = True
            snap("20_post_gary")
        if not enter_to(TOWER[idx + 1], f"floor{idx + 1}-up"):
            if tuple(tv.map_id(b)) == here:      # still on the floor — wild-interrupt flake?
                wedges[here] = wedges.get(here, 0) + 1
                if wedges[here] >= 3:            # 3 consecutive true wedges = a real wall
                    snap(f"30_floor{idx + 1}_fail")
                    L(f"!! climb wedged x3 on floor {idx + 1} (at {tv.map_id(b)}@{tv.coords(b)})")
                    return 1
                drain()
                L(f"   floor {idx + 1} retry {wedges[here]}/3")
            continue                             # bounced (heal) or retry — re-dispatch
        wedges.pop(here, None)
        _stage_save(f"floor{idx + 2}")
    if tuple(tv.map_id(b)) != TOWER[6]:
        L(f"!! climb never reached 7F (at {tv.map_id(b)}@{tv.coords(b)}) — abort")
        return 1

    # ── 3. 7F: the grunt gauntlet (LoS, travel fights through), then MR. FUJI
    snap("40_7f")
    for gx, gy in ((9, 10), (13, 8), (9, 6)):
        rg = engage((gx, gy + 1), "UP", f"grunt{gx}x{gy}")
        L(f"   7F grunt ({gx},{gy}) -> {rg}")
    r = engage(FUJI_FRONT, "UP", "fuji", drains=6)
    L(f"   FUJI -> {r}; now {tv.map_id(b)}@{tv.coords(b)}")
    for _ in range(600):                      # the scripted warp to his house + the give
        b.run_frame()
        if dd_box(b):
            b.press("A", 8, 12, camp.render, owner="agent")
    drain()
    if not got_flute() and tuple(tv.map_id(b)) != (1, 94):
        # in his house — he may need one more talk; he stands beside the arrival
        L(f"   in Fuji's house ({tv.map_id(b)}@{tv.coords(b)}) — talking")
        npcs = sorted(camp.trav._npc_tiles())
        L(f"   npcs here: {npcs}")
        cur = tuple(tv.coords(b) or (0, 0))
        for nt in sorted(npcs, key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))[:3]:
            for front, face in (((nt[0], nt[1] + 1), "UP"), ((nt[0] - 1, nt[1]), "RIGHT"),
                                ((nt[0] + 1, nt[1]), "LEFT"), ((nt[0], nt[1] - 1), "DOWN")):
                engage(front, face, "fuji-house", drains=4)
                if got_flute():
                    break
            if got_flute():
                break

    ok = got_flute()
    L(f"   POKE FLUTE: item={POKE_FLUTE in key_items()} flag={fm.read_flag(b, FLAG_FLUTE)} "
      f"(key_items={key_items()})")
    snap("50_final")
    if not ok:
        L("!! flute NOT confirmed — NOT banking; read the frames")
        return 1

    _stage_save("flute")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} poke_flute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
