"""recon_mansion.py — THE SECRET KEY: Cinnabar -> Pokemon Mansion -> B1F key -> out.

Prereq canonical = cinnabar_reach. Unlocks the Cinnabar Gym door (Blaine, badge 7).

THE DERIVATION (offline toggle-state meta-BFS over pret layout bins + the
setmetatile diffs in data/scripts/pokemon_mansion.inc — G:\\temp\\longrun\\pret\\
mansion_route.json): the Mansion is ONE global switch state (FLAG 0x26C) whose
statues toggle barrier sets on all four floors (setmetatile arg 4 IS the collision
bit — no tileset resolution needed). Route to the key (nodes = floor x state x
region, edges = warps + statue toggles):
  1F entrance -> TOGGLE statue (5,5) ON -> stairs (10,13) -> 2F -> (27,17) -> 3F
  -> (18,18) -> 1F balcony -> (25,27) -> B1F -> TOGGLE (24,29) OFF -> TOGGLE
  (27,5) ON -> SECRET KEY ball (5,7) -> exit (34,29) -> 1F -> south doors -> city.
Statue actuation: face + A -> "press the switch?" YES (default) -> flag 0x26C
flips + DrawWholeMapView applies collision instantly. Key pickup verified by
FLAG_HIDE_POKEMON_MANSION_B1F_SECRET_KEY (0x1A8). Floors: 1F (1,59) 2F (1,60)
3F (1,61) B1F (1,62). Wilds (Koffing/Grimer/Rattata) + mansion trainers ride
BattleAgent via the interrupt path.

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_mansion.py
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
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bridge import Bridge            # noqa: E402
import travel as tv                  # noqa: E402
import firered_ram as ram            # noqa: E402
import field_moves as fm             # noqa: E402
from battle_agent import BattleAgent  # noqa: E402
from campaign import Campaign        # noqa: E402
import campaign as C                 # noqa: E402
from dialogue_drive import box_open as dd_box  # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CANON = os.path.join(_HERE, "states", "campaign")
SCRATCH = os.path.join(os.environ.get("TEMP", _HERE), "longrun")
STAGE = os.path.join(SCRATCH, "stage_mansion")
BANK = os.path.join(SCRATCH, "banked_SECRETKEY")
DBG = os.path.join(SCRATCH, "mansion_probe")

CINNABAR = (3, 8)
M1F, M2F, M3F, MB1F = (1, 59), (1, 60), (1, 61), (1, 62)
FLAG_SWITCH = 0x26C
FLAG_KEY_TAKEN = 0x1A8
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# CORRECTED (run6 + anchor truth): a warp EVENT with plain behavior is a LANDING
# ANCHOR and never fires (2F (27,17) = the 3F fall-holes' landing). Route uses only
# trigger-behavior tiles: stairs 0x6C/0x6F, falls 0x66, arrows 0x65.
MISSION = [
    ("door", (8, 3), M1F),          # mansion front door (Cinnabar overworld)
    ("door", (10, 13), M2F),        # 0x6C up-right stair
    ("door", (9, 3), M3F),          # 0x6C up-right stair
    ("toggle", (12, 5), True),      # 3F statue
    ("door", (18, 18), M1F),        # 0x66 fall hole -> 1F balcony (19,22)
    ("door", (25, 27), MB1F),       # 0x6F down-left stair
    ("toggle", (24, 29), False),
    ("toggle", (27, 5), True),
    ("pickup", (5, 7)),
    ("toggle", (27, 5), False),     # re-open the way back (ON seals the stair side)
    ("toggle", (24, 29), True),     # 1F SE pocket needs ON (Press opens (27-29,25))
    ("door", (34, 29), M1F),        # 0x6C stair up
    ("door", (34, 33), CINNABAR),   # the SE BACK DOOR (0x65) — the front hall is
]                                   # unreachable from the stair pocket in EITHER state


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

    state_path, sc_dir, sc_pref = _resolve_state(os.environ.get("MANSION_STATE", ""))
    b = Bridge(ROM)
    with open(state_path, "rb") as f:
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

    def fight_open():
        return ram.valid_ewram_ptr(b.rd32(ram.GBATTLE_RES_PTR))

    def drain(max_n=40, key="B"):
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
        """'battle' / 'box' / False — walk() refunds budget ONLY for battles (a box
        that reopens every cycle must BURN budget or the loop has infinite fuel —
        the run1 (20,6) 527-replan wedge)."""
        if fight_open():
            fight()
            drain()
            return "battle"
        if dd_box(b):
            drain()
            return "box"
        return False

    def settle(n=90):
        for _ in range(n):
            b.run_frame()

    def live_npc_tiles():
        OB, SZ = 0x02036E38, 0x24
        out = set()
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.add((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                     b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    def coord_event_tiles():
        """Script-trigger tiles (MapHeader.events coordEvents, stride 0x10) — the
        run1-3 Cinnabar wedge: (20,5) fires GymDoorLocked, boxes + bounces her;
        BFS must route AROUND script tiles like warps."""
        try:
            ev = b.rd32(tv.GMAPHEADER + 0x04)
            n = b.rd8(ev + 0x02)
            arr = b.rd32(ev + 0x0C)
            if not (0 < n <= 32) or arr < 0x08000000:
                return set()
            return {(b.rds16(arr + i * 0x10), b.rds16(arr + i * 0x10 + 2))
                    for i in range(n)}
        except Exception:
            return set()

    def walk(goal_test, label, tries=10, avoid=()):
        budget = tries
        stuck = [None, 0]                       # same-coord replan wedge detector
        while budget > 0:
            budget -= 1
            it = handle_interrupts()
            if it == "battle":
                budget += 1
                continue
            if it:
                continue                        # box drained: burns budget
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            if cur == stuck[0]:
                stuck[1] += 1
                if stuck[1] >= 4:
                    L(f"!! [{label}] WEDGE: 4 same-coord replans at {cur} "
                      f"(dd_box={dd_box(b)} fight={fight_open()})")
                    snap(f"wedge_{label[:12]}_{cur[0]}_{cur[1]}")
                    return False
            else:
                stuck[0], stuck[1] = cur, 0
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)}
            npcs = live_npc_tiles() | coord_event_tiles() | {
                tuple(o[0]) for o in tv.read_object_templates(b) if o[2]}
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: g.walkable(sx, sy) and (sx, sy) not in wts
                       and (sx, sy) not in npcs and (sx, sy) not in avoid)
            L(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                L(f"   [{label}] no path from {cur}")
                snap(f"nopath_{label[:12]}_{cur[0]}_{cur[1]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                it2 = handle_interrupts()
                if it2:
                    L(f"   [{label}] interrupt={it2} at {tuple(tv.coords(b) or ())} "
                      f"before step {tuple(t)}")
                    if it2 == "battle":
                        budget += 1
                    break
                if not step_to(tuple(t)):
                    L(f"   [{label}] step blocked {tuple(tv.coords(b) or ())} -> {tuple(t)}")
                    snap(f"blocked_{t[0]}_{t[1]}")
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
            if tuple(tv.coords(b) or ()) != cur:
                budget += 1
        return goal_test(tuple(tv.coords(b) or ()))

    def step_to(tile):
        cur = tuple(tv.coords(b) or (0, 0))
        d = (tile[0] - cur[0], tile[1] - cur[1])
        if d in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            d = (d[0] // 2, d[1] // 2)
        key = KEY_OF.get(d)
        if key is None:
            return camp._step_to(tile)
        for _attempt in range(3):
            b.press(key, 8, 6, camp.render, owner="agent")
            for _ in range(50):
                b.run_frame()
                if tuple(tv.coords(b) or ()) == tile:
                    break
            if fight_open() or dd_box(b):
                return True
            if tuple(tv.coords(b) or ()) == tile:
                return True
        return False

    # arrow warps (0x62-0x65) + DIRECTIONAL STAIR WARPS (0x6C-0x6F: UP_RIGHT /
    # UP_LEFT / DOWN_RIGHT / DOWN_LEFT — fire only when WALKED INTO along their
    # direction; the run5 (10,13) wedge stood ON one, the UGP 0x6F class).
    ARROW_KEY = {0x62: "RIGHT", 0x63: "LEFT", 0x64: "UP", 0x65: "DOWN",
                 0x6C: "RIGHT", 0x6D: "LEFT", 0x6E: "RIGHT", 0x6F: "LEFT"}
    OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    def tile_behavior(t):
        try:
            ml = b.rd32(tv.GMAPHEADER)
            attr = (b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14))
            bw = b.rd32(tv.BACKUP_LAYOUT)
            mp0 = b.rd32(tv.BACKUP_LAYOUT + 8)
            mid = b.rd16(mp0 + ((t[1] + tv.MAP_OFFSET) * bw
                                + (t[0] + tv.MAP_OFFSET)) * 2) & 0x3FF
            base, idx = (attr[0], mid) if mid < tv.NUM_PRIMARY else (attr[1],
                                                                     mid - tv.NUM_PRIMARY)
            return b.rd32(base + idx * 4) & 0xFF
        except Exception:
            return 0

    def go_warp(tile, dest, label):
        m0 = tuple(tv.map_id(b))
        if m0 == dest:
            return True
        beh = tile_behavior(tile)
        arrow = ARROW_KEY.get(beh)
        nbs = [(tile[0] + dx, tile[1] + dy) for dx, dy in
               ((0, 1), (0, -1), (1, 0), (-1, 0))]
        if arrow:
            d = DELTA[arrow]
            nbs = [(tile[0] - d[0], tile[1] - d[1])]
        for attempt in range(4):
            if arrow and tuple(tv.coords(b) or ()) == tile:
                b.press(OPP[arrow], 26, 10, camp.render, owner="agent")  # step OFF first
                settle(40)
            if tuple(tv.coords(b) or ()) not in nbs and tuple(tv.coords(b) or ()) != tile:
                if not walk(lambda c, s=set(nbs): c in s, f"{label}-approach"):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            key = (arrow if arrow and cur == tile
                   else KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1])) or arrow)
            if key is None:
                continue
            for _press in range(4 if arrow else 1):
                b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    break
            if handle_interrupts():
                continue
            if tuple(tv.map_id(b)) == dest:
                settle(180)
                L(f"   [{label}] {m0} -> {dest} @ {tv.coords(b)} (beh {hex(beh)})")
                _stage_save(label)
                return True
            if tuple(tv.map_id(b)) != m0:
                L(f"!! [{label}] warped to {tuple(tv.map_id(b))}, wanted {dest}")
                settle(180)
                return False
        L(f"!! [{label}] never fired (at {tv.map_id(b)}@{tv.coords(b)}, beh {hex(beh)})")
        snap(f"warpfail_{label[:16]}")
        return False

    def interact(tile, label, key="A", verify=None, tries=3, stand=None):
        """Face `tile` from an adjacent square, press A, drain (A = YES default).
        `stand`: required standing tile (the Mansion statues are bg events with
        BG_EVENT_PLAYER_FACING_NORTH — they fire ONLY from below, facing UP; run4
        interacted from the east forever)."""
        nbs = ([stand] if stand else
               [(tile[0] + dx, tile[1] + dy) for dx, dy in
                ((0, 1), (0, -1), (1, 0), (-1, 0))])
        for attempt in range(tries):
            if tuple(tv.coords(b) or ()) not in nbs:
                if not walk(lambda c, s=set(nbs): c in s, f"{label}-approach",
                            avoid={tile}):
                    return False
            cur = tuple(tv.coords(b) or (0, 0))
            face = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            if face is None:
                continue
            b.press(face, 8, 10, camp.render, owner="agent")
            b.press("A", 8, 12, camp.render, owner="agent")
            settle(30)
            drain(key=key)
            settle(30)
            if verify is None or verify():
                return True
            L(f"   [{label}] verify failed (attempt {attempt + 1})")
        L(f"!! [{label}] interaction never verified")
        snap(f"interact_fail_{label[:14]}")
        return False

    # ── preflight ────────────────────────────────────────────────────────────
    L(f"boot map={tv.map_id(b)} coords={tv.coords(b)} money=${camp.money()} "
      f"switch={fm.read_flag(b, FLAG_SWITCH)} key_taken={fm.read_flag(b, FLAG_KEY_TAKEN)}")
    if tuple(tv.map_id(b)) != CINNABAR:
        L("!! not on Cinnabar — abort")
        return 1
    if fm.read_flag(b, FLAG_KEY_TAKEN):
        L("Secret Key already taken — nothing to strike")
        return 0
    r = camp.heal_nearest()
    L(f"   heal_nearest -> {r} (post-Seafoam heal, the run10 gap now mapped)")
    while handle_interrupts():
        pass

    # ── the mission ──────────────────────────────────────────────────────────
    deadline = time.time() + 1800
    for op in MISSION:
        if time.time() > deadline:
            L("!! deadline")
            return 1
        while handle_interrupts():
            pass
        L(f"-- op {op} (map {tv.map_id(b)} @ {tv.coords(b)})")
        if op[0] == "door":
            if not go_warp(op[1], op[2], f"door{op[1]}"):
                return 1
        elif op[0] == "toggle":
            want = op[2]
            if fm.read_flag(b, FLAG_SWITCH) == want:
                L(f"   [toggle] switch already {want}")
                continue
            if not interact(op[1], f"statue{op[1]}", key="A",
                            verify=lambda w=want: fm.read_flag(b, FLAG_SWITCH) == w,
                            stand=(op[1][0], op[1][1] + 1)):    # FACING_NORTH class
                return 1
            L(f"   [toggle] switch -> {fm.read_flag(b, FLAG_SWITCH)}")
            _stage_save("toggle")
        elif op[0] == "pickup":
            if not interact(op[1], "secret-key", key="B",
                            verify=lambda: fm.read_flag(b, FLAG_KEY_TAKEN)):
                return 1
            L("   [pickup] SECRET KEY in bag (flag 0x1A8)")
            _stage_save("secret_key")

    L(f"   OUT with the key @ {tv.map_id(b)}{tv.coords(b)} after {n_battles[0]} battles")
    r = camp.heal_nearest()
    L(f"   heal_nearest -> {r}")
    snap("90_key_out")
    _stage_save("secret_key_out")
    _stage_continuity()
    if os.path.isdir(BANK):
        shutil.rmtree(BANK)
    shutil.copytree(STAGE, BANK)
    L(f"BANKED -> {BANK}")
    L(f"promote: python pokemon_agent/promote_bank.py {BANK} secret_key")
    return 0


if __name__ == "__main__":
    sys.exit(main())
