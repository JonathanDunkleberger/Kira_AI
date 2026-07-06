"""recon_bill_track.py — Bill sequence with WANDERER TRACKING: re-read the NPC's live coords each
attempt, approach the CURRENT tile, face + A (x3, box-checked). Drives the full ticket sequence."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge
from battle_agent import BattleAgent
from dialogue_drive import DialogueDriver, box_open, player_facing
import field_moves as fm
import travel as tv

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SAVE = os.path.join(_HERE, "states", "campaign", "r24_for_bill.state")
SS_TICKET = 0x234
OB, SZ = 0x02036E38, 0x24
TOWARD = {(1, 0): "RIGHT", (-1, 0): "LEFT", (0, 1): "DOWN", (0, -1): "UP"}
FACING = {"DOWN": 1, "UP": 2, "LEFT": 3, "RIGHT": 4}


def face_verified(key, tries=6):
    """Readback-verified turn: press until the facing byte matches (the first press gets eaten
    right after travel — the exact eaten-press class; same doctrine as the cursor readbacks)."""
    want = FACING[key]
    for _ in range(tries):
        if player_facing(b) == want:
            return True
        b.press(key, 8, 8, None, owner="agent")
        for _f in range(16):
            b.run_frame()
    return player_facing(b) == want

b = Bridge(ROM)
with open(SAVE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.press("B", 2, 2, None, owner="agent")
agent = BattleAgent(b, log=print, render=None)
trav = tv.Traveler(b, battle_runner=lambda: agent.run(max_seconds=120), render=None, log=print)


def flag():
    return bool(fm.read_flag(b, SS_TICKET))


def obj_coord(i):
    o = OB + i * SZ
    if not (b.rd8(o) & 1):
        return None
    return (b.rds16(o + 0x10) - 7, b.rds16(o + 0x12) - 7)


def talk_tracked(i, label, tries=6):
    """Approach obj i's LIVE position each try (wanderers move mid-approach), face, A x3 box-checked."""
    for t in range(tries):
        body = obj_coord(i)
        if body is None:
            print(f"   [{label}] obj#{i} gone"); return False
        cur = tv.coords(b)
        grid = tv.Grid(b)
        for adj in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            front = (body[0] + adj[0], body[1] + adj[1])
            if front == cur or tv.bfs(grid, cur, lambda tt, f=front: tt == f, walkable=grid.walkable):
                if front != cur:
                    trav.travel(target_map=None, arrive_coord=front, max_steps=40, max_seconds=25)
                if tv.coords(b) != front:
                    break                                  # walk failed; retry with fresh coords
                if obj_coord(i) != body:
                    break                                  # he moved mid-walk; retry with fresh coords
                face_verified(TOWARD[(-adj[0], -adj[1])])
                for _a in range(3):
                    b.press("A", 6, 12, None, owner="agent")
                    for _f in range(40):
                        b.run_frame()
                    if box_open(b):
                        print(f"   [{label}] BOX OPEN (try {t}, A#{_a}) — driving")
                        r = DialogueDriver(b, render=None, log=print).drive(label=label)
                        print(f"   [{label}] drive -> {r}  flag={flag()}")
                        return True
                break                                      # faced + 3 A's, no box -> re-read coords, retry
    print(f"   [{label}] no box after {tries} tracked tries")
    return False


print(f"BOOT map={tv.map_id(b)} coords={tv.coords(b)} flag={flag()}")
trav.travel(target_map=(3, 44), edge="east", max_steps=400, max_seconds=300)
trav.travel(target_map=None, arrive_coord=(51, 5), max_steps=400, max_seconds=900)
for _ in range(6):
    b.press("UP", 8, 8, None, owner="agent")
    for _f in range(70):
        b.run_frame()
    if tuple(tv.map_id(b)) == (30, 0):
        break
print(f"inside map={tv.map_id(b)} coords={tv.coords(b)}")

talk_tracked(1, "bill-monster")
print(f"after talk1: obj1 now {obj_coord(1)}  flag={flag()}")
if not flag():
    for (bx, by), kind in tv.read_bg_events(b):
        if kind > 4:
            continue
        print(f"-- console BG ({bx},{by}) kind={kind}: trying ALL sides")
        for adj in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            cur = tv.coords(b)
            grid = tv.Grid(b)
            front = (bx + adj[0], by + adj[1])
            if not grid.walkable(*front):
                print(f"   side {front} unwalkable"); continue
            if front != cur and not tv.bfs(grid, cur, lambda tt, f=front: tt == f, walkable=grid.walkable):
                print(f"   side {front} unreachable"); continue
            if front != cur:
                trav.travel(target_map=None, arrive_coord=front, max_steps=40, max_seconds=25)
            if tv.coords(b) != front:
                continue
            face_verified(TOWARD[(-adj[0], -adj[1])])
            for _a in range(2):
                b.press("A", 6, 12, None, owner="agent")
                for _f in range(90):                       # cutscene-length settle
                    b.run_frame()
                bo = box_open(b)
                print(f"   [console] side {front} A#{_a}: box={bo} obj1={obj_coord(1)} flag={flag()}")
                if bo:
                    r = DialogueDriver(b, render=None, log=print).drive(label="console")
                    print(f"   [console] drive -> {r}  flag={flag()} obj1={obj_coord(1)}")
            if flag() or obj_coord(1) is not None:
                break
        if flag() or obj_coord(1) is not None:
            break
if not flag():
    # SWEEP the top-left desk row: Bill's PC is a FURNITURE/metatile interaction, not the BG event.
    # Stand at (x,6), face UP, A — box-check each. (General 'work the room' probe.)
    grid = tv.Grid(b)
    for x in range(0, 9):
        spot = (x, 6)
        if not grid.walkable(*spot):
            continue
        cur = tv.coords(b)
        if spot != cur and not tv.bfs(grid, cur, lambda tt, s=spot: tt == s, walkable=grid.walkable):
            continue
        if spot != cur:
            trav.travel(target_map=None, arrive_coord=spot, max_steps=40, max_seconds=20)
        if tv.coords(b) != spot:
            continue
        b.press("UP", 8, 8, None, owner="agent")
        b.press("A", 6, 12, None, owner="agent")
        for _f in range(40):
            b.run_frame()
        if box_open(b):
            print(f"   [sweep] BOX at ({x},6) facing UP — driving")
            r = DialogueDriver(b, render=None, log=print).drive(label=f"sweep{x}")
            print(f"   [sweep{x}] drive -> {r}  flag={flag()}")
        if flag():
            break
if not flag():
    talk_tracked(1, "bill-again")
else:
    talk_tracked(1, "bill-human-ticket")     # after separation: talk human Bill -> S.S. Ticket
print(f"FINAL flag={flag()}  map={tv.map_id(b)}@{tv.coords(b)}")
b.frame_rgb().save(r"G:\temp\claude\G--JonnyD-NeuroAI-Bot\92df62cc-b82e-4538-8fdb-4a238758a51f\scratchpad\bill_final.png")
print("PASS" if flag() else "FAIL")
