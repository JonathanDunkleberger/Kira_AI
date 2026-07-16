"""recon_bill_sequence.py — GROUND TRUTH on Bill's Cell-Separation sequence (the S.S. Ticket).
Boot the run-7 bank (Route 24 south, trainers beaten), walk to the cottage, then drive the FULL scripted
sequence with logged dialogue: talk monster-Bill (A-only → YES), interact the BG-event console, talk
human Bill → read FLAG_GOT_SS_TICKET after each step. Proves which primitives complete it. No canonical
mutation."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["POKEMON_TRAVEL_MUSE_GAP_S"] = "0"
from bridge import Bridge
from battle_agent import BattleAgent
from dialogue_drive import DialogueDriver, box_open
import field_moves as fm
import travel as tv

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SAVE = os.path.join(_HERE, "states", "campaign", "r24_for_bill.state")
SS_TICKET = 0x234
OB, SZ = 0x02036E38, 0x24
TOWARD = {(1, 0): "RIGHT", (-1, 0): "LEFT", (0, 1): "DOWN", (0, -1): "UP"}

b = Bridge(ROM)
with open(SAVE, "rb") as f:
    b.load_state(f.read())
for _ in range(20):
    b.press("B", 2, 2, None, owner="agent")

agent = BattleAgent(b, log=print, render=None)
trav = tv.Traveler(b, battle_runner=lambda: agent.run(max_seconds=120), render=None, log=print)


def flag():
    return bool(fm.read_flag(b, SS_TICKET))


def objects():
    out = []
    for i in range(1, 16):
        o = OB + i * SZ
        if b.rd8(o) & 1:
            out.append((i, (b.rds16(o + 0x10) - 7, b.rds16(o + 0x12) - 7), b.rd8(o + 0x07)))
    return out


def talk_at(body, label):
    grid = tv.Grid(b)
    cur = tv.coords(b)
    for adj in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        front = (body[0] + adj[0], body[1] + adj[1])
        if front == cur or tv.bfs(grid, cur, lambda t, f=front: t == f, walkable=grid.walkable):
            if front != cur:
                trav.travel(target_map=None, arrive_coord=front, max_steps=60, max_seconds=40)
            if tv.coords(b) != front:
                continue
            face = TOWARD[(-adj[0], -adj[1])]
            b.press(face, 8, 8, None, owner="agent")
            b.press("A", 6, 12, None, owner="agent")
            for _ in range(30):
                b.run_frame()
            print(f"   [{label}] box_open={box_open(b)}")
            r = DialogueDriver(b, render=None, log=print).drive(label=label)
            print(f"   [{label}] drive -> {r}  flag={flag()}")
            return True
    print(f"   [{label}] could not reach any side of {body}")
    return False


print(f"BOOT map={tv.map_id(b)} coords={tv.coords(b)} flag={flag()}")
# 1) Route 24 -> Route 25 (east)
if tuple(tv.map_id(b)) != (3, 44):
    r = trav.travel(target_map=(3, 44), edge="east", max_steps=400, max_seconds=300)
    print(f"cross to Route 25: {r} -> {tv.map_id(b)}@{tv.coords(b)}")
# 2) to the cottage door approach (51,5), then UP into (30,0)
r = trav.travel(target_map=None, arrive_coord=(51, 5), max_steps=400, max_seconds=900)
print(f"approach door: {r} -> {tv.coords(b)}")
for _ in range(6):
    b.press("UP", 8, 8, None, owner="agent")
    for _f in range(70):                    # door animation + warp transition settle
        b.run_frame()
    if tuple(tv.map_id(b)) == (30, 0):
        break
print(f"inside? map={tv.map_id(b)} coords={tv.coords(b)}")
if tuple(tv.map_id(b)) != (30, 0):
    print("FAIL: never entered the cottage"); sys.exit(1)

print(f"objects: {objects()}")
print(f"bg_events: {tv.read_bg_events(b)}")

# 3) talk monster-Bill (obj table body coords), A-only driver answers the YES/NO
objs = objects()
for i, body, ttype in objs:
    talk_at(body, f"bill-talk-obj{i}")
    if flag():
        break
print(f"after talks: objects now {objects()}  flag={flag()}")

# 4) interact each BG event (kind 0-4): the Cell-Separation console
if not flag():
    for (bx, by), kind in tv.read_bg_events(b):
        if kind > 4:
            continue
        print(f"-- BG event ({bx},{by}) kind={kind}")
        talk_at((bx, by), f"bg({bx},{by})")
        if flag():
            break

# 5) talk everyone again (human Bill hands the ticket)
if not flag():
    for i, body, ttype in objects():
        talk_at(body, f"bill-again-obj{i}")
        if flag():
            break

print(f"FINAL: flag={flag()}  map={tv.map_id(b)}@{tv.coords(b)}")
print("PASS" if flag() else "FAIL")
