"""recon_b2f_sim.py — OFFLINE glide-BFS over the dumped B2F grid (no emulator).
Parses hideout_probe/grid_b2f.txt, replays the exact slide mechanics (spinners redirect,
0x58 dots stop, plain floor carries momentum, walls stop), and searches for a press plan
from the B3F-arrival (21,2) / B1F-stairs (28,2) to the elevator-door approach (28,17)/(27,17)
or the B1F-south stairs pocket (23,12). Prints every reachable rest tile + the winning plan.
"""
import os
from collections import deque

GRID = os.path.join(os.environ.get("TEMP", "."), "longrun", "hideout_probe", "grid_b2f.txt")
SPIN = {0x54: (1, 0), 0x55: (-1, 0), 0x56: (0, -1), 0x57: (0, 1)}

rows = {}
npcs = set()
with open(GRID, encoding="utf-8") as f:
    header = f.readline().strip()
    print(header)
    # grunt + the disasm item balls (RocketHideout_B2F map.json: X Speed (15,3),
    # Moon Stone (2,5), TM12 (5,7), Super Potion (0,14)) — object templates block glides
    npcs = {(20, 6), (15, 3), (2, 5), (5, 7), (0, 14)}
    for line in f:
        line = line.rstrip()
        if not line.startswith("y"):
            continue
        y = int(line[1:3])
        toks = line[4:].split()
        rows[y] = toks

H = max(rows) + 1
W = max(len(t) for t in rows.values())
print(f"parsed {H} rows, widest {W} tokens")


def cell(x, y):
    if y not in rows or x >= len(rows[y]) or x < 0 or y < 0:
        return None
    return rows[y][x]


def walkable(x, y):
    c = cell(x, y)
    if c is None or c in ("##", "N!"):
        return False
    if c == "..":
        return True
    v = int(c, 16)
    # spinners + stop dots + stair warps are walkable; 69 elevator door, 9x decor = blocked
    return v in SPIN or v in (0x58, 0x6C, 0x6D, 0x6E, 0x6F, 0x62, 0x63, 0x64, 0x65)


def bh(x, y):
    c = cell(x, y)
    if c in (None, "##", "N!", ".."):
        return 0
    return int(c, 16)


def glide(frm, d):
    x, y = frm
    dx, dy = d
    sliding = False
    v0 = bh(x, y)
    if v0 in SPIN:
        dx, dy = SPIN[v0]
        sliding = True
    for _ in range(300):
        nx, ny = x + dx, y + dy
        if not walkable(nx, ny) or (nx, ny) in npcs:
            return (x, y) if (x, y) != frm else None
        x, y = nx, ny
        v = bh(x, y)
        if v in SPIN:
            dx, dy = SPIN[v]
            sliding = True
            continue
        if v == 0x58 or not sliding:
            return (x, y)
    return (x, y)


def bfs(start, targets):
    prev = {start: None}
    q = deque([start])
    goal = None
    while q:
        cur = q.popleft()
        if cur in targets:
            goal = cur
            break
        for key, d in (("RIGHT", (1, 0)), ("LEFT", (-1, 0)), ("UP", (0, -1)), ("DOWN", (0, 1))):
            dst = glide(cur, d)
            if dst and dst not in prev:
                prev[dst] = (cur, key)
                q.append(dst)
    return prev, goal


TARGETS = {(28, 17), (27, 17), (29, 17), (23, 13), (22, 12), (23, 12), (24, 12)}
for start in ((21, 2), (28, 2), (15, 8)):
    prev, goal = bfs(start, TARGETS)
    print(f"\nfrom {start}: {len(prev)} rest tiles reachable; goal={goal}")
    east_room = sorted(t for t in prev if 15 <= t[0] <= 24 and 12 <= t[1] <= 18)
    print(f"  east-room rest tiles reached: {east_room}")
    if goal:
        plan = []
        n = goal
        while prev[n] is not None:
            p, k = prev[n]
            plan.append((k, n))
            n = p
        plan.reverse()
        print(f"  PLAN ({len(plan)} presses): {plan}")
    else:
        # what's the reachable frontier east of x=14?
        east = sorted(t for t in prev if t[0] >= 14)
        print(f"  reachable rest tiles x>=14: {east}")
