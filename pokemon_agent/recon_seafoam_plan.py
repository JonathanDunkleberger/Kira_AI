"""recon_seafoam_plan.py — OFFLINE Seafoam interior route derivation (the pad_plan port,
planning half). Runs on pret ground truth (layout bins + map.json warp tables cached in
G:\\temp\\longrun\\pret\\) — no emulator needed.

WHY OFFLINE: recon_sabrina.pad_plan floods regions from LIVE RAM, which only works for
same-map teleport pads. Seafoam's pads are CROSS-FLOOR ladders — the meta-BFS needs all
five floors' grids before she has visited them, so the plan is derived here from the
disassembly and the runtime vehicle (recon_seafoam.py) executes the hop list.

MODEL:
  - map.bin word: metatile id bits0-9, collision bits10-11, elevation bits12-15.
  - behavior: metatile_attributes.bin u32 low 9 bits (primary gTileset_General < 0x280,
    secondary gTileset_SeafoamIslands >= 0x280).
  - walk edges: collision-0 both ends + the per-edge elevation law (equal or either
    0/0xF) — the safari_bfs truth.
  - WATER IS A ROAD (she has Surf): pond/deep water (MB 0x10/0x12/0x15) is walkable;
    land<->water edges are ALWAYS allowed (the mount toll / auto-dismount, elevation law
    does not apply across the shoreline). CURRENTS (MB 0x50-0x53) + waterfalls are
    BLOCKED — they sweep the surfer (the Articuno boulder puzzle exists to stop them;
    crossing must avoid them entirely).
  - ladders/holes = warp events; step-on triggers. Boulders + Articuno are masked solid.

OUTPUT: the ordered hop list (floor, warp_idx, step-on tile, dest floor, landing tile)
from the R20 east door landing (1F (6,21)) to the 1F exit (32,21) -> R20 (72,14) WEST sea.

RUN: python pokemon_agent\\recon_seafoam_plan.py
"""
import json
import os
import struct
import sys
from collections import deque

PRET = r"G:\temp\longrun\pret"

FLOORS = {
    "F1":  ("Seafoam1F.json",  "szf1f.bin",   38, 24),
    "B1F": ("SeafoamB1F.json", "szfb1f.bin",  38, 23),
    "B2F": ("SeafoamB2F.json", "szfbb2f.bin", 38, 24),
    "B3F": ("SeafoamB3F.json", "szfbb3f.bin", 38, 24),
    "B4F": ("SeafoamB4F.json", "szfbb4f.bin", 38, 24),
}
NAME2KEY = {
    "MAP_SEAFOAM_ISLANDS_1F": "F1", "MAP_SEAFOAM_ISLANDS_B1F": "B1F",
    "MAP_SEAFOAM_ISLANDS_B2F": "B2F", "MAP_SEAFOAM_ISLANDS_B3F": "B3F",
    "MAP_SEAFOAM_ISLANDS_B4F": "B4F",
}
# emulator (group,num) ids — verified against pret map_groups.json 2026-07-07
MAPNUM = {"F1": (1, 83), "B1F": (1, 84), "B2F": (1, 85), "B3F": (1, 86), "B4F": (1, 87)}

WATER_BEH = {0x10, 0x11, 0x12, 0x15}      # pond/deep/semi-deep (travel.py's water set + 0x11)
BLOCK_BEH = {0x13, 0x50, 0x51, 0x52, 0x53}  # waterfall + the four currents


def load():
    attrs = {}
    for name, fn in (("gen", "attr_general.bin"), ("sea", "attr_seafoam.bin")):
        raw = open(os.path.join(PRET, fn), "rb").read()
        attrs[name] = [struct.unpack_from("<I", raw, i * 4)[0] & 0x1FF
                       for i in range(len(raw) // 4)]

    def behavior(mt):
        if mt < 0x280:
            return attrs["gen"][mt] if mt < len(attrs["gen"]) else 0
        s = mt - 0x280
        return attrs["sea"][s] if s < len(attrs["sea"]) else 0

    grids, warps, solids = {}, {}, {}
    for k, (jf, bf, w, h) in FLOORS.items():
        raw = open(os.path.join(PRET, bf), "rb").read()
        col, elev, beh = {}, {}, {}
        for i in range(w * h):
            v = struct.unpack_from("<H", raw, i * 2)[0]
            t = (i % w, i // w)
            col[t] = (v >> 10) & 3
            elev[t] = (v >> 12) & 0xF
            beh[t] = behavior(v & 0x3FF)
        grids[k] = (col, elev, beh, w, h)
        d = json.load(open(os.path.join(PRET, jf)))
        warps[k] = [((wv["x"], wv["y"]), NAME2KEY.get(wv["dest_map"], wv["dest_map"]),
                     int(wv["dest_warp_id"])) for wv in d["warp_events"]]
        solids[k] = {(o["x"], o["y"]) for o in d.get("object_events", [])
                     if "BOULDER" in o.get("graphics_id", "")
                     or "ARTICUNO" in o.get("graphics_id", "")}
    return grids, warps, solids


def nb4(t):
    return [(t[0] + 1, t[1]), (t[0] - 1, t[1]), (t[0], t[1] + 1), (t[0], t[1] - 1)]


def main():
    grids, warps, solids = load()

    def tile_ok(k, t):
        col, elev, beh, w, h = grids[k]
        if not (0 <= t[0] < w and 0 <= t[1] < h):
            return False
        if col.get(t, 1) != 0 or t in solids[k]:
            return False
        return beh[t] not in BLOCK_BEH

    def is_water(k, t):
        return grids[k][2].get(t) in WATER_BEH

    def edge_ok(k, a, bt):
        _, elev, _, _, _ = grids[k]
        wa, wb = is_water(k, a), is_water(k, bt)
        if wa != wb:
            return True                     # shoreline: mount / auto-dismount
        if wa and wb:
            return True                     # open water glide
        e1, e2 = elev[a], elev[bt]
        return e1 == e2 or 0 in (e1, e2) or 0xF in (e1, e2)

    def region(k, seed):
        if not tile_ok(k, seed):
            return frozenset()
        wtiles = {t for t, _, _ in warps[k]}
        seen, q = {seed}, [seed]
        while q:
            cur = q.pop()
            for n in nb4(cur):
                if n in seen or n in wtiles:
                    continue
                if not tile_ok(k, n) or not edge_ok(k, cur, n):
                    continue
                seen.add(n)
                q.append(n)
        return frozenset(seen)

    def seed_region(k, tile):
        best = frozenset()
        for n in nb4(tile):
            r = region(k, n)
            if len(r) > len(best):
                best = r
        return best

    start_land = warps["F1"][3][0]          # R20 east door lands here: (6,21)
    goal_tile = warps["F1"][4][0]           # (32,21) -> R20 (72,14) WEST sea

    start_reg = region("F1", start_land) or seed_region("F1", start_land)
    print(f"start region ({start_land}) size {len(start_reg)}")
    seen_reg = {("F1", start_reg)}
    q = deque([("F1", start_reg, [])])
    route = None
    while q:
        k, reg, path = q.popleft()
        if k == "F1" and any(n in reg for n in nb4(goal_tile)):
            route = path
            break
        for i, (t, dk, dwid) in enumerate(warps[k]):
            if dk not in grids or not any(n in reg for n in nb4(t)):
                continue
            land = warps[dk][dwid][0]
            lr = region(dk, land) or seed_region(dk, land)
            if not lr or (dk, lr) in seen_reg:
                continue
            seen_reg.add((dk, lr))
            q.append((dk, lr, path + [(k, i, t, dk, land)]))

    if route is None:
        print(f"NO ROUTE ({len(seen_reg)} regions swept) — dump follows")
        for k, reg in sorted(seen_reg, key=lambda x: (x[0], -len(x[1]))):
            print(f"  {k} region size {len(reg)} sample {sorted(reg)[:3]}")
        return 1
    print(f"ROUTE — {len(route)} ladder hops (emulator ids: {MAPNUM}):")
    hops = []
    for k, i, t, dk, land in route:
        water_note = " (region includes WATER)" if any(
            is_water(k, x) for x in nb4(t)) else ""
        print(f"  {k} warp{i}: step-on {t} -> {dk} land {land}{water_note}")
        hops.append({"floor": k, "map": MAPNUM[k], "warp_tile": list(t),
                     "dest": dk, "dest_map": MAPNUM[dk], "land": list(land)})
    print(f"  then walk to F1 exit {goal_tile} -> R20 (72,14) WEST SEA")
    hops.append({"floor": "F1", "map": MAPNUM["F1"], "warp_tile": list(goal_tile),
                 "dest": "R20W", "dest_map": [3, 38], "land": [72, 14]})
    out = os.path.join(PRET, "seafoam_route.json")
    with open(out, "w") as f:
        json.dump(hops, f, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
