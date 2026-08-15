"""recon_mewtwo_gate.py — settle the Cerulean Cave gate EMPIRICALLY (does the champion flag alone
move the guard?).

WHY THIS EXISTS (2026-08-14): `legendary_strikes.py:96` encodes the gate as champion-flag-only —
`FLAG_SYS_GAME_CLEAR = 0x82C  # champion — the Cerulean Cave guard steps aside` — and `MewtwoHunt.run`
checks nothing else before walking onto the cave-mouth warp. But in retail FRLG the Unknown-Dungeon
guard is widely held to need the NATIONAL DEX (Celio's Ruby + Sapphire, Sevii 4-7) — a questline that
does NOT exist anywhere in this repo (islands 1-3 only, as Moltres' ferry). If that is true,
`_mewtwo_gate` arms on the champion flag, the hunt walks to the mouth, the guard is still there, and
run_mewtwo returns 'failed' forever.

Guessing is not allowed here. Every object-event TEMPLATE in FRLG carries its own hide-flag id at
template+20 (travel.read_object_templates reads exactly that). So: load a Cerulean state, dump every
object on the map with its hide flag, and find the guard that stands between her and the cave mouth
(1,12). The flag id IS the answer:
    * flag == 0x82C            -> the repo is right, champion is enough, Mewtwo is a small fix
    * flag == something else   -> that flag is the real gate; find out what sets it

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_mewtwo_gate.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                 # noqa: E402
import firered_ram as ram                 # noqa: E402
import travel as tv                       # noqa: E402
import field_moves as fm                  # noqa: E402
import legendary_strikes as ls            # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
CERULEAN_STATE = next(
    (p for p in (os.path.join(_HERE, "states", "workshop", "seg_cerulean.state"),
                 os.path.join(_HERE, "states", "kira", "seg_cerulean.state"),
                 os.path.join(_HERE, "states", "seg_cerulean.state"))
     if os.path.exists(p)), os.path.join(_HERE, "states", "seg_cerulean.state"))
CAVE_MOUTH = (1, 12)          # ls.MewtwoHunt walks onto this tile to enter Cerulean Cave 1F

# Flags worth naming if we hit them (pret include/constants/flags.h).
KNOWN = {
    0x82C: "FLAG_SYS_GAME_CLEAR (champion — what the repo assumes)",
    0x82D: "FLAG_SYS_POKEDEX_GET",
    0x846: "FLAG_SYS_NATIONAL_DEX (pret: national dex obtained)",
}


def log(m):
    print(m, flush=True)


def dump_objects(b, label):
    """Every object-event template on the CURRENT map + its hide flag + present/gone."""
    ev = b.rd32(tv.GMAPHEADER + 0x04)
    n = b.rd8(ev)
    arr = b.rd32(ev + 0x04)
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    log("")
    log(f"--- {label}: map={tv.map_id(b)} objects={n} ---")
    log(f"{'#':>2} {'tile(save)':>12} {'tile(-off)':>12} {'gfx':>4} {'hideflag':>9} {'set?':>5} present")
    rows = []
    for i in range(n):
        t = arr + i * 0x18
        x, y = b.rds16(t + 4), b.rds16(t + 6)
        gfx = b.rd8(t + 1)
        flag = b.rd16(t + 20)
        fset = bool(flag) and bool(b.rd8(sb1 + 0x0EE0 + (flag >> 3)) & (1 << (flag & 7)))
        rows.append((i, (x, y), (x - tv.MAP_OFFSET, y - tv.MAP_OFFSET), gfx, flag, fset))
        log(f"{i:>2} {str((x, y)):>12} {str((x - tv.MAP_OFFSET, y - tv.MAP_OFFSET)):>12} "
            f"{gfx:>4} {('0x%03X' % flag) if flag else '     -':>9} {str(fset):>5} {not fset}")
    return rows


def nearest_to_mouth(rows):
    """Objects sorted by distance to the cave-mouth warp tile (offset-corrected coords)."""
    mx, my = CAVE_MOUTH

    def d(r):
        x, y = r[2]
        return abs(x - mx) + abs(y - my)
    return sorted(rows, key=d)


def main():
    if not os.path.exists(CERULEAN_STATE):
        log(f"!! missing {CERULEAN_STATE}")
        return 1
    b = Bridge(ROM)
    with open(CERULEAN_STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(120):
        b.run_frame()
    log(f"boot map={tv.map_id(b)} coords={tv.coords(b)} "
        f"champion_flag(0x82C)={fm.read_flag(b, ls.FLAG_SYS_GAME_CLEAR)}")
    if tuple(tv.map_id(b)) != ls.CERULEAN:
        log(f"!! not on Cerulean {ls.CERULEAN} — this probe needs the Cerulean map")
        return 1

    rows = dump_objects(b, "CERULEAN, champion flag CLEAR (pre-credits)")
    log("")
    log(f"--- objects nearest the cave mouth {CAVE_MOUTH} (the guard is whatever stands here) ---")
    for r in nearest_to_mouth(rows)[:6]:
        i, save_xy, off_xy, gfx, flag, fset = r
        mx, my = CAVE_MOUTH
        dist = abs(off_xy[0] - mx) + abs(off_xy[1] - my)
        name = KNOWN.get(flag, "")
        log(f"   obj#{i} at {off_xy} dist={dist} gfx={gfx} "
            f"hideflag={('0x%03X' % flag) if flag else 'NONE (always present)'} "
            f"set={fset}{('  <-- ' + name) if name else ''}")

    # Now SET the champion flag and re-read. Templates are ROM (flags don't move them), but the
    # present/gone column is live save-flag state — this proves whether 0x82C alone clears the guard.
    log("")
    log("=" * 78)
    log("SETTING FLAG_SYS_GAME_CLEAR (0x82C) and re-reading the present/gone column")
    log("=" * 78)
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    addr = sb1 + 0x0EE0 + (ls.FLAG_SYS_GAME_CLEAR >> 3)
    b.core.memory.u8.raw_write(addr, b.rd8(addr) | (1 << (ls.FLAG_SYS_GAME_CLEAR & 7)))
    log(f"champion_flag(0x82C) now = {fm.read_flag(b, ls.FLAG_SYS_GAME_CLEAR)}")
    rows2 = dump_objects(b, "CERULEAN, champion flag SET")

    log("")
    log("--- VERDICT ---")
    changed = [(r1, r2) for r1, r2 in zip(rows, rows2) if r1[5] != r2[5]]
    if changed:
        for r1, r2 in changed:
            log(f"   obj#{r1[0]} at {r1[2]} gfx={r1[3]} flag=0x{r1[4]:03X}: "
                f"present {not r1[5]} -> {not r2[5]}  <-- the champion flag moved THIS object")
    else:
        log("   NO object on Cerulean changed presence when 0x82C was set.")
        log("   => the champion flag alone does NOT remove anything on this map.")
    guard = nearest_to_mouth(rows)[0]
    log("")
    log(f"   Closest object to the cave mouth: obj#{guard[0]} at {guard[2]} gfx={guard[3]} "
        f"hideflag={('0x%03X' % guard[4]) if guard[4] else 'NONE'}")
    if guard[4] == ls.FLAG_SYS_GAME_CLEAR:
        log("   => REPO IS RIGHT: the guard is hidden by FLAG_SYS_GAME_CLEAR. Champion is enough.")
    elif guard[4] == 0:
        log("   => the mouth-side object has NO hide flag: it is scenery/permanent, so the guard "
            "is NOT a hide-flag NPC — access is script/var gated (needs a live post-credits probe).")
    else:
        log(f"   => REPO IS WRONG about the gate: the mouth guard is hidden by 0x{guard[4]:03X}, "
            f"NOT 0x82C. Find what sets 0x{guard[4]:03X} before trusting MewtwoHunt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
