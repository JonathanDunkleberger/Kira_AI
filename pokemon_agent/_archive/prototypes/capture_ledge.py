"""capture_ledge.py - drive FireRed by hand to BANK a ledge savestate + map the
route. STANDALONE (zero Kira imports), same input firewall as session.py.

Why: the boot map (3,19) is ledgeless, so we can't measure FireRed's one-way ledge
behavior value from it. This lets Jonny walk ~20s to a hop-down ledge on Route 1
and save the position; recon_lock_ledge.py then runs the proven hop-test on the
banked state to LOCK the ledge behavior value + direction. While driving it logs
map-ID + coords + facing on every change, so we learn EXACTLY where she boots and
what maps/ledges sit between her and Viridian.

INPUT FIREWALL (identical doctrine to session.py): EXACTLY ONE input owner
("human"), enforced by Bridge's owner guard. Held keys are EVENT-tracked (not
get_pressed) so the launch-Enter stuck-key can't strand START; focus loss clears
held keys. Any non-owner press is logged loudly and dropped (Constraint #3).

RUN: .venv\\Scripts\\python.exe pokemon_agent\\capture_ledge.py
KEYS: arrows=D-pad, Z=A, X=B, Enter=START, Backspace=SELECT,
      S=save states/ledge_sample.state,  G=dump local collision+behavior grid.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bridge import Bridge        # noqa: E402
import firered_ram as ram        # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
STATES = os.path.join(_HERE, "states")
BOOT_STATE = os.path.join(STATES, "after_pick_bulbasaur.state")
SAVE_STATE = os.path.join(STATES, "ledge_sample.state")
SCALE = 3

# ── memory map (all validated in recon) ──────────────────────────────────────
GMAPHEADER = 0x02036DFC          # struct MapHeader; +0x00 -> ROM MapLayout
BACKUP_LAYOUT = 0x03005040       # {s32 width, s32 height, u16 *map} runtime grid
MAP_OFFSET = 7                   # border: save-coord + 7 = buffer index
NUM_PRIMARY = 640                # metatile ids < 640 use primary tileset
OBJ_PLAYER = 0x02036E38          # gObjectEvents[0]; facing nibble at +0x18
FACING = {1: "DOWN", 2: "UP", 3: "LEFT", 4: "RIGHT"}


def log(m):
    print(f"   [capture] {m}", flush=True)


def map_id(b):
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    return b.rd8(sb1 + 0x04), b.rd8(sb1 + 0x05)


def facing(b):
    return FACING.get(b.rd8(OBJ_PLAYER + 0x18) & 0x0F, "?")


def attrs(b):
    """(primary, secondary) metatileAttributes ROM pointers for the current map."""
    ml = b.rd32(GMAPHEADER)
    return b.rd32(b.rd32(ml + 0x10) + 0x14), b.rd32(b.rd32(ml + 0x14) + 0x14)


def tile(b, attr, w, mp, sx, sy):
    """(behavior, collision) at SAVE coords (sx,sy)."""
    bx, by = sx + MAP_OFFSET, sy + MAP_OFFSET
    entry = b.rd16(mp + (bx + w * by) * 2)
    mid = entry & 0x3FF
    col = (entry & 0x0C00) >> 10
    base, idx = (attr[0], mid) if mid < NUM_PRIMARY else (attr[1], mid - NUM_PRIMARY)
    return (b.rd16(base + idx * 2) & 0xFF), col


def dump_grid(b, radius=5):
    """Print collision + behavior around the player so a ledge is visible before
    saving. '.'=walkable 0x00, '#'=blocked, 'g'=grass, 'P'=player; each cell also
    shows the raw behavior hex so the ledge value stands out."""
    co = ram.read_player_coords(b)
    if co is None:
        log("grid: no coords"); return
    attr = attrs(b)
    w = b.rd32(BACKUP_LAYOUT); mp = b.rd32(BACKUP_LAYOUT + 8)
    g, n = map_id(b)
    log(f"grid around {co} map=({g},{n}) facing={facing(b)} (cell = behavior_hex/collision):")
    for dy in range(-radius, radius + 1):
        cells = []
        for dx in range(-radius, radius + 1):
            be, cl = tile(b, attr, w, mp, co[0] + dx, co[1] + dy)
            mark = "P" if (dx == 0 and dy == 0) else ""
            cells.append(f"{mark}{be:02x}/{cl}")
        print("      " + " ".join(f"{c:>6}" for c in cells))


def save_sample(b):
    co = ram.read_player_coords(b)
    g, n = map_id(b)
    try:
        with open(SAVE_STATE, "wb") as f:
            f.write(bytes(b.save_state()))
        log(f"SAVED {SAVE_STATE}  (map=({g},{n}) coords={co} facing={facing(b)})")
        log("  -> next: .venv\\Scripts\\python.exe pokemon_agent\\recon_lock_ledge.py")
    except Exception as e:
        log(f"!! SAVE FAILED: {e}")


def boot(b):
    if not os.path.exists(BOOT_STATE):
        log(f"FAIL - boot state missing: {BOOT_STATE}"); return False
    with open(BOOT_STATE, "rb") as f:
        b.load_state(f.read())
    for _ in range(40):
        b.run_frame()
    co = ram.read_player_coords(b)
    g, n = map_id(b)
    log(f"booted {os.path.basename(BOOT_STATE)}: map=({g},{n}) coords={co} "
        f"party={ram.read_party_count(b)}")
    if co is None:
        log("FAIL - not in overworld control"); return False
    return True


def main():
    import pygame
    pygame.init()
    b = Bridge(ROM)
    log(f"loaded {b.game_code} {b.game_title!r}")
    if not boot(b):
        pygame.quit(); return
    win = (b.width * SCALE, b.height * SCALE)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("Capture ledge - arrows/Z/X/Enter | S=save | G=grid")
    keymap = {pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT",
              pygame.K_RIGHT: "RIGHT", pygame.K_z: "A", pygame.K_x: "B",
              pygame.K_RETURN: "START", pygame.K_BACKSPACE: "SELECT"}

    def blit():
        surf = pygame.image.fromstring(b.frame_rgb().tobytes(),
                                       (b.width, b.height), "RGB")
        screen.blit(pygame.transform.scale(surf, win), (0, 0))
        pygame.display.flip()

    b.set_input_owner("human")     # Bridge enforces: keyboard is the sole writer
    log("DRIVE: arrows=D-pad Z=A X=B Enter=START | S=save ledge_sample | G=grid dump")
    log("walk to a hop-DOWN ledge on Route 1, press G to confirm it's below you, then S")

    WFL = getattr(pygame, "WINDOWFOCUSLOST", None)
    held = set()
    last_log = None
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if (WFL is not None and ev.type == WFL) or \
                   (ev.type == pygame.ACTIVEEVENT and getattr(ev, "gain", 1) == 0):
                    if held:
                        held.clear(); log("focus lost -> held keys cleared")
                elif ev.type == pygame.KEYDOWN and ev.key in keymap:
                    held.add(keymap[ev.key])
                elif ev.type == pygame.KEYUP and ev.key in keymap:
                    held.discard(keymap[ev.key])
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
                    save_sample(b)
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_g:
                    dump_grid(b)
            b.set_keys(*held, owner="human") if held else b.release(owner="human")
            b.run_frame(); blit()
            # log map/coords/facing on every change -> the route trace
            co = ram.read_player_coords(b)
            sig = (map_id(b), co, facing(b))
            if co is not None and sig != last_log:
                g, n = sig[0]
                log(f"map=({g},{n}) coords={co} facing={sig[2]}")
                last_log = sig
    except KeyboardInterrupt:
        log("window closed")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
