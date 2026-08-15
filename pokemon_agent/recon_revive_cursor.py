"""recon_revive_cursor.py — GROUND TRUTH for the in-battle item-use ("Use on which POKEMON?") cursor.

THE BUG (live 17:3x Lorelei, docs/soak-reports/20260814_173116):
    use_item: revive RIGHT off living lead (same pick as switch - not insta-A on the fighter)
    use_item: REFUSE A - still on living lead after RIGHT/DOWN (never insta-A the fighter)
    use_item: REFUSE A on alive slot (wanted fainted row 2, cursor=0 not 0-HP; tried rows [2])
6 Revives in the bag, a 0-HP wincon on the floor, and every single attempt bailed. Two rival
hypotheses that source-reading CANNOT separate:
    H1  the D-pad presses are EATEN on this screen (cursor really never moves)
    H2  the cursor DOES move but both readers are blind here (_party_cursor_slot() finds no
        right-column orange, and gPartyMenu.slotId @0x02020777 is not the item-use screen's byte)

This measures it. Boots the baked Lorelei fixture (recon_revive_stage.py — Blastoise out, Zapdos
dead at row 2, 6 Revives: the live position byte-for-byte), drives the REAL BattleAgent bag path
to the target screen, then steps the D-pad one tap at a time and after every tap dumps:
    * a PNG                                     (eyeball truth)
    * every EWRAM byte that changed             (finds the real slotId, if any)
    * _party_cursor_slot / _party_cursor_on_lead / _item_slot_id / _ram_party_cursor
    * a full-frame orange-row histogram         (where the highlight ACTUALLY is)

RUN: .venv\\Scripts\\python.exe -u pokemon_agent\\recon_revive_cursor.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge                 # noqa: E402
import firered_ram as ram                 # noqa: E402
import pokemon_state as st                # noqa: E402
import battle_agent as ba                 # noqa: E402
from battle_agent import BattleAgent      # noqa: E402

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
SRC = os.path.join(_HERE, "states", "workshop", "e4_revive_stage.state")
OUT = os.path.join(os.environ.get("TEMP", _HERE), "longrun", "revive_cursor")

REVIVE = 24
# EWRAM diff window. Wide on the first tap (find the byte), narrow after (keep it quick).
WLO, WHI = 0x02000000, 0x02040000


def log(m):
    print(m, flush=True)


# ── party / bag surgery ───────────────────────────────────────────────────────
def party_rows(b):
    out = []
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    for i in range(min(cnt, 6)):
        base = ram.GPLAYER_PARTY + i * 100
        out.append((i, st.read_party_species(b, i), b.rd8(base + 0x54),
                    b.rd16(base + 0x56), b.rd16(base + 0x58)))
    return out


# ── screen forensics ──────────────────────────────────────────────────────────
def orange_rows(b, ag):
    """Full-frame histogram of the selection orange: for every scanline, how many of a
    coarse x-sample are 'cursor orange', split into the LEFT (lead panel) and RIGHT
    (slot column) halves. This is where the highlight really is, independent of anchors."""
    p = b.frame_rgb().load()
    left_x = (6, 14, 22, 30, 40, 50, 60, 70, 80, 90)
    right_x = (104, 116, 128, 140, 152, 164, 176, 188, 200, 212, 224, 234)
    hits = []
    for y in range(160):
        nl = sum(1 for x in left_x if ag._cursor_orange(p[x, y]))
        nr = sum(1 for x in right_x if ag._cursor_orange(p[x, y]))
        if nl >= 3 or nr >= 3:
            hits.append((y, nl, nr))
    return hits


def probe_pts(b, ag):
    """Raw RGB at the exact points _party_cursor_slot() samples, per slot."""
    p = b.frame_rgb().load()
    out = {}
    for slot in (1, 2, 3, 4, 5):
        y0 = 10 + 24 * (slot - 1)
        out[slot] = [(dy, [tuple(p[x, y0 + dy]) for x in (110, 140, 170, 200, 225)])
                     for dy in (0,)]
    return out


def readers(b, ag):
    return {
        "pix_slot": ag._party_cursor_slot(),
        "pix_lead": ag._party_cursor_on_lead(),
        "item_slot_id": ag._item_slot_id(),
        "ram_cursor": ag._ram_party_cursor(),
        "party_pix": ag._party_screen(),
        "party_cb2": ag._party_menu_cb2(),
        "bag_pix": ag._bag_screen(),
        "bag_cb2": ag._bag_menu_cb2(),
        "cb2": hex(b.rd32(ram.GMAIN_CB2)),
        "menu_rows": [(r["row"], r["hp"]) for r in ag._menu_rows()],
    }


def snapshot(b, lo=WLO, hi=WHI):
    u8 = b.core.memory.u8
    return bytearray(u8[a] for a in range(lo, hi))


def diff(before, after, lo):
    return [(lo + i, before[i], after[i])
            for i in range(len(before)) if before[i] != after[i]]


def dump(b, tag):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.png")
    b.frame_rgb().save(path)
    return path


# ── staging ───────────────────────────────────────────────────────────────────
def reach_target_screen(b, ag):
    """Drive the REAL bag path to the 'Use on which POKEMON?' target screen.
    Mirrors use_item_in_battle stages A+B exactly, minus the (broken) aim walk."""
    ids = [i for i, _ in ag._items_pocket()]
    if REVIVE not in ids:
        log(f"!! Revive not in pocket {ids}")
        return False
    row = ids.index(REVIVE)
    log(f"   pocket={ag._items_pocket()}  revive display row={row}")
    if not ag._settle_action_menu():
        log("!! no action menu")
        return False
    if not ag._open_bag():
        log("!! bag wouldn't open")
        return False
    # pocket clamp to Items (0)
    for _ in range(6):
        if b.rd8(ram.GBAG_POCKET) == 0:
            break
        ag._tap("LEFT")
        ag._wait(12)

    def sel():
        return b.rd8(ba.BAG_CURSOR) + b.rd16(ba.BAG_SCROLL)
    for _ in range(14):
        if sel() == row:
            break
        ag._tap("DOWN" if sel() < row else "UP")
        ag._wait(10)
    log(f"   bag row selected={sel()} (want {row})  bag_pix={ag._bag_screen()}")
    dump(b, "00_bag_on_revive")
    # A until the target screen owns input. NOTE: the in-battle bag shows a
    # USE/CANCEL sub-box after the item A ("REVIVE is selected." + USE/CANCEL at
    # 210..230,130) — the SAME pixels _party_submenu() samples. Never B here: A#1
    # selects the item, A#2 confirms USE and opens "Use on which POKEMON?".
    for i in range(6):
        b.press("A", ag.hold, ag.hold, lambda: None, owner=ag.owner)
        ag._wait(30)
        log(f"   A#{i + 1}: {readers(b, ag)}")
        log(f"           png={dump(b, f'00_A{i + 1}')}")
        if ag._party_screen() or ag._party_menu_cb2():
            log(f"   target screen up after A x{i + 1}  cb2={hex(b.rd32(ram.GMAIN_CB2))}")
            return True
    log("!! target screen never opened")
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    b = Bridge(ROM)
    with open(SRC, "rb") as f:
        b.load_state(f.read())
    for _ in range(60):
        b.run_frame()
    b.set_input_owner("agent")
    log(f"boot in_battle={st.in_battle(b)}")
    log(f"party={party_rows(b)}")
    log(f"battler_party_idx={b.rd16(ram.GBATTLER_PARTY_IDX)}")

    ag = BattleAgent(b, on_event=lambda *a, **k: None, render=lambda: None, log=log)
    ag.owner = "agent"
    ag._reach_first_menu(__import__("time").time(), 30)
    ag._settle()
    b.set_input_owner("agent")

    if not reach_target_screen(b, ag):
        log("STAGE FAIL")
        return 1

    log("")
    log("=" * 78)
    log("ITEM-USE TARGET SCREEN IS UP — stepping the D-pad, one tap at a time")
    log("=" * 78)
    r = readers(b, ag)
    log(f"[t=0 home] {r}")
    log(f"           png={dump(b, '01_target_home')}")
    log(f"           orange rows (y, left_hits, right_hits): {orange_rows(b, ag)}")
    for slot, pts in probe_pts(b, ag).items():
        log(f"           probe slot{slot} y={10 + 24 * (slot - 1)}: {pts[0][1]}")

    keys = ["RIGHT", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "UP", "UP"]
    wide = True
    for i, key in enumerate(keys, start=1):
        lo, hi = (WLO, WHI) if wide else (0x02020000, 0x02021000)
        before = snapshot(b, lo, hi)
        ag._tap(key)
        ag._wait(18)
        after = snapshot(b, lo, hi)
        d = diff(before, after, lo)
        # Candidate cursor bytes: small values in slot range, and few of them.
        cand = [(hex(a), o, n) for a, o, n in d if n <= 7 and o <= 7 and o != n]
        r = readers(b, ag)
        log("")
        log(f"[t={i} after {key}] {r}")
        log(f"           png={dump(b, f'{i + 1:02d}_after_{key}_{i}')}")
        log(f"           changed bytes in [{hex(lo)},{hex(hi)}): {len(d)}"
            f"   slot-range candidates: {cand[:40]}")
        log(f"           orange rows: {orange_rows(b, ag)}")
        wide = False
    log("")
    log("DONE — read the PNGs in " + OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
