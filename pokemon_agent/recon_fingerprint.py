"""recon_fingerprint.py - RECON ONLY (read-only): empirically confirm every world-fingerprint
field is readable from CURRENT RAM before the watchdog primitive is built. No input, no writes.

Doctrine: no claim without a control. We don't GUESS the unwired addresses (money / bag / badge
count) - we load banked states whose ground truth we KNOW and check the decode lands a sane value:
  - after_pick_bulbasaur : 0 badges, party 1   (just got the starter)
  - brock_done           : 1 badge  (Boulder), Pewter
  - misty_done           : 2 badges (Boulder, Cascade), Cerulean
If badge_count reads 0/1/2 across those three, the SaveBlock1 flag decode is proven. Money + bag
counts are XOR'd with the SaveBlock2 encryption key; we verify the DECRYPTED value is in a sane
range AND stable across frames (a wrong key/offset yields garbage or drift).

Run:  .venv\\Scripts\\python.exe pokemon_agent\\recon_fingerprint.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import firered_ram as ram          # noqa: E402
import pokemon_state as st         # noqa: E402
from bridge import Bridge          # noqa: E402

# ── CANDIDATE addresses (pokefirered disasm; UNVERIFIED until this script's controls pass) ──
SB2_ENCRYPTION_KEY = 0x0F20        # SaveBlock2.encryptionKey (u32) - XOR mask for money + item qty
SB1_MONEY          = 0x0290        # SaveBlock1.money (u32, stored money ^ key)
SB1_MAP_GROUP, SB1_MAP_NUM = 0x04, 0x05
# Bag pockets within SaveBlock1 (FRLG fixed layout). PokeBalls@0x430 is ALREADY control-verified
# (the 2026-06-27 catch fix read balls here), which anchors the whole pocket table below.
POCKETS = {                        # name: (offset, slot_count)
    "Items":     (0x0310, 42),
    "KeyItems":  (0x03B8, 30),
    "PokeBalls": (0x0430, 13),
    "TMHM":      (0x0464, 58),
    "Berries":   (0x054C, 43),
}
BADGE_FLAG_BASE = 0x820            # FLAG_BADGE01_GET .. +7; flag array @ SaveBlock1 + 0x0EE0
SB1_FLAGS_OFF   = 0x0EE0
P_LEVEL, P_HP, P_MAXHP = 0x54, 0x56, 0x58

STATES = ["after_pick_bulbasaur", "brock_done", "misty_done", "route3_caught"]
EXPECT_BADGES = {"after_pick_bulbasaur": 0, "brock_done": 1, "misty_done": 2}


def _sb(b, ptr_addr):
    p = b.rd32(ptr_addr)
    return p if ram.valid_ewram_ptr(p) else None


def badge_count(b):
    sb1 = _sb(b, ram.GSAVEBLOCK1_PTR)
    if sb1 is None:
        return None
    n = 0
    for i in range(8):
        flag = BADGE_FLAG_BASE + i
        if b.rd8(sb1 + SB1_FLAGS_OFF + (flag >> 3)) & (1 << (flag & 7)):
            n += 1
    return n


def enc_key(b):
    sb2 = _sb(b, ram.GSAVEBLOCK2_PTR)
    return b.rd32(sb2 + SB2_ENCRYPTION_KEY) if sb2 else None


def money(b):
    sb1, key = _sb(b, ram.GSAVEBLOCK1_PTR), enc_key(b)
    if sb1 is None or key is None:
        return None
    return (b.rd32(sb1 + SB1_MONEY) ^ key) & 0xFFFFFFFF


def bag_items(b):
    """[(pocket, itemId, qty)] for every non-empty slot; qty decrypted with the low-16 key."""
    sb1, key = _sb(b, ram.GSAVEBLOCK1_PTR), enc_key(b)
    if sb1 is None or key is None:
        return None
    k16 = key & 0xFFFF
    out = []
    for name, (off, cnt) in POCKETS.items():
        for s in range(cnt):
            slot = sb1 + off + s * 4
            iid = b.rd16(slot)
            if iid == 0:
                continue
            qty = b.rd16(slot + 2) ^ k16
            out.append((name, iid, qty))
    return out


def party(b):
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    out = []
    for s in range(min(cnt, 6)):
        base = ram.GPLAYER_PARTY + s * st.PARTY_MON_SIZE
        out.append((st.read_party_species(b, s),
                    b.rd8(base + P_LEVEL), b.rd16(base + P_HP), b.rd16(base + P_MAXHP)))
    return cnt, out


def main():
    rom = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
    ok = True
    for name in STATES:
        path = os.path.join(_HERE, "states", "workshop", name + ".state")
        if not os.path.exists(path):
            print(f"[{name}] MISSING state file - skip")
            continue
        b = Bridge(rom)
        with open(path, "rb") as f:
            b.load_state(f.read())
        for _ in range(20):
            b.run_frame()

        bc = badge_count(b)
        key = enc_key(b)
        m0 = money(b)
        for _ in range(30):
            b.run_frame()
        m1 = money(b)
        items = bag_items(b)
        cnt, pty = party(b)
        sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
        mp = (b.rd8(sb1 + SB1_MAP_GROUP), b.rd8(sb1 + SB1_MAP_NUM))

        print(f"\n==== {name} ====")
        print(f"  map={mp} party_count={cnt}")
        for sp, lv, hp, mx in pty:
            print(f"    {st.SPECIES_NAME.get(sp, sp)!s:>12} L{lv} HP {hp}/{mx}")
        print(f"  badge_count = {bc}   (expect {EXPECT_BADGES.get(name, '?')})")
        print(f"  enc_key = {key:#010x}" if key is not None else "  enc_key = None")
        print(f"  money(decrypted) = {m0}  (re-read +30f = {m1})")
        print(f"  bag ({len(items) if items is not None else '?'} slots): "
              + ", ".join(f"{n}:id{i}x{q}" for n, i, q in (items or [])[:14]))

        # ── controls ──
        exp = EXPECT_BADGES.get(name)
        if exp is not None and bc != exp:
            print(f"  !! BADGE CONTROL FAIL: read {bc}, expected {exp}"); ok = False
        if m0 is None or not (0 <= m0 <= 999999):
            print(f"  !! MONEY CONTROL FAIL: {m0} out of [0,999999] (wrong offset/key?)"); ok = False
        if m0 != m1:
            print(f"  !! MONEY UNSTABLE across frames: {m0} -> {m1}"); ok = False
        if items is not None:
            bad = [(n, i, q) for (n, i, q) in items if not (1 <= i <= 600 and 0 < q <= 999)]
            if bad:
                print(f"  !! BAG CONTROL FAIL (insane id/qty): {bad[:6]}"); ok = False

    print("\n==== RESULT:", "ALL CONTROLS PASS" if ok else "SOME CONTROLS FAILED", "====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
