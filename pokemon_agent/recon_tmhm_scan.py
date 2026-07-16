"""recon_tmhm_scan.py — locate gTMHMLearnsets in the FireRed ROM by constraint signature.

The table is u64[species] (8 bytes/species), bit 0 = TM01 … bit 49 = TM50, bit 50 = HM01 (Cut)
… bit 54 = HM05 (Flash) … bit 57 = HM08. We scan the raw ROM file for a base address where a
set of ROCK-SOLID Gen-3 compat facts all hold. One hit = the table; its GBA address is
0x08000000 + file offset. Read-only recon, no emulator core needed.
"""
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")

BIT_CUT = 50      # HM01
BIT_FLY = 51      # HM02
BIT_SURF = 52     # HM03
BIT_STRENGTH = 53 # HM04
BIT_FLASH = 54    # HM05

# (species_id, bit, expected) — only facts beyond doubt (classic + this project's own verified
# teaches: Cut->Raticate worked live; Persian/meowth line is in the verified _CUT_OK).
FACTS = [
    (1,   BIT_CUT,   1),   # Bulbasaur learns Cut
    (4,   BIT_CUT,   1),   # Charmander learns Cut
    (7,   BIT_CUT,   1),   # Squirtle learns Cut
    (20,  BIT_CUT,   1),   # Raticate learns Cut (taught LIVE, badge-3 cascade)
    (52,  BIT_CUT,   1),   # Meowth learns Cut
    (21,  BIT_CUT,   0),   # Spearow does NOT
    (25,  BIT_FLASH, 1),   # Pikachu learns Flash
    (63,  BIT_FLASH, 1),   # Abra learns Flash
    (100, BIT_FLASH, 1),   # Voltorb learns Flash
    (20,  BIT_FLASH, 0),   # Raticate does NOT learn Flash
    (7,   BIT_SURF,  1),   # Squirtle learns Surf
    (25,  BIT_SURF,  0),   # Pikachu does NOT learn Surf
    (17,  BIT_FLY,   1),   # Pidgeotto learns Fly
    (0,   BIT_CUT,   0),   # species 0 row is empty
]


def main():
    data = open(ROM, "rb").read()
    n = len(data)
    # fast pre-filter fact: Bulbasaur (species 1) u64 with cut bit set, flash per scan (unknown),
    # but we can at least require the u64 to be nonzero and < 2^58.
    hits = []
    for base in range(0, n - 8 * 412, 4):
        ok = True
        for sp, bit, want in FACTS:
            off = base + 8 * sp
            v = struct.unpack_from("<Q", data, off)[0]
            if v >> 58:                      # bits above HM08 must be clear for every checked row
                ok = False
                break
            if ((v >> bit) & 1) != want:
                ok = False
                break
        if ok:
            hits.append(base)
    print(f"hits: {[hex(0x08000000 + h) for h in hits]}")
    for h in hits:
        # dump her live-relevant species for eyeballing
        for sp, nm in ((3, "venusaur"), (53, "persian"), (22, "fearow"), (20, "raticate"),
                       (23, "ekans"), (56, "mankey"), (100, "voltorb"), (25, "pikachu"),
                       (50, "diglett"), (79, "slowpoke"), (35, "clefairy")):
            v = struct.unpack_from("<Q", data, h + 8 * sp)[0]
            print(f"  {nm:10} cut={(v >> BIT_CUT) & 1} fly={(v >> BIT_FLY) & 1} "
                  f"surf={(v >> BIT_SURF) & 1} strength={(v >> BIT_STRENGTH) & 1} "
                  f"flash={(v >> BIT_FLASH) & 1}")
    return 0 if len(hits) == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
