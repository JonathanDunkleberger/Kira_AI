"""recon_boxscan.py — dump PC box occupants of a savestate (NS#28 Lapras-in-box check)."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge
import firered_ram as ram
import pokemon_state as st
import hm_teach as ht
ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
path = sys.argv[1]
b = Bridge(ROM)
with open(path, "rb") as f: b.load_state(f.read())
for _ in range(8): b.run_frame()
n = ram.read_party_count(b)
print(f"PARTY ({n}):")
for s in range(n):
    sp = st.read_party_species(b, s)
    print(f"  slot{s} {st.SPECIES_NAME.get(sp,sp)} (id {sp}) surf={ht.hm_compatible(b,'surf',sp)}")
GSTORAGE_PTR, BOX_MON_SIZE, BOXES, PER_BOX = 0x03005010, 80, 14, 30
base0 = b.rd32(GSTORAGE_PTR); cur_box = b.rd8(base0) if base0 else -1
print(f"BOX (open box={cur_box}):")
found=0
for bx in range(BOXES):
    for sl in range(PER_BOX):
        mbase = base0 + 4 + (bx*PER_BOX+sl)*BOX_MON_SIZE
        pid = b.rd32(mbase)
        if pid==0 and b.rd32(mbase+4)==0: continue
        key = pid ^ b.rd32(mbase+4)
        order = st._SUBSTRUCT_ORDER[pid%24]
        sp = (b.rd32(mbase+32+order.index("G")*12) ^ key) & 0xFFFF
        if 1<=sp<=411:
            found+=1
            print(f"  box{bx} slot{sl} {st.SPECIES_NAME.get(sp,sp)} (id {sp}) surf={ht.hm_compatible(b,'surf',sp)}")
if not found: print("  (empty)")
# HM03 in case?
print(f"HM03 in TM case: {ht.tm_case_row(b, ht.HM_ITEM['surf']) is not None}")
