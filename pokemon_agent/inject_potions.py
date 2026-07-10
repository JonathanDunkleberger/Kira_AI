"""inject_potions.py — SHIFT-10 EXPERIMENT (throwaway): stuff Super/Hyper Potions + a couple Revives
into the Items pocket of a boot state, save it as a new workshop fixture, so recon_longrun can answer
the ONE question gating Plan B: does Venusaur + potions actually stall-and-win Koga, and does in-battle
potion actuation fire on the long core?  Reads/writes SaveBlock1 Items pocket directly in RAM.

RUN: python pokemon_agent/inject_potions.py <src_fixture> <dst_fixture>
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bridge import Bridge
import firered_ram as ram
from campaign import resolve_state, STATES_WORKSHOP

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
ITEMS_POCKET_OFF = 0x0310     # SaveBlock1 Items pocket, 42 slots x 4 bytes (id u16 + qty^key u16)
N_SLOTS = 42

# (item_id, qty) to ensure present. Super Potion=22, Hyper Potion=21, Revive=24, Antidote=11.
GIVE = [(21, 20), (22, 20), (24, 5), (11, 10)]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "fuchsia_gate.state"
    dst = sys.argv[2] if len(sys.argv) > 2 else "fuchsia_potions.state"
    if not dst.endswith(".state"):
        dst += ".state"

    b = Bridge(ROM)
    with open(resolve_state(src), "rb") as f:
        b.load_state(f.read())
    for _ in range(20):
        b.run_frame()

    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
    mem = b.core.memory

    # snapshot current pocket
    def read_pocket():
        out = []
        for s in range(N_SLOTS):
            slot = sb1 + ITEMS_POCKET_OFF + s * 4
            iid = b.rd16(slot)
            qty = b.rd16(slot + 2) ^ key if iid else 0
            out.append((s, iid, qty))
        return out

    print("BEFORE items pocket:")
    for s, iid, qty in read_pocket():
        if iid:
            print(f"  slot{s}: id={iid} qty={qty}")

    # write each GIVE into an existing slot for that id (bump qty) or the first empty slot
    pocket = read_pocket()
    used = {iid: s for s, iid, qty in pocket if iid}
    empties = [s for s, iid, qty in pocket if not iid]
    for iid, qty in GIVE:
        if iid in used:
            slot = sb1 + ITEMS_POCKET_OFF + used[iid] * 4
            cur = b.rd16(slot + 2) ^ key
            newq = min(cur + qty, 99)
            mem.u16.raw_write(slot + 2, (newq ^ key) & 0xFFFF)
            print(f"  bump id={iid} slot{used[iid]} {cur}->{newq}")
        elif empties:
            s = empties.pop(0)
            slot = sb1 + ITEMS_POCKET_OFF + s * 4
            mem.u16.raw_write(slot, iid & 0xFFFF)
            mem.u16.raw_write(slot + 2, (qty ^ key) & 0xFFFF)
            print(f"  set id={iid} slot{s} qty={qty}")
        else:
            print(f"  !! no empty slot for id={iid} — skipped")

    print("AFTER items pocket:")
    for s, iid, qty in read_pocket():
        if iid:
            print(f"  slot{s}: id={iid} qty={qty}")

    data = b.save_state()
    out_path = os.path.join(STATES_WORKSHOP, dst)
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"WROTE {out_path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
