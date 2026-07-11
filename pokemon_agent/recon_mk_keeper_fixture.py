"""recon_mk_keeper_fixture.py — build a CLEAN keeper-router proof fixture from bill_done:
she stands in Bill's house (post-Nugget-Bridge, NO gauntlet between her and Route 25 grass), party
dropped 4->3 (removes the Abra she already holds so the plan re-wants Abra), and ~15 Poke Balls injected.
The router then routes Bill's house -> Route 25 (1 warp hop) and catches an Abra. Read bill_done, write
states/workshop/bill_house_noabra.state. Run: ../.venv/Scripts/python.exe recon_mk_keeper_fixture.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bridge import Bridge
import firered_ram as ram
import pokemon_state as st
from campaign import resolve_state

ROM = os.path.join(os.path.dirname(_HERE), "roms", "firered.gba")
OUT = os.path.join("states", "workshop", "bill_house_noabra.state")
N_BALLS = 15
ITEM_POKE_BALL = 4


def main():
    b = Bridge(ROM)
    with open(resolve_state("bill_done.state"), "rb") as f:
        b.load_state(f.read())
    for _ in range(30):
        b.run_frame()

    cnt0 = b.rd8(ram.GPLAYER_PARTY_CNT)
    print("party count before:", cnt0)
    # DROP the last slot (Abra @ slot3) by decrementing the count. Party data past the counted slots is
    # ignored by the game; the savestate captures live RAM, no checksum on the in-RAM party count.
    b.core.memory.u8.raw_write(ram.GPLAYER_PARTY_CNT, cnt0 - 1)
    print("party count after:", b.rd8(ram.GPLAYER_PARTY_CNT))

    # inject Poke Balls into the BALLS pocket (SaveBlock1+0x430, 16 slots; qty XOR the low-16 key) —
    # NOT the items pocket; _balls_pocket_count/catch only see the balls pocket.
    sb1 = b.rd32(ram.GSAVEBLOCK1_PTR)
    key = b.rd32(b.rd32(ram.GSAVEBLOCK2_PTR) + 0xF20) & 0xFFFF
    placed = False
    for s in range(16):
        slot = sb1 + 0x0430 + s * 4
        iid = b.rd16(slot)
        dec_qty = (b.rd16(slot + 2) ^ key) & 0xFFFF
        if iid == ITEM_POKE_BALL:                      # top up an existing stack
            b.core.memory.u16.raw_write(slot + 2, (max(dec_qty, N_BALLS)) ^ key)
            placed = True
            print(f"topped Poke Balls to {max(dec_qty, N_BALLS)} at item slot {s}")
            break
        if iid == 0:                                   # empty slot -> new stack
            b.core.memory.u16.raw_write(slot, ITEM_POKE_BALL)
            b.core.memory.u16.raw_write(slot + 2, N_BALLS ^ key)
            placed = True
            print(f"placed {N_BALLS} Poke Balls at empty item slot {s}")
            break
    if not placed:
        print("!! could not place Poke Balls (items pocket full?)")

    # verify party
    for si in range(b.rd8(ram.GPLAYER_PARTY_CNT)):
        sp = st.read_party_species(b, si)
        print(f"  slot{si}: species {st.SPECIES_NAME.get(sp, sp)} L{b.rd8(ram.GPLAYER_PARTY + si*st.PARTY_MON_SIZE + 0x54)}")

    data = b.save_state()
    with open(OUT, "wb") as f:
        f.write(data)
    print("wrote", OUT, len(data), "bytes")


if __name__ == "__main__":
    main()
