"""firered_ram.py - FireRed (USA, AGB-BPRE) RAM offset CANDIDATES.

These are from the community / pokefirered disassembly map. They are CANDIDATES
until empirically confirmed (M0 confirms party-count + coords by watching them
change). Nothing here is trusted silently - callers print CANDIDATE vs CONFIRMED.

DMA note: FireRed shuffles SaveBlock1's location, so player coords are NOT at a
fixed address. The pointer at gSaveBlock1Ptr is fixed and always points to the
current SaveBlock1; coords are read as *(SaveBlock1Ptr) + offset.
"""

# ── Fixed-address candidates (EWRAM/IWRAM) ───────────────────────────────────
GSAVEBLOCK1_PTR   = 0x03005008   # pointer -> current SaveBlock1 (DMA-shuffled target)
GSAVEBLOCK2_PTR   = 0x0300500C   # pointer -> SaveBlock2
GPLAYER_PARTY_CNT = 0x02024029   # u8: number of Pokémon in the player's party (0..6)
GPLAYER_PARTY     = 0x02024284   # start of party data (100 bytes / mon)

# ── Offsets WITHIN SaveBlock1 (add to the dereferenced pointer) ───────────────
SB1_OFF_POS_X     = 0x0000       # s16 player map X
SB1_OFF_POS_Y     = 0x0002       # s16 player map Y

# ── Battle (CANDIDATES - confirmed in M1, not M0) ────────────────────────────
# gBattleMainFunc != 0 while a battle is running is a common in-battle signal;
# gBattleTypeFlags / gBattleMons hold the active battle data. Confirm in M1.
GBATTLE_TYPE_FLAGS = 0x02022B4C  # u32 battle type bitflags (0 when not in battle) - CANDIDATE
GBATTLE_MONS       = 0x02024084  # gBattleMons[4] (88 bytes each) - CANDIDATE
GBATTLE_MON_SIZE   = 88

EWRAM_LO, EWRAM_HI = 0x02000000, 0x02040000


def valid_ewram_ptr(p: int) -> bool:
    return EWRAM_LO <= p < EWRAM_HI


def read_player_coords(bridge):
    """Returns (x, y) or None if SaveBlock1 isn't allocated yet (pre-game)."""
    sb1 = bridge.rd32(GSAVEBLOCK1_PTR)
    if not valid_ewram_ptr(sb1):
        return None
    x = bridge.rds16(sb1 + SB1_OFF_POS_X)
    y = bridge.rds16(sb1 + SB1_OFF_POS_Y)
    return (x, y)


def read_party_count(bridge):
    return bridge.rd8(GPLAYER_PARTY_CNT)
