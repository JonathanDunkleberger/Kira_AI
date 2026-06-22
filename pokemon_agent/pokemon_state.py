"""pokemon_state.py - read BATTLE state from FireRed RAM. The HANDS' eyes.

ALL battle offsets here are CANDIDATES from the pokefirered map - UNVERIFIED until
checked against a real battle savestate (M0 only confirmed overworld coords/party).
dump_battle_state() prints raw + parsed so we can verify/correct each field. Nothing
is trusted silently. Move type/power are read from the ROM's gBattleMoves table
(no shipped move table needed).
"""
import firered_ram as ram

# Gen-3 type IDs (the in-RAM type byte order) -> names. This ordering is fixed.
TYPE_BY_ID = {0: "normal", 1: "fighting", 2: "flying", 3: "poison", 4: "ground",
              5: "rock", 6: "bug", 7: "ghost", 8: "steel", 9: "???", 10: "fire",
              11: "water", 12: "grass", 13: "electric", 14: "psychic", 15: "ice",
              16: "dragon", 17: "dark"}

# ── CANDIDATE battle offsets (pokefirered) ───────────────────────────────────
GBATTLE_TYPE_FLAGS = ram.GBATTLE_TYPE_FLAGS   # u32 != 0 while in battle  (CANDIDATE)
GBATTLE_MONS       = ram.GBATTLE_MONS         # gBattleMons[4] base       (CANDIDATE)
MON_SIZE           = ram.GBATTLE_MON_SIZE     # 88                        (CANDIDATE)

# BattlePokemon field offsets within an 88-byte entry (pokefirered) - CANDIDATES:
F_SPECIES = 0x00   # u16
F_MOVES   = 0x0C   # u16[4]
F_PP      = 0x24   # u8[4]
F_TYPE1   = 0x21   # u8
F_TYPE2   = 0x22   # u8
F_HP      = 0x28   # u16
F_MAXHP   = 0x2A   # u16
F_LEVEL   = 0x2C   # u8

# gBattleMoves table in ROM (struct = 12 bytes: effect, power, type, accuracy, pp, ...)
GBATTLE_MOVES = 0x08250C04   # ROM addr (CANDIDATE)
MOVE_SIZE     = 12
MOVE_F_POWER  = 0x01         # u8
MOVE_F_TYPE   = 0x02         # u8


# ── Party species (Gen-3 encrypted substructures) ───────────────────────────
PARTY_MON_SIZE = 100
# substructure order by personality%24 (G=Growth, A=Attacks, E=EVs, M=Misc)
_SUBSTRUCT_ORDER = ["GAEM", "GAME", "GEAM", "GEMA", "GMAE", "GMEA",
                    "AGEM", "AGME", "AEGM", "AEMG", "AMGE", "AMEG",
                    "EGAM", "EGMA", "EAGM", "EAMG", "EMGA", "EMAG",
                    "MGAE", "MGEA", "MAGE", "MAEG", "MEGA", "MEAG"]
# FireRed internal species == National Dex for #1-251; the three starters:
SPECIES_NAME = {1: "bulbasaur", 4: "charmander", 7: "squirtle"}
STARTER_SPECIES = {v: k for k, v in SPECIES_NAME.items()}


def read_party_species(bridge, slot=0):
    """Decrypt the species of party slot N. CANDIDATE (depends on GPLAYER_PARTY).
    Species lives in the encrypted Growth substructure: XOR key = PID ^ OTID."""
    base = ram.GPLAYER_PARTY + slot * PARTY_MON_SIZE
    pid = bridge.rd32(base + 0)
    otid = bridge.rd32(base + 4)
    key = pid ^ otid
    order = _SUBSTRUCT_ORDER[pid % 24]
    growth_addr = base + 32 + order.index("G") * 12
    species = (bridge.rd32(growth_addr) ^ key) & 0xFFFF
    return species


def in_battle(bridge) -> bool:
    return bridge.rd32(GBATTLE_TYPE_FLAGS) != 0


def _type_name(b):
    return TYPE_BY_ID.get(b, f"id{b}")


def move_info(bridge, move_id):
    """(type_name, power) for a move id, read from ROM gBattleMoves. CANDIDATE offsets."""
    if move_id == 0:
        return (None, 0)
    base = GBATTLE_MOVES + move_id * MOVE_SIZE
    power = bridge.rd8(base + MOVE_F_POWER)
    tname = _type_name(bridge.rd8(base + MOVE_F_TYPE))
    return (tname, power)


def read_mon(bridge, index):
    """Parse one gBattleMons entry (0 = our active, 1 = enemy active in singles)."""
    base = GBATTLE_MONS + index * MON_SIZE
    species = bridge.rd16(base + F_SPECIES)
    hp = bridge.rd16(base + F_HP)
    maxhp = bridge.rd16(base + F_MAXHP)
    t1 = _type_name(bridge.rd8(base + F_TYPE1))
    t2 = _type_name(bridge.rd8(base + F_TYPE2))
    moves = []
    for i in range(4):
        mid = bridge.rd16(base + F_MOVES + i * 2)
        mt, mp = move_info(bridge, mid)
        moves.append({"id": mid, "name": f"move#{mid}", "type": mt or "normal",
                      "power": mp, "pp": bridge.rd8(base + F_PP + i)})
    types = [t for t in (t1, t2) if t and t != "normal" or t == t1]
    return {"species": species, "hp": hp, "maxhp": maxhp, "level": bridge.rd8(base + F_LEVEL),
            "types": [t1, t2], "moves": moves}


def read_battle(bridge):
    """Full battle snapshot or None if not in battle. CANDIDATE - verify with dump."""
    if not in_battle(bridge):
        return None
    return {"ours": read_mon(bridge, 0), "enemy": read_mon(bridge, 1)}


def dump_battle_state(bridge):
    """Loud verification dump - run this against a real battle savestate to confirm
    every CANDIDATE offset before trusting read_battle()."""
    print(f"   [PkState] in_battle(flags@{hex(GBATTLE_TYPE_FLAGS)}) = {in_battle(bridge)} "
          f"(raw={hex(bridge.rd32(GBATTLE_TYPE_FLAGS))})")
    for idx, who in ((0, "OURS"), (1, "ENEMY")):
        m = read_mon(bridge, idx)
        print(f"   [PkState] {who}: species={m['species']} lvl={m['level']} "
              f"hp={m['hp']}/{m['maxhp']} types={m['types']}")
        for mv in m["moves"]:
            print(f"             move id={mv['id']} type={mv['type']} pow={mv['power']} pp={mv['pp']}")
    print("   [PkState] ^ VERIFY these against what the battle screen shows; "
          "fix offsets in pokemon_state.py if wrong.")
