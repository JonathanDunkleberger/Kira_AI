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

# BattlePokemon field offsets within an 88-byte entry - ✅ VERIFIED 2026-06-22 vs
# battle.state (Bulbasaur L6 21/21 Tackle/Growl grass-poison vs Pidgey L3 normal-fly).
# NOTE: level is a u8 @0x2A and maxHP a u16 @0x2C (the banked code had these swapped).
F_SPECIES = 0x00   # u16  ✅
F_MOVES   = 0x0C   # u16[4]  ✅ (Tackle=33, Growl=45)
F_PP      = 0x24   # u8[4]  ✅ (35, 40)
F_TYPE1   = 0x21   # u8  ✅ (grass=12)
F_TYPE2   = 0x22   # u8  ✅ (poison=3)
F_HP      = 0x28   # u16  ✅ (21)
F_LEVEL   = 0x2A   # u8   ✅ (6)  - was 0x2C (WRONG)
F_MAXHP   = 0x2C   # u16  ✅ (21) - was 0x2A (WRONG)
F_STATUS  = 0x4C   # u32  status1 (sleep bits 0-2, poison 0x08, burn 0x10, frz 0x20, par 0x40)

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
# FireRed internal species == National Dex for #1-251. Names for what she'll meet on
# the road to Brock (so events use real names, never an index).
# FULL Kanto dex 1-151 (2026-07-06: the 35-entry table made her SAY "species#52" about her own new
# Meowth — a voice lie — and wrote the bond as "a new Pokemon". Names feed voice, bonds AND the
# catch-judgment reasons, so the table must be complete.)
SPECIES_NAME = {
    1: "bulbasaur", 2: "ivysaur", 3: "venusaur", 4: "charmander", 5: "charmeleon",
    6: "charizard", 7: "squirtle", 8: "wartortle", 9: "blastoise", 10: "caterpie",
    11: "metapod", 12: "butterfree", 13: "weedle", 14: "kakuna", 15: "beedrill",
    16: "pidgey", 17: "pidgeotto", 18: "pidgeot", 19: "rattata", 20: "raticate",
    21: "spearow", 22: "fearow", 23: "ekans", 24: "arbok", 25: "pikachu",
    26: "raichu", 27: "sandshrew", 28: "sandslash", 29: "nidoran", 30: "nidorina",
    31: "nidoqueen", 32: "nidoran", 33: "nidorino", 34: "nidoking", 35: "clefairy",
    36: "clefable", 37: "vulpix", 38: "ninetales", 39: "jigglypuff", 40: "wigglytuff",
    41: "zubat", 42: "golbat", 43: "oddish", 44: "gloom", 45: "vileplume",
    46: "paras", 47: "parasect", 48: "venonat", 49: "venomoth", 50: "diglett",
    51: "dugtrio", 52: "meowth", 53: "persian", 54: "psyduck", 55: "golduck",
    56: "mankey", 57: "primeape", 58: "growlithe", 59: "arcanine", 60: "poliwag",
    61: "poliwhirl", 62: "poliwrath", 63: "abra", 64: "kadabra", 65: "alakazam",
    66: "machop", 67: "machoke", 68: "machamp", 69: "bellsprout", 70: "weepinbell",
    71: "victreebel", 72: "tentacool", 73: "tentacruel", 74: "geodude", 75: "graveler",
    76: "golem", 77: "ponyta", 78: "rapidash", 79: "slowpoke", 80: "slowbro",
    81: "magnemite", 82: "magneton", 83: "farfetch'd", 84: "doduo", 85: "dodrio",
    86: "seel", 87: "dewgong", 88: "grimer", 89: "muk", 90: "shellder",
    91: "cloyster", 92: "gastly", 93: "haunter", 94: "gengar", 95: "onix",
    96: "drowzee", 97: "hypno", 98: "krabby", 99: "kingler", 100: "voltorb",
    101: "electrode", 102: "exeggcute", 103: "exeggutor", 104: "cubone", 105: "marowak",
    106: "hitmonlee", 107: "hitmonchan", 108: "lickitung", 109: "koffing", 110: "weezing",
    111: "rhyhorn", 112: "rhydon", 113: "chansey", 114: "tangela", 115: "kangaskhan",
    116: "horsea", 117: "seadra", 118: "goldeen", 119: "seaking", 120: "staryu",
    121: "starmie", 122: "mr. mime", 123: "scyther", 124: "jynx", 125: "electabuzz",
    126: "magmar", 127: "pinsir", 128: "tauros", 129: "magikarp", 130: "gyarados",
    131: "lapras", 132: "ditto", 133: "eevee", 134: "vaporeon", 135: "jolteon",
    136: "flareon", 137: "porygon", 138: "omanyte", 139: "omastar", 140: "kabuto",
    141: "kabutops", 142: "aerodactyl", 143: "snorlax", 144: "articuno", 145: "zapdos",
    146: "moltres", 147: "dratini", 148: "dragonair", 149: "dragonite", 150: "mewtwo",
    151: "mew",
}
STARTER_SPECIES = {"bulbasaur": 1, "charmander": 4, "squirtle": 7}   # the 3 starters only

# HUD — species_id -> Gen-3 type(s). FireRed-era typings (Steel exists: Magnemite line is Electric/
# Steel; NO Fairy yet: Clefairy/Jigglypuff/Mr.Mime stay Normal/Psychic). Kanto #1-151, the full
# realistic playthrough roster. Used by the stream HUD for per-mon type badges (color-coded).
SPECIES_TYPES = {
    1: ["grass", "poison"], 2: ["grass", "poison"], 3: ["grass", "poison"], 4: ["fire"],
    5: ["fire"], 6: ["fire", "flying"], 7: ["water"], 8: ["water"], 9: ["water"], 10: ["bug"],
    11: ["bug"], 12: ["bug", "flying"], 13: ["bug", "poison"], 14: ["bug", "poison"],
    15: ["bug", "poison"], 16: ["normal", "flying"], 17: ["normal", "flying"], 18: ["normal", "flying"],
    19: ["normal"], 20: ["normal"], 21: ["normal", "flying"], 22: ["normal", "flying"], 23: ["poison"],
    24: ["poison"], 25: ["electric"], 26: ["electric"], 27: ["ground"], 28: ["ground"], 29: ["poison"],
    30: ["poison"], 31: ["poison", "ground"], 32: ["poison"], 33: ["poison"], 34: ["poison", "ground"],
    35: ["normal"], 36: ["normal"], 37: ["fire"], 38: ["fire"], 39: ["normal"], 40: ["normal"],
    41: ["poison", "flying"], 42: ["poison", "flying"], 43: ["grass", "poison"], 44: ["grass", "poison"],
    45: ["grass", "poison"], 46: ["bug", "grass"], 47: ["bug", "grass"], 48: ["bug", "poison"],
    49: ["bug", "poison"], 50: ["ground"], 51: ["ground"], 52: ["normal"], 53: ["normal"],
    54: ["water"], 55: ["water"], 56: ["fighting"], 57: ["fighting"], 58: ["fire"], 59: ["fire"],
    60: ["water"], 61: ["water"], 62: ["water", "fighting"], 63: ["psychic"], 64: ["psychic"],
    65: ["psychic"], 66: ["fighting"], 67: ["fighting"], 68: ["fighting"], 69: ["grass", "poison"],
    70: ["grass", "poison"], 71: ["grass", "poison"], 72: ["water", "poison"], 73: ["water", "poison"],
    74: ["rock", "ground"], 75: ["rock", "ground"], 76: ["rock", "ground"], 77: ["fire"], 78: ["fire"],
    79: ["water", "psychic"], 80: ["water", "psychic"], 81: ["electric", "steel"],
    82: ["electric", "steel"], 83: ["normal", "flying"], 84: ["normal", "flying"],
    85: ["normal", "flying"], 86: ["water"], 87: ["water", "ice"], 88: ["poison"], 89: ["poison"],
    90: ["water"], 91: ["water", "ice"], 92: ["ghost", "poison"], 93: ["ghost", "poison"],
    94: ["ghost", "poison"], 95: ["rock", "ground"], 96: ["psychic"], 97: ["psychic"], 98: ["water"],
    99: ["water"], 100: ["electric"], 101: ["electric"], 102: ["grass", "psychic"],
    103: ["grass", "psychic"], 104: ["ground"], 105: ["ground"], 106: ["fighting"], 107: ["fighting"],
    108: ["normal"], 109: ["poison"], 110: ["poison"], 111: ["ground", "rock"], 112: ["ground", "rock"],
    113: ["normal"], 114: ["grass"], 115: ["normal"], 116: ["water"], 117: ["water"], 118: ["water"],
    119: ["water"], 120: ["water"], 121: ["water", "psychic"], 122: ["psychic"], 123: ["bug", "flying"],
    124: ["ice", "psychic"], 125: ["electric"], 126: ["fire"], 127: ["bug"], 128: ["normal"],
    129: ["water"], 130: ["water", "flying"], 131: ["water", "ice"], 132: ["normal"], 133: ["normal"],
    134: ["water"], 135: ["electric"], 136: ["fire"], 137: ["normal"], 138: ["rock", "water"],
    139: ["rock", "water"], 140: ["rock", "water"], 141: ["rock", "water"], 142: ["rock", "flying"],
    143: ["normal"], 144: ["ice", "flying"], 145: ["electric", "flying"], 146: ["fire", "flying"],
    147: ["dragon"], 148: ["dragon"], 149: ["dragon", "flying"], 150: ["psychic"], 151: ["psychic"],
}


def species_types(species_id):
    """Gen-3 type list for a species id (1-2 entries), or [] if unknown. For the HUD's type badges."""
    return list(SPECIES_TYPES.get(int(species_id) if species_id else 0, []))

# Move id -> name (so events say "Tackle", never "move#33"). Covers the starters'
# movepools + common wild moves + Metal Claw (the Brock answer).
MOVE_NAMES = {
    0: "nothing", 10: "Scratch", 33: "Tackle", 45: "Growl", 39: "Tail Whip",
    43: "Leer", 28: "Sand-Attack", 98: "Quick Attack", 64: "Peck", 16: "Gust",
    52: "Ember", 108: "Smokescreen", 82: "Dragon Rage", 232: "Metal Claw",
    22: "Vine Whip", 74: "Growth", 73: "Leech Seed", 77: "PoisonPowder",
    79: "Sleep Powder", 75: "Razor Leaf", 55: "Water Gun", 145: "Bubble",
    44: "Bite", 99: "Rage", 84: "Thunder Shock", 88: "Rock Throw", 111: "Defense Curl",
}


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


def read_enemy_species(bridge, slot=0):
    """Decrypt the species of ENEMY party slot N (gEnemyParty). Same struct/encryption as the player
    party — used to recognize a rival/Gary fight (his starter-line ace) at ANY encounter, not just by
    the active lead. Returns 0 on any read error (never raises)."""
    try:
        base = ram.GENEMY_PARTY + slot * PARTY_MON_SIZE
        pid = bridge.rd32(base + 0)
        otid = bridge.rd32(base + 4)
        if pid == 0 and otid == 0:
            return 0
        key = pid ^ otid
        order = _SUBSTRUCT_ORDER[pid % 24]
        growth_addr = base + 32 + order.index("G") * 12
        return (bridge.rd32(growth_addr) ^ key) & 0xFFFF
    except Exception:
        return 0


def read_enemy_level(bridge, slot=0):
    """Plaintext level of ENEMY party slot N (gEnemyParty base +0x54 — the unencrypted
    battle-stats block, same layout as the player party). 0 on any read error. Pairs with
    read_enemy_species as the AUTHORITATIVE at-encounter foe read: gEnemyParty is written
    when the wild mon is CREATED (before the battle intro even fades in), while
    gBattleMons[1] lags a beat and can still hold the PREVIOUS battle's foe."""
    try:
        return bridge.rd8(ram.GENEMY_PARTY + slot * PARTY_MON_SIZE + 0x54)
    except Exception:
        return 0


def read_party_moves(bridge, slot=0):
    """Decrypt the 4 move IDs of party slot N. Sibling of read_party_species, but reads the
    ATTACKS substructure instead of Growth (same XOR key = PID ^ OTID, same personality%24 order).
    Moves are stored ONLY here (encrypted) — there is no plaintext move cache in the party struct
    (sourced: pret/pokefirered struct PokemonSubstruct1 { u16 moves[4]; u8 pp[4]; }). Returns a
    4-int list (0 = empty slot). This is the read that answers 'does this mon know an HM?'."""
    base = ram.GPLAYER_PARTY + slot * PARTY_MON_SIZE
    pid = bridge.rd32(base + 0)
    otid = bridge.rd32(base + 4)
    key = pid ^ otid
    order = _SUBSTRUCT_ORDER[pid % 24]
    a = base + 32 + order.index("A") * 12     # Attacks substructure (12 bytes = 3 u32 words)
    w0 = bridge.rd32(a + 0) ^ key             # moves[0] (low u16), moves[1] (high u16)
    w1 = bridge.rd32(a + 4) ^ key             # moves[2] (low u16), moves[3] (high u16)
    return [w0 & 0xFFFF, (w0 >> 16) & 0xFFFF, w1 & 0xFFFF, (w1 >> 16) & 0xFFFF]


def party_knows_move(bridge, move_id, count=6):
    """Which party slot (0-based) knows move_id, or None. Scans up to `count` members.
    Defensive: a decrypt/read error on one slot is skipped, never raised (a bad read must
    not crash the field-move check). count should be the live party count when known."""
    for s in range(min(count, 6)):
        try:
            if move_id in read_party_moves(bridge, s):
                return s
        except Exception:
            continue
    return None


def is_shiny(pid, otid):
    """Gen-3 shininess: shiny iff (TID ^ SID ^ pidHi ^ pidLo) < 8, where otId packs SID<<16 | TID.
    Pure function (testable without a bridge)."""
    return (((otid & 0xFFFF) ^ (otid >> 16) ^ (pid & 0xFFFF) ^ (pid >> 16)) & 0xFFFF) < 8


def enemy_is_shiny(bridge, slot=0):
    """Is the enemy party mon at `slot` SHINY? Reads PID/otId from gEnemyParty (CONFIRMED base; PID@+0,
    otId@+4 are unencrypted). slot 0 = the active wild foe / a trainer's lead. Defensive: any read
    error -> False (never a false shiny freak-out). The most clippable moment the game can produce, so
    this is read source-first off verified addresses, not guessed."""
    try:
        base = ram.GENEMY_PARTY + slot * PARTY_MON_SIZE
        pid = bridge.rd32(base + 0)
        otid = bridge.rd32(base + 4)
        if pid == 0 and otid == 0:
            return False
        return is_shiny(pid, otid)
    except Exception:
        return False


def in_battle(bridge) -> bool:
    """✅ VERIFIED gate. gBattleTypeFlags is STALE out of battle (false-positive), so
    we gate on the battle-RESOURCES pointer (valid EWRAM addr only during battle) AND
    a sanity check on gBattleMons[0]. True for battle.state, False for every overworld
    state tested (incl. immediately post-rival-battle)."""
    if not ram.valid_ewram_ptr(bridge.rd32(ram.GBATTLE_RES_PTR)):
        return False
    base = GBATTLE_MONS
    sp = bridge.rd16(base + F_SPECIES)
    mhp = bridge.rd16(base + F_MAXHP)
    lvl = bridge.rd8(base + F_LEVEL)
    hp = bridge.rd16(base + F_HP)
    return 1 <= sp <= 411 and 0 < mhp <= 999 and 1 <= lvl <= 100 and hp <= mhp


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
        moves.append({"id": mid, "name": MOVE_NAMES.get(mid, f"move#{mid}"),
                      "type": mt or "normal", "power": mp, "pp": bridge.rd8(base + F_PP + i)})
    types = [t for t in (t1, t2) if t and t != "normal" or t == t1]
    # status1 (u32 @ 0x4C in BattlePokemon): sleep = bits 0-2 (turn counter), poison 0x08, burn 0x10,
    # freeze 0x20, paralysis 0x40, bad-poison 0x80. Exposed so the engine can sleep-LOCK a foe (re-apply
    # only when it's awake) instead of wasting a turn re-sleeping an already-asleep mon.
    status1 = bridge.rd32(base + F_STATUS)
    return {"species": species, "hp": hp, "maxhp": maxhp, "level": bridge.rd8(base + F_LEVEL),
            "types": [t1, t2], "moves": moves, "status1": status1,
            "asleep": bool(status1 & 0x07)}


def read_battle(bridge):
    """Full battle snapshot or None if not in battle. CANDIDATE - verify with dump."""
    if not in_battle(bridge):
        return None
    return {"ours": read_mon(bridge, 0), "enemy": read_mon(bridge, 1)}


def dump_battle_state(bridge):
    """Loud verification dump - run this against a real battle savestate to confirm
    every CANDIDATE offset before trusting read_battle()."""
    print(f"   [PkState] in_battle(res_ptr@{hex(ram.GBATTLE_RES_PTR)}) = {in_battle(bridge)} "
          f"(ptr={hex(bridge.rd32(ram.GBATTLE_RES_PTR))})")
    for idx, who in ((0, "OURS"), (1, "ENEMY")):
        m = read_mon(bridge, idx)
        print(f"   [PkState] {who}: species={m['species']} lvl={m['level']} "
              f"hp={m['hp']}/{m['maxhp']} types={m['types']}")
        for mv in m["moves"]:
            print(f"             move id={mv['id']} type={mv['type']} pow={mv['power']} pp={mv['pp']}")
    print("   [PkState] ^ VERIFY these against what the battle screen shows; "
          "fix offsets in pokemon_state.py if wrong.")
