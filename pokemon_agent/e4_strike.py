"""e4_strike.py — THE LAST VEHICLE, in-loop: League mart stock-up + Elite Four + Champion -> CREDITS.

A FAITHFUL port of the proven recon_e4.py vehicle into an in-loop module driven by the live `camp` bridge,
so the endgame push can call it as ONE decision (the same shape as victory_road / giovanni_gym / blaine_gym).
Entering the Hall of Fame is the point of no return: the game auto-saves and the credits roll.

Ground truth (pret; recon_e4's champion-clear-proven constants, verbatim):
- IndigoPlateau exterior (3,9): the League center door is the only warp -> center (13,0).
- League center: clerk stand (2,7) FACE LEFT to shop; nurse heals; League door (4,1) -> Lorelei's room.
- Mart rows (0-based): 2 FULL RESTORE(19), 4 REVIVE(24), 5 FULL HEAL(23) — the true-index engine
  (row+scroll @0x02039940/42), each unit verified by money drop + bag qty (camp._mart_buy_one).
- E4 rooms are ONE TEMPLATE: arrive south warp, trainer talk-triggered, north door OPENS on the
  FLAG_DEFEATED_* the win sets. Chain: Lorelei -> Bruno -> Agatha -> Lance -> Champion (Gary) ->
  HALL OF FAME -> CREDITS. Whiteout = respawn at THIS center (last heal); DEFEATED flags persist so
  cleared rooms pass straight through -> the dispatch loop re-heals/re-shops and re-enters.

WHITEOUT-TOLERANT + resume-safe: the loop keys on the CURRENT map each iteration (exterior/center/room/
HoF) so a mid-gauntlet whiteout costs a re-lap of the UNCLEARED rooms, never solved ground; a resume from
disk anywhere on the League maps picks up in place. Battles run through camp.battle_runner() (the same
battle-brain that cracked the E4 wall on the champion climb — commit 23487e7: never sleep-lock a 2x-SE foe;
field the super-effective specialist), and the item instinct takes the offered heal/cure/revive.

run_strike returns:
  'credits'     — the Hall of Fame reached; auto-save fired, the credits are rolling. (THE SUMMIT.)
  'battle_loss' — a persistent whiteout the caller's recovery should own (rare; the loop self-recovers most).
  'stuck'       — the deadline hit without the Hall of Fame (a real wall — usually team-depth: a thin team
                  can't out-attrition Lance/Gary). Surfaces LOUD so the caller can grind + retry.
"""
import json
import os
import time

import travel as tv
import firered_ram as ram
import field_moves as fm
from dialogue_drive import box_open as dd_box

# ── FireRed Indigo-Plateau / League fact table (game-knowledge layer; rule 14 portability debt) ─────────
INDIGO_EXT = (3, 9)                       # the Indigo Plateau exterior
LEAGUE_CENTER = (13, 0)                   # the League Pokémon Center (KB ground truth, learned live e4_run3)
CLERK_STAND = (2, 7)                      # stand here, FACE LEFT, A to shop
LEAGUE_DOOR = (4, 1)                      # the center's door into Lorelei's room
FLAG_BADGE_EARTH = 0x827                  # badge 8 — the strike's preflight guard
FULL_RESTORE, MAX_POTION, REVIVE, FULL_HEAL = 19, 20, 24, 23
# (item id, mart true-index row, want, unit price) — FR-first tuning (recon_e4 runs 5-9 postmortem: every
# lap died at Lance with FR x0; 4 FR tanks the 5-dragon wave, 2 Revive for the type-answer comeback, 1 Full
# Heal for the Jynx/Hypnosis sleep). The stock_up scaler clamps each line to money (poverty-safe order).
# WANTS RAISED 4/2/1 -> 6/3/2 (2026-08-04, Jonny: 'OP and steamrolling'): the old wants were sized for a
# ~$13k pauper arrival; with mart loot-selling + the badge 6-8 payouts she arrives rich, and the postmortem
# failure mode was ALWAYS an empty kit at Lance/Gary, never an over-full bag. The comeback FLOOR pass below
# keeps the poverty order identical when money is short — extra wants only ever spend SURPLUS.
SHOPPING = [(FULL_RESTORE, 2, 8, 3000), (REVIVE, 4, 8, 1500), (FULL_HEAL, 5, 2, 600)]
# Jonny 2026-08-14: "she needs more revives" — buy a STACK at the Center
# BEFORE Lorelei. Live 11:38: Revive x1 FR x0 in Bruno's Room, then
# restock-south walked into CloseEntry's sealed door at (6,10) forever.
# pret PokemonLeague_EventScript_CloseEntry sets (5-7,11-12) collision=1
# on room entry. South is NEVER a shop path mid-gauntlet. Shop here;
# between-room bag Revives are the only mid-gauntlet recover.
# Jonny 12:11: leave the Center with Revive ≥6 and FR ≥2. CloseEntry
# seals south forever after Lorelei — this is the only shop window.
RESTOCK_REVIVE_MIN = 6
RESTOCK_FR_MIN = 2
SHOP_REVIVE_FLOOR = 6
# Kit cash: 6 Revive × $1500 + 2 FR × $3000. Broke Center ($364) cannot
# buy that. FireRed money = trainer payouts (wilds pay nothing). Gym
# rematches are post-credits. VS Seeker (item 362) from the Vermilion
# Center girl (6,4) unlocks Route 11 rematches — the lobby sidequest.
ITEM_VS_SEEKER = 362
NUGGET = 110
# Endgame sell-everything-cashable: Nugget/pearls/dust PLUS mushrooms
# (Two Island tutors are post-credits). NEVER heals/Revives/balls/HMs.
# TMs live in the TM Case — the mart SELL list is the Items pocket only.
# ID ground truth (campaign.MART_SELL_LOOT verified + live screen 16:43):
# 102 Tiny Mushroom / 103 Big Mushroom. 93/94 are SUN/MOON STONE — the
# mart REFUSES them ("Oh, no. I can't buy that."), price $0; the bag's
# x2 stack is MOON STONE, not a mushroom. Never list stones here.
SELL_LOOT = (110, 109, 107, 108, 106, 103, 102)  # + Big/Tiny Mushroom
CENTER_MAT = (11, 15)  # tile NORTH of the street warp; DOWN exits (live 16:35)
VERMILION_CITY = (3, 5)
VERMILION_PC_DOOR = (15, 6)
VS_SEEKER_WOMAN = (6, 4)              # pret VermilionCity_PokemonCenter_1F
ROUTE11 = (3, 29)
CENTER_STREET_EXIT = (11, 16)         # pret IndigoPlateau_PokemonCenter_1F → exterior
# 12:11 wipe: 17/240 Blastoise walked into Lance because between-room
# logged "heal pocket EMPTY" and did not field a healthy bird.
E4_LEAD_CRIT_FRAC = 0.25
HEAL_POCKET_IDS = (13, 22, 21, 20, 19)  # Potion, Super, Hyper, Max, FR
KEY_OF = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}

# pret/pokefirered include/constants/flags.h — persist across whiteout, cleared in Hall of Fame.
FLAG_DEFEATED_LORELEI, FLAG_DEFEATED_BRUNO = 0x4B8, 0x4B9
FLAG_DEFEATED_AGATHA, FLAG_DEFEATED_LANCE = 0x4BA, 0x4BB
FLAG_DEFEATED_CHAMP = 0x4BC
# Map-keyed seats (campaign._PLACE_NAMES). Never trust seen_rooms count after a whiteout.
ROOM_SEAT = {
    (1, 75): "Lorelei", (1, 76): "Bruno", (1, 77): "Agatha",
    (1, 78): "Lance", (1, 79): "Gary",
}
# Room #6 past the Champion — the Hall of Fame itself. The credits drain uses it to prove that
# CONTINUE actually re-entered the world (GameClear warps a continued save HOME, so staying on
# this map means we are still sitting in the save-summary menu).
HALL_OF_FAME = (1, 80)
# Exterior + Center + the five rooms. Roam must offer enter_league here
# (live 09:27: Agatha's Room offered head_to_league + heal, then heal
# walked NORTH into Lance with Revive x0).
LEAGUE_CHAIN_MAPS = frozenset({INDIGO_EXT, LEAGUE_CENTER, *ROOM_SEAT})
SEAT_FLAG = {
    "Lorelei": FLAG_DEFEATED_LORELEI, "Bruno": FLAG_DEFEATED_BRUNO,
    "Agatha": FLAG_DEFEATED_AGATHA, "Lance": FLAG_DEFEATED_LANCE,
    "Gary": FLAG_DEFEATED_CHAMP,
}
SEAT_ORDER = ("Lorelei", "Bruno", "Agatha", "Lance", "Gary")
# HARD RULE (Jonny 2026-08-13, revised same-day live Lorelei): Zapdos leads Lorelei
# ONLY if he actually has an Electric damaging move (TM24 Thunderbolt / TM25 Thunder /
# Shock Wave). Wild Power-Plant Zapdos is TWave/Agility/Detect/Drill Peck — Drill Peck
# is resisted by Ice and he dies in 3 turns. Without the gun, Blastoise Earthquake is
# the honest water-smash. Articuno (Ice-into-Ice) stays banned. Species ids so a
# name-table miss cannot soften this.
ZAPDOS_SP, MOLTRES_SP, ARTICUNO_SP, BLASTOISE_SP = 145, 146, 144, 9
JYNX_SP = 124
GYARADOS_SP = 130
PIDGEOT_SP, VENUSAUR_SP, RHYDON_SP, ARCANINE_SP = 18, 3, 112, 59
# Dewgong, Cloyster, Slowbro, Lapras — Electric DESTROYS these IF the move exists.
LORELEI_WATER_SP = frozenset({87, 91, 80, 131})
LORELEI_LEAD_ORDER = (ZAPDOS_SP, BLASTOISE_SP, MOLTRES_SP)
LORELEI_BANNED = frozenset({ARTICUNO_SP})  # Ice-into-Ice always; Blastoise is conditional
# Damaging Electric move ids (Gen 3). Thunder Wave (86) is status — never counts.
# Stale power-byte on Thunderbolt still matches by id.
ELECTRIC_DAMAGE_IDS = frozenset({
    84,   # ThunderShock
    85,   # Thunderbolt
    87,   # Thunder
    9,    # ThunderPunch
    192,  # Zap Cannon
    209,  # Spark
    351,  # Shock Wave
})
MOVE_THUNDERBOLT, MOVE_THUNDER = 85, 87
MOVE_DRILL_PECK, MOVE_AGILITY, MOVE_DETECT, MOVE_THUNDER_WAVE = 65, 97, 197, 86
MOVE_SURF, MOVE_EARTHQUAKE, MOVE_SKULL_BASH, MOVE_ICE_BEAM = 57, 89, 130, 58
MOVE_FLAMETHROWER = 53
ITEM_TM24, ITEM_TM25, ITEM_TM34 = 312, 313, 322  # TM01=289; TM34 Shock Wave
# Setup / stall — a free turn for the foe (live 20:28: Moltres Agility while
# Flamethrower sat at 14 PP). Never pick these when a connecting hit exists.
E4_SETUP_IDS = frozenset({
    97,   # Agility
    104,  # Double Team
    116,  # Focus Energy
    14,   # Swords Dance
    197,  # Detect
    182,  # Protect
    54,   # Mist
    115,  # Reflect
    113,  # Light Screen
    203,  # Endure
    170,  # Mind Reader
    86,   # Thunder Wave
})
# Ground is 0x Flying (chat: no EQ on flyers). Species backfill if types flake.
E4_FLYING_SP = frozenset({
    16, 17, 18, 21, 22, 41, 42, 83, 84, 85, 130, 142, 144, 145, 146, 149,
})
LEGENDARY_BIRDS = frozenset({ARTICUNO_SP, ZAPDOS_SP, MOLTRES_SP})
LAPRAS_SP = 131
# STAB types for the E4 four. max-damage without these burned EQ on Bruno's
# Hitmons (100 > Surf 95) then Arbok ate Skull Bash (live 21:08).
E4_SELF_TYPES = {
    BLASTOISE_SP: ["water"],
    ARTICUNO_SP: ["ice", "flying"],
    ZAPDOS_SP: ["electric", "flying"],
    MOLTRES_SP: ["fire", "flying"],
}
# Elixir/Max Elixir restore ALL moves (EQ may not be slot 0). Ether/Max Ether = one move.
ITEM_ELIXIR, ITEM_MAX_ELIXIR, ITEM_ETHER, ITEM_MAX_ETHER = 36, 37, 34, 35
E4_ELIXIR_PREF = (ITEM_ELIXIR, ITEM_MAX_ELIXIR, ITEM_ETHER, ITEM_MAX_ETHER)
# Other seats: preferred living species in order, then the type-chart scorer.
# Lance LEADS Gyarados: Zapdos-with-gun is the 4x answer (Jonny 19:26). Articuno
# next (Ice vs the dragon wave). Blastoise 1x Skull Bash. NEVER Moltres
# (Fire 0.5x Gyarados — live wipe 17:23). Drill Peck Zapdos is 0.5x — gun gate
# in preferred_lead_slot skips him. Gary: Articuno Ice vs Pidgeot; Moltres = Venusaur.
PREFERRED_LEAD_IDS = {
    "Lorelei": LORELEI_LEAD_ORDER,
    # Articuno Ice/Flying vs Onix is 4x Rock — live 19:52 one-shot. Surf is 4x.
    "Bruno": (BLASTOISE_SP, ZAPDOS_SP, MOLTRES_SP, ARTICUNO_SP),
    "Agatha": (BLASTOISE_SP, ZAPDOS_SP, MOLTRES_SP, ARTICUNO_SP),
    "Lance": (ZAPDOS_SP, ARTICUNO_SP, BLASTOISE_SP),
    "Gary": (ARTICUNO_SP, MOLTRES_SP, ZAPDOS_SP, BLASTOISE_SP),
}
_STARTER_LINE = {
    1: "bulbasaur", 2: "bulbasaur", 3: "bulbasaur",
    4: "charmander", 5: "charmander", 6: "charmander",
    7: "squirtle", 8: "squirtle", 9: "squirtle",
}
_ROSTERS_CACHE = None


def _load_rosters():
    global _ROSTERS_CACHE
    if _ROSTERS_CACHE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "gamedata", "frlg_rosters.json")
        with open(path, encoding="utf-8") as f:
            _ROSTERS_CACHE = json.load(f)
    return _ROSTERS_CACHE


def detect_starter_branch(b):
    """HER starter family from the live party (Blastoise => squirtle). Fail-open squirtle
    (this Sherpa timeline) so we never invent a Charizard Champion against a water starter."""
    import pokemon_state as st
    try:
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(min(cnt, 6)):
            br = _STARTER_LINE.get(st.read_party_species(b, s))
            if br:
                return br
    except Exception:
        pass
    return "squirtle"   # this Sherpa timeline; never invent a Charizard champ


def roster_for_seat(b, seat_name):
    """Ground-truth E4/Champion team from gamedata. Champion branches on HER starter."""
    kb = _load_rosters()
    if seat_name == "Gary":
        champ = kb.get("champion") or {}
        by_s = champ.get("by_starter") or {}
        team = by_s.get(detect_starter_branch(b)) or champ.get("team") or []
        return team
    return (kb.get("e4") or {}).get(seat_name) or []


def next_uncleared_seat(b):
    """First E4/Champ seat whose DEFEATED flag is still clear. Whiteout keeps flags."""
    for name in SEAT_ORDER:
        try:
            if not fm.read_flag(b, SEAT_FLAG[name]):
                return name
        except Exception:
            return name
    return "Lorelei"


def e4_room_entry_sealed(here):
    """PURE. True inside an E4/Champ room: EnterRoom → CloseEntry seals
    the south door (metatiles 5-7,11-12 collision). PreventExit even
    force-walks you north ('Don't run away!'). Cannot shop from here."""
    return tuple(here or ()) in ROOM_SEAT


def e4_must_restock(revive_count, money, next_seat, full_restore_count=0,
                    here=None):
    """PURE. Thin kit + cash: stay at the League Center and buy a STACK
    before walking into Lorelei. NEVER a reason to walk south from an
    E4 room — CloseEntry sealed that door (live 11:38 Bruno (6,10) loop).
    A stack is RESTOCK_REVIVE_MIN (6) Revives AND RESTOCK_FR_MIN (2)
    Full Restores — the only shop window before CloseEntry seals south."""
    if here is not None and e4_room_entry_sealed(here):
        return False
    # Broke + NOT at the Center: cannot shop, do not retreat. Broke AT
    # the Center: hold the Lorelei door and earn (Jonny 2026-08-14).
    if int(money or 0) < 1500 and (
            here is None or tuple(here) != LEAGUE_CENTER):
        return False
    rev = int(revive_count or 0)
    fr = int(full_restore_count or 0)
    # After Nugget+mushroom sale she can afford 6 Revives + 1 FR and be
    # ~$136 short of FR #2. Do not hold the door for an item she cannot
    # buy — Revive stack + 1 FR + Super x6 is a real kit.
    thin = rev < RESTOCK_REVIVE_MIN or (
        fr < RESTOCK_FR_MIN and int(money or 0) >= 3000)
    # Live 12:43 whiteout: next uncleared is Lorelei, here is League
    # Center (13,0), kit Revive x1 FR x0. The Agatha/Lance/Gary-only
    # gate skipped the shop and would walk her back in empty. Center
    # is the only shop window — any next seat, if the kit is thin.
    if here is not None and tuple(here) == LEAGUE_CENTER:
        return thin
    if next_seat in ("Agatha", "Lance", "Gary"):
        return thin
    return False


def e4_kit_cash_needed(revive_count, fr_count, money):
    """PURE. Yen still needed to buy the missing Revive/FR stack units."""
    need_rev = max(0, RESTOCK_REVIVE_MIN - int(revive_count or 0))
    need_fr = max(0, RESTOCK_FR_MIN - int(fr_count or 0))
    cost = need_rev * 1500 + need_fr * 3000
    return max(0, cost - int(money or 0))


def e4_must_earn(revive_count, money, fr_count=0, here=None):
    """PURE. League Center + thin kit + cannot afford the missing stack.

    Sealed E4 rooms never leave (CloseEntry). Rich + thin shops in place.
    """
    if here is None or e4_room_entry_sealed(here):
        return False
    if tuple(here) != LEAGUE_CENTER:
        return False
    rev = int(revive_count or 0)
    fr = int(fr_count or 0)
    thin = rev < RESTOCK_REVIVE_MIN or fr < RESTOCK_FR_MIN
    if not thin:
        return False
    # Earn only when she cannot buy the NEXT needed unit. $5364 after a
    # Nugget sale shops Revives in place — do not Fly Vermilion for $136.
    if rev < RESTOCK_REVIVE_MIN:
        return int(money or 0) < 1500
    # FR shortfall alone is NOT an earn reason (Jonny 2026-08-14: "whatever
    # Full Restores she can afford"). Revive stack met + can't afford FR
    # ($864, no Fly — the Vermilion errand is a dead loop) = walk into
    # Lorelei with the stack; e4_must_restock already clears that door.
    return False


def e4_warp_approach_cluster(tile):
    """PURE. Door tile + cardinal neighbors. Live restock-south: (6,11) is
    also a warp, BFS excluded it, no path from (6,7) to any neighbor of
    (6,12), then 'falling through' walked her north into Lance empty."""
    if not tile or len(tile) < 2:
        return ()
    x, y = int(tile[0]), int(tile[1])
    return ((x, y), (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y))


def e4_south_heal_warp(here, warps, next_uncleared):
    """PURE. warps = [(xy, dest), ...]. Warp toward Center / a previous seat.
    Never the door into `next_uncleared` (Agatha (6,2) is Lance).
    None = no safe warp on this floor."""
    scored = []
    nxt = next_uncleared
    nxt_i = SEAT_ORDER.index(nxt) if nxt in SEAT_ORDER else len(SEAT_ORDER)
    for xy, dest in warps or []:
        dest = tuple(dest or ())
        xy = tuple(xy or ())
        if len(dest) < 2 or len(xy) < 2:
            continue
        seat = ROOM_SEAT.get(dest)
        if seat == nxt:
            continue
        if dest == LEAGUE_CENTER or dest == INDIGO_EXT:
            scored.append((0, -xy[1], xy))
        elif seat and SEAT_ORDER.index(seat) < nxt_i:
            scored.append((1, -xy[1], xy))
        else:
            scored.append((2, -xy[1], xy))
    if not scored:
        return None
    scored.sort()
    return scored[0][2]


def _alive_slot_of(b, species_id):
    """Party slot of a living `species_id`, or None."""
    import pokemon_state as st
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    for s in range(min(cnt, 6)):
        if st.read_party_species(b, s) != species_id:
            continue
        if b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) > 0:
            return s
    return None


def move_is_electric_damage(mid, mtype=None, power=None):
    """PURE. Thunder Wave (86) is Electric status — never a gun. Thunderbolt id 85
    still counts if the power byte is stale."""
    if mid in ELECTRIC_DAMAGE_IDS:
        return True
    return (str(mtype or "").lower() == "electric" and (power or 0) > 0)


def moveset_has_electric_damage(move_ids):
    """PURE. True iff any move id is a real Electric attack (not Thunder Wave)."""
    return any(m in ELECTRIC_DAMAGE_IDS for m in (move_ids or []) if m)


def slot_has_electric_damage(b, slot):
    """RAM: this party slot has a damaging Electric move."""
    import pokemon_state as st
    try:
        for mid in st.read_party_moves(b, slot):
            if not mid:
                continue
            if mid in ELECTRIC_DAMAGE_IDS:
                return True
            mt, mp = st.move_info(b, mid)
            if move_is_electric_damage(mid, mt, mp):
                return True
    except Exception:
        pass
    return False


def zapdos_forget_idx(move_ids):
    """PURE. Overwrite Agility or Detect (TWave last). Never Drill Peck. None = abort."""
    moves = list(move_ids or [])
    for pref in (MOVE_AGILITY, MOVE_DETECT, MOVE_THUNDER_WAVE):
        if pref in moves:
            return moves.index(pref)
    for i, m in enumerate(moves):
        if m and m != MOVE_DRILL_PECK and m not in ELECTRIC_DAMAGE_IDS:
            return i
    return None


def lorelei_banned_species(zap_has_electric):
    """PURE. Articuno always (Ice-into-Ice). Blastoise only when Zapdos actually has the gun."""
    banned = {ARTICUNO_SP}
    if zap_has_electric:
        banned.add(BLASTOISE_SP)
    return frozenset(banned)


def preferred_lead_slot(b, seat_name):
    """HARD preferred living species for this seat. Lorelei: Zapdos ONLY with an
    Electric damaging move; else Blastoise (Earthquake); else Moltres. Never Articuno.
    Lance: Zapdos-with-gun first (4x Gyarados lead). Else Articuno Ice, then
    Blastoise. Never Moltres vs Gyarados (Fire 0.5x). Drill Peck is 0.5x.
    Gary / squirtle-start: Articuno Ice vs Pidgeot; Zapdos only with Electric.
    Agatha: Blastoise Surf vs Gengar (Skull Bash 0x Ghost, EQ 0x Levitate).
    Never Zapdos Drill Peck into Gengar (live wipe: potion loop then bag latch).
    Bruno: Blastoise Surf vs Onix. Never Articuno (4x Rock — live 19:52).
    Other seats: PREFERRED_LEAD_IDS in order. None = scorer."""
    if seat_name == "Agatha":
        blast = _alive_slot_of(b, BLASTOISE_SP)
        if blast is not None:
            return blast
        for sp in (ZAPDOS_SP, MOLTRES_SP, ARTICUNO_SP):
            slot = _alive_slot_of(b, sp)
            if slot is not None:
                return slot
        return None
    if seat_name == "Lorelei":
        zap = _alive_slot_of(b, ZAPDOS_SP)
        if zap is not None and slot_has_electric_damage(b, zap):
            return zap
        blast = _alive_slot_of(b, BLASTOISE_SP)
        if blast is not None:
            return blast
        molt = _alive_slot_of(b, MOLTRES_SP)
        if molt is not None:
            return molt
        return None
    if seat_name == "Lance":
        zap = _alive_slot_of(b, ZAPDOS_SP)
        if zap is not None and slot_has_electric_damage(b, zap):
            return zap
        for sp in PREFERRED_LEAD_IDS.get(seat_name) or ():
            if sp == ZAPDOS_SP:
                continue
            slot = _alive_slot_of(b, sp)
            if slot is not None:
                return slot
        return None
    if seat_name == "Gary":
        # Pidgeot lead: Articuno Ice, not Zapdos-first (that's the Lance Gyarados gun).
        for sp in PREFERRED_LEAD_IDS.get(seat_name) or ():
            if sp == ZAPDOS_SP:
                zap = _alive_slot_of(b, ZAPDOS_SP)
                if zap is None or not slot_has_electric_damage(b, zap):
                    continue
                return zap
            slot = _alive_slot_of(b, sp)
            if slot is not None:
                return slot
        return None
    for sp in PREFERRED_LEAD_IDS.get(seat_name) or ():
        slot = _alive_slot_of(b, sp)
        if slot is not None:
            return slot
    return None


def lorelei_foe_is_water(enemy_types, enemy_species=None):
    """True for Lorelei's waters (and any Water-typed foe). Species-id backfill
    so an Ice-only RAM flake on Dewgong still counts as Water."""
    if enemy_species in LORELEI_WATER_SP:
        return True
    return any(str(t).lower() == "water" for t in (enemy_types or []) if t)


def lorelei_foe_is_jynx(enemy_types, enemy_species=None):
    """Jynx is the one Lorelei mon Zapdos does not smash (Ice/Psychic, not Water)."""
    if enemy_species == JYNX_SP:
        return True
    types = {str(t).lower() for t in (enemy_types or []) if t}
    return "ice" in types and "psychic" in types and "water" not in types


def lorelei_inbattle_switch(active_sp, enemy_types, enemy_species, zap_slot, molt_slot,
                            zap_has_electric=True, blast_slot=None, locked=()):
    """PURE in-battle Lorelei policy. Returns 'stay' or a party slot.

    Zapdos vs Water → STAY only if he has an Electric damaging move.
    Without the gun, Blastoise Earthquake is the water smash (live 13:33: Dewgong
    + Cloyster fell to EQ after Zapdos Drill-Pecked into Ice and fainted).
    Moltres is the Jynx answer. Never Articuno vs waters (Ice into Lapras is 0.25x).
    `locked` = species that cannot act (sleep/freeze) — do not yank back to a
    sleeper (live 14:00 MUST-LEAVE Blastoise↔Articuno ping-pong).
    Caller invokes this ONLY in Lorelei's room (1, 75).
    """
    locked = frozenset(locked or ())
    jynx = lorelei_foe_is_jynx(enemy_types, enemy_species)
    if jynx:
        # L75 Blastoise Surf 2HKOs Jynx. Sending L50 Moltres into Lovely Kiss
        # is the 18:23 wipe (4 Super Potions on a sleeper, then whiteout).
        if active_sp == BLASTOISE_SP and BLASTOISE_SP not in locked:
            return "stay"
        if blast_slot is not None and BLASTOISE_SP not in locked:
            return blast_slot
        if active_sp == MOLTRES_SP:
            return "stay"
        if molt_slot is not None and MOLTRES_SP not in locked:
            return molt_slot
        if zap_has_electric and active_sp == ZAPDOS_SP:
            return "stay"
        if zap_has_electric and zap_slot is not None and ZAPDOS_SP not in locked:
            return zap_slot
        return "stay"
    # Waters: electric Zapdos holds; otherwise Blastoise (EQ / Skull Bash).
    # Blastoise stays legal EVEN IF frozen — live 18:44 Moltres KO'd Jynx then
    # died on Lapras because frozen Blastoise was treated as unfieldable.
    # MUST-LEAVE already used its one pull on Jynx; he stays vs Lapras.
    if zap_has_electric and ZAPDOS_SP not in locked:
        if active_sp == ZAPDOS_SP:
            return "stay"
        if zap_slot is not None:
            return zap_slot
    blast_ok = blast_slot is not None
    if blast_ok:
        if active_sp == BLASTOISE_SP:
            return "stay"
        return blast_slot
    # Blastoise asleep/fainted: never Articuno Ice into Lapras (0.25x).
    if active_sp == ARTICUNO_SP:
        if molt_slot is not None and MOLTRES_SP not in locked:
            return molt_slot
        if zap_slot is not None and ZAPDOS_SP not in locked:
            return zap_slot
        return "stay"
    if active_sp in (MOLTRES_SP, ZAPDOS_SP, BLASTOISE_SP):
        return "stay"
    if molt_slot is not None and MOLTRES_SP not in locked:
        return molt_slot
    if zap_slot is not None and ZAPDOS_SP not in locked:
        return zap_slot
    return "stay"


# Agatha ghosts (FRLG): Gengar 94, Haunter 93. Skull Bash is 0x Ghost.
# Earthquake is 0x Levitate (Gen 3 Gengar/Haunter). Surf is the ONLY hit.
# Live 20:08 wipe: last Surf spent on Arbok, then Skull Bash vs Gengar forever.
GENGAR_SP, HAUNTER_SP = 94, 93
AGATHA_GHOST_SP = frozenset({94, 93})
GOLBAT_SP, ARBOK_SP, ONIX_SP = 42, 24, 95

# Lance dragons (FRLG): Dragonair 148, Dragonite 149. Water is 0.5x into Dragon.
LANCE_DRAGON_SP = frozenset({148, 149})
AERODACTYL_SP = 142


def agatha_foe_is_ghost(enemy_types, enemy_species=None):
    """True for Agatha's Gengar / Haunter (and any Ghost-typed foe)."""
    if enemy_species in AGATHA_GHOST_SP:
        return True
    return any(str(t).lower() == "ghost" for t in (enemy_types or []) if t)


def agatha_foe_is_golbat(enemy_types, enemy_species=None):
    """Golbat is Poison/Flying. Shock Wave 2x Flying; EQ is 0x Flying."""
    if enemy_species == GOLBAT_SP:
        return True
    types = {str(t).lower() for t in (enemy_types or []) if t}
    return "poison" in types and "flying" in types


def agatha_foe_is_arbok(enemy_types, enemy_species=None):
    """Arbok is Poison. EQ is 4x. Zapdos Drill Peck is a 1x waste (live 20:03 faint)."""
    if enemy_species == ARBOK_SP:
        return True
    types = {str(t).lower() for t in (enemy_types or []) if t}
    return ("poison" in types and "flying" not in types and "ghost" not in types)


def bruno_foe_is_onix(enemy_types, enemy_species=None):
    """Onix is Rock/Ground. Surf 4x. Articuno Ice/Flying eats 4x Rock (live 19:52)."""
    if enemy_species == ONIX_SP:
        return True
    types = {str(t).lower() for t in (enemy_types or []) if t}
    return "rock" in types and "ground" in types


def agatha_inbattle_switch(active_sp, enemy_types, enemy_species, blast_slot,
                           zap_slot=None, zap_has_electric=False, locked=(),
                           frail=()):
    """PURE. Ghost → Blastoise Surf (EQ is 0x Levitate — never EQ Gengar).
    Golbat → healthy Zapdos-with-gun, else Blastoise. NEVER half-HP Articuno
    (live 20:00: 78 HP Articuno walked in and died).
    Arbok → Blastoise (EQ 4x). Zapdos staying is the 20:03 faint."""
    locked = frozenset(locked or ())
    frail = frozenset(frail or ())
    if agatha_foe_is_ghost(enemy_types, enemy_species):
        if blast_slot is not None and BLASTOISE_SP not in locked:
            if active_sp == BLASTOISE_SP:
                return "stay"
            return blast_slot
        return "stay"
    if agatha_foe_is_golbat(enemy_types, enemy_species):
        zap_ok = (zap_slot is not None and zap_has_electric
                  and ZAPDOS_SP not in locked and ZAPDOS_SP not in frail)
        if zap_ok:
            if active_sp == ZAPDOS_SP:
                return "stay"
            return zap_slot
        if blast_slot is not None and BLASTOISE_SP not in locked:
            if active_sp == BLASTOISE_SP:
                return "stay"
            return blast_slot
        return "stay"
    if agatha_foe_is_arbok(enemy_types, enemy_species):
        if blast_slot is not None and BLASTOISE_SP not in locked:
            if active_sp == BLASTOISE_SP:
                return "stay"
            return blast_slot
        return "stay"
    return None


def bruno_inbattle_switch(active_sp, enemy_types, enemy_species, blast_slot,
                          locked=()):
    """PURE. vs Onix: Blastoise Surf. Never send Articuno (4x Rock)."""
    locked = frozenset(locked or ())
    if not bruno_foe_is_onix(enemy_types, enemy_species):
        return None
    if blast_slot is not None and BLASTOISE_SP not in locked:
        if active_sp == BLASTOISE_SP:
            return "stay"
        return blast_slot
    return "stay"


def e4_is_ghost_wincon(sp, enemy_types, enemy_species=None):
    """PURE. Fainted Blastoise is the Agatha-ghost revive (Surf vs Gengar)."""
    return sp == BLASTOISE_SP and agatha_foe_is_ghost(enemy_types, enemy_species)


def e4_is_arbok_wincon(sp, enemy_types, enemy_species=None):
    """PURE. Fainted Blastoise is the Arbok revive (EQ 4x). Live 20:28 left
    him fainted and the oracle picked keep_fighting."""
    return sp == BLASTOISE_SP and agatha_foe_is_arbok(enemy_types, enemy_species)


def e4_is_setup_move(move_id, power=None):
    """PURE. Agility / Detect / Endure / any 0-power stall. Never the E4 pick
    when a connecting attack still has PP (Jonny: foe sets up while we could
    have Flamethrowered twice)."""
    if move_id in E4_SETUP_IDS:
        return True
    return (power or 0) <= 0 and bool(move_id)


def lance_foe_is_dragon(enemy_types, enemy_species=None):
    """True for Lance's Dragonair / Dragonite (and any Dragon-typed foe)."""
    if enemy_species in LANCE_DRAGON_SP:
        return True
    return any(str(t).lower() == "dragon" for t in (enemy_types or []) if t)


def lance_foe_is_gyarados(enemy_types, enemy_species=None):
    """True for Lance's Gyarados lead (Water/Flying). Species-id backfill so a
    Water-only RAM flake still counts — Electric is 4x, Drill Peck is 0.5x."""
    if enemy_species == GYARADOS_SP:
        return True
    types = {str(t).lower() for t in (enemy_types or []) if t}
    return "water" in types and "flying" in types


def articuno_ice_into_water_ice(enemy_types, enemy_species=None):
    """True ONLY for Water/Ice (Lapras/Dewgong/Cloyster) where Ice Beam is 0.25x,
    plus Slowbro (Water/Psychic, Ice 0.5x). Gyarados is Water/Flying — Ice is 1x
    and Articuno's ONLY damaging move. Live 21:08 banned Ice Beam, fired Agility
    three times, whiteout. Never treat Gyarados as a Lorelei water for this ban."""
    if lance_foe_is_gyarados(enemy_types, enemy_species):
        return False
    if enemy_species in LORELEI_WATER_SP:
        return True
    types = {str(t).lower() for t in (enemy_types or []) if t}
    return "water" in types and "ice" in types


def e4_between_room_revive_species(seat):
    """Scarce-Revive order before this room. Live 21:08: 1 Revive, revived
    Blastoise (slot 0) before Lance, Zapdos stayed dead, Articuno Agility'd
    Gyarados. Zapdos-with-gun is the 4x Gyarados answer — he stands up first."""
    if seat == "Lance":
        return (ZAPDOS_SP, ARTICUNO_SP, BLASTOISE_SP, MOLTRES_SP)
    if seat == "Lorelei":
        return (ZAPDOS_SP, BLASTOISE_SP, MOLTRES_SP, ARTICUNO_SP)
    if seat in ("Bruno", "Agatha"):
        return (BLASTOISE_SP, ZAPDOS_SP, MOLTRES_SP, ARTICUNO_SP)
    if seat == "Gary":
        return (ARTICUNO_SP, ZAPDOS_SP, MOLTRES_SP, BLASTOISE_SP)
    return (BLASTOISE_SP, ZAPDOS_SP, ARTICUNO_SP, MOLTRES_SP)


def lance_inbattle_switch(active_sp, enemy_types, enemy_species, zap_slot, art_slot,
                          zap_has_electric=False, blast_slot=None, locked=()):
    """PURE. vs Gyarados: Zapdos-with-gun NOW (4x). Never stay on Moltres
    (Fire 0.5x — live 17:23). Without the gun, Articuno / Blastoise, not
    Drill-Peck Zapdos. vs Dragonair/Dragonite: Articuno Ice. None = Aerodactyl
    (generic matchup). `locked` = sleep/freeze — do not yank to a sleeper."""
    locked = frozenset(locked or ())
    if lance_foe_is_gyarados(enemy_types, enemy_species):
        if zap_has_electric and zap_slot is not None and ZAPDOS_SP not in locked:
            if active_sp == ZAPDOS_SP:
                return "stay"
            return zap_slot
        if art_slot is not None and ARTICUNO_SP not in locked:
            if active_sp == ARTICUNO_SP:
                return "stay"
            return art_slot
        if blast_slot is not None and BLASTOISE_SP not in locked:
            if active_sp == BLASTOISE_SP:
                return "stay"
            return blast_slot
        if active_sp == MOLTRES_SP:
            if zap_slot is not None and ZAPDOS_SP not in locked:
                return zap_slot
        return "stay"
    if lance_foe_is_dragon(enemy_types, enemy_species):
        if art_slot is not None and ARTICUNO_SP not in locked:
            if active_sp == ARTICUNO_SP:
                return "stay"
            return art_slot
        return "stay"
    return None


def e4_pick_send_cand(pref, cands, crit_frac=E4_LEAD_CRIT_FRAC):
    """PURE. First pref species that is living AND not dying, if a
    healthier pref exists. Live 12:34: Zapdos fainted vs Gyarados →
    E4 SEND blastoise at 17/240 → then switched to Articuno (free hit).
    Never seat a dying out-typed mon as a stepping stone.
    cands = [{species, hp, maxhp}, ...] living only. Returns the cand
    dict or None (caller falls back to highest-level)."""
    by_sp = {}
    for c in cands or []:
        if not isinstance(c, dict):
            continue
        sp = c.get("species")
        hp = int(c.get("hp") or 0)
        if not sp or hp <= 0:
            continue
        by_sp[int(sp)] = c
    if not pref or not by_sp:
        return None
    healthy, dying = [], []
    for want in pref:
        c = by_sp.get(int(want))
        if not c:
            continue
        mx = int(c.get("maxhp") or 0) or 1
        if (int(c.get("hp") or 0) / mx) > float(crit_frac):
            healthy.append(c)
        else:
            dying.append(c)
    if healthy:
        return healthy[0]
    if dying:
        return dying[0]
    return None


def e4_defender_4x_weak(sp, enemy_types, enemy_species=None):
    """PURE. True if this species takes 4x from the foe's STAB types.
    Never revive Zapdos/Moltres/Articuno into Rock (Aerodactyl) when a
    better answer is also fainted — they are OHKO bait."""
    import pokemon_policy as pp
    ours = E4_SELF_TYPES.get(int(sp or 0)) or []
    foe = [str(t).lower() for t in (enemy_types or []) if t]
    if enemy_species == GYARADOS_SP:
        foe = ["water", "flying"]
    elif enemy_species in LANCE_DRAGON_SP:
        foe = ["dragon"] + (["flying"] if enemy_species == 149 else [])
    elif enemy_species == AERODACTYL_SP:
        foe = ["rock", "flying"]
    if not ours or not foe:
        return False
    return any(pp.effectiveness(ft, ours) >= 4.0 for ft in foe)


def e4_revive_pref(seat, enemy_types, enemy_species, zap_has_electric=False):
    """PURE. Ordered fainted species to Revive for THIS foe — not
    first-fainted-row, not highest-level (live 12:34: Zapdos L52 over
    Articuno vs Dragonair). Gyarados + gun: Zapdos (4x). Dragons:
    Articuno (Ice 4x), then Blastoise. Never revive a 4x-weak bait
    when a better answer is also down."""
    # Gyarados is Water/Flying — check Lance answers BEFORE generic Water
    # (Lorelei waters would otherwise steal the order and drop Articuno).
    if lance_foe_is_gyarados(enemy_types, enemy_species):
        if zap_has_electric:
            return (ZAPDOS_SP, ARTICUNO_SP, BLASTOISE_SP)
        return (ARTICUNO_SP, BLASTOISE_SP, ZAPDOS_SP)
    if lance_foe_is_dragon(enemy_types, enemy_species):
        return (ARTICUNO_SP, BLASTOISE_SP, ZAPDOS_SP)
    if lorelei_foe_is_jynx(enemy_types, enemy_species):
        return (MOLTRES_SP, BLASTOISE_SP, ZAPDOS_SP)
    if lorelei_foe_is_water(enemy_types, enemy_species):
        if zap_has_electric:
            return (ZAPDOS_SP, BLASTOISE_SP, MOLTRES_SP)
        return (BLASTOISE_SP, MOLTRES_SP, ZAPDOS_SP)
    if enemy_species == AERODACTYL_SP or (
            {str(t).lower() for t in (enemy_types or []) if t} >= {"rock", "flying"}):
        return (BLASTOISE_SP, ARTICUNO_SP, MOLTRES_SP)
    if agatha_foe_is_ghost(enemy_types, enemy_species):
        return (BLASTOISE_SP, MOLTRES_SP, ARTICUNO_SP)
    if agatha_foe_is_arbok(enemy_types, enemy_species):
        return (BLASTOISE_SP, ZAPDOS_SP, MOLTRES_SP)
    if enemy_species == PIDGEOT_SP:
        return (ARTICUNO_SP, ZAPDOS_SP, BLASTOISE_SP)
    if enemy_species == VENUSAUR_SP:
        return (MOLTRES_SP, BLASTOISE_SP, ARTICUNO_SP)
    if enemy_species == RHYDON_SP:
        return (BLASTOISE_SP, ARTICUNO_SP, MOLTRES_SP)
    if seat:
        return e4_between_room_revive_species(seat)
    return (BLASTOISE_SP, ARTICUNO_SP, ZAPDOS_SP, MOLTRES_SP)


def e4_revive_target_from_rows(rows, seat, enemy_types, enemy_species,
                               zap_has_electric=False):
    """PURE. Party row to Revive. Wincon / type-answer order.
    Never 'first fainted row' and never highest-level (Zapdos L52 beat
    Articuno vs Dragonair). Never revive a 4x-weak mon if a better
    answer is also fainted. Thin party (alive<=2): still wincon-first,
    then any fainted. None = nobody down or no wincon with 3+ alive."""
    parsed = []
    for i, r in enumerate(rows or []):
        if isinstance(r, dict):
            idx = r.get("row", i)
            sp, hp = r.get("species"), r.get("hp") or 0
        elif r:
            idx = i
            sp = r[0]
            hp = r[1] if len(r) > 1 else 0
        else:
            continue
        if not sp:
            continue
        parsed.append((int(idx), int(sp), int(hp)))
    fainted = [(idx, sp) for idx, sp, hp in parsed if hp <= 0]
    alive_n = sum(1 for _i, _s, hp in parsed if hp > 0)
    if not fainted:
        return None
    pref = e4_revive_pref(seat, enemy_types, enemy_species, zap_has_electric)
    down_set = {sp for _idx, sp in fainted}
    for want in pref or ():
        if want not in down_set:
            continue
        if e4_defender_4x_weak(want, enemy_types, enemy_species):
            better = [sp for sp in down_set
                      if sp != want and sp in (pref or ())
                      and not e4_defender_4x_weak(sp, enemy_types, enemy_species)]
            if better:
                continue
        for idx, sp in fainted:
            if sp == want:
                return idx
    if alive_n <= 2:
        return fainted[0][0]
    return None


def e4_should_hold_north(lead_hp, lead_max, heal_pocket, revive_count, fainted_n,
                         crit_frac=E4_LEAD_CRIT_FRAC):
    """PURE. True = items can still save a wincon; do not walk north yet.
    Revive cannot heal a living ace. Empty bag = switch + go (log thin)."""
    try:
        hp, mx = int(lead_hp or 0), int(lead_max or 0)
    except (TypeError, ValueError):
        return False
    if int(heal_pocket or 0) > 0 and mx > 0 and hp > 0 and (hp / mx) <= float(crit_frac):
        return True
    if int(revive_count or 0) > 0 and int(fainted_n or 0) > 0:
        return True
    return False


def e4_force_send_pref(seat, enemy_types, enemy_species, zap_has_electric=False):
    """PURE. Ordered species to SEND after a faint in this E4 room.
    First living HEALTHY match wins (see e4_pick_send_cand). None = fall
    back to highest-level live.
    Never send Articuno into Lorelei waters (Ice Beam 0.25x Lapras).
    Never send Moltres into Gyarados (Fire 0.5x)."""
    if seat == "Lorelei":
        if lorelei_foe_is_jynx(enemy_types, enemy_species):
            return (BLASTOISE_SP, MOLTRES_SP, ZAPDOS_SP)
        if zap_has_electric:
            return (ZAPDOS_SP, BLASTOISE_SP, MOLTRES_SP)
        return (BLASTOISE_SP, MOLTRES_SP, ZAPDOS_SP)
    if seat == "Lance":
        if lance_foe_is_gyarados(enemy_types, enemy_species):
            if zap_has_electric:
                return (ZAPDOS_SP, BLASTOISE_SP, ARTICUNO_SP)
            return (ARTICUNO_SP, BLASTOISE_SP, ZAPDOS_SP)
        if lance_foe_is_dragon(enemy_types, enemy_species):
            return (ARTICUNO_SP, BLASTOISE_SP, ZAPDOS_SP)
        return (ARTICUNO_SP, ZAPDOS_SP, BLASTOISE_SP)
    if seat == "Agatha":
        if agatha_foe_is_ghost(enemy_types, enemy_species):
            return (BLASTOISE_SP, MOLTRES_SP, ARTICUNO_SP)
        if agatha_foe_is_golbat(enemy_types, enemy_species) and zap_has_electric:
            return (ZAPDOS_SP, BLASTOISE_SP, MOLTRES_SP)
        if agatha_foe_is_arbok(enemy_types, enemy_species):
            return (BLASTOISE_SP, ZAPDOS_SP, MOLTRES_SP)
        return (BLASTOISE_SP, ZAPDOS_SP, MOLTRES_SP)
    if seat == "Bruno":
        return (BLASTOISE_SP, ZAPDOS_SP, MOLTRES_SP)
    if seat == "Gary":
        types = {str(t).lower() for t in (enemy_types or []) if t}
        if lance_foe_is_gyarados(enemy_types, enemy_species):
            if zap_has_electric:
                return (ZAPDOS_SP, ARTICUNO_SP, BLASTOISE_SP)
            return (ARTICUNO_SP, BLASTOISE_SP, ZAPDOS_SP)
        if enemy_species == PIDGEOT_SP or ("flying" in types and "normal" in types):
            return (ARTICUNO_SP, ZAPDOS_SP, BLASTOISE_SP)
        if enemy_species == VENUSAUR_SP or "grass" in types:
            return (MOLTRES_SP, BLASTOISE_SP, ARTICUNO_SP)
        if enemy_species == RHYDON_SP or ("ground" in types and "rock" in types):
            return (BLASTOISE_SP, ARTICUNO_SP, MOLTRES_SP)
        if enemy_species == ARCANINE_SP or "fire" in types:
            return (BLASTOISE_SP, ARTICUNO_SP, ZAPDOS_SP)
        return (ARTICUNO_SP, MOLTRES_SP, ZAPDOS_SP, BLASTOISE_SP)
    return None


def e4_banned_move(move_id, move_type, active_sp, enemy_types, enemy_species=None):
    """PURE. Blastoise never Surfs a Water-type in E4 (0.5x; live Lapras wipe).
    Blastoise never Surfs a Dragon (0.5x; live Dragonite wipe). Articuno never
    Ice-Beams Water/Ice (Lapras 0.25x). ANY mon: no Ground vs Flying (chat),
    no Normal/Fighting vs Ghost (chat — Skull Bash 0x). Gengar also 0x EQ
    (Levitate) — Surf is the only Blastoise hit, never Earthquake."""
    mtype = str(move_type or "").lower()
    water = lorelei_foe_is_water(enemy_types, enemy_species)
    if active_sp == BLASTOISE_SP and water:
        if move_id == MOVE_SURF or mtype == "water":
            return True
    types = {str(t).lower() for t in (enemy_types or []) if t}
    if enemy_species in LORELEI_WATER_SP:
        types.add("water")
        types.add("ice")
    if enemy_species in E4_FLYING_SP or agatha_foe_is_golbat(enemy_types, enemy_species):
        types.add("flying")
    if enemy_species in AGATHA_GHOST_SP:
        types.add("ghost")
    if active_sp == BLASTOISE_SP and lance_foe_is_dragon(enemy_types, enemy_species):
        if move_id == MOVE_SURF or mtype == "water":
            return True
    # Chat: no EQ on flyers. Ground is 0x Flying (Golbat / Gyarados / birds).
    if (move_id == MOVE_EARTHQUAKE or mtype == "ground") and "flying" in types:
        return True
    # Chat: no physical (Normal/Fighting) on ghosts. Gen 3 immunity.
    if "ghost" in types:
        if mtype in ("normal", "fighting") or move_id == MOVE_SKULL_BASH:
            return True
        # Levitate: Ground is 0x Gengar/Haunter even though Ghost is not Ground-immune.
        if move_id == MOVE_EARTHQUAKE or mtype == "ground":
            return True
    # Agatha non-ghost: Surf is the Gengar gun. Ban it when EQ or Skull Bash
    # can still chip (live 20:08: last 2 Surf on Arbok, then 0x Skull Bash).
    if (active_sp == BLASTOISE_SP
            and not agatha_foe_is_ghost(enemy_types, enemy_species)
            and (agatha_foe_is_arbok(enemy_types, enemy_species)
                 or agatha_foe_is_golbat(enemy_types, enemy_species))
            and (move_id == MOVE_SURF or mtype == "water")):
        return True
    # Ice Beam vs Water/Ice is 0.25x. Gyarados is Water/Flying — Ice is 1x.
    # Live 21:08: this used to ban ANY water, so Articuno Agility'd Gyarados
    # with Ice Beam at 10 PP and the run died.
    if (active_sp == ARTICUNO_SP
            and articuno_ice_into_water_ice(enemy_types, enemy_species)):
        if mtype == "ice" or move_id == MOVE_ICE_BEAM:
            return True
    return False


def e4_filter_moves(moves, active_sp, enemy_types, enemy_species=None):
    """PURE. Copy with banned damaging moves treated as pp=0 unless nothing else can hit.
    Earthquake PP>0 still wins the picker; when EQ is dry this makes Skull Bash/Bite
    beat Surf-into-Water (STAB Surf otherwise outscores charge Skull Bash)."""
    moves = list(moves or [])
    if not moves:
        return moves
    banned_i = []
    for i, m in enumerate(moves):
        if (m.get("pp") or 0) <= 0 or (m.get("power") or 0) <= 0:
            continue
        if e4_banned_move(m.get("id") or 0, m.get("type"), active_sp,
                          enemy_types, enemy_species):
            banned_i.append(i)
    others = [i for i, m in enumerate(moves)
              if (m.get("pp") or 0) > 0 and (m.get("power") or 0) > 0
              and i not in banned_i]
    out = []
    for i, m in enumerate(moves):
        mm = dict(m)
        if i in banned_i:
            mm["pp"] = 0
        out.append(mm)
    if not others:
        # Articuno Ice into Lapras is NEVER a last-resort hit (0.25x Water/Ice).
        # Fail-open kept Ice Beam in the 18:44 whiteout. Gyarados is NOT that
        # case — Ice Beam must fail-open (live 21:08 Agility whiteout).
        if (active_sp == ARTICUNO_SP
                and articuno_ice_into_water_ice(enemy_types, enemy_species)):
            return out
        return moves
    return out


def e4_max_damage_index(moves, active_sp, enemy_types, enemy_species=None,
                        avoid_move_id=None):
    """PURE. Highest connecting damage. Never setup (Agility), never banned
    (EQ-on-Flying / Skull-Bash-on-Ghost), never 0-power. Live 20:28: Moltres
    rotated Flamethrower-miss into Agility and the foe lived two extra turns."""
    import pokemon_policy as pp
    types = [t for t in (enemy_types or []) if t]
    best_i, best = None, -1.0
    for i, m in enumerate(moves or []):
        if not m:
            continue
        mid = m.get("id") or 0
        if mid == avoid_move_id:
            continue
        if (m.get("pp") or 0) <= 0 or (m.get("power") or 0) <= 0:
            continue
        if e4_is_setup_move(mid, m.get("power")):
            continue
        if e4_banned_move(mid, m.get("type"), active_sp, enemy_types, enemy_species):
            continue
        # STAB in: Surf 142.5 beats EQ 100 on Bruno's Hitmons, so EQ PP
        # survives for Arbok / Lorelei waters (live 21:08 EQ-dry Skull Bash).
        score = pp.move_score(m, types, our_types=E4_SELF_TYPES.get(active_sp))
        if score > best:
            best_i, best = i, score
    return best_i


def e4_preferred_move_index(moves, active_sp, enemy_types, enemy_species=None,
                            seat=None, avoid_move_id=None):
    """PURE. The one move that actually wins THIS foe.
    Water → EQ (never Surf). Onix → Surf 4x (EQ is only 2x). Arbok → EQ 4x.
    Ghost → Surf (EQ is 0x Levitate — Jonny, this is why we do not EQ Gengar).
    Agatha non-ghost → EQ or Skull Bash, never the last Surf.
    Everyone else → max connecting damage (never Agility / Detect).
    `avoid_move_id` = Cloyster Protect just ate this move; rotate off it."""
    def _idx(mid):
        if mid == avoid_move_id:
            return None
        for i, m in enumerate(moves or []):
            if (m.get("id") == mid and (m.get("pp") or 0) > 0
                    and (m.get("power") or 0) > 0):
                return i
        return None

    if active_sp == BLASTOISE_SP:
        ghost = agatha_foe_is_ghost(enemy_types, enemy_species)
        if ghost:
            return _idx(MOVE_SURF)
        if lorelei_foe_is_water(enemy_types, enemy_species):
            return _idx(MOVE_EARTHQUAKE)
        if bruno_foe_is_onix(enemy_types, enemy_species):
            return _idx(MOVE_SURF) if _idx(MOVE_SURF) is not None else _idx(MOVE_EARTHQUAKE)
        if agatha_foe_is_arbok(enemy_types, enemy_species):
            eq = _idx(MOVE_EARTHQUAKE)
            if eq is not None:
                return eq
            if seat == "Agatha":
                return _idx(MOVE_SKULL_BASH)
        elif seat == "Agatha" and not ghost:
            eq = _idx(MOVE_EARTHQUAKE)
            if eq is not None and not agatha_foe_is_golbat(enemy_types, enemy_species):
                return eq
            bash = _idx(MOVE_SKULL_BASH)
            if bash is not None:
                return bash
    if active_sp == ZAPDOS_SP and lance_foe_is_gyarados(enemy_types, enemy_species):
        # Live Lance wipe: Shock Wave "didn't fire" then Agility. The gun
        # (Thunderbolt / Thunder / Shock Wave) is the 4x — never Drill Peck.
        for mid in (MOVE_THUNDERBOLT, MOVE_THUNDER, 351):
            i = _idx(mid)
            if i is not None:
                return i
    if active_sp == ARTICUNO_SP:
        # Ice Beam is the only connecting hit (Mist/Agility/Detect are setup).
        # vs Gyarados 1x STAB still beats Agility; vs dragons it is 4x.
        ib = _idx(MOVE_ICE_BEAM)
        if ib is not None and not articuno_ice_into_water_ice(enemy_types, enemy_species):
            return ib
    return e4_max_damage_index(moves, active_sp, enemy_types, enemy_species,
                               avoid_move_id=avoid_move_id)


def e4_eq_pp_dry(moves, active_sp, enemy_types, enemy_species=None):
    """PURE. Blastoise vs Water/Arbok and Earthquake exists at 0 PP (Surf still
    having PP is NOT a reason to skip Elixir — Cloyster/Slowbro/Lapras wipe)."""
    if active_sp != BLASTOISE_SP:
        return False
    if not (lorelei_foe_is_water(enemy_types, enemy_species)
            or agatha_foe_is_arbok(enemy_types, enemy_species)):
        return False
    found = False
    for m in moves or []:
        if m.get("id") == MOVE_EARTHQUAKE:
            found = True
            if (m.get("pp") or 0) > 0:
                return False
    return found


def e4_thin_party_revive_index(rows, prefer_species=()):
    """PURE. When 1-2 are alive and anyone is fainted, spend the turn.
    Returns the party index of the preferred fainted species, else the
    highest-level fainted. None = 3+ still standing, nobody down, or wipe.

    Jonny 2026-08-14: 1 alive / 3 dead or 2 alive / 2 dead -> Revive, not
    a random attack (live Agatha oracle keep_fighting; live Lance empty kit).
    rows = [{row, species, hp, level}] or (species, hp, level) aligned to slot.
    """
    parsed = []
    for i, r in enumerate(rows or []):
        if isinstance(r, dict):
            idx = r.get("row", i)
            sp, hp, lv = r.get("species"), r.get("hp") or 0, r.get("level") or 0
        elif r:
            idx = i
            sp = r[0]
            hp = r[1] if len(r) > 1 else 0
            lv = r[2] if len(r) > 2 else 0
        else:
            continue
        if not sp:
            continue
        parsed.append((int(idx), int(sp), int(hp), int(lv)))
    alive = [p for p in parsed if p[2] > 0]
    dead = [p for p in parsed if p[2] <= 0]
    if not dead or not alive or len(alive) > 2:
        return None
    for want in tuple(prefer_species or ()):
        for idx, sp, _hp, _lv in dead:
            if sp == want:
                return idx
    dead.sort(key=lambda p: -p[3])
    return dead[0][0]


def e4_fainted_wincon_slot(b, enemy_types, enemy_species=None, seat=None):
    """Fainted party slot that is the E4 wincon for THIS foe, or None.
    Blastoise with Earthquake PP vs Water; Moltres vs Jynx; Articuno vs
    Lance dragons (Ice Beam 4x Dragonite — live wipe left him fainted
    with 3 Revives unused). Agatha: fainted Blastoise (Surf vs Ghost).
    Any fainted legendary bird when alive<=2 is also a wincon (live
    Agatha: Zapdos down, Revive x2, worthy=None because L74 Blastoise
    outleveled the L50 birds). Order is e4_revive_pref — not first
    fainted row, not highest-level (live 12:34 Zapdos over Articuno)."""
    import pokemon_state as st
    try:
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        rows = []
        zap_elec = False
        for s in range(min(cnt, 6)):
            sp = st.read_party_species(b, s)
            hp = b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56)
            rows.append({
                "row": s, "species": sp, "hp": hp,
                "level": b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54),
            })
            if sp == ZAPDOS_SP:
                zap_elec = slot_has_electric_damage(b, s)
        hit = e4_revive_target_from_rows(
            rows, seat, enemy_types, enemy_species, zap_elec)
        if hit is not None:
            return hit
        water = lorelei_foe_is_water(enemy_types, enemy_species)
        jynx = lorelei_foe_is_jynx(enemy_types, enemy_species)
        dragon = lance_foe_is_dragon(enemy_types, enemy_species)
        ghost = agatha_foe_is_ghost(enemy_types, enemy_species)
        alive_n = 0
        blast_ghost = None
        bird_down = None
        for s in range(min(cnt, 6)):
            hp = b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56)
            sp = st.read_party_species(b, s)
            if not sp:
                continue
            if hp > 0:
                alive_n += 1
                continue
            if water and sp == BLASTOISE_SP:
                # EQ PP=0 still: Skull Bash/Bite beat Articuno Ice 0.25x into Lapras
                # (live 18:44: wincon skipped dry-EQ Blastoise, Revive aimed wrong).
                return s
            if jynx and sp == MOLTRES_SP:
                return s
            if dragon and sp == ARTICUNO_SP:
                return s
            if (lance_foe_is_gyarados(enemy_types, enemy_species)
                    and sp == ZAPDOS_SP and slot_has_electric_damage(b, s)):
                return s
            if agatha_foe_is_arbok(enemy_types, enemy_species) and sp == BLASTOISE_SP:
                return s
            if enemy_species == PIDGEOT_SP and sp == ARTICUNO_SP:
                return s
            if enemy_species == VENUSAUR_SP and sp == MOLTRES_SP:
                return s
            if enemy_species == RHYDON_SP and sp == BLASTOISE_SP:
                return s
            if ghost and e4_is_ghost_wincon(sp, enemy_types, enemy_species):
                blast_ghost = s
            if sp in LEGENDARY_BIRDS and bird_down is None:
                bird_down = s
        if blast_ghost is not None:
            return blast_ghost
        if alive_n <= 2 and bird_down is not None:
            return bird_down
        # Thin party: ANY fainted teammate is worth the turn (Blastoise
        # down, two birds standing — live "she attacks instead of Revive").
        if alive_n <= 2:
            rows = []
            for s in range(min(cnt, 6)):
                rows.append({
                    "row": s,
                    "species": st.read_party_species(b, s),
                    "hp": b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56),
                    "level": b.rd8(ram.GPLAYER_PARTY + s * 100 + 0x54),
                })
            return e4_thin_party_revive_index(
                rows, prefer_species=(BLASTOISE_SP,) + tuple(LEGENDARY_BIRDS))
    except Exception:
        return None
    return None


def log_e4_kit(camp, log):
    """LOUD kit line. At the Center a thin broke kit is a money errand;
    mid-gauntlet (CloseEntry) still fights with what's aboard."""
    try:
        fr = camp.bag_count(FULL_RESTORE)
        mxp = camp.bag_count(MAX_POTION)
        hyper = camp.bag_count(21)
        superp = camp.bag_count(22)
        pot = camp.bag_count(13)
        rev = camp.bag_count(REVIVE)
        ether = camp.bag_count(ITEM_ETHER)
        elixir = camp.bag_count(ITEM_ELIXIR)
        money = camp.money()
        log(f"   [e4] kit: FR x{fr} MaxPot x{mxp} Hyper x{hyper} Super x{superp} "
            f"Potion x{pot} Revive x{rev} Ether x{ether} Elixir x{elixir} "
            f"money=${money}")
        here = None
        try:
            here = tuple(tv.map_id(camp.b))
        except Exception:
            here = None
        if (fr + mxp + hyper + superp + pot + rev + ether + elixir) <= 0:
            if here == LEAGUE_CENTER:
                log(f"   [e4] !! KIT EMPTY money=${money} at Center — "
                    "sell loot / VS Seeker rematches before Lorelei (LOUD)")
            else:
                log(f"   [e4] !! KIT EMPTY money=${money} — no potions/revives/ethers; "
                    f"fight with what's standing (cannot shop mid-E4) (LOUD)")
    except Exception as e:
        log(f"   [e4] kit read failed ({e}) — LOUD")


def log_box_help(camp, log):
    """Recon the PC. Keep the team of 4 unless the box has something that
    actually helps E4 (not L19 Kadabra / L26 Lapras). Log-only — no withdraw
    mid-gauntlet (live stream; empty seats stay empty)."""
    _chaff = {63, 64, 131, 50, 51}  # Abra, Kadabra, Lapras, Diglett, Dugtrio
    try:
        _cb, occ = camp._box_scan()
        names = []
        useful = []
        for (_bx, _sl), sp in sorted(occ.items()):
            nm = None
            try:
                import pokemon_state as st
                nm = st.SPECIES_NAME.get(sp, str(sp))
            except Exception:
                nm = str(sp)
            names.append(nm)
            if sp not in _chaff and sp not in LEGENDARY_BIRDS and sp != BLASTOISE_SP:
                useful.append(nm)
        log(f"   [e4] box recon: {len(occ)} boxed ({', '.join(names[:12]) or 'empty'})")
        if useful:
            log(f"   [e4] box has possible help {useful[:6]} — NOT withdrawing mid-gauntlet "
                f"(team of 4 is the plan; empty seats stay empty)")
        else:
            log("   [e4] box has nothing E4-useful (Kadabra/Lapras/chaff) — keep 4, make the 4 work")
    except Exception as e:
        log(f"   [e4] box recon skipped ({e})")


def _revive_seat_answer_first(camp, log, seat):
    """Revive EVERY fainted wincon in seat order before walking north.
    Zapdos first before Lance, then Blastoise, then birds. Live 21:08:
    1 Revive, slot-order stood Blastoise up, Zapdos stayed dead, Articuno
    Agility'd Gyarados. A stack + this loop consumes Revives on the
    overworld so she does not enter the next seat with corpses."""
    import pokemon_state as st
    import hm_teach as ht
    if not seat:
        return 0
    order = e4_between_room_revive_species(seat)
    cnt = camp.b.rd8(ram.GPLAYER_PARTY_CNT)
    n = 0
    for want_sp in order:
        rid = next((i for i in (24, 25) if camp.bag_count(i) > 0), None)
        if rid is None:
            break
        for s in range(min(cnt, 6)):
            if st.read_party_species(camp.b, s) != want_sp:
                continue
            if camp.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) > 0:
                continue
            nm = st.SPECIES_NAME.get(want_sp, f"slot{s}").title()
            log(f"   [e4] BETWEEN-ROOM: Revive -> {nm} "
                f"(seat={seat}; every fainted wincon, Zapdos first before Lance)")
            rr = ht.TeachFlow(camp, log=log,
                              on_event=getattr(camp, "on_event", None)).field_revive(rid, s)
            if rr == "revived":
                n += 1
            else:
                log(f"   [e4] BETWEEN-ROOM priority revive -> {rr} — trying next wincon")
            break
    return n


def e4_heal_pocket_empty(fr=0, max_potion=0, hyper=0, super_potion=0, potion=0):
    """PURE. True when no HP bottle is in the bag. field_heal already
    considers Potion/Super — the 12:11 'EMPTY' skip was a true empty,
    not a Super-ignore. Tests lock that Super/Potion count."""
    return (int(fr or 0) + int(max_potion or 0) + int(hyper or 0)
            + int(super_potion or 0) + int(potion or 0)) <= 0


def e4_should_switch_dying_lead(lead_hp, lead_max, healthy_reserve,
                                crit_frac=E4_LEAD_CRIT_FRAC):
    """PURE. After the item pass: dying lead + a healthy reserve = SWITCH.
    Empty bag ≠ send the dying ace (live 12:11: 17/240 Blastoise, three
    healthy legendaries, walked into Lance)."""
    try:
        hp, mx = int(lead_hp or 0), int(lead_max or 0)
    except (TypeError, ValueError):
        return False
    if not healthy_reserve or mx <= 0 or hp <= 0:
        return False
    return (hp / mx) <= float(crit_frac)


def _healthy_reserve_exists(b, min_frac=0.35):
    """Another living party mon (not slot 0) at/above min_frac HP."""
    try:
        cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(1, min(cnt, 6)):
            hp = b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56)
            mx = b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x58)
            if mx > 0 and hp > 0 and (hp / mx) >= min_frac:
                return True
    except Exception:
        return False
    return False


def switch_dying_lead(camp, log, seat=None):
    """Between-room / E4-room: if the lead is still critical after items,
    field the next seat's answer. Log loudly — this is the 12:11 wipe-prevention."""
    if st_in_battle(camp.b):
        return 0
    try:
        hp = camp.b.rd16(ram.GPLAYER_PARTY + 0x56)
        mx = camp.b.rd16(ram.GPLAYER_PARTY + 0x58)
    except Exception:
        return 0
    if not e4_should_switch_dying_lead(hp, mx, _healthy_reserve_exists(camp.b)):
        return 0
    name = seat or next_uncleared_seat(camp.b)
    log(f"   [e4] BETWEEN-ROOM SWITCH: lead {hp}/{mx} ({(hp / mx) if mx else 0:.0%}) "
        f"and the heal pocket cannot save them — fielding the {name} answer "
        f"(empty bag ≠ send the dying ace; live 12:11 Lance 17 HP)")
    sp = apply_answer_lead(camp, log, seat_name=name)
    return 1 if sp else 0


def between_room_heal(camp, log, seat=None):
    """BETWEEN-ROOM HEAL. After an E4 trainer faints, BEFORE the next LOS:
    Revive every fainted wincon + Hyper/Max/Full Restore the party, Elixir EQ
    if dry. Bag items only — CloseEntry sealed the south door, so there is
    no Center walk mid-gauntlet (live 11:38 Bruno (6,10) restock-south loop).
    `seat` = the room she is ABOUT to fight (scarce-Revive order)."""
    if st_in_battle(camp.b):
        log("   [e4] between-room heal skipped — still in battle")
        return 0
    log_e4_kit(camp, log)
    n = 0
    try:
        n += _revive_seat_answer_first(camp, log, seat)
    except Exception as e:
        log(f"   [e4] BETWEEN-ROOM priority revive skipped ({e}) — LOUD")
    try:
        n += camp.field_heal_check(reason="e4-between-room", party_wide=True, force=True)
        log(f"   [e4] BETWEEN-ROOM HEAL: field_heal_check -> {n} "
            f"(items between rooms, not the Center)")
    except Exception as e:
        log(f"   [e4] BETWEEN-ROOM HEAL field pass skipped ({e}) — LOUD")
    try:
        import pokemon_state as st
        import hm_teach as ht
        slot = _alive_slot_of(camp.b, BLASTOISE_SP)
        if slot is not None:
            moves = list(st.read_party_moves(camp.b, slot) or [])
            pps = list(st.read_party_pp(camp.b, slot) or [])
            eq_i = next((i for i, m in enumerate(moves) if m == MOVE_EARTHQUAKE), None)
            if eq_i is not None and (pps[eq_i] if eq_i < len(pps) else 1) <= 0:
                item = next((i for i in E4_ELIXIR_PREF if camp.bag_count(i) > 0), None)
                if item is None:
                    log("   [e4] BETWEEN-ROOM: Blastoise Earthquake PP=0 and NO Ether/Elixir "
                        "in bag — Skull Bash next fight, never Surf (LOUD)")
                else:
                    log(f"   [e4] BETWEEN-ROOM: Earthquake PP=0 — field Elixir/Ether item {item} "
                        f"on Blastoise slot {slot}")
                    res = ht.TeachFlow(camp, log=log,
                                       on_event=getattr(camp, "on_event", None)).field_pp_restore(
                        item, slot)
                    log(f"   [e4] BETWEEN-ROOM PP restore -> {res}")
                    if res == "restored":
                        n += 1
    except Exception as e:
        log(f"   [e4] BETWEEN-ROOM PP restore skipped ({e}) — LOUD")
    # field_heal_check -> 0 is NOT the end. Empty/failed heal + dying lead
    # + a healthy reserve = SWITCH (live 12:11 17/240 into Lance).
    try:
        n += switch_dying_lead(camp, log, seat=seat)
    except Exception as e:
        log(f"   [e4] BETWEEN-ROOM SWITCH skipped ({e}) — LOUD")
    # HOLD: dying ace + potions still in bag, or a corpse + Revive still
    # in bag. Revive cannot heal a living ace — that is switch + go.
    try:
        hp = camp.b.rd16(ram.GPLAYER_PARTY + 0x56)
        mx = camp.b.rd16(ram.GPLAYER_PARTY + 0x58)
        pots = sum(camp.bag_count(i) for i in HEAL_POCKET_IDS)
        rev = camp.bag_count(REVIVE) + camp.bag_count(25)
        fainted_n = 0
        cnt = camp.b.rd8(ram.GPLAYER_PARTY_CNT)
        for s in range(min(cnt, 6)):
            if camp.b.rd16(ram.GPLAYER_PARTY + s * 100 + 0x56) <= 0:
                if __import__("pokemon_state").read_party_species(camp.b, s):
                    fainted_n += 1
        if e4_should_hold_north(hp, mx, pots, rev, fainted_n):
            log(f"   [e4] BETWEEN-ROOM HOLD north — items still on the wincon "
                f"(lead {hp}/{mx}, heals x{pots}, Revive x{rev}, fainted={fainted_n})")
            n += _revive_seat_answer_first(camp, log, seat)
            n += camp.field_heal_check(reason="e4-between-room-hold",
                                       party_wide=True, force=True)
            n += switch_dying_lead(camp, log, seat=seat)
        elif seat == "Lance" and (rev + pots) <= 1:
            log(f"   [e4] BETWEEN-ROOM: Lance will be attempted on a thin kit "
                f"(Revive x{rev} heals x{pots}; lead {hp}/{mx}) — CloseEntry "
                f"sealed the shop; do not invent items")
    except Exception as e:
        log(f"   [e4] BETWEEN-ROOM hold/thin-kit log skipped ({e}) — LOUD")
    return n


def _best_move_eff(move_types_powers, defender_types):
    import pokemon_policy as pp
    best = 0.0
    for mt, mpow in move_types_powers:
        if mpow and mpow > 0 and defender_types:
            best = max(best, pp.effectiveness(mt or "normal", defender_types))
    return best


def pick_lead_slot(b, roster, banned=()):
    """Best living party slot vs a known roster: first-mon smash, then rest-of-team coverage,
    then not-getting-wrecked, then HP, then level. MOVE-GATED (an SE typing with no SE move
    is a decoy). `banned` species ids are skipped (Lorelei: Blastoise + Articuno)."""
    import pokemon_state as st
    import pokemon_policy as pp
    if not roster:
        return None
    first_types = [t for t in (roster[0].get("types") or []) if t]
    rest = roster[1:]
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    best, best_key = None, None
    banned = frozenset(banned or ())
    for s in range(min(cnt, 6)):
        base = ram.GPLAYER_PARTY + s * 100
        hp, mx = b.rd16(base + 0x56), b.rd16(base + 0x58)
        if hp <= 0 or not mx:
            continue
        if st.read_party_species(b, s) in banned:
            continue
        lv = b.rd8(base + 0x54)
        moves = []
        for mid in st.read_party_moves(b, s):
            if not mid:
                continue
            mt, mpow = st.move_info(b, mid)
            moves.append((mt or "normal", mpow or 0))
        first_eff = _best_move_eff(moves, first_types)
        rest_scores = []
        for mon in rest:
            dt = [t for t in (mon.get("types") or []) if t]
            if dt:
                rest_scores.append(_best_move_eff(moves, dt))
        rest_eff = (sum(rest_scores) / len(rest_scores)) if rest_scores else 0.0
        my_types = st.species_types(st.read_party_species(b, s)) or []
        cdef = 1.0
        for et in first_types:
            cdef = max(cdef, pp.effectiveness(et, my_types) if my_types else 1.0)
        key = (first_eff, rest_eff, -cdef, hp / mx >= 0.5, lv)
        if best_key is None or key > best_key:
            best, best_key = s, key
    return best, best_key


def apply_answer_lead(camp, log, seat_name=None):
    """Overworld-only save-safe swap of slot 0 to the seat's answer. Called from
    _enter_league (before the door) AND from EliteFour (each room + post-whiteout heal).
    Lorelei: Zapdos iff he has Electric damage; else Blastoise. Never Articuno."""
    try:
        if st_in_battle(camp.b):
            return None
        name = seat_name or next_uncleared_seat(camp.b)
        import pokemon_state as st
        zap = _alive_slot_of(camp.b, ZAPDOS_SP)
        zap_elec = bool(zap is not None and slot_has_electric_damage(camp.b, zap))
        banned = lorelei_banned_species(zap_elec) if name == "Lorelei" else frozenset()
        best = preferred_lead_slot(camp.b, name)
        why, key = "preferred", None
        if best is None:
            roster = roster_for_seat(camp.b, name)
            picked = pick_lead_slot(camp.b, roster, banned=banned)
            if not picked or picked[0] is None:
                if name == "Lorelei":
                    log("   [e4] ANSWER-LEAD Lorelei: no living Zapdos-with-gun / "
                        "Blastoise / Moltres — fighting the standing order (LOUD)")
                return None
            best, key = picked
            why = "scorer"
        sp_id = st.read_party_species(camp.b, best)
        if name == "Lorelei" and sp_id in banned:
            log(f"   [e4] ANSWER-LEAD Lorelei: BLOCKED {st.SPECIES_NAME.get(sp_id, sp_id)} "
                f"— Articuno never / Blastoise only without Zapdos's gun")
            return None
        sp = st.SPECIES_NAME.get(sp_id, f"slot{best}")
        if best == 0:
            log(f"   [e4] ANSWER-LEAD {name}: {sp} already leads [{why}]")
            return sp
        first = "?"
        try:
            first = (roster_for_seat(camp.b, name) or [{}])[0].get("species") or "?"
        except Exception:
            pass
        log(f"   [e4] ANSWER-LEAD {name}: fielding {sp} vs opener {first} [{why}]")
        camp._swap_party_slots(0, best)
        return sp
    except Exception as e:
        log(f"   [e4] answer-lead skipped ({e}) — fighting with the standing order (LOUD)")
        return None


def _party_slot_of(b, species_id):
    """Any party slot of species_id (alive or fainted), or None."""
    import pokemon_state as st
    cnt = b.rd8(ram.GPLAYER_PARTY_CNT)
    for s in range(min(cnt, 6)):
        if st.read_party_species(b, s) == species_id:
            return s
    return None


def teach_zapdos_electric(camp, log):
    """Overworld-only: teach TM24 Thunderbolt (else TM25 Thunder) to Zapdos
    BEFORE the League door, using the existing TeachFlow. Overwrite Agility or
    Detect — never Drill Peck. No-op in battle / if already has Electric damage /
    if neither TM is in the case. Returns taught|already|not_in_case|in_battle|..."""
    if st_in_battle(camp.b):
        log("   [e4] zapdos-TM: skip — in battle")
        return "in_battle"
    import pokemon_state as st
    import hm_teach as ht
    slot = _party_slot_of(camp.b, ZAPDOS_SP)
    if slot is None:
        log("   [e4] zapdos-TM: no Zapdos in party")
        return "no_zapdos"
    moves = list(st.read_party_moves(camp.b, slot) or [])
    if moveset_has_electric_damage(moves):
        log("   [e4] zapdos-TM: already has Electric damage — skip")
        return "already"
    cand = None
    for item, move_id, label in (
            (ITEM_TM24, MOVE_THUNDERBOLT, "TM24 Thunderbolt"),
            (ITEM_TM25, MOVE_THUNDER, "TM25 Thunder"),
            (ITEM_TM34, 351, "TM34 Shock Wave")):
        if ht.tm_case_row(camp.b, item) is not None:
            cand = (item, move_id, label)
            break
    if cand is None:
        log("   [e4] zapdos-TM: no TM24/TM25/TM34 in case — Blastoise leads Lorelei")
        return "not_in_case"
    item, move_id, label = cand
    forget = zapdos_forget_idx(moves)
    if forget is None and len([m for m in moves if m]) >= 4:
        log("   [e4] zapdos-TM: no safe forget slot (won't overwrite Drill Peck)")
        return "no_forget"
    log(f"   [e4] zapdos-TM: teaching {label} -> slot {slot} forget_idx={forget}")
    try:
        tf = ht.TeachFlow(camp, log=log, on_event=getattr(camp, "on_event", None))
        res = tf.teach("_tm", slot, forget, item_override=item, move_override=move_id)
    except Exception as e:
        log(f"   [e4] zapdos-TM: TeachFlow crashed ({e}) — LOUD")
        return "failed"
    log(f"   [e4] zapdos-TM: {label} -> {res}")
    return res


def st_in_battle(b):
    try:
        import pokemon_state as st
        return bool(st.in_battle(b))
    except Exception:
        return False


class EliteFour:
    def __init__(self, camp, log, dbg_dir=None):
        self.camp = camp
        self.b = camp.b
        self.log = log
        self.dbg = dbg_dir
        if dbg_dir:
            try:
                os.makedirs(dbg_dir, exist_ok=True)
            except Exception:
                self.dbg = None
        self.n_battles = 0
        # Whiteout laps compound the ace's XP; give the gauntlet room to converge (recon_e4 ran a 4h window,
        # but in-loop this is ONE decision — bound it so a genuinely thin team returns 'stuck' to grind).
        self.deadline = time.time() + int(os.getenv("POKEMON_E4_DEADLINE_S", "5400"))

    # ── snap / battle / dialogue drains (victory_road/giovanni shape) ────────────────────────────────────
    def snap(self, name):
        if not self.dbg:
            return
        try:
            self.b.frame_rgb().resize((480, 320)).save(os.path.join(self.dbg, name + ".png"))
        except Exception as e:
            self.log(f"   snap {name} failed: {e}")

    def fight(self):
        self.n_battles += 1
        return self.camp.battle_runner()

    def fight_open(self):
        # pointer validity + LIVENESS (gMain.callback2) — a stale res_ptr after a whiteout is a corpse, not a
        # fight (the run3 phantom re-attach livelock).
        return (ram.valid_ewram_ptr(self.b.rd32(ram.GBATTLE_RES_PTR))
                and not ram.battle_cb2_dead(self.b))

    def drain(self, max_n=60, key="B"):
        b, camp = self.b, self.camp
        stable = 0
        for _ in range(max_n):
            if self.fight_open():
                return
            if dd_box(b):
                stable = 0
                b.press(key, 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    return
                for _ in range(30):
                    b.run_frame()

    def handle_interrupts(self):
        if self.fight_open():
            self.fight()
            self.drain()
            return True
        if dd_box(self.b):
            self.drain()
            return True
        return False

    def settle(self, n=90):
        for _ in range(n):
            self.b.run_frame()

    def lead_frac(self):
        b = self.b
        cur, mx = b.rd16(ram.GPLAYER_PARTY + 0x56), b.rd16(ram.GPLAYER_PARTY + 0x58)
        return (cur / mx) if mx else 1.0

    def party_alive(self):
        b = self.b
        n = 0
        for i in range(6):
            base = ram.GPLAYER_PARTY + i * 100
            if b.rd16(base + 0x56) > 0 and b.rd16(base + 0x58) > 0:
                n += 1
        return n

    # Room # -> the seat's threat-KB name (frlg_strategy.json 'threats'; seat order is fixed).
    _SEAT_NAME = {1: "Lorelei", 2: "Bruno", 3: "Agatha", 4: "Lance", 5: "Gary"}

    def answer_lead(self, room_no=None, seat_name=None):
        """ANSWER-LEAD THE SEAT: overworld reorder so the smash vs that trainer's KNOWN
        first mon (then rest-of-roster coverage) leads — Zapdos into Lorelei's Dewgong,
        Articuno into Bruno's Onix / Lance's dragons, Moltres into Gary's Venusaur-line.
        Map/flag keyed (not seen_rooms count) so a whiteout retry still fields the opener.
        Called BEFORE walking into LOS. Existing _swap_party_slots; fail-open."""
        try:
            if self.fight_open():
                return
            here = tuple(tv.map_id(self.b))
            name = (seat_name or ROOM_SEAT.get(here)
                    or self._SEAT_NAME.get(room_no)
                    or next_uncleared_seat(self.b))
            sp = apply_answer_lead(self.camp, self.log, name)
            if sp:
                self.on_event_safe(f"{name} next — {sp} takes point. that's the matchup.", tier=2)
        except Exception as e:
            self.log(f"   [e4] answer-lead skipped ({e}) — fighting with the standing order (LOUD)")

    def live_npc_tiles(self):
        b = self.b
        OB, SZ = 0x02036E38, 0x24
        out = []
        for i in range(1, 16):
            o = OB + i * SZ
            if not (b.rd8(o) & 1):
                continue
            out.append((b.rds16(o + 0x10) - tv.MAP_OFFSET,
                        b.rds16(o + 0x12) - tv.MAP_OFFSET))
        return out

    # ── interior walk / warp (plain — the League maps have no water/boulders) ────────────────────────────
    def walk(self, goal_test, label, tries=12, allow=()):
        b, camp = self.b, self.camp
        budget = tries
        frozen, last_pos = 0, None
        map_start = tuple(tv.map_id(b))
        while budget > 0:
            budget -= 1
            if self.handle_interrupts():
                budget += 1
                continue
            if tuple(tv.map_id(b)) != map_start:
                self.log(f"   [{label}] map changed {map_start} -> {tuple(tv.map_id(b))} — bail")
                return False
            cur = tuple(tv.coords(b) or (0, 0))
            if goal_test(cur):
                return True
            # FREEZE ARMOR (e4_run1: post-whiteout she planned len-20 paths for 22 replans without moving —
            # a hidden modal box eating inputs dd_box can't see). 3 no-move replans -> blind B/A drain.
            if cur == last_pos:
                frozen += 1
                if frozen >= 3:
                    self.log(f"!! [{label}] FROZEN at {cur} x{frozen} replans — hidden box? blind B/A drain")
                    self.snap(f"frozen_{label[:12]}")
                    for k in ("B", "B", "A", "B"):
                        b.press(k, 8, 12, camp.render, owner="agent")
                        for _ in range(30):
                            b.run_frame()
                    self.drain()
                    frozen = 0
                    budget += 1
            else:
                frozen = 0
            last_pos = cur
            g = tv.Grid(b)
            wts = {tuple(w[0]) for w in tv.read_warps(b)} - set(allow)
            npcs = set(self.live_npc_tiles())
            p = tv.bfs(g, cur, goal_test,
                       walkable=lambda sx, sy: g.walkable(sx, sy)
                       and (sx, sy) not in wts and (sx, sy) not in npcs)
            self.log(f"   [{label}] replan at {cur} (len {len(p) if p else 0}, budget {budget})")
            if not p:
                self.log(f"   [{label}] no path from {cur}")
                self.snap(f"nopath_{label[:12]}")
                return False
            m0 = tuple(tv.map_id(b))
            for t in p[1:]:
                if self.handle_interrupts():
                    budget += 1
                    break
                if not camp._step_to(tuple(t)):
                    break
                if tuple(tv.map_id(b)) != m0:
                    return True
            if goal_test(tuple(tv.coords(b) or ())):
                return True
        return goal_test(tuple(tv.coords(b) or ()))

    def go_warp(self, tile, label):
        """Walk onto `tile` or a neighbor, step in; True on any map change.
        Door cluster (tile + cardinals) is both the goal AND walkable — live
        restock-south: (6,11) is also a warp, BFS excluded it, no path from
        (6,7) to any neighbor of (6,12)."""
        b, camp = self.b, self.camp
        m0 = tuple(tv.map_id(b))
        cluster = e4_warp_approach_cluster(tile)
        nbs = list(cluster[1:]) if cluster else []
        for _attempt in range(6):
            cur = tuple(tv.coords(b) or ())
            if cur != tile and cur not in nbs:
                reached = self.walk(
                    lambda c, s=set(cluster): c in s,
                    f"{label}-approach", allow=cluster)
                if not reached and tuple(tv.map_id(b)) == m0:
                    # BFS refused (dual-warp / NPC in the 1-tile hall).
                    # Step toward the door — never return False on first miss.
                    self.log(f"   [{label}-approach] BFS empty at {cur} — "
                             f"stepping toward {tile} (never abort restock)")
                    dx = tile[0] - (cur[0] if cur else 0)
                    dy = tile[1] - (cur[1] if cur else 0)
                    key = None
                    if abs(dy) >= abs(dx) and dy != 0:
                        key = "DOWN" if dy > 0 else "UP"
                    elif dx != 0:
                        key = "RIGHT" if dx > 0 else "LEFT"
                    if not key:
                        return False
                    b.press(key, 26, 10, camp.render, owner="agent")
                    for _ in range(40):
                        b.run_frame()
                        if tuple(tv.map_id(b)) != m0:
                            break
                    continue
            cur = tuple(tv.coords(b) or (0, 0))
            key = KEY_OF.get((tile[0] - cur[0], tile[1] - cur[1]))
            for _press in range(4):
                if key:
                    b.press(key, 26, 10, camp.render, owner="agent")
                for _ in range(120):
                    b.run_frame()
                    if tuple(tv.map_id(b)) != m0:
                        break
                if tuple(tv.map_id(b)) != m0:
                    break
            if self.handle_interrupts():
                continue
            if tuple(tv.map_id(b)) != m0:
                self.settle(180)
                self.log(f"   [{label}] {m0} -> {tuple(tv.map_id(b))} @ {tv.coords(b)}")
                return True
        self.log(f"!! [{label}] never fired (at {tv.map_id(b)}@{tv.coords(b)})")
        self.snap(f"warpfail_{label[:16]}")
        return False

    # ── the mart engine (tm_errand's true-index machinery, Items-pocket verify) ──────────────────────────
    def shop_index(self):
        return self.b.rd16(0x02039940) + self.b.rd16(0x02039942)

    def list_live(self):
        b, camp = self.b, self.camp
        c0 = self.shop_index()
        b.press("DOWN", 8, 10, camp.render, owner="agent")
        for _ in range(20):
            b.run_frame()
        if self.shop_index() != c0:
            return True
        b.press("UP", 8, 10, camp.render, owner="agent")
        for _ in range(12):
            b.run_frame()
        return False

    def shop_goto_index(self, target, tries=20):
        b, camp = self.b, self.camp
        for _ in range(tries):
            idx = self.shop_index()
            if idx == target:
                for _ in range(20):
                    b.run_frame()
                if self.shop_index() == target:
                    return True
                continue
            b.press("DOWN" if idx < target else "UP", 8, 10, camp.render, owner="agent")
            for _ in range(20):
                b.run_frame()
        return self.shop_index() == target

    def _league_sell_loot(self):
        """Sell cashable Items-pocket loot at the League clerk.

        Live 16:35/16:43: the BUY/SELL/QUIT nav was BLIND — one eaten DOWN on
        the long-running core left the cursor on BUY, the follow-up A opened the
        BUY list, and the sell-bag row walk then bounced off bytes that never
        answer in that screen ('You don't have enough money' / 'couldn't reach
        loot row 0'). Doctrine: never navigate a multichoice blind. Greeting =
        max 2 A's, then DOWN to SELL, then A — and PROVE where we are after the
        A: the pocket byte flips on RIGHT only in the live bag UI (mart SELL
        included, _mart_sell_loot-proven); the BUY-list cursor pair answers
        d-pads only in the buy list. Recover by state (B out of the buy list /
        re-engage the clerk), never by hope. TMs are in the TM Case — this list
        is Items only (Nugget, mushrooms). Never sells Revives/potions.
        """
        b, camp, L = self.b, self.camp, self.log
        have = [(iid, camp.bag_count(iid)) for iid in SELL_LOOT if camp.bag_count(iid) > 0]
        if not have:
            return 0
        m0 = camp.money()
        L(f"   [shop] SELL loot {have} before the kit buy (money ${m0})")
        if not self.walk(lambda c: c == CLERK_STAND, "clerk-sell"):
            return 0
        import hm_teach as ht

        def _pet():
            # Bounded outcome-verified menu drive — never a wedge; keep the roam
            # watchdog from latching on the legitimately-frozen menu fingerprint.
            try:
                if getattr(camp, "_stuckwatch", None) is not None:
                    camp._stuckwatch.reset()
            except Exception:
                pass

        def _open_clerk():
            for _ in range(6):
                b.press("LEFT", 8, 8, camp.render, owner="agent")
                b.press("A", 8, 10, camp.render, owner="agent")
                for _ in range(30):
                    b.run_frame()
                    if dd_box(b):
                        return True
            return False

        def _dismiss_greeting():
            # 'Hi, there! May I help you?' — max 2 A's while the bottom box is
            # up. A third A on the BUY/SELL/QUIT menu confirms BUY (16:35).
            dismissed = 0
            for _ in range(6):
                if dd_box(b) and dismissed < 2:
                    b.press("A", 8, 12, camp.render, owner="agent")
                    dismissed += 1
                    for _ in range(18):
                        b.run_frame()
                    continue
                break
            for _ in range(24):                    # let the multichoice spawn
                b.run_frame()

        def _in_sell_bag():
            # The pocket byte flips on RIGHT ONLY in the live bag UI (mart
            # SELL included). RIGHT probe, then LEFT home to the Items pocket
            # (0), readback each press. Any box-up lap skips probing (the
            # caller gates on dd_box first), so RIGHT never touches a qty box.
            p0 = b.rd8(ram.GBAG_POCKET)
            b.press("RIGHT", 8, 10, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
            p1 = b.rd8(ram.GBAG_POCKET)
            live = p0 is not None and p1 is not None and p1 != p0
            for _ in range(4):
                if b.rd8(ram.GBAG_POCKET) == 0:
                    break
                b.press("LEFT", 8, 10, camp.render, owner="agent")
                for _ in range(16):
                    b.run_frame()
            return live and b.rd8(ram.GBAG_POCKET) == 0

        def _in_buy_list():
            # The BUY-list cursor pair answers d-pads ONLY in the buy list.
            # Probe DOWN (restore with UP); if clamped, probe UP (restore DOWN).
            # At the BUY/SELL menu these presses jiggle the menu cursor invisibly
            # and net zero — harmless. money never moves (no A in here).
            c0 = self.shop_index()
            b.press("DOWN", 8, 10, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
            moved = self.shop_index() != c0
            b.press("UP", 8, 10, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
            if moved:
                return True
            c2 = self.shop_index()
            b.press("UP", 8, 10, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
            moved = self.shop_index() != c2
            b.press("DOWN", 8, 10, camp.render, owner="agent")
            for _ in range(16):
                b.run_frame()
            return moved

        def _menu_select_sell():
            # BUY/SELL/QUIT is up (or the overworld is): DOWN to SELL, then A.
            # The lead LEFT tap is a no-op at the vertical multichoice and a
            # face-the-clerk turn at world control; it also guarantees DOWN is
            # always a FRESH direction (tap-turn law — never an accidental step).
            # At world control the DOWN tap only TURNS and A talks to the clerk
            # again — never harmful, always converges back to the menu.
            b.press("LEFT", 8, 8, camp.render, owner="agent")
            for _ in range(12):
                b.run_frame()
            b.press("DOWN", 8, 12, camp.render, owner="agent")
            for _ in range(18):
                b.run_frame()
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(80):
                b.run_frame()

        if not _open_clerk():
            L("!! [shop] sell: clerk never opened a dialog (LOUD)")
            self.snap("sell_clerk_never_opened")
            return 0
        in_bag = False
        for _try in range(6):
            _pet()
            if camp.money() < m0:
                L(f"!! [shop] sell: money DROPPED ${m0}->${camp.money()} during "
                  "entry — we are buying somewhere; B out LOUD")
                self.snap("sell_entry_money_drop")
                for _ in range(12):
                    b.press("B", 6, 12, camp.render, owner="agent")
                    for _ in range(14):
                        b.run_frame()
                return 0
            if _try and tuple(tv.coords(b) or ()) != CLERK_STAND:
                L(f"   [shop] sell: drifted to {tv.coords(b)} on a retry — "
                  "re-standing at the clerk")
                self.walk(lambda c: c == CLERK_STAND, "clerk-sell-re")
                _open_clerk()
                continue
            if not dd_box(b):
                if _in_sell_bag():
                    in_bag = True
                    if _try:
                        L(f"   [shop] SELL bag confirmed after {_try} retr(ies)")
                    break
                if _in_buy_list():
                    L("   [shop] sell: DOWN was eaten — landed in the BUY list "
                      "(BUY-list cursor answered a probe); B out, retry DOWN to SELL")
                    b.press("B", 8, 12, camp.render, owner="agent")
                    for _ in range(40):
                        b.run_frame()
                    continue
            else:
                _dismiss_greeting()                 # closes to the BUY/SELL menu
            _menu_select_sell()                     # prove WHERE it landed next lap
            continue
        if not in_bag:
            L("!! [shop] sell: SELL bag never confirmed after 6 tries — aborting "
              "LOUD (kit buy continues with what's in the wallet)")
            self.snap("sell_bag_never_confirmed")
            for _ in range(10):
                b.press("B", 6, 12, camp.render, owner="agent")
                for _ in range(14):
                    b.run_frame()
            return 0
        gained = 0
        for _u in range(12):
            _pet()
            if dd_box(b):                           # stray qty/message box from a late A — B it away
                b.press("B", 6, 12, camp.render, owner="agent")
                for _ in range(14):
                    b.run_frame()
            rows = [(i, iid) for i, (iid, _q) in enumerate(ht.items_pocket_rows(b))
                    if iid in SELL_LOOT]
            if not rows:
                break
            rows.sort(key=lambda t: SELL_LOOT.index(t[1]))
            row, iid = rows[0]
            if not camp._sell_goto_row(row):
                L(f"!! [shop] sell: couldn't reach loot row {row} with the SELL "
                  "bag CONFIRMED open — cursor wedged; B out LOUD")
                break
            _pet()
            b.press("A", 8, 12, camp.render, owner="agent")   # select the row
            for _ in range(24):
                b.run_frame()
            sel = b.rd16(ram.GSPECIALVAR_ITEMID)
            if sel not in SELL_LOOT:
                L(f"   [shop] sell: selected id {sel} not loot ({SELL_LOOT}) "
                  "— B, retry the row (select A eaten or stale read)")
                b.press("B", 6, 12, camp.render, owner="agent")
                for _ in range(14):
                    b.run_frame()
                continue
            q0 = ht.items_pocket_qty(b, iid)
            mm = camp.money()
            got = 0
            _dbg = os.getenv("POKEMON_SELL_DEBUG") == "1"
            for _k in range(16):
                b.press("A", 6, 10, camp.render, owner="agent")
                for _ in range(12):
                    b.run_frame()
                got = camp.money() - mm
                if _dbg:
                    L(f"   [shop] sell A#{_k}: money={camp.money()} "
                      f"sel={b.rd16(ram.GSPECIALVAR_ITEMID)} box={dd_box(b)} "
                      f"ow_idx={camp._sell_list_index()}")
                    if _k in (0, 2, 4):
                        self.snap(f"sell_dbg_A{_k}")
                if got > 0:
                    break
            for _ in range(8):                      # drain the payout message back to the list
                if not dd_box(b):
                    break
                b.press("A", 6, 10, camp.render, owner="agent")
                for _ in range(12):
                    b.run_frame()
            if got <= 0 or ht.items_pocket_qty(b, iid) != q0 - 1:
                L(f"!! [shop] sell verify failed item {iid} (+${got}, "
                  f"x{q0}->x{ht.items_pocket_qty(b, iid)}) — stopping LOUD")
                try:
                    L(f"   [shop] sell-fail debug: pocket={b.rd8(ram.GBAG_POCKET)} "
                      f"ow_index={camp._sell_list_index()} "
                      f"sel={b.rd16(ram.GSPECIALVAR_ITEMID)} box={dd_box(b)}")
                except Exception as _de:
                    L(f"   [shop] sell-fail debug read error: {_de}")
                self.snap("sell_verify_fail")
                break
            gained += got
            L(f"   [shop] sold item {iid} for ${got} (money ${camp.money()})")
        for _ in range(10):
            b.press("B", 6, 12, camp.render, owner="agent")
            for _ in range(14):
                b.run_frame()
        self.drain()
        _pet()
        m1 = camp.money()
        L(f"   [shop] sold loot ${m0} -> ${m1} (gained ${gained})")
        return max(0, m1 - m0)

    def _exit_center_to_indigo(self):
        """Leave the League Center the way heal_nearest already does:
        walk to (11,15), DOWN onto (11,16). go_warp(11,16) oscillated
        (4,16)<->(18,16) live 16:35 and never fired."""
        b, camp = self.b, self.camp
        here = tuple(tv.map_id(b))
        if here == INDIGO_EXT:
            return True
        if here != LEAGUE_CENTER:
            return False
        if tuple(tv.coords(b) or ()) != CENTER_MAT:
            if not self.walk(lambda c: c == CENTER_MAT, "center-mat"):
                self.log("!! [e4] center-mat (11,15) unreachable")
                return False
        m0 = tuple(tv.map_id(b))
        b.press("DOWN", 26, 10, camp.render, owner="agent")
        for _ in range(180):
            b.run_frame()
            if tuple(tv.map_id(b)) != m0:
                self.settle(90)
                self.log(f"   [center-exit] {m0} -> {tuple(tv.map_id(b))} "
                         f"@ {tv.coords(b)} via (11,15) DOWN")
                return True
        self.log(f"!! [e4] center-exit DOWN from (11,15) did not warp "
                 f"(still {tv.map_id(b)}@{tv.coords(b)})")
        return tuple(tv.map_id(b)) == INDIGO_EXT

    def _use_vs_seeker(self):
        """Bag → Key Items → VS Seeker → USE. Only works on route/city/town."""
        import hm_teach as ht
        camp, L = self.camp, self.log
        if not camp._key_item_owned(ITEM_VS_SEEKER):
            return False
        flow = ht.TeachFlow(camp, log=L)
        opened = False
        flow._press("START", settle=60)
        for _ in range(4):
            c0 = self.b.rd8(ht.START_CURSOR)
            flow._press("DOWN", settle=24)
            if self.b.rd8(ht.START_CURSOR) != c0:
                opened = True
                break
            flow._press("START", settle=60)
        if not opened or not flow._nav_byte(ht.START_CURSOR, 2):
            L("   [e4] VS Seeker: START menu never opened")
            flow._b_cascade()
            return False
        flow._press("A", settle=80)
        for _ in range(4):
            if self.b.rd8(ht.BAG_POCKET) == 1:
                break
            flow._press("RIGHT" if self.b.rd8(ht.BAG_POCKET) < 1 else "LEFT", settle=20)
        if self.b.rd8(ht.BAG_POCKET) != 1:
            L("   [e4] VS Seeker: Key Items pocket missed")
            flow._b_cascade()
            return False
        ki = ht.pocket_items(self.b, ht.KEY_ITEMS_OFF, 30)
        if ITEM_VS_SEEKER not in ki:
            L("   [e4] VS Seeker: not in Key Items after all")
            flow._b_cascade()
            return False
        if not flow._nav_byte(ht.BAG_LIST_CURSOR, ki.index(ITEM_VS_SEEKER)):
            L("   [e4] VS Seeker: list cursor missed")
            flow._b_cascade()
            return False
        flow._press("A", settle=40)
        flow._press("A", settle=80)
        self.drain()
        flow._b_cascade()
        flow._confirm_world_back("vs-seeker")
        L("   [e4] VS Seeker used — rematch trainers should flash")
        return True

    def _fetch_vs_seeker(self):
        """Vermilion Center girl at (6,4). Stand (6,5), face UP, A."""
        b, camp, L = self.b, self.camp, self.log
        if camp._key_item_owned(ITEM_VS_SEEKER):
            return True
        here = tuple(tv.map_id(b))
        if here == VERMILION_CITY:
            try:
                camp.enter_warp(pick=VERMILION_PC_DOOR)
            except Exception as e:
                L(f"   [e4] VS Seeker fetch: PC door ({e}) — LOUD")
        self.settle(60)
        self.drain()
        if not self.walk(lambda c: c == (6, 5), "vs-seeker-girl"):
            L("!! [e4] VS Seeker fetch: couldn't stand in front of the girl")
            return False
        b.press("UP", 8, 8, camp.render, owner="agent")
        for _ in range(12):
            b.run_frame()
        for _ in range(8):
            b.press("A", 8, 12, camp.render, owner="agent")
            for _ in range(30):
                b.run_frame()
            if dd_box(b):
                self.drain(key="A")
                break
        self.drain(key="A")
        got = camp._key_item_owned(ITEM_VS_SEEKER)
        L(f"   [e4] VS Seeker fetch -> {'GOT' if got else 'MISSED'}")
        return got

    def _leave_to_overworld(self):
        """If indoors, take a warp whose dest is Kanto overworld (group 3)."""
        b = self.b
        here = tuple(tv.map_id(b))
        if here[0] == 3:
            return True
        warps = [(tuple(xy), tuple(d)) for xy, d, _w in tv.read_warps(b)]
        street = [xy for xy, dest in warps if dest and dest[0] == 3]
        if not street:
            street = [xy for xy, _d in warps]
        if not street:
            return False
        return self.go_warp(street[0], "leave-building")

    def earn_kit_cash(self):
        """Lobby money sidequest: sell already happened; Fly Vermilion, get
        VS Seeker, rematch Route 11, Fly back to Indigo. Trainer payouts
        are how FireRed makes money. Wilds and gym rematches (post-E4)
        are not this errand."""
        camp, L = self.camp, self.log
        m0 = camp.money()
        need = e4_kit_cash_needed(camp.bag_count(REVIVE),
                                  camp.bag_count(FULL_RESTORE), m0)
        L(f"   [e4] MONEY ERRAND: ${m0}, need ${need} more for the Revive/FR stack")
        self.on_event_safe(
            "four mons and a thin wallet. trainers pay cash in this game. "
            "Vermilion Center has a girl who hands out a rematch gadget — "
            "that's the sidequest, then we buy Revives and come back.",
            tier=2)
        if not self._exit_center_to_indigo():
            L("!! [e4] money errand: can't leave the Center")
            return False
        r = camp.fly_to("vermilion")
        if r != "arrived" and tuple(tv.map_id(self.b)) != VERMILION_CITY:
            L(f"!! [e4] money errand: fly Vermilion -> {r}")
            try:
                camp.fly_to("indigo")
            except Exception:
                pass
            return False
        if not camp._key_item_owned(ITEM_VS_SEEKER):
            if not self._fetch_vs_seeker():
                L("!! [e4] money errand: VS Seeker girl missed — abort, fly home")
                self._leave_to_overworld()
                camp.fly_to("indigo")
                return False
        self._leave_to_overworld()
        try:
            camp.walk_to_map(ROUTE11, "east")
        except Exception as e:
            L(f"   [e4] money errand: walk Route 11 ({e}) — fighting wherever we are")
        battles = 0
        for _lap in range(14):
            if e4_kit_cash_needed(camp.bag_count(REVIVE),
                                  camp.bag_count(FULL_RESTORE),
                                  camp.money()) == 0:
                break
            if battles >= 8:
                break
            if tuple(tv.map_id(self.b))[0] == 3:
                self._use_vs_seeker()
            if self.handle_interrupts():
                battles += 1
                continue
            try:
                camp.talk_npc()
            except Exception:
                pass
            if self.handle_interrupts():
                battles += 1
                continue
            for key in ("RIGHT", "LEFT", "DOWN", "UP"):
                self.b.press(key, 18, 10, camp.render, owner="agent")
                for _ in range(20):
                    self.b.run_frame()
                if self.handle_interrupts():
                    battles += 1
                    break
        self._leave_to_overworld()
        back = camp.fly_to("indigo")
        L(f"   [e4] money errand done ${m0} -> ${camp.money()} "
          f"(battles {battles}, fly-home {back})")
        return camp.money() > m0

    def stock_up(self):
        b, camp, L = self.b, self.camp, self.log
        try:
            self._league_sell_loot()
        except Exception as e:
            L(f"   [shop] sell-first skipped ({e}) — LOUD")
        need = [(iid, row, want - camp.bag_count(iid), price)
                for iid, row, want, price in SHOPPING
                if camp.bag_count(iid) < want]
        money = camp.money()
        # ── THE LANCE LADDER (2026-08-14, measured on the 40-min gauntlet smoke) ──
        # The old COMEBACK FLOOR was {REVIVE: 6, FULL_HEAL: 1}: pass 1 reserved $9,600 of
        # Revives before Full Restores saw a single yen. Measured consequence, EVERY lap:
        # a $6,440 whiteout return bought 4 Revives and ZERO heals, and she walked into
        # Lance with "FR x0 MaxPot x0 Hyper x0 Super x0 Potion x0 Revive x2 money=$17240"
        # — an empty heal pocket with $17k stranded in her pocket (CloseEntry forbids
        # shopping mid-gauntlet, so the kit she carries in is all she gets).
        # A Revive returns an L52 bird at half HP into a wave of L54-62 dragons that one-shot
        # it; a Full Restore returns the L88 ace's 266 HP *and* clears the poison/paralysis
        # the dragons stack. So the floor must never be single-item: walk a ROUND-ROBIN
        # ladder so ANY budget buys a usable MIX. $6,440 -> FR x1 + Revive x2 (a real kit
        # slice) instead of Revive x4 + no heals. Pass 2 below still tops up in SHOPPING
        # order with whatever surplus is left.
        LADDER = (FULL_RESTORE, REVIVE, FULL_RESTORE, REVIVE, FULL_HEAL,
                  FULL_RESTORE, REVIVE, FULL_RESTORE, REVIVE)
        by_id = {iid: (row, n, price) for iid, row, n, price in need}
        alloc = {iid: 0 for iid, _r, _n, _p in need}
        for iid in LADDER:                                    # pass 1: balanced kit first
            if iid not in by_id:
                continue
            _row, n, price = by_id[iid]
            if alloc[iid] >= n or money < price:
                continue
            alloc[iid] += 1
            money -= price
        for iid, row, n, price in need:                       # pass 2: SHOPPING order takes the surplus
            extra = min(n - alloc[iid], max(0, money // price))
            alloc[iid] += extra
            money -= extra * price
        plan = [(iid, row, alloc[iid], price) for iid, row, n, price in need if alloc[iid] > 0]
        if not plan:
            L(f"   [shop] {'kit SHORT but broke' if need else 'stocked already'} (money ${camp.money()})")
            return True
        L(f"   [shop] plan {plan} (money ${camp.money()})")
        if not self.walk(lambda c: c == CLERK_STAND, "clerk-approach"):
            return False
        opened = False
        for _ in range(8):
            b.press("LEFT", 8, 8, camp.render, owner="agent")
            b.press("A", 8, 10, camp.render, owner="agent")
            for _ in range(40):
                b.run_frame()
                if dd_box(b):
                    opened = True
                    break
            if opened:
                break
        if not opened:
            L(f"!! [shop] clerk never opened a dialog (coords {tv.coords(b)})")
            self.snap("shop_no_greeting")
            return False
        stable = 0
        for _ in range(30):
            if dd_box(b):
                stable = 0
                b.press("A", 8, 12, camp.render, owner="agent")
                for _ in range(20):
                    b.run_frame()
            else:
                stable += 1
                if stable >= 2:
                    break
                for _ in range(30):
                    b.run_frame()
        if not self.list_live():
            b.press("A", 8, 10, camp.render, owner="agent")   # BUY (top of BUY/SELL)
            for _ in range(120):
                b.run_frame()
            if not self.list_live():
                L("!! [shop] BUY list didn't confirm — abort shop")
                self.snap("shop_entry_fail")
                return False
        for iid, row, n, price in plan:
            for _u in range(n):
                q0 = camp.bag_count(iid)
                if not self.shop_goto_index(row) or self.shop_index() != row:
                    L(f"!! [shop] couldn't hold index {row} — abort shop")
                    return False
                got = camp._mart_buy_one()
                q1 = camp.bag_count(iid)
                if got <= 0 or got > price + 500 or q1 != q0 + 1:
                    L(f"!! [shop] buy-verify FAILED item {iid} (price={got}, bag x{q0}->x{q1}) — abort shop")
                    self.snap("shop_buy_fail")
                    return False
            L(f"   [shop] bought {n} x item {iid} (bag x{camp.bag_count(iid)}, money ${camp.money()})")
        for _ in range(8):
            b.press("B", 6, 12, camp.render, owner="agent")
            for _ in range(14):
                b.run_frame()
        self.drain()
        try:
            camp._save_campaign("e4_shopped")
        except Exception:
            pass
        return True

    # ── the dispatch loop: exterior -> center (heal+shop) -> door chain -> CREDITS ───────────────────────
    def run(self):
        b, camp, L = self.b, self.camp, self.log
        try:
            if not fm.read_flag(b, FLAG_BADGE_EARTH):
                L("!! [e4] badge 8 not held — not the endgame; abort")
                return "stuck"
        except Exception:
            return "stuck"
        L(f"[e4] boot map={tv.map_id(b)} coords={tv.coords(b)} lead={self.lead_frac():.0%} "
          f"alive={self.party_alive()} money=${camp.money()} FR x{camp.bag_count(FULL_RESTORE)}")
        log_e4_kit(camp, L)
        log_box_help(camp, L)
        center = LEAGUE_CENTER
        shopped = False
        healed = False
        seen_rooms = []               # map ids in door-chain order (Lorelei..Champion)
        prev_here = None
        while time.time() < self.deadline:
            if self.handle_interrupts():
                continue
            here = tuple(tv.map_id(b))
            came_from, prev_here = prev_here, here
            if here == center and seen_rooms and came_from is not None \
                    and came_from not in (center, INDIGO_EXT):
                # back at the center FROM the chain = a whiteout — respawn healed the party but the Full
                # Restore kit is drained: re-check the shop (money-aware; no-op if the bag is still stocked).
                # Walk the gauntlet FROM THE START (Lorelei door). DEFEATED flags persist so beaten
                # rooms pass through; never resume from a wedged mid-gauntlet tile.
                L(f"   [e4] back at the center from the chain (whiteout) — re-checking the kit "
                  f"[money ${camp.money()}, FR x{camp.bag_count(FULL_RESTORE)}]; "
                  f"retry from the League door (next uncleared: {next_uncleared_seat(b)})")
                shopped = False
                healed = False
                seen_rooms = []
                try:
                    import victory_road as _vr
                    _vr.VictoryRoad(camp, L)._restore_escort_party()
                except Exception as _re:
                    L(f"   [e4] steamroll restore after whiteout skipped ({_re}) — LOUD")
            warps = [(tuple(xy), tuple(d)) for xy, d, _w in tv.read_warps(b)]
            if here == INDIGO_EXT:
                if not warps:
                    self.settle(60)
                    continue
                tgt = min(warps, key=lambda w: abs(w[0][0] - 11))[0]
                if not self.go_warp(tgt, "enter-center"):
                    L("!! [e4] can't enter the League center — stuck")
                    return "stuck"
            elif here == center:
                if not healed:
                    r = camp.heal_nearest()
                    L(f"   [e4] heal_nearest -> {r} (lead {self.lead_frac():.0%})")
                    self.drain()
                    healed = self.lead_frac() > 0.99
                    continue
                if not shopped:
                    if not self.stock_up():
                        L("!! [e4] shopping failed — proceeding with what's aboard (LOUD)")
                    shopped = True
                    continue
                # Broke + thin at the Center: do NOT walk into Lorelei.
                # Sell already ran inside stock_up. Earn the rest on Route 11.
                if e4_must_earn(camp.bag_count(REVIVE), camp.money(),
                                camp.bag_count(FULL_RESTORE), here=here):
                    tries = getattr(self, "_earn_tries", 0)
                    if tries < 2:
                        self._earn_tries = tries + 1
                        L(f"   [e4] HOLD THE DOOR: broke + thin kit "
                          f"(Revive x{camp.bag_count(REVIVE)} FR x{camp.bag_count(FULL_RESTORE)} "
                          f"${camp.money()}) — money errand {self._earn_tries}/2 "
                          f"(VS Seeker rematches, then shop)")
                        self.earn_kit_cash()
                        shopped = False
                        healed = False
                        continue
                    L("   [e4] money errand already tried x2 — shop what's left, "
                      "then a poverty walk if Revive stack exists (LOUD)")
                # Last gate before Lorelei: if the kit is still a stub and she
                # has cash, shop AGAIN. Once CloseEntry slams, she cannot come
                # back (live 11:38 Bruno sealed-door loop).
                if e4_must_restock(camp.bag_count(REVIVE), camp.money(),
                                   next_uncleared_seat(b) or "Agatha",
                                   camp.bag_count(FULL_RESTORE),
                                   here=here) and camp.money() >= 1500:
                    L(f"   [e4] HOLD THE DOOR: Center kit still thin "
                      f"(Revive x{camp.bag_count(REVIVE)} FR x{camp.bag_count(FULL_RESTORE)}; "
                      f"need >={RESTOCK_REVIVE_MIN}/{RESTOCK_FR_MIN}) — "
                      f"shop again before Lorelei (cannot buy 1 and walk)")
                    shopped = False
                    continue
                # PRE-DOOR: teach Zapdos TM24/TM25 if we have it, THEN field the honest lead.
                # Without the TM, preferred_lead_slot picks Blastoise (Earthquake), not the pecker.
                try:
                    teach_zapdos_electric(camp, L)
                except Exception as _te:
                    L(f"   [e4] zapdos-TM skipped ({_te}) — LOUD")
                self.answer_lead(seat_name=next_uncleared_seat(b))
                if not self.go_warp(LEAGUE_DOOR, "league-door"):
                    L("!! [e4] League door failed — stuck")
                    return "stuck"
            elif here != center:
                # CloseEntry sealed the south door on room entry (pret
                # PokemonLeague_EventScript_CloseEntry: metatiles 5-7,11-12
                # collision). Live 11:38: KIT THIN → restock-south froze at
                # Bruno (6,10) bumping a closed door. Shop ONLY at the Center
                # before Lorelei. Mid-gauntlet = bag items, then north.
                _nxt = next_uncleared_seat(b)
                if here in ROOM_SEAT:
                    L(f"   [e4] entry door sealed (CloseEntry) in {ROOM_SEAT.get(here)} "
                      f"— cannot shop mid-gauntlet (Revive x{camp.bag_count(REVIVE)} "
                      f"FR x{camp.bag_count(FULL_RESTORE)} ${camp.money()}); "
                      f"between-room bag only, going north toward {_nxt}")
                # inside the door chain: an E4 room, the Champion's room, or the HoF
                if here not in seen_rooms:
                    seen_rooms.append(here)
                    L(f"   [e4] room #{len(seen_rooms)}: map {here} @ {tv.coords(b)} "
                      f"[lead {self.lead_frac():.0%}, alive {self.party_alive()}]")
                    self.snap(f"room{len(seen_rooms)}_enter")
                    # Heal BEFORE walking into the next trainer's LOS (Bruno+ after Lorelei).
                    # Center is a gauntlet restart — items between rooms is the only heal.
                    # seat=this room so a scarce Revive stands up Zapdos before Lance
                    # (live 21:08 revived Blastoise first, Zapdos stayed dead).
                    between_room_heal(camp, L, seat=ROOM_SEAT.get(here))
                if len(seen_rooms) >= 6:
                    # room #6 past the champion = HALL OF FAME — the credits are rolling
                    L("   *** [e4] HALL OF FAME — CREDITS INBOUND ***")
                    self.snap("hall_of_fame")
                    self.on_event_safe("...that's it. that's the whole thing. eight badges, the Elite Four, "
                                       "the Champion — I actually did it. we did it.", tier=2)
                    break
                npcs = self.live_npc_tiles()
                room_warps = sorted([t for t, _d in warps], key=lambda t: t[1])
                north = room_warps[0] if room_warps else (6, 2)
                if npcs:
                    # Field the seat's answer BEFORE the approach (map-keyed; idempotent —
                    # no-ops once the right mon leads; re-fires after a whiteout re-lap).
                    self.answer_lead(seat_name=ROOM_SEAT.get(here))
                    trainer = min(npcs, key=lambda t: t[1])
                    stand = (trainer[0], trainer[1] + 1)
                    if tuple(tv.coords(b) or ()) != stand:
                        if not self.walk(lambda c, s=stand: c == s, "trainer-approach"):
                            self.settle(180)
                            self.drain()
                            continue
                    fought = False
                    for _try in range(3):
                        b.press("UP", 8, 8, camp.render, owner="agent")
                        b.press("A", 8, 12, camp.render, owner="agent")
                        for _ in range(120):
                            b.run_frame()
                            if self.fight_open():
                                break
                        if self.fight_open():
                            L(f"   [e4] battle #{len(seen_rooms)} OPENS [lead {self.lead_frac():.0%}, "
                              f"alive {self.party_alive()}, FR x{camp.bag_count(FULL_RESTORE)}]")
                            self.fight()
                            self.drain()
                            fought = True
                            break
                        self.drain(key="A")
                        if not dd_box(b) and not self.fight_open():
                            break
                    for _ in range(30):
                        if self.handle_interrupts():
                            continue
                        if tuple(tv.map_id(b)) != here:
                            break
                        self.settle(30)
                        if not dd_box(b) and not self.fight_open():
                            break
                    if tuple(tv.map_id(b)) != here:
                        continue
                    if fought:
                        # Room clear: revive EVERY fainted wincon BEFORE the
                        # north door. Prep the NEXT seat (Agatha won → Lance
                        # wants Zapdos up, then Blastoise, then birds).
                        _nxt = None
                        _cur = ROOM_SEAT.get(here)
                        if _cur in SEAT_ORDER:
                            _i = SEAT_ORDER.index(_cur)
                            _nxt = SEAT_ORDER[_i + 1] if _i + 1 < len(SEAT_ORDER) else _cur
                        between_room_heal(camp, L, seat=_nxt)
                        try:
                            camp._save_campaign(f"e4_room{len(seen_rooms)}")
                        except Exception:
                            pass
                        # Do NOT walk south / refuse north. CloseEntry sealed
                        # the entry door. Thin kit mid-gauntlet = bag only.
                if not self.go_warp(north, "north-door"):
                    if self.party_alive() == 0 or self.lead_frac() == 0:
                        L("   [e4] whiteout state — loop recovers via the center")
                    self.settle(120)
            else:
                L(f"   [e4] off-route at {here} — exiting")
                camp.enter_warp(prefer="south")
                self.settle(80)
        if len(seen_rooms) < 6:
            L(f"!! [e4] deadline without the Hall of Fame (rooms {len(seen_rooms)}/5, battles {self.n_battles}) "
              f"— usually team-depth: a thin team can't out-attrition Lance/Gary. Grind + retry.")
            return "stuck"
        # ── CREDITS: ceremony → credits → the post-credits SoftReset → back INTO the world ───────────────
        # 2026-08-14 (Jonny: "she goes and says fuck yeah I beat this game"): the old drain just
        # A-mashed for a flat 600s and returned "credits" — but FRLG's Hall of Fame is a POINT OF NO
        # RETURN that ends in a SoftReset to the TITLE SCREEN (recon_postgame_bank calls the mid-
        # ceremony bank "a live grenade — the QW-4 void core"). So the old path ended the show parked
        # on the title screen with a dead world, and free_roam kept ticking against (0,0). That is not
        # an ending. This ports recon_postgame_bank's proven sequence: drain the ceremony, run the
        # credits out to the reset, START → CONTINUE, drain the "Previously on your quest…" recap, and
        # only THEN take the victory line — with her back in the world, in control, as Champion.
        L("   [e4] draining the Hall of Fame ceremony (A while a box is up)")
        _quiet = 0
        for _ in range(600):
            if dd_box(b):
                _quiet = 0
                b.press("A", 8, 12, camp.render, owner="agent")
                self.settle(20)
            else:
                _quiet += 1
                self.settle(30)
                if _quiet >= 20:                      # ~10s with no box — the credits own the screen
                    break
        self.snap("97_credits_rolling")
        L("   [e4] credits rolling — running them out to the post-credits SoftReset")
        t_cred = time.time()
        while self._world_alive() and time.time() - t_cred < 900:
            self.settle(600)
            b.press("A", 8, 10, camp.render, owner="agent")   # FRLG parks on "THE END" until a press
        if self._world_alive():
            L("!! [e4] credits never ended in 15 min — returning 'credits' anyway (LOUD)")
            self.snap("98_credits_stuck")
            return "credits"
        L(f"   [e4] world went dark after {time.time() - t_cred:.0f}s — title screen next")
        for _ in range(40):                                   # START → the main menu
            b.press("START", 8, 12, camp.render, owner="agent")
            self.settle(40)
            if dd_box(b):
                break
        L("   [e4] selecting CONTINUE")
        for _ in range(40):
            b.press("A", 8, 12, camp.render, owner="agent")
            self.settle(90)
            if self._world_alive() and tuple(tv.map_id(b)) != HALL_OF_FAME:
                break
        if not self._world_alive():
            L("!! [e4] CONTINUE never re-entered the world — LOUD (supervisor will re-resume)")
            self.snap("98_continue_stuck")
            return "credits"
        self.settle(240)                                      # let the GameClear home-warp finish
        L("   [e4] draining the post-credits recap")
        for _ in range(80):
            if not dd_box(b):
                break
            b.press("A", 8, 10, camp.render, owner="agent")
            self.settle(30)
        self.snap("99_post_credits")
        L(f"   [e4] BACK IN THE WORLD as Champion: map={tv.map_id(b)} coords={tv.coords(b)} "
          f"party={b.rd8(ram.GPLAYER_PARTY_CNT)}")
        # THE line. tier=3 (same weight the legendary hunts use for a doorstep moment) — this is the
        # single biggest beat in the whole project and it was firing at tier 2, before the credits,
        # from inside a menu drain. It fires HERE: credits done, world back, she is the Champion.
        self.on_event_safe(
            "the credits rolled. that's the whole game — Pallet Town to the Hall of Fame, "
            "every step of it mine. I BEAT POKEMON. no walkthrough, no one steering me. "
            "I did that.", tier=3)
        L(f"   [e4] CREDITS SEQUENCE DRAINED | battles {self.n_battles} | money ${camp.money()}")
        return "credits"

    def _world_alive(self):
        """False once the post-credits SoftReset has torn the world down (title screen).
        recon_postgame_bank's test: a real map + real coords + a party still in RAM."""
        try:
            return (tuple(tv.map_id(self.b)) != (0, 0)
                    and tv.coords(self.b) is not None
                    and self.b.rd8(ram.GPLAYER_PARTY_CNT) > 0)
        except Exception:
            return False

    def on_event_safe(self, msg, tier=2):
        try:
            self.camp.on_event(msg, kind="milestone", tier=tier)
        except Exception:
            pass


def run_strike(camp, log, dbg_dir=None):
    """Run the Elite-Four gauntlet (Indigo Plateau -> Lorelei..Champion -> Hall of Fame -> CREDITS) from
    wherever she stands on the League maps, in ONE call. Whiteout-tolerant + resume-safe (map-keyed dispatch;
    DEFEATED flags ratchet in the save). Returns 'credits' (the summit) | 'battle_loss' | 'stuck'."""
    return EliteFour(camp, log, dbg_dir).run()
