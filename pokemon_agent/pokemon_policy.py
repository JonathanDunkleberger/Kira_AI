"""pokemon_policy.py - the HANDS' move-picker. Dumb-but-correct, NO LLM, fast/free.

Gen-3 type chart + "pick the highest expected-damage move; flag low HP". Returns a
move index + a NEUTRAL descriptor (for the event summary - never her dialogue).
Pure logic, fully unit-testable without the emulator.

STAB + accuracy (2026-08-02, Surge/Raichu chalk): without Same-Type Attack Bonus a
Dark Bite (60) beat Water Gun (40) on raw power vs Electric — Blastoise never touched
the water STAB in the top-right FIGHT slot. Score is now power × STAB × type-eff ×
accuracy so she uses the whole 2×2 matrix for the quickest/safest KO.
"""

TYPES = ["normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
         "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"]

STAB_MULT = 1.5

# attacker -> {defender: multiplier}; unlisted = 1.0. Gen 3 (no Fairy).
_X = {
    "normal":   {"rock": .5, "ghost": 0, "steel": .5},
    "fighting": {"normal": 2, "flying": .5, "poison": .5, "rock": 2, "bug": .5, "ghost": 0,
                 "steel": 2, "psychic": .5, "ice": 2, "dark": 2},
    "flying":   {"fighting": 2, "rock": .5, "bug": 2, "steel": .5, "grass": 2, "electric": .5},
    "poison":   {"poison": .5, "ground": .5, "rock": .5, "ghost": .5, "steel": 0, "grass": 2},
    "ground":   {"flying": 0, "poison": 2, "rock": 2, "bug": .5, "steel": 2, "fire": 2,
                 "grass": .5, "electric": 2},
    "rock":     {"fighting": .5, "flying": 2, "ground": .5, "bug": 2, "steel": .5, "fire": 2, "ice": 2},
    "bug":      {"fighting": .5, "flying": .5, "poison": .5, "ghost": .5, "steel": .5, "fire": .5,
                 "grass": 2, "psychic": 2, "dark": 2},
    "ghost":    {"normal": 0, "ghost": 2, "psychic": 2, "dark": .5},
    "steel":    {"rock": 2, "steel": .5, "fire": .5, "water": .5, "electric": .5, "ice": 2},
    "fire":     {"rock": .5, "bug": 2, "steel": 2, "fire": .5, "water": .5, "grass": 2,
                 "ice": 2, "dragon": .5},
    "water":    {"ground": 2, "rock": 2, "fire": 2, "water": .5, "grass": .5, "dragon": .5},
    "grass":    {"flying": .5, "poison": .5, "ground": 2, "rock": 2, "bug": .5, "steel": .5,
                 "fire": .5, "water": 2, "grass": .5, "dragon": .5},
    "electric": {"flying": 2, "ground": 0, "water": 2, "grass": .5, "electric": .5, "dragon": .5},
    "psychic":  {"fighting": 2, "poison": 2, "steel": .5, "psychic": .5, "dark": 0},
    "ice":      {"flying": 2, "ground": 2, "steel": .5, "fire": .5, "water": .5, "grass": 2,
                 "ice": .5, "dragon": 2},
    "dragon":   {"steel": .5, "dragon": 2},
    "dark":     {"fighting": .5, "ghost": 2, "steel": .5, "psychic": 2, "dark": .5},
}

LOW_HP_FRAC = 0.25   # below this, flag a switch/heal (M1: just flag; M2 handles switching)

# TWO-TURN THROUGHPUT LAW (2026-08-04 LIVE, the Gary/Silph wipe): Skull Bash's flat
# power-100 outscored Bite(60)/Water Pulse(45) vs Venusaur, so Blastoise spent its last
# 12 HP "lowering its head" on the charge turn while Venusaur swung freely — that IS the
# "passive move instead of ending the fight" Jonny watched. A charge (or recharge) move
# only deals its power every SECOND turn, so its honest expected damage is half.
# Gen-3 move ids.
CHARGE_MOVES = {13, 19, 76, 91, 130, 143, 291, 340}   # RazorWind Fly SolarBeam Dig SkullBash SkyAttack Dive Bounce
RECHARGE_MOVES = {63, 307, 308, 338}                  # HyperBeam BlastBurn HydroCannon FrenzyPlant
SUICIDE_MOVES = {120, 153}                            # Selfdestruct, Explosion — never a voluntary pick


def effectiveness(move_type, defender_types):
    m = 1.0
    for d in defender_types:
        if d:
            m *= _X.get(move_type, {}).get(d, 1.0)
    return m


def stab_mult(move_type, our_types):
    """Gen-3 Same-Type Attack Bonus: 1.5x when the move shares a type with the user."""
    if not our_types or not move_type:
        return 1.0
    ours = {t for t in our_types if t and t != "???"}
    return STAB_MULT if move_type in ours else 1.0


def accuracy_frac(mv):
    """Gen-3 accuracy byte → hit chance. 0 / missing = always-hit / unknown → 1.0
    (Swift-class and unread ROM rows must not zero the score)."""
    acc = mv.get("accuracy")
    if acc is None or acc <= 0:
        return 1.0
    return min(100, int(acc)) / 100.0


def move_score(mv, enemy_types, our_types=None):
    """Expected-damage score for a damaging move: power × STAB × type-eff × accuracy,
    halved for charge/recharge moves (per-turn throughput — the Skull Bash law above).
    Status / 0-power / suicide → 0 (never wins a damage pick over a real attack)."""
    power = int(mv.get("power", 0) or 0)
    if power <= 0:
        return 0.0
    mid = int(mv.get("id", 0) or 0)
    if mid in SUICIDE_MOVES:
        return 0.0
    eff = effectiveness(mv.get("type", "normal"), enemy_types)
    score = power * stab_mult(mv.get("type"), our_types) * eff * accuracy_frac(mv)
    if mid in CHARGE_MOVES or mid in RECHARGE_MOVES:
        score *= 0.5
    return score


# ── CATCH-FLOW CHIP PICK (2026-08-04 LIVE, the badge-8 full-HP ball-burn) ─────────────────────
# The catch flow needs the INVERSE of move_score: "chip, don't kill". A raw lowest-base-power
# sort ignores STAB/type-eff AND the level gap — a L59 Blastoise "gentle" Bite still one-shots a
# L20 wild. These are TIER HEURISTICS, not a damage calc: Gen-3 damage ≈ level × power × Atk/Def,
# and stats + HP all scale ~linearly with level, so one hit's fraction of the foe's max HP scales
# ~ power × (our_level/foe_level)². Calibrated so a same-level neutral 50-power hit ≈ 35% of max
# HP. Accuracy is deliberately EXCLUDED — a miss doesn't make a move safer against the overkill
# KO, only a landed hit matters for "did I just kill the catch target".
CHIP_KO_SAFETY = 0.70       # a chip is safe only if est. damage <= this × the foe's CURRENT hp
_CHIP_BASE_FRAC = 0.35      # same-level neutral 50-power calibration point
_CHIP_RATIO_CAP = 3.5       # level-ratio clamp (beyond this everything one-shots anyway)


def chip_hit_frac(mv, enemy_types, our_level, enemy_level, our_types=None):
    """Estimated fraction of the foe's MAX HP one LANDED hit of `mv` removes (0.0 for
    status/immune/suicide — those never chip). Pure + unit-testable."""
    power = int(mv.get("power", 0) or 0)
    if power <= 0 or int(mv.get("id", 0) or 0) in SUICIDE_MOVES:
        return 0.0
    eff = effectiveness(mv.get("type", "normal"), enemy_types)
    if eff <= 0:
        return 0.0
    ratio = max(1.0 / _CHIP_RATIO_CAP,
                min(_CHIP_RATIO_CAP, (our_level or 1) / max(1, enemy_level or 1)))
    return (power / 50.0) * stab_mult(mv.get("type"), our_types) * eff \
        * ratio * ratio * _CHIP_BASE_FRAC


def chip_move_pick(our_moves, enemy_types, our_level, enemy_level,
                   foe_hp_frac=1.0, our_types=None):
    """The 'weaken, don't kill' pick: (index, est_frac, safe) of the GENTLEST usable damaging
    move (lowest estimated hit, PP>0). safe=True when that hit likely leaves the foe alive
    with margin (est <= CHIP_KO_SAFETY × its current HP fraction) — safe=False means EVERY
    usable move risks the overkill KO, so the caller should sleep/switch or throw early
    (a wasted ball beats a dead catch target). (None, None, False) = nothing damages at all."""
    best = None
    for i, mv in enumerate(our_moves):
        if mv.get("pp", 1) <= 0:
            continue
        est = chip_hit_frac(mv, enemy_types, our_level, enemy_level, our_types)
        if est <= 0:
            continue
        if best is None or est < best[1]:
            best = (i, est)
    if best is None:
        return None, None, False
    return best[0], best[1], best[1] <= max(0.0, foe_hp_frac) * CHIP_KO_SAFETY


def choose_move(our_moves, enemy_types, our_hp_frac=1.0, our_types=None):
    """our_moves: list of dicts {name, type, power, pp[, accuracy]}.
    enemy_types / our_types: type strings. Returns (index, descriptor, low_hp).
    Picks max expected damage (STAB + type chart + accuracy) among PP>0 damaging
    moves; falls back to first PP-having move if nothing damages."""
    best_i, best_score, best_eff = -1, -1.0, 1.0
    for i, mv in enumerate(our_moves):
        if mv.get("pp", 1) <= 0:
            continue
        score = move_score(mv, enemy_types, our_types)
        if score <= 0:
            continue                                    # status — only if nothing damages
        eff = effectiveness(mv.get("type", "normal"), enemy_types)
        # Tie-break when expected damage matches (Bite 60 == Water Gun 40×1.5): prefer
        # STAB first (her typing is the point), then accuracy, then raw power.
        tie = (stab_mult(mv.get("type"), our_types), accuracy_frac(mv),
               int(mv.get("power", 0) or 0))
        best_tie = (
            stab_mult(our_moves[best_i].get("type"), our_types),
            accuracy_frac(our_moves[best_i]),
            int(our_moves[best_i].get("power", 0) or 0),
        ) if best_i >= 0 else (-1, -1, -1)
        if score > best_score or (score == best_score and tie > best_tie):
            best_i, best_score, best_eff = i, score, eff
    if best_i < 0:
        # No damaging PP — any remaining status / Struggle slot
        for i, mv in enumerate(our_moves):
            if mv.get("pp", 1) > 0:
                best_i = i
                break
        if best_i < 0:
            return 0, "out of options (Struggle)", our_hp_frac < LOW_HP_FRAC
    if our_moves[best_i].get("power", 0) <= 0:
        # Only status moves have PP left (or one is genuinely best) — never bill it as a "hit"
        # (erika_run2 logged 'Growl - a solid hit' for hours while a 60/60 Gloom never moved).
        word = "a status move (nothing damaging left)"
    else:
        _stab = stab_mult(our_moves[best_i].get("type"), our_types) > 1.0
        word = ("a super-effective hit" if best_eff >= 2 else
                "a not-very-effective hit" if 0 < best_eff < 1 else
                "no effect" if best_eff == 0 else
                ("a STAB hit" if _stab else "a solid hit"))
    return best_i, f"{our_moves[best_i]['name']} - {word}", our_hp_frac < LOW_HP_FRAC
