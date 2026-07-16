"""pokemon_policy.py - the HANDS' move-picker. Dumb-but-correct, NO LLM, fast/free.

Gen-3 type chart + "pick the highest expected-damage move; flag low HP". Returns a
move index + a NEUTRAL descriptor (for the event summary - never her dialogue).
Pure logic, fully unit-testable without the emulator.
"""

TYPES = ["normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
         "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"]

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


def effectiveness(move_type, defender_types):
    m = 1.0
    for d in defender_types:
        if d:
            m *= _X.get(move_type, {}).get(d, 1.0)
    return m


def choose_move(our_moves, enemy_types, our_hp_frac=1.0):
    """our_moves: list of dicts {name, type, power, pp}. enemy_types: list of type strings.
    Returns (index, descriptor, low_hp). descriptor is NEUTRAL ("a super-effective Water
    move") for the event summary - NOT dialogue. Picks max power*effectiveness among
    moves with PP > 0; falls back to first move with PP if all score 0."""
    best_i, best_score, best_eff = -1, -1.0, 1.0
    for i, mv in enumerate(our_moves):
        if mv.get("pp", 1) <= 0:
            continue
        eff = effectiveness(mv.get("type", "normal"), enemy_types)
        score = max(mv.get("power", 0), 1) * eff
        if score > best_score:
            best_i, best_score, best_eff = i, score, eff
    if best_i < 0:                       # no PP anywhere -> Struggle / first move
        return 0, "out of options (Struggle)", our_hp_frac < LOW_HP_FRAC
    if our_moves[best_i].get("power", 0) <= 0:
        # Only status moves have PP left (or one is genuinely best) — never bill it as a "hit"
        # (erika_run2 logged 'Growl - a solid hit' for hours while a 60/60 Gloom never moved).
        word = "a status move (nothing damaging left)"
    else:
        word = ("a super-effective hit" if best_eff >= 2 else
                "a not-very-effective hit" if 0 < best_eff < 1 else
                "no effect" if best_eff == 0 else "a solid hit")
    return best_i, f"{our_moves[best_i]['name']} - {word}", our_hp_frac < LOW_HP_FRAC
