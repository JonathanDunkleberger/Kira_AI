"""fly_nav.py — HM02 Fly region-map fast travel (FireRed).

pret/pokefirered src/region_map.c + region_map_sections.json:
  Fly opens REGIONMAP_TYPE_FLY; D-pad moves the map cursor; A confirms a
  VISITED city/PC spawn. Cannot Fly FROM underground/indoor (MAP_TYPE_*).

Coordinates are the Kanto region-map grid (sMapSectionTopLeftCorners).
Cursor starts on the player's current mapsec — we D-pad relative to that.
"""

from __future__ import annotations

import time

import field_moves as fm
import firered_ram as ram
import hm_teach as ht
import pokemon_state as st
import travel as tv

# Region-map (x, y) + overworld map_id after a successful Fly warp.
# Route 10 Poké Center is the Zapdos staging pad (door at (7,40) on R10).
FLY_DEST = {
    "pallet":     {"map": (3, 0),  "xy": (4, 11), "label": "Pallet Town"},
    "viridian":   {"map": (3, 1),  "xy": (4, 8),  "label": "Viridian City"},
    "pewter":     {"map": (3, 2),  "xy": (4, 4),  "label": "Pewter City"},
    "cerulean":   {"map": (3, 3),  "xy": (14, 3), "label": "Cerulean City"},
    "lavender":   {"map": (3, 4),  "xy": (18, 6), "label": "Lavender Town"},
    "vermilion":  {"map": (3, 5),  "xy": (14, 9), "label": "Vermilion City"},
    "celadon":    {"map": (3, 6),  "xy": (11, 6), "label": "Celadon City"},
    "fuchsia":    {"map": (3, 7),  "xy": (12, 12), "label": "Fuchsia City"},
    "cinnabar":   {"map": (3, 8),  "xy": (4, 14), "label": "Cinnabar Island"},
    "saffron":    {"map": (3, 10), "xy": (14, 6), "label": "Saffron City"},
    # Live-verified FRLG map ids (pokemon_world / campaign): Indigo=(3,9),
    # Route 4=(3,22), Route 10=(3,28). (3,45) is Kindle Road — NEVER indigo.
    "indigo":     {"map": (3, 9),  "xy": (2, 3),  "label": "Indigo Plateau"},
    "route4":     {"map": (3, 22), "xy": (8, 3),  "label": "Route 4 Center"},
    "route10":    {"map": (3, 28), "xy": (18, 3), "label": "Route 10 Center"},
}

# Outdoor map_id → approximate region-map cursor start (player icon).
_HERE_XY = {
    (3, 0): (4, 11), (3, 1): (4, 8), (3, 2): (4, 4), (3, 3): (14, 3),
    (3, 4): (18, 6), (3, 5): (14, 9), (3, 6): (11, 6), (3, 7): (12, 12),
    (3, 8): (4, 14), (3, 9): (2, 3), (3, 10): (14, 6),
    (3, 19): (4, 9),    # Route 1
    (3, 22): (8, 3),    # Route 4 (Center pad)
    (3, 28): (18, 3),   # Route 10 (Center / Power Plant approach)
    (3, 24): (14, 9),   # Route 6 near Vermilion
    (3, 34): (7, 6),    # Route 16 west of Celadon (HM02 house)
    (3, 38): (8, 14),   # Route 20 (Seafoam mouth) — mid strip
    (3, 39): (12, 14),  # Route 19
    (3, 41): (4, 13),   # Route 21
    (3, 45): (4, 14),   # Kindle Road (Sevii)
}

_SEAFOAM = {(1, 83), (1, 84), (1, 85), (1, 86), (1, 87)}

# Prefer these species when teaching HM02 (Fearow already on the endgame roster).
_FLY_TEACH_PREFER = {22, 18, 17, 16, 83, 142, 144, 145, 146}  # fearow..birds


def resolve_dest(city):
    """city name / map_id / alias -> FLY_DEST entry or None."""
    if city is None:
        return None
    if isinstance(city, (tuple, list)) and len(city) >= 2:
        mid = (int(city[0]), int(city[1]))
        for _k, d in FLY_DEST.items():
            if d["map"] == mid:
                return d
        return None
    key = str(city).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "pallet": "pallet", "pallettown": "pallet",
        "viridian": "viridian", "viridiancity": "viridian",
        "pewter": "pewter", "pewtercity": "pewter",
        "cerulean": "cerulean", "ceruleancity": "cerulean",
        "lavender": "lavender", "lavendertown": "lavender",
        "vermilion": "vermilion", "vermilioncity": "vermilion",
        "celadon": "celadon", "celadoncity": "celadon",
        "fuchsia": "fuchsia", "fuchsiacity": "fuchsia",
        "cinnabar": "cinnabar", "cinnabarisland": "cinnabar",
        "saffron": "saffron", "saffroncity": "saffron",
        "indigo": "indigo", "indigoplateau": "indigo", "league": "indigo",
        "route4": "route4", "r4": "route4", "route4center": "route4",
        "route10": "route10", "r10": "route10", "route10center": "route10",
        "powerplant": "route10", "zapdos": "route10",
    }
    k = aliases.get(key)
    return FLY_DEST.get(k) if k else FLY_DEST.get(key)


def can_fly_here(b):
    """Fly refuses underground/indoor. Seafoam interiors are underground."""
    try:
        mid = tuple(tv.map_id(b))
    except Exception:
        return False
    if mid in _SEAFOAM:
        return False
    if mid[0] != 3:
        return False
    return True


def fly_slot(b):
    """Party slot that can use Fly, or None."""
    try:
        cnt = int(b.rd8(ram.GPLAYER_PARTY_CNT) or 0)
    except Exception:
        return None
    return fm.can_use(b, "fly", cnt)


def teach_plan(b):
    """Prefer Fearow/birds for HM02. Returns (slot, forget_idx, reason) or None."""
    try:
        cnt = int(b.rd8(ram.GPLAYER_PARTY_CNT) or 0)
    except Exception:
        return None
    # Prefer known flyers with a free slot / weak forget.
    prefer = []
    for s in range(cnt):
        sp = st.read_party_species(b, s)
        if not ht.hm_compatible(b, "fly", sp):
            continue
        moves = st.read_party_moves(b, s) or []
        free = 0 in moves or len([m for m in moves if m]) < 4
        prefer.append((0 if sp in _FLY_TEACH_PREFER else 1, 0 if free else 1, s, sp, moves, free))
    if not prefer:
        return ht.default_plan(b, "fly", cnt)
    prefer.sort()
    _t, _f, s, sp, moves, free = prefer[0]
    if free:
        return s, None, f"slot {s} ({st.SPECIES_NAME.get(sp, sp)}) has room"
    # forget weakest non-precious
    scored = []
    for i, m in enumerate(moves):
        if not m or m in ht._PRECIOUS:
            continue
        try:
            _ty, power = st.move_info(b, m)
        except Exception:
            _ty, power = "", 0
        scored.append((power or 0, i, m))
    scored.sort()
    if not scored:
        return s, 0, "overwrite move 0"
    return s, scored[0][1], f"forget move {scored[0][2]} (power {scored[0][0]})"


def _here_xy(b):
    mid = tuple(tv.map_id(b))
    if mid in _HERE_XY:
        return _HERE_XY[mid]
    # Fallback: Cinnabar / R20 band if y-ish unknown — center of Kanto mid-south.
    return (8, 10)


def _press(camp, key, settle=12):
    b = camp.b
    b.press(key, max(settle // 2, 4), max(settle // 2, 4),
            getattr(camp, "render", None), owner="agent")
    for _ in range(settle):
        b.run_frame()


def _open_fly_map(camp, slot, log, max_seconds=45):
    """START → party → slot → Fly field-move → wait until CB2 leaves overworld."""
    b = camp.b
    flow = ht.TeachFlow(camp, log=log, on_event=getattr(camp, "on_event", None))
    t0 = time.time()
    for attempt in range(3):
        if time.time() - t0 > max_seconds:
            break
        flow._b_cascade(6)
        if not ram.battle_cb2_dead(b):
            # already off-overworld (map may be open)
            return True
        flow._press("START", settle=60)
        if not flow._nav_byte(ht.START_CURSOR, 1):
            continue
        flow._press("A", settle=90)
        ok_party = False
        for _ in range(10):
            if flow._classify() == "party":
                ok_party = True
                break
            flow._press("A", settle=20)
        if not ok_party:
            continue
        if not flow._party_goto(slot):
            continue
        flow._press("A", settle=40)                 # submenu
        # Fly is the first field move on a single-HM mon (Fearow post-teach).
        flow._press("A", settle=40)
        # Drain fade into region map — CB2 leaves overworld.
        for _ in range(400):
            b.run_frame()
            if not ram.battle_cb2_dead(b) and not st.in_battle(b):
                log(f"   [fly] region map open (attempt {attempt})")
                return True
        log(f"   [fly] attempt {attempt}: map did not open — backing out")
        flow._b_cascade(8)
    return False


def fly_to(camp, city, log=print, max_seconds=90):
    """Use Fly to `city`. Returns 'arrived' | 'not_owned' | 'not_outdoors' |
    'unknown_city' | 'failed'."""
    dest = resolve_dest(city)
    if dest is None:
        log(f"   [fly] !! unknown destination {city!r}")
        return "unknown_city"
    b = camp.b
    if not can_fly_here(b):
        log(f"   [fly] !! not outdoors (map={tv.map_id(b)}) — climb out before Fly (LOUD)")
        return "not_outdoors"
    slot = fly_slot(b)
    if slot is None:
        log("   [fly] !! nobody can use Fly (need HM02 taught + Thunder Badge)")
        return "not_owned"
    target = tuple(dest["map"])
    if tuple(tv.map_id(b)) == target:
        log(f"   [fly] already at {dest['label']}")
        return "arrived"
    sx, sy = _here_xy(b)
    tx, ty = dest["xy"]
    dx, dy = tx - sx, ty - sy
    log(f"   [fly] ✈ {dest['label']} from {tv.map_id(b)}@{tv.coords(b)} "
        f"(map Δ {dx:+d},{dy:+d})")
    b.set_input_owner("agent")
    if not _open_fly_map(camp, slot, log, max_seconds=min(45, max_seconds)):
        return "failed"
    # Settle the map UI.
    for _ in range(40):
        b.run_frame()
    # D-pad to destination (relative from current mapsec).
    for _ in range(abs(dx)):
        _press(camp, "RIGHT" if dx > 0 else "LEFT", settle=10)
    for _ in range(abs(dy)):
        _press(camp, "DOWN" if dy > 0 else "UP", settle=10)
    for _ in range(20):
        b.run_frame()
    # SINGLE A — never mash (overshoots / cancels).
    _press(camp, "A", settle=20)
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        b.run_frame()
        try:
            if getattr(camp, "render", None):
                camp.render()
        except Exception:
            pass
        if st.in_battle(b):
            continue
        if tuple(tv.map_id(b)) == target and ram.battle_cb2_dead(b):
            log(f"   [fly] VERIFIED arrived {dest['label']} "
                f"{tv.map_id(b)}@{tv.coords(b)}")
            return "arrived"
        # Cancel button / failed select — still overworld elsewhere
        if ram.battle_cb2_dead(b) and time.time() - t0 > 25:
            log(f"   [fly] !! warp did not reach {dest['label']} "
                f"(now {tv.map_id(b)}) (LOUD)")
            return "failed"
    log(f"   [fly] !! timed out flying to {dest['label']} (LOUD)")
    return "failed"
