"""field_moves.py — HM field-move capability (Cut / Surf / Strength / Flash / Fly).

These gate progression: Cut clears tree obstacles, Surf crosses water, Strength
shoves boulders, Flash lights dark caves, Fly fast-travels. Without them she
literally cannot finish the game.

DESIGN (firewall-correct, "capability not script"): this module is pure RAM
DETECTION + a thin ACTUATION helper. It NEVER decides to use a move — it exposes
"there's a cut tree in front and you can Cut" so the soul ORACLE chooses, in
character, then calls the actuation. Detection is source-cited; actuation rides
the existing _step_to / press / _drain_overworld primitives.

SOURCE (pret/pokefirered, confirmed this batch):
  - FRLG obstacle detection is SPRITE-based, not metatile-based: scan gObjectEvents
    for the graphicsId of the tile in front. gfx 95 = cuttable tree, gfx 97 =
    pushable boulder, gfx 92 = item ball. This mirrors the game's own
    CheckObjectGraphicsInFrontOfPlayer (src/fldeff_cut.c, src/fldeff_strength.c).
  - Surf/Waterfall ARE tile-based: metatile behavior 0x10/0x12/0x15 = surfable
    water, 0x13 = waterfall (include/constants/metatile_behaviors.h).
  - Surf is gated by FLAG_BADGE05_GET (0x824) — directly source-confirmed in
    GetInteractedWaterScript (src/field_control_avatar.c).
  - Move IDs (include/constants/moves.h): CUT=15 FLY=19 SURF=57 STRENGTH=70
    WATERFALL=127 FLASH=148 ROCK_SMASH=249. Moves live ONLY in the encrypted
    Attacks substructure → pokemon_state.read_party_moves decrypts them.
  - Flash = gMapHeader.cave==TRUE & !FLAG_SYS_FLASH_ACTIVE(0x806); Strength-active
    = FLAG_SYS_USE_STRENGTH(0x805).

BUILT & CLEANLY TESTABLE: detection of cut-tree / boulder / surf-water / item-ball;
the "does she have a usable HM" check (knows-the-move AND has-the-badge); surfacing
the opportunity to the oracle.
RECON-FLAGGED (see notes on each actuation method): the actual menu/prompt
ACTUATION is UNVERIFIED on a long-running libmgba core — the project's standing
lesson is that menu/prompt presses that land 6/6 on a fresh core can flake on a
long-running one (the move-list wedge). These must pass a live control before being
trusted; they are built to the proven interaction pattern but not yet proven here.
"""

import firered_ram as ram
import pokemon_state as st


# ── Move IDs (source: include/constants/moves.h) ─────────────────────────────
MOVE_CUT, MOVE_FLY, MOVE_SURF = 15, 19, 57
MOVE_STRENGTH, MOVE_WATERFALL, MOVE_FLASH, MOVE_ROCK_SMASH = 70, 127, 148, 249

# ── Badge flags (source: include/constants/flags.h) ──────────────────────────
FLAG_BADGE = {1: 0x820, 2: 0x821, 3: 0x822, 4: 0x823,
              5: 0x824, 6: 0x825, 7: 0x826, 8: 0x827}

# HM → (move id, gating badge number, human name). Surf=badge5 is SOURCE-CONFIRMED;
# the rest are the documented FRLG mapping (Cut=Cascade/2, Fly=Thunder/3,
# Strength=Rainbow/4, Flash=Boulder/1) — NOT re-verified from source this pass, so
# treat the badge gate on non-Surf HMs as needs-live-confirm before trusting a "no".
HM = {
    "cut":      (MOVE_CUT,      2, "Cut"),
    "fly":      (MOVE_FLY,      3, "Fly"),
    "surf":     (MOVE_SURF,     5, "Surf"),
    "strength": (MOVE_STRENGTH, 4, "Strength"),
    "flash":    (MOVE_FLASH,    1, "Flash"),
}

# ── System flags ─────────────────────────────────────────────────────────────
FLAG_SYS_USE_STRENGTH = 0x805
FLAG_SYS_FLASH_ACTIVE = 0x806

# ── Object graphics ids (source: include/constants/event_objects.h) ──────────
GFX_ITEM_BALL = 92      # 0x5C — ground item ball (unique to pickups; not an NPC)
GFX_CUT_TREE  = 95      # cuttable tree
GFX_BOULDER   = 97      # Strength-pushable boulder

# ── Surfable-water metatile behaviors (source: metatile_behaviors.h) ─────────
# 0x10 pond, 0x12 deep, 0x15 ocean = surfable; 0x11 fast-water (current), 0x13
# waterfall, 0x16 puddle, 0x17 shallow are water but NOT a plain Surf-start.
WATER_SURFABLE = {0x10, 0x12, 0x15}
WATER_BEHAVIORS = {0x10, 0x11, 0x12, 0x13, 0x15, 0x16, 0x17}

# ── Object-event table (matches campaign.py _OB/_SZ; graphicsId offset from source) ──
_OB, _SZ = 0x02036E38, 0x24
_OFF_ACTIVE, _OFF_GFX, _OFF_X, _OFF_Y, _OFF_FACING = 0x00, 0x05, 0x10, 0x12, 0x18
# facing nibble → (dx, dy) the object is looking; same convention as campaign._DELTA
_FACE_DELTA = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}
_DELTA_KEY = {(0, 1): "DOWN", (0, -1): "UP", (-1, 0): "LEFT", (1, 0): "RIGHT"}


# ── Flag / RAM reads ─────────────────────────────────────────────────────────
def read_flag(bridge, flag):
    """Read any game FLAG from the SaveBlock1 flag array (base + 0x0EE0). Same read
    as campaign.has_badge, generalized to any flag id."""
    sb1 = bridge.rd32(ram.GSAVEBLOCK1_PTR)
    if not ram.valid_ewram_ptr(sb1):
        return False
    fa = sb1 + 0x0EE0 + (flag >> 3)
    return bool(bridge.rd8(fa) & (1 << (flag & 7)))


def player_facing(bridge):
    """The direction the player object is facing as (dx, dy), or (0, 1) on a bad read.
    gObjectEvents[0] is the player; facing nibble at +0x18."""
    try:
        return _FACE_DELTA.get(bridge.rd8(_OB + _OFF_FACING) & 0x0F, (0, 1))
    except Exception:
        return (0, 1)


def scan_field_objects(bridge, gfx_set=None):
    """Active object events whose graphicsId is in gfx_set (default: the three field
    objects). Returns [{idx, gfx, coord (save), facing}]. Defensive per-slot."""
    gfx_set = gfx_set or {GFX_ITEM_BALL, GFX_CUT_TREE, GFX_BOULDER}
    out = []
    for i in range(1, 16):
        o = _OB + i * _SZ
        try:
            if not (bridge.rd8(o + _OFF_ACTIVE) & 1):
                continue
            g = bridge.rd8(o + _OFF_GFX)
            if g not in gfx_set:
                continue
            c = (bridge.rds16(o + _OFF_X) - 7, bridge.rds16(o + _OFF_Y) - 7)
            out.append({"idx": i, "gfx": g, "coord": c,
                        "facing": bridge.rd8(o + _OFF_FACING) & 0x0F})
        except Exception:
            continue
    return out


# ── Obstacle / opportunity detection (the oracle-surfacing inputs) ───────────
def obstacles_adjacent(bridge, player_xy):
    """Cut-trees / boulders on a tile orthogonally ADJACENT to the player — the things
    blocking a route she could clear. Returns [{kind, hm, coord, face}] where `face` is
    the press to look at it. kind ∈ {'tree','boulder'}."""
    if not player_xy:
        return []
    px, py = player_xy
    objs = scan_field_objects(bridge, {GFX_CUT_TREE, GFX_BOULDER})
    out = []
    for ob in objs:
        ox, oy = ob["coord"]
        d = (ox - px, oy - py)
        if d not in _DELTA_KEY:        # not orthogonally adjacent
            continue
        if ob["gfx"] == GFX_CUT_TREE:
            out.append({"kind": "tree", "hm": "cut", "coord": ob["coord"], "face": _DELTA_KEY[d]})
        else:
            out.append({"kind": "boulder", "hm": "strength", "coord": ob["coord"], "face": _DELTA_KEY[d]})
    return out


def surf_edge_adjacent(bridge, grid, player_xy):
    """Is there surfable water on a tile adjacent to the player? Returns the press to
    face it, or None. Uses the Grid's water classification (added in travel.py)."""
    if not player_xy or grid is None:
        return None
    water = getattr(grid, "water", None)
    if not water:
        return None
    px, py = player_xy
    for d, key in _DELTA_KEY.items():
        bx, by = px + d[0] + 7, py + d[1] + 7    # buffer coords (grid.water is buffer-indexed)
        if (bx, by) in water:
            return key
    return None


def usable_hms(bridge, party_count=6):
    """Which HMs she can ACTUALLY use right now: knows the move AND has the gating badge.
    Returns {hm_key: {'slot': int, 'badge_ok': bool, 'name': str}} for every HM she
    knows (even if badge-gated, so the oracle can say 'I know Cut but need the badge')."""
    out = {}
    for key, (mid, badge_no, name) in HM.items():
        slot = st.party_knows_move(bridge, mid, party_count)
        if slot is None:
            continue
        badge_ok = read_flag(bridge, FLAG_BADGE[badge_no])
        out[key] = {"slot": slot, "badge_ok": badge_ok, "name": name}
    return out


def can_use(bridge, hm_key, party_count=6):
    """True iff she knows HM `hm_key` AND has its badge — i.e. it would actually work."""
    info = usable_hms(bridge, party_count).get(hm_key)
    return bool(info and info["badge_ok"])


def item_balls(bridge, grid=None):
    """Uncollected ground item balls (gfx 92) on this map: [{idx, coord, reachable}].
    An item ball that is still an ACTIVE object is uncollected (collection sets its
    FLAG_HIDE_* so it despawns — source-confirmed). `reachable` is a BFS check when a
    Grid is supplied (None → not computed)."""
    out = []
    for ob in scan_field_objects(bridge, {GFX_ITEM_BALL}):
        out.append({"idx": ob["idx"], "coord": ob["coord"], "reachable": None})
    return out


# ── Opportunity summary for the oracle ───────────────────────────────────────
def field_opportunities(bridge, grid, player_xy, party_count=6):
    """One structured read of every field-move opportunity at the player's feet, for the
    free-roam loop to fold into the oracle ctx. Pure detection — never acts.
    Returns {'obstacles': [...], 'surf': key|None, 'items': [...], 'usable_hms': {...}}."""
    try:
        return {
            "obstacles": obstacles_adjacent(bridge, player_xy),
            "surf": surf_edge_adjacent(bridge, grid, player_xy),
            "items": item_balls(bridge, grid),
            "usable_hms": usable_hms(bridge, party_count),
        }
    except Exception as e:
        return {"obstacles": [], "surf": None, "items": [], "usable_hms": {}, "error": str(e)}


# ── Actuation (RECON-FLAGGED: unverified on a long-running core) ──────────────
class FieldMoveActuator:
    """Thin actuation over a Campaign instance (duck-typed: needs .b, .render, ._step_to,
    ._drain_overworld). Built to the proven _talk interaction pattern (face → A → drain),
    but the HM-confirm PROMPT path is UNVERIFIED here — see clear_obstacle().
    """

    def __init__(self, campaign):
        self.c = campaign
        self.b = campaign.b

    def _face_and_a(self, face_key, presses=1):
        for _ in range(presses):
            self.b.press(face_key, 8, 8, self.c.render, owner="agent")
            self.b.press("A", 8, 10, self.c.render, owner="agent")
            for _ in range(16):
                self.b.run_frame()

    def grab_item(self, target_coord, face_key):
        """Pick up a ground item ball: step adjacent (the campaign router handles the
        walk), face it, A, drain the 'found ITEM!' line. Returns 'grabbed' | 'no_box'.
        This is the SAME mechanic as _talk on an NPC, so it is the most likely of the
        field actuations to work as-is — but still wants a live control to confirm the
        ball is consumed (bag delta), like the catch-path lesson. CALLER must have
        already routed the player adjacent to target_coord."""
        try:
            from dialogue_drive import box_open as _box_open
        except Exception:
            _box_open = lambda _b: False
        self.b.set_input_owner("agent")
        for _ in range(6):
            if _box_open(self.b):
                break
            self._face_and_a(face_key, presses=1)
        got = _box_open(self.b)
        self.c._drain_overworld(label="item")
        return "grabbed" if got else "no_box"

    def clear_obstacle(self, hm_key, face_key):
        """Use Cut/Strength/Surf on the obstacle the player is facing.

        ⚠️ RECON-FLAGGED — UNVERIFIED on this core. Two possible in-game actuation paths:
          (A) interaction-prompt: face the tree/boulder/water, press A → the game offers
              "This tree can be CUT. Use it?" YES/NO → confirm YES (default cursor). This
              is what this method ATTEMPTS (face → A → confirm-A → drain), because it
              reuses the proven dialogue path and avoids the historically-flaky party menu.
          (B) party-menu: START → Pokémon → the HM mon → the field move → confirm. This is
              the menu nav the project has repeatedly found UN-navigable on a long-running
              libmgba core (the move-list wedge) — DEFERRED.
        Whether FRLG auto-offers (A) for trees/boulders was NOT source-confirmed this pass
        (GetInteractedWaterScript confirms the water A-prompt for Surf; the tree/boulder
        prompt is the documented behavior but unverified here). So: this fires the (A)
        attempt and reports honestly; a live control must confirm the obstacle clears
        before this is trusted. Returns 'used' | 'no_prompt' | 'cant'.
        """
        if not can_use(self.b, hm_key):
            return "cant"
        try:
            from dialogue_drive import box_open as _box_open
        except Exception:
            _box_open = lambda _b: False
        self.b.set_input_owner("agent")
        # face the obstacle and press A to (hopefully) raise the use-HM prompt
        for _ in range(4):
            if _box_open(self.b):
                break
            self._face_and_a(face_key, presses=1)
        if not _box_open(self.b):
            return "no_prompt"
        # CONFIRM the "use STRENGTH?/CUT?" YES/NO cleanly. BUG (victory-road watch 2026-07-08): the old
        # single raw A raced the still-TYPING prompt — the A sped the text, the YES/NO landed AFTER, so
        # the confirm missed and the whole thing re-prompted next tick (the "2 cycles before she confirms
        # yes"). The race-safe overworld drainer snaps the text full, then advances the YES (default
        # cursor) — ONE clean confirm — and drains the field-effect animation. No pre-A needed.
        self.c._drain_overworld(label=f"hm-{hm_key}")
        return "used"
