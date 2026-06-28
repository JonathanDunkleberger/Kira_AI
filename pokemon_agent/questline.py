"""questline.py — GENERAL gate-unlock capability (recognise → derive → execute).

THE PATTERN (capability, not script): she hits something she can't pass, identifies WHAT she's missing,
DERIVES how to get it from game knowledge, and EXECUTES that as ordinary nav/interaction goals. Cut is
the first instance; the SAME pipeline handles Surf/Strength/Fly/Flash and item gates (Poké Flute, Silph
Scope, the S.S. Ticket story-gate, …). Nothing here is hard-coded to Cut.

THREE STAGES:
  1. GATE RECOGNIZER (this file, Phase 1): a blocked forward goal / an adjacent obstacle becomes a typed
     `Gate{kind, missing, where, human}`. HM obstacles come from field_moves (source-cited gfx/metatile);
     STORY/exit gates come from the curated KB (gamedata/frlg_gates.json) + a LIVE flag read.
  2. QUESTLINE DERIVER (Phase 2): Gate.missing -> ordered steps, graph-searched over the KB capability
     chain, every destination/flag LIVE-CROSS-CHECKED before routing (disasm route-nums are unreliable).
  3. QUESTLINE EXECUTOR (Phase 3): run the steps as head goals over the existing travel/head_to_gym/talk.

FIREWALL: harness-side game-mechanics. It RETURNS structured facts (a Gate, a plan) the free-roam loop
folds into the oracle ctx via the existing `place` seam — she reasons + narrates in character, chooses
herself. No core edit, never decides for her.
"""
import json
import os

import field_moves as fm

_HERE = os.path.dirname(os.path.abspath(__file__))
_KB_PATH = os.path.join(_HERE, "gamedata", "frlg_gates.json")

# ── Gate kinds ───────────────────────────────────────────────────────────────
HM_OBSTACLE = "hm_obstacle"   # cut tree / boulder / surfable water — needs an HM she may not have
STORY_NPC = "story_npc"        # an exit a flag-gated map script walls until a story flag is set (Slowbro)
ITEM_GATE = "item_gate"        # blocked until she holds an item (Poké Flute, Silph Scope, Tea, Bike…)
BADGE_GATE = "badge_gate"      # she KNOWS the HM but lacks the gating badge

_DIRS = ("north", "south", "east", "west")


def _k(map_id):
    g, n = map_id
    return f"{int(g)},{int(n)}"


class Gate:
    """A recognised obstacle she can't pass yet.

    kind     : one of HM_OBSTACLE / STORY_NPC / ITEM_GATE / BADGE_GATE
    missing  : the capability/flag/item key she lacks — a KEY into KB['capabilities'] (so the deriver can
               look up how to obtain it). e.g. 'cut', 'strength', 'FLAG_GOT_SS_TICKET'.
    where    : map id (g,n) the gate is on.
    human    : a short in-character description of the obstacle (the oracle colours it; never 'I searched').
    detail   : structured evidence (tile, gfx, direction, flag id, blocker) for logs + the executor.
    """
    __slots__ = ("kind", "missing", "where", "human", "detail")

    def __init__(self, kind, missing, where, human, detail=None):
        self.kind = kind
        self.missing = missing
        self.where = where
        self.human = human
        self.detail = detail or {}

    def __repr__(self):
        return f"Gate({self.kind}, missing={self.missing!r}, where={self.where}, {self.human!r})"

    def to_ctx(self):
        """One line for the oracle ctx (folded via the `place` seam). Honest + in-character, never a search."""
        return f"You've hit a wall: {self.human}."


def load_kb(path=_KB_PATH, log=print):
    """Load the curated gate/questline KB. Returns {} (degrade to GuideSearch-only) on any failure — a
    missing KB must never crash the run (CLAUDE.md constraint 3: announce, don't swallow silently)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"   [questline] !! KB load failed ({e}) — running KB-less (search-only derivation) (LOUD)")
        return {}


class GateRecognizer:
    """Stage 1. Turn 'I'm blocked' into a typed Gate. KB-driven for story/exit gates; field_moves for HM
    obstacles. Pure detection + a LIVE flag read — never acts, never decides."""

    def __init__(self, bridge, world, kb=None, party_count_fn=None, log=print):
        self.b = bridge
        self.world = world
        self.kb = kb if kb is not None else load_kb(log=log)
        self.log = log
        # how many party slots to scan for HM moves; default 6 (safe over-scan)
        self._pc = party_count_fn or (lambda: 6)

    # ── live flag read (the cross-check: a SET flag means the gate is already open) ──────────
    def _flag_set(self, flag_name):
        ent = (self.kb.get("flags") or {}).get(flag_name)
        if not ent or "id" not in ent:
            return None                       # unknown flag -> can't confirm; treat as "not satisfied"
        try:
            return fm.read_flag(self.b, int(ent["id"]))
        except Exception as e:
            self.log(f"   [questline] flag read {flag_name} failed: {e}")
            return None

    # ── stage 1a: a flag-gated EXIT (story gate) on the current map ──────────────────────────
    def exit_gate(self, map_id, direction):
        """If the KB says `direction` off `map_id` is gated by a story flag that is currently UNSET,
        return a STORY_NPC Gate. None if there's no such gate or the flag is already set (cross-check)."""
        eg = (self.kb.get("exit_gates") or {}).get(_k(map_id))
        if not eg:
            return None
        g = eg.get(direction)
        if not g:
            return None
        flag = g.get("flag")
        is_set = self._flag_set(flag)
        if is_set:                            # LIVE cross-check: flag already set -> the road is open now
            return None
        return Gate(STORY_NPC, missing=flag, where=tuple(map_id), human=g.get("human", "the way is blocked"),
                    detail={"direction": direction, "flag": flag, "flag_set": is_set,
                            "blocker": g.get("blocker")})

    # ── stage 1b: an HM obstacle (cut tree / boulder / surf water) at her feet ────────────────
    def hm_obstacle(self, player_xy, grid=None):
        """Return an HM_OBSTACLE / BADGE_GATE Gate for an adjacent obstacle she can't currently clear,
        else None. Uses field_moves' source-cited detection + her live HM capability."""
        try:
            obs = fm.obstacles_adjacent(self.b, player_xy)
        except Exception:
            obs = []
        # surfable water counts as an obstacle she may lack the HM for
        if grid is not None:
            try:
                if fm.surf_edge_adjacent(self.b, grid, player_xy):
                    obs = obs + [{"kind": "water", "hm": "surf", "coord": None, "face": None}]
            except Exception:
                pass
        for ob in obs:
            hm = ob.get("hm")
            usable = (self.kb.get("capabilities") or {})
            human = (usable.get(hm) or {}).get("human") or f"HM {hm.title()}"
            try:
                info = fm.usable_hms(self.b, self._pc()).get(hm)
            except Exception:
                info = None
            if info and info.get("badge_ok"):
                continue                      # she can actually clear it — not a gate
            if info and not info.get("badge_ok"):
                # she KNOWS the move but lacks the badge -> a BADGE gate (different unlock)
                return Gate(BADGE_GATE, missing=f"badge:{hm}", where=tuple(map_id_or_none(self.b)),
                            human=f"a {ob['kind']} I could {hm.title()}, but I don't have the badge for it yet",
                            detail={**ob, "knows_move": True, "badge_ok": False})
            # she doesn't know the move at all -> needs to OBTAIN the HM (the common case)
            return Gate(HM_OBSTACLE, missing=hm, where=tuple(map_id_or_none(self.b)),
                        human=f"a {ob['kind']} blocking the way — I'd need {human} to get past it",
                        detail={**ob, "knows_move": False})
        return None

    # ── the one-call recogniser the free-roam loop uses ───────────────────────────────────────
    def recognize(self, map_id, player_xy=None, blocked_dir=None, grid=None):
        """Classify why she's blocked, best gate first. STORY/exit gates (the immediate, often-invisible
        cause) take precedence over an adjacent HM obstacle (which may be a secondary blocker — e.g. the
        Cerulean cut tree behind the Slowbro story-gate). Returns a Gate or None."""
        if blocked_dir:
            g = self.exit_gate(map_id, blocked_dir)
            if g:
                return g
        if player_xy is not None:
            g = self.hm_obstacle(player_xy, grid=grid)
            if g:
                return g
        return None


def map_id_or_none(bridge):
    try:
        import travel as tv
        return tv.map_id(bridge)
    except Exception:
        return (0, 0)


# ── Stage 2: QUESTLINE DERIVER ───────────────────────────────────────────────
class Step:
    """One concrete sub-goal of a questline: go somewhere + do something to obtain a capability/flag/item.

    missing    : the capability/flag/item this step obtains (a KB capability key).
    via        : how — 'talk_npc' | 'use_hm' | 'board' (reuses existing interaction primitives).
    npc        : who to interact with (display/log).
    from_map   : the map the destination is reached FROM (KB 'from', a 'g,n' string) — context for `dir`.
    dir        : coarse, RELIABLE direction to head from there ('north'..) — NOT a map number.
    place_name : human destination name.
    success    : ('flag', name) or ('cap', hm_key) — the LIVE condition that means this step is DONE.
    satisfied  : live-checked — True if already obtained (skip it).
    resolved   : True if the KB knew how to obtain `missing`; False -> needs the GuideSearch fallback.
    human      : in-character one-liner.
    """
    __slots__ = ("missing", "kind", "via", "npc", "from_map", "dir", "place_name",
                 "success", "satisfied", "resolved", "human")

    def __init__(self, missing, kind=None, via=None, npc=None, from_map=None, dir=None,
                 place_name=None, success=None, satisfied=False, resolved=True, human=None):
        self.missing = missing
        self.kind = kind
        self.via = via
        self.npc = npc
        self.from_map = from_map
        self.dir = dir
        self.place_name = place_name
        self.success = success
        self.satisfied = satisfied
        self.resolved = resolved
        self.human = human or f"get {missing}"

    def __repr__(self):
        flag = "✓" if self.satisfied else ("·" if self.resolved else "?")
        return f"Step[{flag}]({self.missing}, via={self.via}, dir={self.dir}, {self.human!r})"


class Questline:
    """An ordered plan to clear a Gate: prereq steps first, each a Step. `actionable` is the first
    not-yet-satisfied step (what to DO now). `narration` is the whole plan in her voice."""

    def __init__(self, gate, steps):
        self.gate = gate
        self.steps = steps
        self.actionable = next((s for s in steps if not s.satisfied), None)

    @property
    def complete(self):
        return all(s.satisfied for s in self.steps)

    @property
    def derivable(self):
        return all(s.resolved for s in self.steps)

    def narration(self):
        """Her derived plan, in character — the CONCLUSION she 'works out', never 'I searched'."""
        todo = [s for s in self.steps if not s.satisfied]
        if not todo:
            return None
        if not todo[0].resolved:
            return (f"I'm stuck on this — {self.gate.human}. I'm not sure how to get past it; "
                    f"let me think / look into it.")
        first = todo[0]
        tail = ""
        if len(todo) > 1:
            tail = " Then " + "; then ".join(s.human for s in todo[1:]) + "."
        return f"Okay — to get past this I need {first.human}.{tail}"


def _flag_id(kb, flag_name):
    ent = (kb.get("flags") or {}).get(flag_name)
    return int(ent["id"]) if ent and "id" in ent else None


def _step_satisfied(step, bridge, kb, party_count_fn, log=print):
    """LIVE cross-check: is this step already done? Flag-step -> flag set; HM/cap-step -> knows the move."""
    kind, val = (step.success or (None, None))
    try:
        if kind == "flag":
            fid = _flag_id(kb, val)
            if fid is None:
                return False
            return bool(fm.read_flag(bridge, fid))
        if kind == "cap":
            mid = (HM_MOVE_IDS or {}).get(val)
            if mid is None:
                return False
            import pokemon_state as st
            return st.party_knows_move(bridge, mid, party_count_fn()) is not None
    except Exception as e:
        log(f"   [questline] satisfied-check {step.missing} failed: {e}")
    return False


HM_MOVE_IDS = {"cut": 15, "fly": 19, "surf": 57, "strength": 70, "flash": 148, "waterfall": 127}


def _step_from_cap(key, cap):
    ob = (cap or {}).get("obtain") or {}
    success = None
    if ob.get("sets_flag"):
        success = ("flag", ob["sets_flag"])
    elif ob.get("gives_cap"):
        success = ("cap", ob["gives_cap"])
    elif key.startswith("FLAG_"):
        success = ("flag", key)
    elif key in HM_MOVE_IDS:
        success = ("cap", key)
    return Step(missing=key, kind=cap.get("kind"), via=ob.get("via"), npc=ob.get("npc"),
                from_map=ob.get("from"), dir=ob.get("dir"), place_name=ob.get("place_name"),
                success=success, resolved=True, human=cap.get("human") or f"get {cap.get('name', key)}")


def derive_questline(gate, kb, bridge, party_count_fn=None, log=print):
    """Stage 2: Gate.missing -> an ordered Questline (prereqs first), every step LIVE-CROSS-CHECKED.
    Walks the KB capability chain via obtain.prereq. An unknown capability yields an UNRESOLVED step
    (handed to the GuideSearch fallback in Phase 4). Returns a Questline (possibly with .actionable=None
    if everything's already satisfied — meaning the gate should now be passable)."""
    caps = kb.get("capabilities") or {}
    pcf = party_count_fn or (lambda: 6)
    chain, key, seen = [], gate.missing, set()
    while key and key not in seen:
        seen.add(key)
        cap = caps.get(key)
        if not cap:
            # KB doesn't know how to obtain this -> unresolved; Phase 4 GuideSearch fills it in.
            chain.append(Step(missing=key, resolved=False,
                              human=f"figure out how to get past {gate.human}"))
            break
        chain.append(_step_from_cap(key, cap))
        key = (cap.get("obtain") or {}).get("prereq")
    chain.reverse()                                    # earliest prerequisite first
    for s in chain:
        if s.resolved:
            s.satisfied = _step_satisfied(s, bridge, kb, pcf, log=log)
    return Questline(gate, chain)
