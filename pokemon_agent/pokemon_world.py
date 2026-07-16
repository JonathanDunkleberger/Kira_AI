"""pokemon_world.py — Kira's WORLD-MODEL + CAPABILITY-MODEL (her missing sense of PLACE).

THE GAP this closes (diagnosed 2026-06-28): the oracle got her position + badges + party +
"next gym = X" — a GOAL and a COMPASS, but no MAP. No memory of where she's been, no idea
what's around her, no concept that the world is an explorable graph, and no model of her own
travel powers. So when the forward path was blocked she had nowhere in her head to go but
into the wall (the Gary death-loop) — and that failure class repeats at every city, cave and
HM-gate. This gives her a real WORLD-MODEL (visited nodes + connectivity + traits) and a
CAPABILITY-MODEL (what travel/field powers she has) she can reason over.

FIREWALL: the SHAPE here is general reasoning — "where am I in an explorable world, what can
I do, where have I been and how do I get back" — and generalises to Emerald or any explorable
game. Only the DATA (FireRed map names/traits) is mode-side, and the ACTUATION (travel) lives
in the harness. Like StrategicMemory, this module EMITS NOTHING on its own and NEVER decides
for her: it RETURNS a short spatial brief + a list of travel-able known places, which the
free-roam loop folds into the oracle ctx via the SAME `place` seam survival/loss/shop use (no
core edit). She still chooses where to go and why, in character.

Persistence: the whole model is saved next to the campaign save (world_model.json) so it
SURVIVES --resume — no more spatial amnesia on every relaunch.
"""
import json
import os


# ── map-id <-> JSON-key helpers (JSON keys must be strings; map ids are (group, num)) ──
def _k(map_id):
    g, n = map_id
    return f"{int(g)},{int(n)}"


def _t(key):
    g, n = key.split(",")
    return (int(g), int(n))


_OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}

# ── FireRed SEED (mode-side DATA): human names + best-known traits for the maps she's already
# been through this run, so she boots already "knowing where she's been". CONNECTIVITY is NOT
# seeded (no guessed edges) — it's learned LIVE from the real map header and persisted, so a
# wrong-direction guess can never mislead routing. Live visits CORRECT traits (a confirmed
# has_grass is never downgraded). Source-first: the seed is a prior; the live read is truth. ──
SEED_NODES = {
    (3, 0):  ("Pallet Town",   {"is_town": True}),
    (3, 1):  ("Viridian City", {"is_town": True, "has_pokecenter": True, "has_mart": True}),
    (3, 2):  ("Pewter City",   {"is_town": True, "has_pokecenter": True, "has_mart": True}),
    (3, 3):  ("Cerulean City", {"is_town": True, "has_pokecenter": True, "has_mart": True}),
    (3, 19): ("Route 1",       {"is_route": True}),
    (3, 20): ("Route 2",       {"is_route": True, "has_grass": True, "has_wild": True}),
    (3, 21): ("Route 3",       {"is_route": True, "has_grass": True, "has_wild": True}),
    (3, 22): ("Route 4",       {"is_route": True, "has_grass": True, "has_wild": True}),
    # ── FORWARD SPINE (the rest of the main quest), map IDs from the pret/pokefirered disassembly
    # (data/maps/map_groups.json, group 3 = TownsAndRoutes) AND cross-checked against LIVE RAM
    # (2026-06-28: Route 3/4 read (3,21)/(3,22) live — the disasm export miscounted routes by one, so
    # the CITY block 0-10 is used verbatim and ROUTE nums follow the live-verified contiguous pattern
    # Route_N=(3,18+N)). These are KNOWN-AHEAD priors (NOT marked visited): they give her an accurate
    # sense that the gym cities exist forward, so "head to the next gym" has a real destination. CONNECTIVITY
    # is still learned LIVE (no fabricated edges — the Cerulean→Vermilion path runs through the Underground
    # Path warp, which the live read models truthfully). Source-first: live reads upgrade/confirm. ──
    (3, 4):  ("Lavender Town",   {"is_town": True, "has_pokecenter": True}),
    (3, 5):  ("Vermilion City",  {"is_town": True, "has_pokecenter": True, "has_mart": True}),  # GYM 3 (Surge)
    (3, 6):  ("Celadon City",    {"is_town": True, "has_pokecenter": True, "has_mart": True}),  # GYM 4 (Erika)
    (3, 7):  ("Fuchsia City",    {"is_town": True, "has_pokecenter": True, "has_mart": True}),  # GYM 5 (Koga)
    (3, 10): ("Saffron City",    {"is_town": True, "has_pokecenter": True, "has_mart": True}),  # GYM 6 (Sabrina)
    (3, 8):  ("Cinnabar Island", {"is_town": True, "has_pokecenter": True, "has_mart": True}),  # GYM 7 (Blaine)
    (3, 23): ("Route 5",         {"is_route": True}),   # Cerulean → (Underground Path) → Route 6 → Vermilion
    (3, 24): ("Route 6",         {"is_route": True}),
    # ── ENDGAME + POST-GAME (map ids VERIFIED against the pret disasm map_groups.json: group 1 =
    # gMapGroup_Dungeons, group 3 = TownsAndRoutes). These name the finale + post-credits maps so a
    # watch from the summit (or through the E4 gauntlet) reads "the Hall of Fame" / "Agatha's Room"
    # instead of the honest-but-flat "an unfamiliar area". The Champion/Hall states are where the
    # canonical pop-in lands, so naming them is what kills the '(1,80) an unfamiliar area' line. ──
    (3, 9):  ("Indigo Plateau",       {"is_town": True, "has_pokecenter": True}),
    (1, 39): ("Victory Road 1F",      {"is_cave": True, "has_wild": True}),
    (1, 40): ("Victory Road 2F",      {"is_cave": True, "has_wild": True}),
    (1, 41): ("Victory Road 3F",      {"is_cave": True, "has_wild": True}),
    (1, 75): ("Lorelei's Room",       {}),   # Elite Four #1
    (1, 76): ("Bruno's Room",         {}),   # Elite Four #2
    (1, 77): ("Agatha's Room",        {}),   # Elite Four #3
    (1, 78): ("Lance's Room",         {}),   # Elite Four #4
    (1, 79): ("the Champion's Room",  {}),   # the rival — the final battle
    (1, 80): ("the Hall of Fame",     {}),   # credits roll here; the canonical summit pop-in point
    (1, 74): ("Cerulean Cave",        {"is_cave": True, "has_wild": True}),  # post-game — Mewtwo at the bottom
}

# THE FIXED MAIN-QUEST SPINE (ordered): the gym cities she must reach in sequence, each as a real
# map id (verified above). The harness already derives `next_gym` from badge count via _GYM_ORDER;
# this maps each gym city to its map id so her world-model + heading point at a CONCRETE place, not a
# vibe. The Elite Four / credits follow gym 8. (Viridian gym 8 = (3,1), reused post-Giovanni.)
GYM_SPINE = [
    ("Pewter City", (3, 2)), ("Cerulean City", (3, 3)), ("Vermilion City", (3, 5)),
    ("Celadon City", (3, 6)), ("Fuchsia City", (3, 7)), ("Saffron City", (3, 10)),
    ("Cinnabar Island", (3, 8)), ("Viridian City", (3, 1)),
]

# badge_count -> the maps her progress PROVES she's traversed (so a --resume marks them visited
# honestly, by badge proof, not by guess). Boulder => she crossed Pallet..Pewter; Cascade =>
# she also crossed Route 3 / Mt Moon / Route 4 to reach Cerulean.
_PROGRESS_VISITED = {
    1: [(3, 0), (3, 19), (3, 1), (3, 20), (3, 2)],
    2: [(3, 0), (3, 19), (3, 1), (3, 20), (3, 2), (3, 21), (3, 22), (3, 3)],
}

_ALL_TRAITS = ("has_grass", "has_wild", "has_mart", "has_pokecenter",
               "is_cave", "is_town", "is_route")

# Travel/field capabilities she reasons over. WALK is always hers; the rest are earned. HM caps
# are read live (knows-the-move AND has-the-badge); bike is item-based (no reliable reader yet,
# so it stays a flagged default — see WorldModel.refresh_caps).
_HM_CAPS = ("cut", "fly", "surf", "strength", "flash")


class WorldModel:
    """Her accumulating mental map + capability registry. Pure data + awareness strings; no
    on_event coupling, headless-safe. Default-constructed on the Campaign and always present."""

    def __init__(self, log=print):
        self.log = log
        self.nodes = {}     # "g,n" -> {"name", "traits": {...}, "edges": {dir: "g,n"}, "visited": bool}
        self.caps = {"walk": True, "cut": False, "fly": False, "surf": False,
                     "strength": False, "flash": False, "bike": False}
        self.social = {}    # F-6 SOCIAL FABRIC: greeted key figures (id -> ts). Rides the sanctity
        #                     bundle via world_model.json so "already said hi to Mom" survives resume.
        self._cap_warned = False
        for mid, (name, traits) in SEED_NODES.items():
            self._ensure(mid, name, traits, visited=False)

    # ── node bookkeeping ─────────────────────────────────────────────────────────────────
    def _ensure(self, map_id, name=None, traits=None, visited=None):
        key = _k(map_id)
        node = self.nodes.get(key)
        if node is None:
            base = {t: False for t in _ALL_TRAITS}
            # `warps` = {"x,y": "g,n"} — the map-header warp tiles on THIS map and where they go. This is
            # what makes the graph traversable THROUGH warps/dungeons (the Underground-Path class), not
            # just map edges. Learned LIVE (travel.read_warps) and seeded for the spine; live confirms.
            node = {"name": name, "traits": base, "edges": {}, "warps": {}, "visited": False}
            self.nodes[key] = node
        node.setdefault("warps", {})       # back-compat for graphs persisted before warps existed
        if name and not node.get("name"):
            node["name"] = name
        if traits:
            for t, v in traits.items():
                if t in node["traits"] and v:
                    node["traits"][t] = True       # only ever UPGRADE a trait to True (never downgrade)
        if visited:
            node["visited"] = True
        return node

    def note_visit(self, map_id, name=None, live_traits=None, edges=None, warps=None):
        """Record (or enrich) the map she's standing on RIGHT NOW: mark visited, fold in the
        live map-header connections + WARPS, and let live signals correct traits. `edges` is
        {direction: (grp,num)}; `warps` is [((x,y), (grp,num), warp_id), ...] from travel.read_warps;
        `live_traits` is what the live read knows. Called every free-roam tick — so warps are LEARNED
        the instant she stands on a map (and a seeded warp is CONFIRMED/corrected by the live read)."""
        node = self._ensure(map_id, name=name, visited=True)
        # interior maps (group != 3 = building/cave); only label a CAVE if a route-like interior
        if map_id[0] != 3 and not node["traits"]["is_town"]:
            node["traits"].setdefault("is_cave", False)
        if live_traits:
            for t, v in live_traits.items():
                if t in node["traits"] and v:
                    node["traits"][t] = True
        if edges:
            for d, m in edges.items():
                if d and m:
                    node["edges"][d] = _k(m)
                    self._ensure(m)                # the neighbour exists in the graph (maybe not visited)
        if warps:
            for xy, dest, _wid in warps:
                if xy and dest:
                    node["warps"][f"{int(xy[0])},{int(xy[1])}"] = _k(dest)   # live = truth, overwrites seed
                    self._ensure(dest)
        return node

    def seed_warps(self, map_id, warp_map):
        """Seed KNOWN warp tiles for a (possibly unvisited) map as a PRIOR — {(x,y): (grp,num)}. Used to
        give her the forward spine's warp path (Underground-Path class) before she's walked it; the live
        read on arrival overwrites these with truth (source-first). Does NOT mark the map visited."""
        node = self._ensure(map_id)
        for xy, dest in (warp_map or {}).items():
            node["warps"][f"{int(xy[0])},{int(xy[1])}"] = _k(dest)
            self._ensure(dest)

    def seed_known(self, badge_count):
        """Mark the maps her badge progress PROVES she's traversed as visited (honest by-proof
        seed), so a fresh --resume already 'knows where she's been' before she re-walks anything.
        Names/traits come from SEED_NODES; connectivity still fills in live as she moves."""
        for mid in _PROGRESS_VISITED.get(min(badge_count, max(_PROGRESS_VISITED)), []):
            self._ensure(mid, visited=True)

    def seed_corridors(self, corridors):
        """Seed KNOWN dungeon-corridor connectivity (warps + edges) for maps she may NOT have
        VISITED this run, so head_to_gym / world.route can plan a path THROUGH an unexplored
        dungeon shortcut toward a goal — Diglett's Cave: Route 2 <-> cave floors <-> Route 11 ->
        Vermilion (NS#16). Without it the canonical graph is blind in the cave region, head_to_gym
        returns no_gym_route, and she gets dumped (via heal) into a sealed Route-2 pocket.
        ADDITIVE-ONLY: setdefault never clobbers a live-learned or already-present connection, and
        nothing is removed — the live read on arrival (note_visit) still confirms/corrects (source-
        first). Game-knowledge is supplied by the caller (gamedata), keeping this engine general
        (rule 14). `corridors` = {area_name: {"nodes": {"g,n": {"warps": {"x,y": [g,n]},
        "edges": {dir: [g,n]}}}}}. Returns the count of connections seeded."""
        n = 0
        for area, spec in (corridors or {}).items():
            for mid, conn in ((spec or {}).get("nodes") or {}).items():
                try:
                    mt = _t(mid) if isinstance(mid, str) else tuple(mid)
                    interior = int(mt[0]) != 3
                except Exception:
                    continue
                node = self._ensure(mt, name=(area if interior else None))
                for xy, dest in (conn.get("warps") or {}).items():
                    if node["warps"].setdefault(str(xy), _k(dest)) == _k(dest):
                        self._ensure(dest); n += 1
                for d, dest in (conn.get("edges") or {}).items():
                    if node["edges"].setdefault(d, _k(dest)) == _k(dest):
                        self._ensure(dest); n += 1
        if n:
            self.log(f"   [world] seeded {n} dungeon-corridor connection(s) — route-through priors "
                     f"(Diglett's Cave class)")
        return n

    # ── capability registry ──────────────────────────────────────────────────────────────
    def refresh_caps(self, can_use_fn=None):
        """Update what she can DO from the live game. can_use_fn(hm_key)->bool answers 'knows
        the HM AND has its badge' (from field_moves.can_use). Walk is always True. Bike has no
        reliable reader yet → stays False with a one-time LOUD note (Constraint #3) so a missing
        bike capability is never silently assumed present."""
        self.caps["walk"] = True
        if can_use_fn is not None:
            for hm in _HM_CAPS:
                try:
                    self.caps[hm] = bool(can_use_fn(hm))
                except Exception as e:
                    self.caps[hm] = False
                    if not self._cap_warned:
                        self.log(f"   [world] !! capability read for {hm!r} failed: {e} — assuming "
                                 f"NOT owned (LOUD); only WALK is certain")
            self._cap_warned = True
        elif not self._cap_warned:
            self.log("   [world] capability reader unavailable — only WALK assumed (HM/Fly/Surf "
                     "gated off until a live read confirms them) (LOUD)")
            self._cap_warned = True

    def has_cap(self, name):
        return bool(self.caps.get(name))

    def caps_summary(self):
        """One readable line on how she can move RIGHT NOW — reflects current capabilities so the
        oracle knows walking-back is always on the table and (once earned) Fly/Surf are too."""
        line = "You can walk to anywhere you've already been."
        if self.caps.get("fly"):
            line += " You also have FLY — you can fast-travel straight to any town you've visited (no need to walk the whole map)."
        if self.caps.get("surf"):
            line += " You have SURF — you can cross water."
        extra = [n.upper() for n in ("cut", "strength", "flash") if self.caps.get(n)]
        if extra:
            line += " Field moves ready: " + ", ".join(extra) + "."
        return line

    # ── routing over the learned graph (the backward/anywhere verb she was missing) ────────
    def route(self, src, dst, avoid=None):
        """BFS the learned connection graph for a path of map ids from `src` to `dst`, never
        stepping onto a map in `avoid` (the gated walls) — except `src` itself is always allowed
        (she's standing on it). Returns [src, ..., dst] or None. This is what lets her turn
        AROUND and walk back to a place behind her, not just push forward."""
        from collections import deque
        avoid = set(_k(m) for m in (avoid or [])) - {_k(src)}
        sk, dk = _k(src), _k(dst)
        if sk == dk:
            return [src]
        seen = {sk}
        q = deque([(sk, [sk])])
        while q:
            cur, path = q.popleft()
            cn = self.nodes.get(cur, {})
            # adjacency = map EDGES (walk across a border) UNION WARPS (step through a door/cave mouth)
            # -> she can route THROUGH the Underground Path etc., not just along map edges.
            for nbr in list(cn.get("edges", {}).values()) + list(cn.get("warps", {}).values()):
                if nbr in seen or nbr in avoid:
                    continue
                if nbr == dk:
                    return [_t(p) for p in path + [nbr]]
                seen.add(nbr)
                q.append((nbr, path + [nbr]))
        return None

    def edge_neighbor(self, map_id, dirword):
        """The learned neighbor map across `map_id`'s edge in `dirword` ('north'..), or None.
        Connections are learned from the map header ON VISIT (all four at once), so a visited map
        knows its neighbors' ids before she ever crosses — the road-binding seam relies on this."""
        nbr = self.nodes.get(_k(map_id), {}).get("edges", {}).get(dirword)
        if not nbr:
            return None
        try:
            g, n = nbr.split(",")
            return (int(g), int(n))
        except Exception:
            return None

    def next_hop(self, src, dst, avoid=None):
        """The FIRST step toward dst: (next_map_id, direction) or None — EDGE hops only (legacy
        callers that only walk map borders). Warp-aware callers use next_step()."""
        path = self.route(src, dst, avoid)
        if not path or len(path) < 2:
            return None
        nxt = path[1]
        for d, nbr in self.nodes.get(_k(src), {}).get("edges", {}).items():
            if nbr == _k(nxt):
                return (nxt, d)
        return None

    def next_step(self, src, dst, avoid=None):
        """The FIRST hop toward dst, WARP-AWARE: (next_map_id, kind, detail) where kind='edge' ->
        detail=direction ('north'..), or kind='warp' -> detail=(x,y) warp tile to step onto. None if
        no route. Re-evaluated each tick (one hop at a time — true free roam, she can still divert)."""
        path = self.route(src, dst, avoid)
        if not path or len(path) < 2:
            return None
        nxt = path[1]
        nk = _k(nxt)
        node = self.nodes.get(_k(src), {})
        for d, nbr in node.get("edges", {}).items():
            if nbr == nk:
                return (nxt, "edge", d)
        for xy, nbr in node.get("warps", {}).items():
            if nbr == nk:
                g = xy.split(",")
                return (nxt, "warp", (int(g[0]), int(g[1])))
        return None

    def warp_tiles(self, src, dst):
        """EVERY learned warp tile on `src` that lands on `dst` — next_step returns only the
        first, but one map can hold several doors to the same neighbor (hideout B4F's twin
        lift-lobby doors) and only some may be walk-reachable from where she stands."""
        node = self.nodes.get(_k(src), {})
        dk = _k(dst)
        out = []
        for xy, nbr in node.get("warps", {}).items():
            if nbr == dk:
                g = xy.split(",")
                out.append((int(g[0]), int(g[1])))
        return out

    def reachable_with_trait(self, src, trait, avoid=None):
        """Visited places she can REACH from here (route exists, not across a wall) that have
        `trait` — e.g. all known grass she could walk back to. [(map_id, name, hops)] nearest
        first. EXCLUDES src itself."""
        out = []
        for key, node in self.nodes.items():
            if key == _k(src) or not node.get("visited") or not node["traits"].get(trait):
                continue
            path = self.route(src, _t(key), avoid)
            if path:
                out.append((_t(key), node.get("name") or "an unfamiliar area", len(path) - 1))
        out.sort(key=lambda r: r[2])
        return out

    def name(self, map_id):
        node = self.nodes.get(_k(map_id))
        return (node and node.get("name")) or "an unfamiliar area"

    def node(self, map_id):
        return self.nodes.get(_k(map_id))

    def is_town(self, map_id):
        node = self.nodes.get(_k(map_id))
        return bool(node and node["traits"].get("is_town"))

    def visited(self, map_id):
        node = self.nodes.get(_k(map_id))
        return bool(node and node.get("visited"))

    # ── the spatial brief folded into the oracle ctx (Phase 2) ─────────────────────────────
    def _trait_tag(self, node):
        tr = node["traits"]
        bits = []
        if tr.get("has_grass") or tr.get("has_wild"):
            bits.append("wild Pokémon to catch/train")
        if tr.get("is_cave"):
            bits.append("a cave")
        if tr.get("has_mart"):
            bits.append("a Mart")
        if tr.get("has_pokecenter"):
            bits.append("a Pokémon Center")
        return ", ".join(bits)

    def spatial_brief(self, src, avoid=None, blocked_dirs=None, max_extra=3, named_already=False):
        """A SHORT, readable 'where am I / what's around me' picture built from her visited-world
        memory — the missing MAP. Lists the adjacent exits (named, tagged, BLOCKED where gated)
        plus a few notable backward places she can travel to. Her options become PLACES, not just
        'advance toward the objective'. Kept to a few lines, never a data dump.
        named_already: the caller's ctx already leads with a grounded 'You're in X' line (the F-8
        location block) — skip the duplicate naming and go straight to the surroundings."""
        avoid_keys = set(_k(m) for m in (avoid or []))
        blocked_dirs = set(blocked_dirs or [])
        node = self.nodes.get(_k(src))
        here = self.name(src)
        if node is None:
            return ("" if named_already else
                    f"You're in {here} — somewhere new; you don't have your bearings here yet.")
        htag = self._trait_tag(node)
        lead = ("" if named_already else
                f"You're in {here}" + (f" ({htag})" if htag else "") + ".")
        # adjacent exits, in a stable readable order
        order = ["north", "south", "east", "west"]
        exits = []
        for d in order:
            nbr = node["edges"].get(d)
            if not nbr:
                continue
            nb = self.nodes.get(nbr) or {}
            nm = nb.get("name") or "an unexplored area"
            blocked = (d in blocked_dirs) or (nbr in avoid_keys)
            if blocked:
                exits.append(f"{d.upper()} → {nm} (BLOCKED — you can't get past there yet)")
            else:
                tag = self._trait_tag(nb) if nb else ""
                seen = "" if nb.get("visited") else " (unexplored)"
                exits.append(f"{d.upper()} → {nm}{seen}" + (f" — {tag}" if tag else ""))
        lines = [lead] if lead else []
        if exits:
            lines.append("From here: " + "; ".join(exits) + ".")
        # notable backward grass she can WALK to (places she knows, not adjacent), for grinding
        far_grass = [r for r in self.reachable_with_trait(src, "has_grass", avoid)
                     if r[2] > 1][:max_extra]
        if far_grass:
            lines.append("Places you've been with wild Pokémon you can walk back to: "
                         + "; ".join(f"{nm}" for _m, nm, _h in far_grass) + ".")
        lines.append(self.caps_summary())
        return " ".join(lines)

    def travel_targets(self, src, avoid=None, want_traits=("has_grass", "has_mart")):
        """Visited places worth offering as a first-class 'travel to X' move (grass to grind,
        a Mart to stock). [(map_id, name, why)] — reachable only, nearest first, deduped."""
        seen, out = set(), []
        why_by_trait = {"has_grass": "wild Pokémon to catch and level up",
                        "has_mart": "a Mart to buy Poké Balls / supplies",
                        "has_pokecenter": "a Pokémon Center to heal"}
        for trait in want_traits:
            for mid, nm, _h in self.reachable_with_trait(src, trait, avoid):
                if _k(mid) in seen:
                    continue
                seen.add(_k(mid))
                out.append((mid, nm, why_by_trait.get(trait, "somewhere useful")))
        return out

    # ── F-6 SOCIAL FABRIC: who she's already greeted (key figures, not every townsperson) ──
    def met(self, sid):
        return sid in self.social

    def mark_met(self, sid):
        import time as _t
        self.social[sid] = _t.time()

    # ── persistence (survive --resume) ─────────────────────────────────────────────────────
    def to_dict(self):
        return {"nodes": self.nodes, "caps": self.caps, "social": self.social}

    def from_dict(self, d):
        nodes = d.get("nodes") or {}
        for key, node in nodes.items():
            traits = {t: bool(node.get("traits", {}).get(t)) for t in _ALL_TRAITS}
            self.nodes[key] = {"name": node.get("name"),
                               "traits": {**self.nodes.get(key, {}).get("traits", traits), **traits},
                               "edges": dict(node.get("edges") or {}),
                               "warps": dict(node.get("warps") or {}),
                               "visited": bool(node.get("visited"))}
        caps = d.get("caps") or {}
        for c, v in caps.items():
            if c in self.caps:
                self.caps[c] = bool(v)
        self.social.update(d.get("social") or {})   # F-6 (back-compat: absent in older bundles)

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, path)
            return True
        except Exception as e:
            self.log(f"   [world] !! world-model save failed: {e} (LOUD)")
            return False

    def load(self, path):
        try:
            if not os.path.exists(path):
                self.log("   [world] no world-model sidecar yet — fresh mental map (seed only)")
                return False
            with open(path, encoding="utf-8") as f:
                self.from_dict(json.load(f))
            visited = sum(1 for n in self.nodes.values() if n.get("visited"))
            self.log(f"   [world] mental map loaded: {visited} place(s) known, "
                     f"{len(self.nodes)} node(s) in the graph")
            return True
        except Exception as e:
            self.log(f"   [world] !! world-model load failed: {e} — starting fresh (LOUD)")
            return False
