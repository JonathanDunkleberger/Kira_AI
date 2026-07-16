"""pokemon_planner.py — THE STRATEGIC PLANNER (mode plumbing, engine-side; not core).

THE GAP THIS FIXES: her strategic layer was entirely REACTIVE — she learned a wall was a wall only
AFTER losing to it (loss_awareness / strategic-stuck). She had a complete in-battle type chart but NO
forward planning: she walked into every gym/E4 blind and adapted only once wrecked. The live proof was
the Elite-Four bench — a L67 ace over an unleveled L15 Ekans + a L26 Lapras (Lapras being the textbook
Lance answer, sitting dead on the bench). That reactive, one-carry pattern is what stretched the climb
to ~67h instead of a watchable ~30h.

WHAT THIS IS: the "she read every strategy guide" layer. Given her live position + party + the next
threat, it reads the game-knowledge KB (gamedata/frlg_strategy.json) and emits ONE proactive, in-character
anticipation beat — what's ahead, whether she already has the answer, and the single smartest prep move
(level her existing counter, catch a keeper type, or stop neglecting the bench). It PROPOSES; the oracle
still CHOOSES (capability-not-script). It is folded into the SAME `place`/_spine_and_history seam the
folklore + questline + loss-awareness notes already use.

FIREWALL (CLAUDE.md rule 12/14): pure mode-side game-knowledge. Reads only the Pokémon KB + the live
state dict the campaign already builds. Returns strings the campaign folds into the POKÉMON oracle ctx —
never touches core Kira identity, never surfaces in unrelated chat. FireRed FACTS live in the swappable
KB; THIS engine is game-agnostic (it would drive Emerald off an emerald_strategy.json unchanged).

VOICE (the Supreme Law): the note is raw material rendered as ONE beat of her anticipation/curiosity —
the excited kid who did her homework ("the E4 opens with an ice-and-water lady… my Lapras's Ice Beam is
my Lance answer — I think I've got a plan"), NEVER a dry data dump. The oracle colours it; this only
hands over the facts + the lean.
"""
import json
import os

import pokemon_state as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_KB_PATH = os.path.join(_HERE, "gamedata", "frlg_strategy.json")

# Default ON — this is a load-bearing GO-readiness feature, not an experiment. Kill-switch for a fast
# revert if it ever misbehaves live (POKEMON_STRATEGIC_PLANNER=0 → the planner note simply never folds).
PLANNER_ENABLED = os.getenv("POKEMON_STRATEGIC_PLANNER", "1") == "1"

# How far BELOW a threat's level floor a mon that HARD-COUNTERS it may be before she's nudged to level it
# (the Lapras-for-Lance case: L26 vs a L54 floor = 28 under → flagged). Tunable.
COUNTER_UNDERLEVEL_GAP = int(os.getenv("POKEMON_COUNTER_UNDERLEVEL_GAP", "10"))
# Bench spread (top level - floor level) at/above which the team reads as ace-heavy and the bench-develop
# nudge fires PROACTIVELY (not just at a wall). 14 ≈ "roughly two evolutions behind" — the ace-overpower
# smell the L67/L15 E4 bench had. Tunable.
BENCH_SPREAD_ALARM = int(os.getenv("POKEMON_BENCH_SPREAD_ALARM", "14"))
# GYM-READINESS (2026-07-09, Fix B) — how the enforced pre-gym prep judges "ready". A type-answer mon
# counts only if it's within this many levels of the leader's ace (a L5 water mon is not a Brock answer);
# the grind target is the ace level + this margin. Party target scales toward ~4 by mid-game. Tunable.
GYM_ANSWER_LEVEL_GAP = int(os.getenv("POKEMON_GYM_ANSWER_LEVEL_GAP", "3"))
GYM_LEVEL_MARGIN     = int(os.getenv("POKEMON_GYM_LEVEL_MARGIN", "1"))


def load_strategy_kb(path=_KB_PATH, log=print):
    """Load the curated strategy KB. Degrades to {} (planner silently no-ops) on any failure — never a
    hard dep, and LOUD on failure so a missing/broken KB can't fail silently (HARD CONSTRAINT #3)."""
    try:
        with open(path, encoding="utf-8") as f:
            kb = json.load(f)
        threats = kb.get("threats") or {}
        log(f"   [planner] strategy KB loaded: {len(threats)} threats, "
            f"{len(kb.get('species_quality') or {})} keeper species, {len(kb.get('key_moves') or {})} key moves")
        return kb
    except Exception as e:
        log(f"   [planner] !! strategy KB load FAILED ({e}) — planner disabled, no forward prep "
            f"(LOUD; falls back to the reactive loss-learning layer)")
        return {}


def _types_of(mon):
    """The Gen-3 type list for a party mon dict (uses species_id if present, else the name)."""
    sid = mon.get("species_id")
    if not sid:
        # map name → id via the reverse of SPECIES_NAME (rare path; live state always carries species_id)
        nm = (mon.get("species") or "").lower()
        sid = next((k for k, v in st.SPECIES_NAME.items() if v == nm), 0)
    return [t for t in st.SPECIES_TYPES.get(int(sid or 0), []) if t]


class StrategicPlanner:
    """Reads the strategy KB and turns 'what's ahead' into a proactive prep beat. Stateless w.r.t. game
    RAM (pure function of the passed-in state + the KB); default-constructed on the Campaign, headless-safe.
    Emits NOTHING on its own — the campaign calls plan_note(state) and folds the result into the oracle ctx."""

    def __init__(self, kb=None, log=print):
        self.log = log
        self.kb = kb if kb is not None else load_strategy_kb(log=log)
        self.threats = self.kb.get("threats") or {}
        self.species_quality = self.kb.get("species_quality") or {}
        self.key_moves = self.kb.get("key_moves") or {}
        self.e4_seq = self.kb.get("e4_sequence") or ["Lorelei", "Bruno", "Agatha", "Lance", "Gary"]
        self._last_sig = None            # de-dupe the LOG line (the ctx fold still happens every tick)

    # ── keeper lookup (used by roster_judgment for opportunistic catching) ────────────────────────────
    def keeper(self, species_name):
        """The species-quality record for a wild species name (or None). tier ∈ rare_strong/strong/
        project/decent. Lets roster_judgment recognise a KEEPER (a Pikachu!) not just a type-hole."""
        if not species_name:
            return None
        return self.species_quality.get(species_name.lower())

    # ── gym readiness (Fix B: the ENFORCED pre-gym prep reads this to decide catch/grind) ──────────────
    def gym_readiness(self, gym_name, party, party_target=3, loss_bump=0):
        """Assess whether `party` is ready for gym `gym_name`, from the KB (pure logic; no game RAM).
        Returns a dict the campaign's prep_for_gym ACTS on (catch a counter / grind to level / develop the
        bench), or None if the KB has no record for this gym. Fields:
          ace, ace_level, level_target (grind-to), top_level, underleveled,
          has_type_answer (a party mon of a weak-to type, within GYM_ANSWER_LEVEL_GAP of the ace),
          want_types (answer types she doesn't field), answer_species (KB counters to seek when catching),
          party_size, thin (party smaller than the target), ready (nothing to prep).
        loss_bump ESCALATES the demand after a loss (grind higher / want a bigger team on the retry)."""
        rec = self.threats.get(gym_name)
        if not rec:
            return None
        band = rec.get("level_band") or [0, 0]
        ace_level = band[-1] if band else 0
        weak = set(rec.get("weak_to") or []) | set(rec.get("answer_types") or [])
        answer_ok_floor = max(1, ace_level - GYM_ANSWER_LEVEL_GAP)
        has_answer = any((weak & set(_types_of(m))) and m.get("level", 0) >= answer_ok_floor
                         for m in party)
        ptypes = self._party_types(party)
        want = [t for t in (rec.get("answer_types") or []) if t not in ptypes]
        level_target = ace_level + GYM_LEVEL_MARGIN + loss_bump
        top_level = max((m.get("level", 0) for m in party), default=0)
        target_size = min(6, party_target + (1 if loss_bump else 0))
        return {
            "gym": gym_name, "ace": rec.get("ace"), "ace_level": ace_level,
            "level_target": level_target, "top_level": top_level, "underleveled": top_level < level_target,
            "has_type_answer": has_answer, "want_types": want,
            "answer_species": rec.get("answer_species") or [],
            "party_size": len(party), "target_size": target_size, "thin": len(party) < target_size,
            "ready": has_answer and top_level >= level_target and len(party) >= target_size,
        }

    # ── threat selection ─────────────────────────────────────────────────────────────────────────────
    def _current_threat(self, state):
        """The threat she should be PREPPING for, as (name, rec). Gym → next_gym leader; all-8-badges
        (pre-credits) → the Elite Four bloc (name '__E4__'); post-game → None (victory lap, no prep)."""
        if state.get("post_game"):
            return None, None
        ng = state.get("next_gym")
        if ng and ng.get("leader") in self.threats:
            return ng["leader"], self.threats[ng["leader"]]
        if state.get("badge_count", 0) >= 8:
            return "__E4__", None                       # the E4 gauntlet — handled specially below
        return None, None

    # ── the plan note (folded into the oracle ctx) ───────────────────────────────────────────────────
    def plan_note(self, state):
        """ONE proactive, in-character anticipation beat for the oracle ctx (or '' when nothing to prep:
        post-game, or no threat identified). Composes: (1) do I have a type answer to what's ahead? →
        confidence or a concrete WANT; (2) the single smartest prep action — level an existing hard-counter
        that's lagging, or fix an ace-heavy bench. Kept to ≤2 clauses so it reads as a beat, never a dump."""
        if not PLANNER_ENABLED:
            return ""
        try:
            name, rec = self._current_threat(state)
            if name is None:
                return ""
            party = state.get("party") or []
            if not party:
                return ""
            # GOAL-PINNED WATCH (2026-07-08): when Jonny pins a focused moment (watch.py --goal /
            # POKEMON_WATCH_GOAL), the doctrine is "go straight at the objective, don't drift onto
            # grinding". So on a pin we keep the in-the-moment matchup ANTICIPATION (the cute homework
            # beat) but DROP the "go prep/level first" directive that would walk her out of the fight.
            # On a real fresh run (no pin) the full prep actionable fires — that's where it belongs.
            pinned = bool(state.get("watch_goal"))
            if name == "__E4__":
                note = self._e4_note(party, pinned=pinned)
            else:
                note = self._gym_note(name, rec, party, pinned=pinned)
            if note:
                sig = note[:60]
                if sig != self._last_sig:
                    self._last_sig = sig
                    self.log(f"   [planner] forward-prep for {name}: folding a proactive plan beat "
                             f"into the oracle ctx")
            return note
        except Exception as e:
            self.log(f"   [planner] plan_note err {e} (LOUD; skipping this tick's prep beat)")
            return ""

    # ── helpers ──────────────────────────────────────────────────────────────────────────────────────
    def _party_types(self, party):
        s = set()
        for m in party:
            s.update(_types_of(m))
        return s

    def _bench_alarm(self, party):
        """A concrete bench-development clause if the team is ace-heavy (the L67-over-L15 smell), else ''.
        This is the whole-team-development lever: it fires PROACTIVELY, not only after a loss."""
        lv = sorted(m.get("level", 0) for m in party)
        if len(lv) >= 3 and (lv[-1] - lv[0]) >= BENCH_SPREAD_ALARM:
            return (f"and be honest — my bench is way behind my ace (L{lv[0]} vs L{lv[-1]}); leaning on one "
                    f"carry is how you get picked apart, so I should level the WHOLE team, not just the star")
        return ""

    def _counter_action(self, rec, party):
        """If she OWNS a hard-counter to this threat that's lagging in level, the smartest single move is
        to bring it up — name it (the Lapras-for-Lance gem). Returns a clause or ''."""
        low = (rec.get("level_band") or [0, 0])[0]
        answers = set(rec.get("answer_species") or [])
        answer_types = set(rec.get("answer_types") or [])
        # prefer a named answer_species she owns; else any owned mon whose TYPE is a listed answer
        owned_named = [m for m in party if (m.get("species") or "").lower() in answers]
        owned_typed = [m for m in party if answer_types & set(_types_of(m))]
        pool = owned_named or owned_typed
        laggards = [m for m in pool if m.get("level", 0) < max(1, low - COUNTER_UNDERLEVEL_GAP)]
        if laggards:
            m = min(laggards, key=lambda x: x.get("level", 0))
            tt = "/".join(_types_of(m)) or "the right type"
            return (f"and here's the thing — my {m['species']} ({tt}) is a textbook answer to them, but it's "
                    f"only L{m.get('level', '?')} and dead on the bench; that's exactly the mon I should be "
                    f"levelling up before I go")
        return ""

    def _gym_note(self, name, rec, party, pinned=False):
        types = "/".join(rec.get("types") or []) or "??"
        low, high = (rec.get("level_band") or [0, 0])[0], (rec.get("level_band") or [0, 0])[-1]
        band = f"~L{low}" if low == high else f"L{low}-{high}"
        ptypes = self._party_types(party)
        have = [t for t in (rec.get("weak_to") or []) if t in ptypes]
        want = [t for t in (rec.get("answer_types") or []) if t not in ptypes]
        # LEAD CLAUSE — matchup foresight (item 5): confidence if I already field an answer, else a WANT.
        if have:
            lead = (f"LOOKING AHEAD: next is {name} in {rec.get('city', 'the next city')} — {types}-types, "
                    f"{band}. Good news: my {have[0]} already has the edge on them, so I've got a real answer")
        elif pinned:
            # pinned: acknowledge the gap as in-the-moment grit, NOT a "go get one first" directive
            lead = (f"LOOKING AHEAD: {name} up next — {types}-types, {band}. I don't have a clean answer for "
                    f"that, honestly… this one's going to be a scrap, but I'm going in")
        else:
            wt = " or ".join(want[:2]) or "a counter"
            lead = (f"LOOKING AHEAD: next is {name} in {rec.get('city', 'the next city')} — {types}-types, "
                    f"{band}. I've got no clean answer to that on my team — before I walk in there I want a "
                    f"{wt} type ({rec.get('counter', '')})")
        # ACTION CLAUSE — the single smartest prep (lagging counter > bench alarm). Suppressed on a pin
        # (a focused watch shouldn't have her wander off to grind mid-objective).
        action = "" if pinned else (self._counter_action(rec, party) or self._bench_alarm(party))
        return (lead + (". " + _cap(action) if action else ".")).strip()

    def _e4_note(self, party, pinned=False):
        """The Elite Four bloc beat — the scaffolding-free hole at the run's climax. Names the gauntlet +
        her per-seat answers in one breath, spotlights the ICE/Lance insight tied to HER party, and folds
        the bench-development alarm (the E4 punishes a lopsided team hardest)."""
        ptypes = self._party_types(party)
        lance = self.threats.get("Lance") or {}
        low = (lance.get("level_band") or [52, 60])[0]
        # ice answer she OWNS (Lapras/Dewgong/Jynx/Cloyster/Articuno)?
        ice_mons = [m for m in party if "ice" in _types_of(m)]
        ice_owned = ice_mons[0] if ice_mons else None
        pieces = ["THE ELITE FOUR IS AHEAD — this is what I've been building toward, so let me think it "
                  "through: Lorelei opens with ice-and-water (lightning and grass hit her), Bruno's the "
                  "fighting muscle (psychic turns him to jelly), Agatha's the ghost-and-poison woman "
                  "(psychic and ground both answer her — and Normal moves do NOTHING to her ghosts), then "
                  "Lance the dragon master"]
        # the signature Lance/ICE beat, grounded in her actual team
        if ice_owned:
            lv = ice_owned.get("level", 0)
            if lv < max(1, low - COUNTER_UNDERLEVEL_GAP) and not pinned:
                # fresh run: the actionable "level her first" (she should prep before the gauntlet)
                pieces.append(f"— and ICE is the dragon-slayer, so my {ice_owned['species']} is literally my "
                              f"Lance answer… except it's only L{lv}. That is the mon I need to level up "
                              f"BEFORE the gauntlet, not the ace that's already fine")
            else:
                # pinned watch (or already-levelled): in-the-moment confidence, no walk-out directive
                pieces.append(f"— and ICE is the dragon-slayer, so my {ice_owned['species']} is my Lance "
                              f"answer. That's the plan when I get to him")
        elif not pinned:
            pieces.append("— and everyone says ICE is the only thing a dragon fears, so an Ice-type or an "
                          "Ice Beam is the piece I really want before I face him")
        else:
            pieces.append("— and everyone says ICE is the only thing a dragon fears; I'll have to out-fight "
                          "him without one")
        # bench alarm — the E4 is where an ace-heavy team dies (fresh-run prep only; a pin fights as-is)
        bench = "" if pinned else self._bench_alarm(party)
        if bench:
            pieces.append(f". {_cap(bench)} — there's no Pokémon Center between the five of them, so the "
                          f"whole six has to pull weight")
        note = pieces[0] + " " + pieces[1] + (pieces[2] if len(pieces) > 2 else "")
        return note.strip().rstrip(",") + ("." if not note.rstrip().endswith(".") else "")


def _cap(s):
    return (s[:1].upper() + s[1:]) if s else s


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# THE FORWARD-PLANNING TEAM BRAIN (mission-pivot 2026-07-09, Part B). Extends the reactive
# StrategicPlanner above with a STANDING, PERSISTENT plan: she picks a balanced-6 archetype at the
# start, then — via whole-game lookahead (union of EVERY future threat incl. the E4) — knows the single
# highest-leverage next acquisition/evolve/teach/grind and does it PROACTIVELY, two towns before the wall.
# Reads the deep KB (Part A). Mode-side; the ONLY voice interface is plan_note() (same firewall).
# ══════════════════════════════════════════════════════════════════════════════════════════════════

TEAM_PLANNER_ENABLED = os.getenv("POKEMON_TEAM_PLANNER", "1") == "1"
# A planned slot becomes ACTIONABLE this many badges before its acquisition deadline (so she grabs a
# keeper as soon as its location is reachable, not exactly at the wall). 1 = "due now or next badge".
PLAN_DUE_WINDOW = int(os.getenv("POKEMON_PLAN_DUE_WINDOW", "1"))
# Bounded, watchable grind: top-of-party must be within this of the next milestone or she's "underleveled".
PLAN_UNDERLEVEL_SLACK = int(os.getenv("POKEMON_PLAN_UNDERLEVEL_SLACK", "0"))

_GYM_SEQ = ["Brock", "Misty", "Lt. Surge", "Erika", "Koga", "Sabrina", "Blaine", "Giovanni"]
_E4_SEQ = ["Lorelei", "Bruno", "Agatha", "Lance", "Champion"]


def _load_json(name, log=print):
    try:
        with open(os.path.join(_HERE, "gamedata", name), encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"   [teamplan] !! KB load FAILED {name} ({e}) — LOUD; that layer degrades")
        return {}


class TeamPlanner:
    """Standing forward-planning team brain. PERSISTENT plan-state (banked in the campaign bundle);
    whole-game lookahead over the deep KB; emits ONE proactive PlanAction + a first-person voiced line.
    Pure function of (party, badges, bag, dex) + the KB + the persisted plan-state — headless-safe.

    Firewall: mode-side only. plan_note(state) is the sole voice seam. NEVER writes states/kira/.
    """

    def __init__(self, kb=None, log=print, soul=None):
        self.log = log
        self.team_plan = _load_json("frlg_team_plan.json", log)
        self.rosters = _load_json("frlg_rosters.json", log)
        self.evolutions = _load_json("frlg_evolutions.json", log)
        self.learnsets = _load_json("frlg_learnsets.json", log)
        self.tms = _load_json("frlg_tms.json", log)
        self.encounters = _load_json("frlg_encounters.json", log)
        self.strategy = (kb or load_strategy_kb(log=log)).get("threats") or {}
        self.soul = soul or {}
        self.archetypes = self.team_plan.get("archetypes") or []
        self.state = None          # the persisted plan-state (set by init_plan / load)
        self._last_sig = None

    # ── archetype selection + plan init ───────────────────────────────────────────────────────────
    def _starter_of(self, party):
        lines = {"bulbasaur": "bulbasaur", "ivysaur": "bulbasaur", "venusaur": "bulbasaur",
                 "charmander": "charmander", "charmeleon": "charmander", "charizard": "charmander",
                 "squirtle": "squirtle", "wartortle": "squirtle", "blastoise": "squirtle"}
        for m in party:
            b = lines.get((m.get("species") or "").lower())
            if b:
                return b
        return "bulbasaur"

    def _pick_archetype(self, party):
        branch = self._starter_of(party)
        for a in self.archetypes:
            if a.get("starter_branch") == branch:
                return a
        return self.archetypes[0] if self.archetypes else {"name": "none", "slots": []}

    def init_plan(self, party, badges, soul=None):
        """Choose the archetype (seeded by her REAL starter) and build the persistent slot-plan. SHE
        chooses (soul may reorder favorites); the archetype is a menu, not a solver's dictation."""
        if soul:
            self.soul = soul
        arch = self._pick_archetype(party)
        slots = []
        for s in arch.get("slots", []):
            slots.append({
                "role": s["role"], "target": (s.get("species") or [s.get("role")])[-1],
                "line": s.get("line") or s.get("species") or [], "covers": s.get("covers") or [],
                "acquire": s.get("acquire") or {}, "evolve": s.get("evolve") or [],
                "teach": s.get("teach") or [], "why": s.get("why", ""),
                "status": "planned",
            })
        self.state = {
            "archetype": arch.get("name"), "starter_branch": arch.get("starter_branch"),
            "slots": slots, "history": [],
            "level_milestones": arch.get("level_milestones") or {},
            "threat_answers": arch.get("threat_answers") or {},
        }
        self._recompute_status(party)
        self.log(f"   [teamplan] plan initialised: archetype={arch.get('name')} "
                 f"({sum(1 for s in slots if s['status'] != 'planned')}/{len(slots)} slots already fielded)")
        return self.state

    def ensure_plan(self, party, badges):
        if self.state is None:
            self.init_plan(party, badges)
        else:
            self._recompute_status(party)
        return self.state

    # ── live status: which slots are already fielded / evolved ────────────────────────────────────
    def _recompute_status(self, party):
        # BOX-AWARE (2026-07-13, order 3d): a slot satisfied by a mon sitting in the PC BOX must read
        # "acquired", not "planned" — else assess() keeps proposing catch_keeper and the router marches
        # her back to re-catch it (the wasted-detour half of the duplicate-Dugtrio bug). The Campaign
        # refreshes _owned_box_names at catch-decision cadence (no per-tick PC scan); absent => party-only.
        names = {(m.get("species") or "").lower() for m in party} | (getattr(self, "_owned_box_names", None) or set())
        for s in self.state["slots"]:
            line = [x.lower() for x in s["line"]]
            target = s["target"].lower()
            if target in names:
                s["status"] = "evolved"
            elif any(n in names for n in line):
                s["status"] = "acquired"
            else:
                # keep a manually-set 'acquired' (e.g. a gift boxed) unless clearly absent
                if s["status"] in ("planned",):
                    s["status"] = "planned"
                elif not any(n in names for n in line + [target]):
                    s["status"] = "planned"

    # ── whole-game lookahead ──────────────────────────────────────────────────────────────────────
    def _upcoming_threats(self, badge_count, post_game=False):
        if post_game:
            return []
        up = [g for i, g in enumerate(_GYM_SEQ) if i >= badge_count]  # gyms not yet beaten
        return up + list(_E4_SEQ)                                     # + the whole E4 + Champion

    def _serves(self, role, upcoming):
        ta = self.state.get("threat_answers") or {}
        return [t for t in upcoming if role in (ta.get(t, {}).get("answer_slots") or [])]

    def _due(self, slot, badge_count):
        by = (slot.get("acquire") or {}).get("by_badge")
        return by is not None and by <= badge_count + PLAN_DUE_WINDOW

    def _next_milestone(self, badge_count, post_game):
        up = self._upcoming_threats(badge_count, post_game)
        ms = self.state.get("level_milestones") or {}
        for t in up:
            if t in ms:
                return t, ms[t]
        return None, 0

    # ── the assessment: the single highest-leverage next action ───────────────────────────────────
    def assess(self, party, badges, bag=None, dex=None, post_game=False):
        """Scan ALL remaining threats and return the ONE highest-leverage PlanAction + a WHY + voice.
        kind ∈ catch_keeper | acquire_special | evolve | grind_to | teach_tm | develop_bench | on_track."""
        if not TEAM_PLANNER_ENABLED:
            return None
        badge_count = badges if isinstance(badges, int) else len(badges or [])
        self.ensure_plan(party, badge_count)
        if post_game:
            return {"kind": "on_track", "why": "post-game victory lap — the team is built",
                    "voice": ""}
        upcoming = self._upcoming_threats(badge_count, post_game)

        # (1) DUE, MISSING keeper — the proactive heart. Earliest deadline, then highest multiplicity.
        planned = [s for s in self.state["slots"] if s["status"] == "planned" and self._due(s, badge_count)]
        if planned:
            def leverage(s):
                by = (s["acquire"] or {}).get("by_badge", 99)
                serves = self._serves(s["role"], upcoming)
                return (by, -len(serves))
            planned.sort(key=leverage)
            s = planned[0]
            acq = s["acquire"] or {}
            serves = self._serves(s["role"], upcoming)
            sp = acq.get("species")
            if sp:
                return {"kind": "catch_keeper", "species": sp, "where": acq.get("where"),
                        "slot": s["role"], "serves": serves,
                        "why": f"{sp} -> {s['target']} answers {', '.join(serves) or 'the road ahead'}",
                        "voice": self._voice_catch(sp, s, serves)}
            return {"kind": "acquire_special", "species": s["target"], "where": acq.get("where"),
                    "method": acq.get("method"), "slot": s["role"], "serves": serves,
                    "why": f"{s['target']} ({acq.get('method')}) — {s['why']}",
                    "voice": self._voice_acquire(s, serves)}

        # (2) READY evolution — a stone-evo whose keeper is in hand (level-evos happen naturally).
        ev = self._ready_evolution(party, bag)
        if ev:
            return ev

        # (3) UNDERLEVELED for the next milestone — bounded, watchable grind.
        threat, target = self._next_milestone(badge_count, post_game)
        top = max((m.get("level", 0) for m in party), default=0)
        if threat and top < target - PLAN_UNDERLEVEL_SLACK:
            return {"kind": "grind_to", "level": target, "threat": threat, "top_level": top,
                    "why": f"top mon L{top} < the L{target} I want for {threat}",
                    "voice": (f"Before {threat} I want to be around L{target} — we're at L{top}, so a "
                              f"little focused training, then straight in. No grinding for its own sake.")}

        # (4) DUE coverage TM the target mon can learn (advisory — bag/coins unverified here).
        tm = self._due_teach(party, badge_count)
        if tm:
            return tm

        # (5) ACE-HEAVY bench (the L67-over-L15 smell) — develop the whole six.
        lv = sorted(m.get("level", 0) for m in party)
        if len(lv) >= 3 and (lv[-1] - lv[0]) >= BENCH_SPREAD_ALARM:
            return {"kind": "develop_bench", "spread": lv[-1] - lv[0],
                    "why": f"bench L{lv[0]} vs ace L{lv[-1]} — leaning on one carry",
                    "voice": (f"My bench is dragging (L{lv[0]}) behind my ace (L{lv[-1]}) — the E4 has no "
                              f"Center between five fights, so I need the WHOLE team pulling weight, not one star.")}

        # (6) on track.
        fielded = sum(1 for s in self.state["slots"] if s["status"] != "planned")
        return {"kind": "on_track", "fielded": fielded, "why": f"{fielded}/6 slots fielded, levels on pace",
                "voice": ""}

    def next_action(self, party, badges, bag=None, dex=None, post_game=False):
        return self.assess(party, badges, bag=bag, dex=dex, post_game=post_game)

    def _ready_evolution(self, party, bag):
        """A deliberate evolution that's READY: a stone-evo keeper in hand whose stone is available, or a
        level-evo mon sitting past its evolution level (should have evolved — nudge). Returns action or None."""
        bag = bag or {}
        for m in party:
            sp = (m.get("species") or "").lower()
            rec = self.evolutions.get(sp)
            if not rec:
                continue
            if rec.get("method") == "stone":
                stones = [rec["stone"]] if rec.get("stone") else list((rec.get("options") or {}).keys())
                have = [s for s in stones if bag.get(s)]
                # only nudge if this species is one of our target lines
                if have and self._is_target_line(sp):
                    into = rec.get("into") or (rec.get("options") or {}).get(have[0], "its evolution")
                    return {"kind": "evolve", "species": sp, "into": into, "method": "stone",
                            "stone": have[0], "why": f"{sp} -> {into} with the {have[0]} I'm holding",
                            "voice": (f"I've got the {have[0].replace('-', ' ')} — time to make my {sp} "
                                      f"into {into}. Been planning this one.")}
            elif rec.get("method") == "level" and m.get("level", 0) >= rec.get("level", 999):
                # normally auto-evolves; if it hasn't, it's fine — don't spam. (informational, low value)
                continue
        return None

    def _is_target_line(self, species):
        s = species.lower()
        for slot in self.state["slots"]:
            if s in [x.lower() for x in slot["line"]] or s == slot["target"].lower():
                return True
        return False

    def _due_teach(self, party, badge_count):
        arch = next((a for a in self.archetypes if a.get("name") == self.state["archetype"]), {})
        names = {(m.get("species") or "").lower() for m in party}
        taught = {(h.get("tm"), h.get("species")) for h in self.state["history"] if h.get("kind") == "teach"}
        for t in arch.get("teach_plan", []):
            if (t.get("when_badge", 99) <= badge_count and t.get("to", "").lower() in names
                    and (t.get("tm"), t.get("to")) not in taught):
                return {"kind": "teach_tm", "tm": t["tm"], "move": t.get("move"), "mon": t.get("to"),
                        "where": t.get("where"), "for": t.get("for") or [],
                        "why": f"teach {t.get('move')} to {t.get('to')} for {', '.join(t.get('for') or [])}",
                        "voice": (f"I should get {t.get('move','that move').replace('-', ' ')} onto my "
                                  f"{t.get('to')} — it's my answer for {', '.join(t.get('for') or ['what''s ahead'])}. "
                                  f"Grab it at {t.get('where','the shop')}.")}
        return None

    # ── voice templates (first-person, forward-looking, guide-literate kid) ────────────────────────
    def _voice_catch(self, sp, slot, serves):
        who = sp.replace("-", " ")
        where = (slot["acquire"] or {}).get("where", "the route ahead")
        if serves:
            tail = (f"— that's my answer for {serves[0]}"
                    + (f" AND {serves[1]}" if len(serves) > 1 else "") + " down the line")
        else:
            tail = "— rounding out the team"
        return (f"Next on my list: catch an {who} around {where}. It grows into my {slot['target']} {tail}. "
                f"Grabbing it now, before I need it.")

    def _voice_acquire(self, slot, serves):
        where = (slot["acquire"] or {}).get("where", "where it lives")
        tail = f" — my {serves[0]} answer" if serves else ""
        return (f"Time to pick up my {slot['target']} from {where}{tail}. "
                f"This has been on the plan since the start.")

    # ── history / persistence ─────────────────────────────────────────────────────────────────────
    def on_acquire(self, species, party=None):
        self.state["history"].append({"kind": "acquire", "species": species})
        if party is not None:
            self._recompute_status(party)

    def on_evolve(self, species, into):
        self.state["history"].append({"kind": "evolve", "species": species, "into": into})

    def on_teach(self, tm, species):
        self.state["history"].append({"kind": "teach", "tm": tm, "species": species})

    def save(self, dir_path):
        """Persist plan-state to the campaign bundle (dev-line only; NEVER states/kira/)."""
        try:
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, "team_plan_state.json"), "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=1)
            return True
        except Exception as e:
            self.log(f"   [teamplan] !! save FAILED ({e}) — LOUD")
            return False

    def load(self, dir_path):
        try:
            p = os.path.join(dir_path, "team_plan_state.json")
            if not os.path.exists(p):
                return False
            with open(p, encoding="utf-8") as f:
                self.state = json.load(f)
            return True
        except Exception as e:
            self.log(f"   [teamplan] !! load FAILED ({e}) — LOUD; will re-init")
            return False

    # ── the voice seam (the ONLY interface into her ctx) ──────────────────────────────────────────
    def plan_note(self, state):
        """ONE forward-looking beat for the oracle ctx (or '' when nothing to do / post-game). Same
        firewall + dedupe-log discipline as StrategicPlanner.plan_note."""
        if not TEAM_PLANNER_ENABLED:
            return ""
        try:
            party = state.get("party") or []
            if not party:
                return ""
            act = self.assess(party, state.get("badge_count", 0),
                              bag=state.get("bag"), dex=state.get("dex_caught"),
                              post_game=bool(state.get("post_game")))
            voice = (act or {}).get("voice", "")
            if voice:
                sig = (act.get("kind"), act.get("species") or act.get("threat") or act.get("mon"))
                if sig != self._last_sig:
                    self._last_sig = sig
                    self.log(f"   [teamplan] next action: {act['kind']} "
                             f"({act.get('why','')}) — folding a forward-plan beat")
            return voice
        except Exception as e:
            self.log(f"   [teamplan] plan_note err {e} (LOUD; skipping this tick)")
            return ""
