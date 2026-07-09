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
            if name == "__E4__":
                note = self._e4_note(party)
            else:
                note = self._gym_note(name, rec, party)
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

    def _gym_note(self, name, rec, party):
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
        else:
            wt = " or ".join(want[:2]) or "a counter"
            lead = (f"LOOKING AHEAD: next is {name} in {rec.get('city', 'the next city')} — {types}-types, "
                    f"{band}. I've got no clean answer to that on my team — before I walk in there I want a "
                    f"{wt} type ({rec.get('counter', '')})")
        # ACTION CLAUSE — the single smartest prep (lagging counter > bench alarm).
        action = self._counter_action(rec, party) or self._bench_alarm(party)
        return (lead + (". " + _cap(action) if action else ".")).strip()

    def _e4_note(self, party):
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
            if lv < max(1, low - COUNTER_UNDERLEVEL_GAP):
                pieces.append(f"— and ICE is the dragon-slayer, so my {ice_owned['species']} is literally my "
                              f"Lance answer… except it's only L{lv}. That is the mon I need to level up "
                              f"BEFORE the gauntlet, not the ace that's already fine")
            else:
                pieces.append(f"— and ICE is the dragon-slayer, so my {ice_owned['species']} (Ice Beam!) is "
                              f"exactly his counter. I think I've actually got a plan")
        else:
            pieces.append("— and everyone says ICE is the only thing a dragon fears, so an Ice-type or an "
                          "Ice Beam is the piece I really want before I face him")
        # bench alarm — the E4 is where an ace-heavy team dies
        bench = self._bench_alarm(party)
        if bench:
            pieces.append(f". {_cap(bench)} — there's no Pokémon Center between the five of them, so the "
                          f"whole six has to pull weight")
        note = pieces[0] + " " + pieces[1] + (pieces[2] if len(pieces) > 2 else "")
        return note.strip().rstrip(",") + ("." if not note.rstrip().endswith(".") else "")


def _cap(s):
    return (s[:1].upper() + s[1:]) if s else s
