# kira_state.py — Cross-mode shared agency / stakes layer
#
# Owns: theories, entity sentiment/attachment, session investment,
#       rolling narrative summary, and the UNIFIED session intensity.
#
# All mode consumers (VN autopilot, game/general observer, media, general)
# read from and write to ONE instance of this object per bot session.
#
# Design constraints:
#   - Pass ai_core directly to __init__ — NEVER reach into bot to avoid
#     circular references (bot → kira_state → bot).
#   - asyncio single-threaded: no locks needed for field access.
#   - Background LLM tasks (summary, theory formation) fire as ensure_future;
#     they never block the caller.

import asyncio
import enum
import json
import os
import re
import time

try:
    from kira.config import CLAUDE_SONNET_MODEL as CLAUDE_CHAT_MODEL
except Exception:
    CLAUDE_CHAT_MODEL = "claude-sonnet-4-6"  # last-resort fallback if config import fails


# ── Unified Session Intensity ──────────────────────────────────────────────────
# Single vocabulary consumed by BOTH the game/general observer (was MomentType)
# and VN autopilot (was INTENSITY_* string constants).
# One source of truth — no two consumers that can disagree and inject
# contradictory prompt signals.
#
# Mapping from old vocabularies:
#   MomentType.NEUTRAL   → CALM
#   MomentType.LULL      → CALM
#   MomentType.TENSE     → TENSE
#   MomentType.EMOTIONAL → EMOTIONAL
#   MomentType.CUTSCENE  → CUTSCENE
#   INTENSITY_CALM       → CALM
#   INTENSITY_BUILDING   → BUILDING
#   INTENSITY_INTENSE    → INTENSE
#   INTENSITY_CLIMACTIC  → CLIMACTIC
#   INTENSITY_AFTERMATH  → AFTERMATH

class SessionIntensity(enum.Enum):
    CALM      = "calm"       # low stakes; riff and theorize freely
    BUILDING  = "building"   # tension rising; lean in, anticipate
    TENSE     = "tense"      # active threat/pressure; react sparingly
    INTENSE   = "intense"    # things happening; react sharply
    EMOTIONAL = "emotional"  # character/emotional beat; give it weight
    CLIMACTIC = "climactic"  # peak moment; silence is usually correct
    AFTERMATH = "aftermath"  # post-big-moment; process quietly
    CUTSCENE  = "cutscene"   # pure narrative transition; advance/skip


# ── VN backward-compatible aliases ────────────────────────────────────────────
# vn_autopilot.py uses these names in comparisons and dict keys.
# Re-aliasing them here means VN code like `if intensity == INTENSITY_CALM:`
# continues to work without any string changes — they now compare enums.
INTENSITY_CALM      = SessionIntensity.CALM
INTENSITY_BUILDING  = SessionIntensity.BUILDING
INTENSITY_INTENSE   = SessionIntensity.INTENSE
INTENSITY_CLIMACTIC = SessionIntensity.CLIMACTIC
INTENSITY_AFTERMATH = SessionIntensity.AFTERMATH


# Cheap heuristic for spotting a concrete, falsifiable prediction in Kira's own
# response — NO LLM call. Deliberately permissive on the marker, conservative on
# length elsewhere; a false positive just means a shot that quietly expires.
_CALLED_SHOT_MARKERS = re.compile(
    r"\b("
    r"calling it"
    r"|i'?m calling it"
    r"|i give (?:it|this|him|her|them|that)"
    r"|i bet\b|bet you\b|bet that\b|twenty bucks says|bucks says"
    r"|mark my words"
    r"|watch[,:]"          # "watch, this ..." / "watch:"
    r"|(?:is|are|he'?s|she'?s|they'?re|that'?s|this is) (?:gonna|going to)"
    r"|\bgonna\b"
    r"|i predict|my prediction|prediction:"
    r"|\bwill (?:die|betray|turn|fail|win|lose|come back|show up|break|leave)\b"
    r")",
    re.IGNORECASE,
)


class KiraState:
    """
    Cross-mode shared agency layer.

    One instance per bot session, shared across all mode consumers:
      - VNAutopilot reads/writes theories, attachment, intensity via this object.
      - VTubeBot game/general observer loop updates context here.
      - process_and_respond injects get_state_block() into dynamic context.

    AI client: pass ai_core; used only for Claude narrative-summary and theory
    calls. No bot reference; no circular dependency.
    """

    def __init__(self, ai_core):
        self.ai_core = ai_core

        # ── A. Stakes ──────────────────────────────────────────────────────────
        # Theories tracked this session.
        # Schema: {theory: str, formed_at: float, context_snapshot: str,
        #          status: "open"|"confirmed"|"busted",
        #          resolved_at: float (optional)}
        self.active_theories: list[dict] = []

        # Sentiment ledger: generalizes VN's character_attachment.
        # Works for game NPCs, real people, stream regulars, recurring entities.
        # entity_name → float 0.0–1.0 (0=stranger, 1=deeply familiar/invested)
        self.sentiment_ledger: dict[str, float] = {}

        # Raw mention counts driving the ledger (sqrt ramp toward 1.0).
        self.entity_familiarity: dict[str, int] = {}

        # Signed VALENCE ledger — ORTHOGONAL to sentiment_ledger/familiarity.
        # familiarity says HOW MUCH she's around an entity (0..1, never negative);
        # valence says WHICH WAY she feels about them (-1 contempt .. +1 trust).
        # "Isola betrayed us" and "Isola I trust" move this in OPPOSITE directions.
        # Derived from the emotional tone of her tagged reactions, NOT mention count.
        self.entity_valence: dict[str, float] = {}        # entity → -1.0..+1.0
        self.entity_valence_prev: dict[str, float] = {}   # last value, for trend (hardening/thawing)
        self._valence_pending: list[str] = []             # spoken reactions queued for the next derivation pass
        self._last_valence_wall: float = 0.0              # time.time() of last valence pass
        self._VALENCE_MIN_INTERVAL: float = 90.0          # ≥090s between passes (matches summary/theory laziness)
        self._VALENCE_MIN_PENDING: int = 3                # need ≥3 queued reactions before a pass fires
        self._VALENCE_ALPHA: float = 0.4                  # EMA weight — feelings evolve, never flip in one line
        self._VALENCE_DECAY: float = 0.98                 # entities absent from a pass soften toward 0
        self._valence_task_running: bool = False

        # Hard cap on tracked entities — least-recently-touched evicted (LRU) so a
        # long session can't grow these ledgers unbounded (proper-noun extraction
        # leaks one-off names). 100 keeps every meaningful recurring entity.
        self._LEDGER_MAX = 100

        # ── B. Investment ──────────────────────────────────────────────────────
        self.story_investment: float = 0.0      # 0.0–1.0; ramps per update_context()
        self.emotional_trajectory: str = ""     # computed descriptor string
        self.session_narrative_summary: str = "" # rolling ~150-word Sonnet summary

        # Internal: context fragments fed by all consumers; drives summary updates.
        self._context_buffer: list[str] = []
        self._contexts_since_summary: int = 0
        self._summary_task_running: bool = False

        # ── C. Intensity ───────────────────────────────────────────────────────
        self.current_intensity: SessionIntensity = SessionIntensity.CALM
        # Aftermath decay: boxes/ticks remaining in AFTERMATH state after a CLIMACTIC.
        self._aftermath_countdown: int = 0

        # ── D. Felt-history ────────────────────────────────────────────────────
        # Archive of theories that resolved (VN previously discarded these).
        self.resolved_theories: list[dict] = []
        # Called shots — concrete falsifiable predictions Kira makes out loud.
        # Schema: {prediction_text: str, formed_at: float, context_snippet: str,
        #          status: "open"|"hit"|"miss"|"expired"|"discarded",
        #          resolution_note: str (set on hit/miss), resolved_at: float}
        self.called_shots: list[dict] = []
        # Archive of shots that resolved hit/miss — fed to session record / lore.
        self.resolved_called_shots: list[dict] = []
        # At most ONE freshly-resolved shot awaiting a payoff line. dict or None.
        self.pending_payoff: "dict | None" = None
        # Session stat counters (surfaceable later).
        self.called_shot_hits: int = 0
        self.called_shot_misses: int = 0
        # Tunables for the called-shot lifecycle.
        self._CALLED_SHOT_MAX_OPEN: int = 5        # cap on simultaneously-open shots
        self._CALLED_SHOT_EXPIRY: float = 45 * 60  # unresolved shots expire after 45 min
        self._PAYOFF_MIN_INTERVAL: float = 600.0   # at most one payoff every ~10 min
        self._last_payoff_wall: float = 0.0        # time.time() of last payoff injection


        # ── Internal counters ──────────────────────────────────────────────────
        self._total_updates: int = 0
        self._updates_since_theory_check: int = 0

        # ── LoadShed integration ───────────────────────────────────────────────
        # Set by bot.py's LoadShed block whenever GPU/encoder saturation is detected.
        # When True, maybe_run_background_tasks() returns immediately — no LLM tasks
        # fire while the encoder is fighting for headroom.
        self.under_load: bool = False

        # Wall-clock floors: LLM tasks can't fire more than once per interval
        # regardless of how many context pushes have accumulated. Keeps a AAA game
        # session lazy even when many interjections happen in quick succession.
        self._last_summary_wall: float = 0.0   # time.time() of last summary call
        self._last_theory_wall:  float = 0.0   # time.time() of last theory call
        # Minimum seconds between background LLM tasks:
        self._SUMMARY_MIN_INTERVAL: float = 60.0   # at most one summary per minute
        self._THEORY_MIN_INTERVAL:  float = 90.0   # at most one theory per 90s

    # ── C. Intensity interface ─────────────────────────────────────────────────

    def set_intensity(self, intensity: SessionIntensity) -> None:
        """Set current session intensity. Called by both mode observers."""
        self.current_intensity = intensity

    def set_intensity_from_weight(self, weight: float) -> SessionIntensity:
        """Convert a narrative-weight float (0.0–1.0) to SessionIntensity.
        Mirrors VN autopilot's _estimate_scene_intensity logic exactly,
        including the aftermath countdown. Updates current_intensity in place."""
        if self._aftermath_countdown > 0:
            self._aftermath_countdown -= 1
            self.current_intensity = SessionIntensity.AFTERMATH
            return SessionIntensity.AFTERMATH

        if weight >= 0.72:
            self._aftermath_countdown = 4       # 4 ticks/boxes of aftermath
            self.current_intensity = SessionIntensity.CLIMACTIC
        elif weight >= 0.50:
            self.current_intensity = SessionIntensity.INTENSE
        elif weight >= 0.28:
            self.current_intensity = SessionIntensity.BUILDING
        else:
            self.current_intensity = SessionIntensity.CALM
        return self.current_intensity

    # ── A. Stakes — entity tracking ────────────────────────────────────────────

    def _touch_entity(self, name: str) -> None:
        """Mark an entity most-recently-used and evict LRU entries past the cap.
        entity_familiarity is the master key set (sentiment is a subset), so we
        evict from its oldest end and drop the same keys from sentiment_ledger."""
        # Move to the most-recently-used (end) position. Dicts preserve insertion
        # order, so pop + reinsert is the move-to-end.
        if name in self.entity_familiarity:
            self.entity_familiarity[name] = self.entity_familiarity.pop(name)
        while len(self.entity_familiarity) > self._LEDGER_MAX:
            oldest = next(iter(self.entity_familiarity))
            self.entity_familiarity.pop(oldest, None)
            self.sentiment_ledger.pop(oldest, None)
            self.entity_valence.pop(oldest, None)
            self.entity_valence_prev.pop(oldest, None)

    def update_entity_mentions(self, text: str) -> None:
        """Update entity familiarity from text via capitalized-word extraction.
        Identical logic to VN autopilot's _update_character_attachment; works
        for any mode since it's purely text-based."""
        _COMMON = {
            "I", "It", "The", "A", "An", "He", "She", "They", "We", "You",
            "But", "And", "Or", "So", "Then", "When", "If", "That", "This",
            "Is", "Was", "Be", "Have", "Do", "Not", "No", "Yes", "What",
            "My", "His", "Her", "Our", "Your", "Their", "Me", "Him",
        }
        for word in text.split():
            cleaned = word.strip(".,!?\"'\u2014\u2026()[]")
            if (cleaned and cleaned[0].isupper() and cleaned not in _COMMON
                    and len(cleaned) > 2 and cleaned.isalpha()):
                self.entity_familiarity[cleaned] = (
                    self.entity_familiarity.get(cleaned, 0) + 1
                )
                count = self.entity_familiarity[cleaned]
                if count >= 3:
                    # Sqrt ramp: ~50 mentions → 0.5, ~200 mentions → ~1.0
                    raw = min(1.0, (count / 200.0) ** 0.5)
                    self.sentiment_ledger[cleaned] = min(
                        1.0, max(self.sentiment_ledger.get(cleaned, 0.0), raw)
                    )
                self._touch_entity(cleaned)
        # Investment ramps slowly per update
        self.story_investment = min(1.0, self.story_investment + 0.001)

    def update_entity_from_speaker(self, speaker: str) -> None:
        """Boost familiarity for a named speaker (VN speaker-attribution path)."""
        if not speaker:
            return
        self.entity_familiarity[speaker] = self.entity_familiarity.get(speaker, 0) + 1
        count = self.entity_familiarity[speaker]
        raw = min(1.0, (count / 200.0) ** 0.5)
        self.sentiment_ledger[speaker] = min(
            1.0, max(self.sentiment_ledger.get(speaker, 0.0), raw)
        )
        self._touch_entity(speaker)

    # ── A. Stakes — signed valence (feeling, not familiarity) ──────────────────
    # Additive layer on top of the familiarity ledger. Intake is cheap and hot-path
    # safe (just queues text); the LLM derivation runs in maybe_run_background_tasks.

    def note_reaction_for_valence(self, text: str) -> None:
        """Queue one of Kira's spoken reaction lines for the next valence pass.
        Cheap and synchronous — safe to call from any hot path. The actual
        tone→valence derivation happens in _update_entity_valence (background)."""
        if not text:
            return
        cleaned = text.strip()
        if len(cleaned) <= 2:
            return
        self._valence_pending.append(cleaned)
        if len(self._valence_pending) > 40:
            self._valence_pending = self._valence_pending[-40:]

    async def _update_entity_valence(self) -> None:
        """Derive signed per-entity valence from the emotional TONE of Kira's
        recently-spoken reactions and evolve the ledger via EMA. Background
        fire-and-forget; uses the cheap tool_inference path (Groq/local), never Opus.

        Loud by design: prints a [Valence] line on every fire AND every no-op /
        parse failure, so a silently-degrading feeling-renderer can't slip past."""
        if self._valence_task_running:
            return
        if not getattr(self.ai_core, "anthropic_client", None) and not hasattr(self.ai_core, "tool_inference"):
            return
        # Drain the queue under the running flag.
        pending = self._valence_pending[:]
        self._valence_pending = []
        if not pending:
            return
        self._valence_task_running = True
        try:
            joined = "\n".join(f"- {r}" for r in pending[-40:])
            system = (
                "You read an AI co-host's spoken lines and rate how SHE feels about each "
                "named character or person mentioned, based ONLY on the emotional tone of "
                "THESE lines. Scale: -1.0 = distrust / contempt / betrayal / hostility, "
                "0.0 = neutral, +1.0 = trust / fondness / affection / loyalty. "
                "Only include PROPER NAMES of characters or people she has a feeling about. "
                "Omit anyone neutral, unnamed, or only mentioned in passing. "
                "Return ONLY a compact JSON object mapping name to float, e.g. "
                '{\"Isola\": -0.6, \"Coldfish\": 0.4}. If nothing qualifies, return {}.'
            )
            user = f"Kira's spoken lines (oldest first):\n{joined}"
            raw = await self.ai_core.tool_inference(system, user, max_tokens=200)
            deltas = self._parse_valence_json(raw)
            if deltas is None:
                print(
                    f"   [Valence] no-op \u2014 could not parse model output "
                    f"({len(pending)} reaction(s) dropped). Raw: {str(raw)[:120]!r}"
                )
                return
            if not deltas:
                print(f"   [Valence] no-op \u2014 model found no named feelings in {len(pending)} reaction(s).")
                # Still decay untouched entities so feelings soften over quiet stretches.
                self._decay_absent_valence(set())
                return

            alpha = self._VALENCE_ALPHA
            touched: set[str] = set()
            updated_pairs = []
            for name, delta in deltas.items():
                name = name.strip()
                if not name:
                    continue
                try:
                    delta = max(-1.0, min(1.0, float(delta)))
                except (TypeError, ValueError):
                    continue
                old = self.entity_valence.get(name, 0.0)
                self.entity_valence_prev[name] = old
                new = max(-1.0, min(1.0, (1.0 - alpha) * old + alpha * delta))
                self.entity_valence[name] = new
                # Keep the entity in the familiarity key set so LRU + persistence
                # track it (valence-only entities would otherwise dodge eviction).
                self.entity_familiarity.setdefault(name, self.entity_familiarity.get(name, 1))
                self._touch_entity(name)
                touched.add(name)
                updated_pairs.append(f"{name} {old:+.2f}\u2192{new:+.2f}")

            self._decay_absent_valence(touched)
            print(f"   [Valence] updated {len(updated_pairs)} entit(ies): " + "; ".join(updated_pairs))
        except Exception as e:
            print(f"   [Valence] pass FAILED (continuing): {e}")
        finally:
            self._valence_task_running = False
            self._last_valence_wall = time.time()

    def _decay_absent_valence(self, touched: set) -> None:
        """Soften feelings toward entities that didn't appear in this pass."""
        decay = self._VALENCE_DECAY
        for name in list(self.entity_valence.keys()):
            if name in touched:
                continue
            self.entity_valence[name] *= decay
            if abs(self.entity_valence[name]) < 0.02:
                # Negligible feeling — drop it so it renders as neutral again.
                self.entity_valence.pop(name, None)
                self.entity_valence_prev.pop(name, None)

    @staticmethod
    def _parse_valence_json(raw: str) -> "dict | None":
        """Extract a {name: float} JSON object from the model output.
        Returns the dict (possibly empty) on success, or None if unparseable."""
        if not raw:
            return {}
        text = raw.strip()
        # Strip code fences if the model wrapped the JSON.
        if text.startswith("```"):
            text = text.strip("`")
            nl = text.find("\n")
            if nl != -1:
                text = text[nl + 1:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    # ── A. Stakes — cross-session persistence ──────────────────────────────────
    # The sentiment ledger and familiarity counts are the ONLY agency state that
    # should COMPOUND across sessions: attachment to a recurring character/person
    # grows the more she's been around them. Theories, called-shots, intensity,
    # and the narrative summary are intentionally session-scoped and NOT persisted.
    LEDGER_PERSIST_PATH = os.path.join("memory_db", "sentiment_ledger.json")

    def save_ledger(self) -> None:
        """Persist sentiment_ledger + entity_familiarity to disk so attachment
        compounds across sessions instead of resetting in-RAM. Called at session
        end. Best-effort: never raises into the caller."""
        try:
            if not self.entity_familiarity and not self.sentiment_ledger:
                return  # nothing earned this session — don't clobber prior file
            os.makedirs(os.path.dirname(self.LEDGER_PERSIST_PATH), exist_ok=True)
            payload = {
                "saved_at": time.time(),
                "entity_familiarity": dict(self.entity_familiarity),
                "sentiment_ledger": {k: round(v, 4) for k, v in self.sentiment_ledger.items()},
                "entity_valence": {k: round(v, 4) for k, v in self.entity_valence.items()},
            }
            with open(self.LEDGER_PERSIST_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"   [KiraState] Sentiment ledger saved ({len(self.sentiment_ledger)} entities) → {self.LEDGER_PERSIST_PATH}")
        except Exception as e:
            print(f"   [KiraState] Ledger save failed: {e}")

    def load_ledger(self) -> None:
        """Restore sentiment_ledger + entity_familiarity from disk at startup so
        prior-session attachment is the baseline this session keeps ramping from.
        Best-effort: a missing/corrupt file leaves the in-RAM ledgers empty."""
        try:
            if not os.path.exists(self.LEDGER_PERSIST_PATH):
                return
            with open(self.LEDGER_PERSIST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            fam = data.get("entity_familiarity", {})
            sent = data.get("sentiment_ledger", {})
            val = data.get("entity_valence", {})
            if isinstance(fam, dict):
                for name, count in fam.items():
                    try:
                        self.entity_familiarity[name] = int(count)
                    except (TypeError, ValueError):
                        continue
            if isinstance(sent, dict):
                for name, v in sent.items():
                    try:
                        self.sentiment_ledger[name] = max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        continue
            # Signed valence carries across sessions purely event-driven (v1: no
            # time-based load-decay — that constant is deferred until live data
            # shows how fast a grudge should actually soften over a day off).
            if isinstance(val, dict):
                for name, v in val.items():
                    try:
                        self.entity_valence[name] = max(-1.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        continue
            # Enforce the LRU cap on the restored set (oldest insertion evicted).
            while len(self.entity_familiarity) > self._LEDGER_MAX:
                oldest = next(iter(self.entity_familiarity))
                self.entity_familiarity.pop(oldest, None)
                self.sentiment_ledger.pop(oldest, None)
                self.entity_valence.pop(oldest, None)
                self.entity_valence_prev.pop(oldest, None)
            if self.sentiment_ledger:
                _vn = len(self.entity_valence)
                _vnote = f", {_vn} with felt valence" if _vn else ""
                print(f"   [KiraState] Sentiment ledger loaded ({len(self.sentiment_ledger)} entities{_vnote}) — attachment compounding from prior sessions")
        except Exception as e:
            print(f"   [KiraState] Ledger load failed: {e}")

    # ── A. Stakes — narrative weight (for VN pacing callers) ──────────────────

    def estimate_narrative_weight(self, text: str) -> float:
        """Heuristic text-based narrative weight 0.0–1.0.
        Ported verbatim from VN autopilot's _estimate_narrative_weight.
        No LLM call — rules only, <1ms."""
        t = text.lower()
        score = 0.20

        if any(w in t for w in ("died", " dead", "death", "killed", "never come back",
                                 "farewell", "goodbye forever", "gone forever", "is gone")):
            score += 0.50
        if any(w in t for w in ("truth", "secret", "real ", "lied", "wasn't",
                                 "has always", "never told", "hidden", "discovered",
                                 "realized", "all along", "knew it", "was lying")):
            score += 0.35
        if any(w in t for w in ("i love", "love you", "always loved", "always felt",
                                 "feelings for", "can't hide", "wanted to tell")):
            score += 0.30
        if text.count("!") >= 2 or text.count("...") >= 3:
            score += 0.15
        if "?" in text and any(w in t for w in ("why", "how could", "what have you", "what did")):
            score += 0.10
        if len(text) < 35:
            score -= 0.15
        if len(text) > 300:
            score += 0.10

        for entity, attachment in self.sentiment_ledger.items():
            if attachment >= 0.3 and entity.lower() in t:
                score += 0.15 * min(attachment, 1.0)

        return max(0.0, min(1.0, score))

    # ── B. Investment — trajectory ─────────────────────────────────────────────

    def update_emotional_trajectory(self) -> None:
        """Recompute trajectory string from current session state. No LLM call."""
        parts = []
        if self._total_updates > 300:
            parts.append("deep in the session")
        elif self._total_updates > 100:
            parts.append("several exchanges in")
        elif self._total_updates > 30:
            parts.append("getting into it")

        if self.story_investment > 0.60:
            parts.append("deeply invested")
        elif self.story_investment > 0.30:
            parts.append("getting invested")

        top = sorted(self.sentiment_ledger.items(), key=lambda x: -x[1])
        attached = [(e, v) for e, v in top if v >= 0.35][:3]
        if attached:
            parts.append("tracking " + ", ".join(e for e, _ in attached))

        self.emotional_trajectory = "; ".join(parts) if parts else ""

    # ── Core update interface ──────────────────────────────────────────────────

    def update_context_sync(self, text: str, speaker: str = "") -> None:
        """Synchronous update: entity mentions, investment, context buffer.
        Call this from any hot path. Does NOT trigger LLM tasks — call
        maybe_run_background_tasks() after to schedule those if appropriate."""
        if not text or not text.strip():
            return
        self._total_updates += 1
        self._updates_since_theory_check += 1
        self._contexts_since_summary += 1

        self.update_entity_mentions(text)
        if speaker:
            self.update_entity_from_speaker(speaker)

        self._context_buffer.append(text)
        if len(self._context_buffer) > 30:
            self._context_buffer = self._context_buffer[-20:]

    async def update_context(self, text: str, speaker: str = "") -> None:
        """Convenience async wrapper around update_context_sync +
        maybe_run_background_tasks. Use this when you can await."""
        self.update_context_sync(text, speaker)
        await self.maybe_run_background_tasks()

    async def maybe_run_background_tasks(self) -> None:
        """Schedule background LLM tasks (summary, theory) when thresholds met.
        Fire-and-forget via ensure_future — never blocks the caller.

        Load-awareness: returns immediately when under_load=True (encoder saturated).
        Wall-clock floors: even if thresholds are met, tasks fire at most once per
        _SUMMARY_MIN_INTERVAL / _THEORY_MIN_INTERVAL so a AAA game session stays lazy.
        Thresholds: summary cold-start=20, rolling=40; theory=50 — all conservative
        to avoid spurious API calls during active gameplay."""
        # Hard bail when GPU/encoder is fighting for headroom.
        if self.under_load:
            return

        now = time.time()

        # Narrative summary: conservative thresholds + wall-clock floor.
        # Cold-start: 20 pushes before first summary (was 5).
        # Rolling: 40 pushes between summaries (was 15).
        first_threshold = 20 if not self.session_narrative_summary else 40
        if (self._contexts_since_summary >= first_threshold
                and len(self._context_buffer) >= 3
                and not self._summary_task_running
                and (now - self._last_summary_wall) >= self._SUMMARY_MIN_INTERVAL):
            try:
                asyncio.ensure_future(self._update_narrative_summary())
                self._last_summary_wall = now
            except Exception as e:
                print(f"   [WARN] kira_state: summary task scheduling error: {e}")

        # Theory check: every 50 updates (was 25) + wall-clock floor, max 5 open.
        # The same background call also resolves open called shots (piggybacked),
        # so it must still fire when theories are capped but shots are open.
        if self._updates_since_theory_check >= 50:
            self._sweep_called_shots()
            open_count = sum(1 for t in self.active_theories if t["status"] == "open")
            open_shots = sum(1 for s in self.called_shots if s["status"] == "open")
            if ((open_count < 5 or open_shots > 0)
                    and self.session_narrative_summary
                    and (now - self._last_theory_wall) >= self._THEORY_MIN_INTERVAL):
                try:
                    asyncio.ensure_future(self._maybe_form_theory())
                    self._last_theory_wall = now
                except Exception as e:
                    print(f"   [WARN] kira_state: theory task scheduling error: {e}")
            # Reset counter whether or not we scheduled.
            self._updates_since_theory_check = 0

        # Valence derivation: tone→feeling pass over queued reactions. Lazy by the
        # same discipline as summary/theory — ≥90s apart AND ≥3 pending reactions.
        if (len(self._valence_pending) >= self._VALENCE_MIN_PENDING
                and not self._valence_task_running
                and (now - self._last_valence_wall) >= self._VALENCE_MIN_INTERVAL):
            try:
                asyncio.ensure_future(self._update_entity_valence())
                self._last_valence_wall = now
            except Exception as e:
                print(f"   [WARN] kira_state: valence task scheduling error: {e}")
            self._updates_since_theory_check = 0

    # ── Theory resolution (callable from hot path) ────────────────────────────

    async def check_theory_resolutions(self, text: str) -> "str | None":
        """Check if current text confirms or busts an open theory.
        Ported from VN autopilot's _check_theory_resolutions.
        Uses a fast keyword heuristic before calling Claude.
        Returns spoken reaction string, or None."""
        open_theories = [t for t in self.active_theories if t["status"] == "open"]
        if not open_theories:
            return None
        if not getattr(self.ai_core, "anthropic_client", None):
            return None

        # Fast heuristic: any significant word overlap between theory and text?
        t_lower = text.lower()
        likely_relevant = False
        for theory in open_theories:
            theory_words = {
                w.lower().strip(".,!?\"\u2019\u2018\u2014\u2026()")
                for w in theory["theory"].split()
                if len(w) > 4
            }
            if sum(1 for w in theory_words if w in t_lower) >= 2:
                likely_relevant = True
                break

        if not likely_relevant:
            return None

        theories_str = "\n".join(
            f"  {i+1}. {t['theory']}" for i, t in enumerate(open_theories[:5])
        )
        prompt = (
            f"You are Kira, paying close attention to the ongoing situation.\n\n"
            f"Your open theories:\n{theories_str}\n\n"
            f"You just encountered this:\n\"{text}\"\n\n"
            f"Does this CONFIRM or BUST any of your theories?\n"
            f"If yes: CONFIRM:N:reaction text  OR  BUST:N:reaction text\n"
            f"('I CALLED it!' energy for confirms; 'I was wrong about that' for busts.)\n"
            f"If no clear connection: NONE\n\n"
            f"Output ONLY that exact format or NONE."
        )
        try:
            resp = await asyncio.wait_for(
                self.ai_core.anthropic_client.messages.create(
                    model=CLAUDE_CHAT_MODEL,
                    max_tokens=120,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=20,
            )
            result = resp.content[0].text.strip()
            if not result or result.upper() == "NONE":
                return None
            parts = result.split(":", 2)
            if len(parts) == 3:
                resolution = parts[0].strip().upper()
                try:
                    idx = int(parts[1].strip()) - 1
                except ValueError:
                    return None
                reaction = parts[2].strip()
                if 0 <= idx < len(open_theories) and reaction:
                    theory = open_theories[idx]
                    theory["status"] = "confirmed" if resolution == "CONFIRM" else "busted"
                    theory["resolved_at"] = time.time()
                    # Archive a copy; active_theories still holds the object.
                    self.resolved_theories.append(dict(theory))
                    print(
                        f"   [KiraState] Theory {resolution}: "
                        f"{theory['theory'][:60]}"
                    )
                    return reaction
        except asyncio.TimeoutError:
            print("   [WARN] kira_state (theory resolution) LLM call timed out after 20s — skipping")
        except Exception as e:
            print(f"   [WARN] kira_state: theory resolution error: {e}")
        return None

    # ── Called shots: capture → resolve → payoff ───────────────────────────────

    def _sweep_called_shots(self) -> None:
        """Expire stale/over-cap open shots. Pure bookkeeping, no LLM.
        - Any open shot older than _CALLED_SHOT_EXPIRY becomes 'expired'.
        - If still over _CALLED_SHOT_MAX_OPEN open, the oldest expire first."""
        now = time.time()
        for shot in self.called_shots:
            if shot["status"] == "open" and (now - shot["formed_at"]) > self._CALLED_SHOT_EXPIRY:
                shot["status"] = "expired"
                print(f"   [CalledShot] EXPIRED: {shot['prediction_text'][:70]}")

        open_shots = [s for s in self.called_shots if s["status"] == "open"]
        if len(open_shots) > self._CALLED_SHOT_MAX_OPEN:
            # Oldest first; expire the overflow.
            open_shots.sort(key=lambda s: s["formed_at"])
            for shot in open_shots[: len(open_shots) - self._CALLED_SHOT_MAX_OPEN]:
                shot["status"] = "expired"
                print(f"   [CalledShot] EXPIRED (over cap): {shot['prediction_text'][:70]}")

    def capture_called_shot(self, response_text: str) -> bool:
        """A) CAPTURE. If Kira's own response contains a concrete falsifiable
        prediction (cheap heuristic — NO LLM call), record it as an open called
        shot. Returns True if a shot was captured. Safe to call on every response."""
        if not response_text:
            return False
        text = response_text.strip()
        if len(text) < 15:
            return False
        if not _CALLED_SHOT_MARKERS.search(text):
            return False

        self._sweep_called_shots()

        prediction_text = text[:200]
        # Dedupe: don't re-capture an identical open prediction.
        for shot in self.called_shots:
            if shot["status"] == "open" and shot["prediction_text"] == prediction_text:
                return False

        # Derive a context snippet from what we know is happening right now.
        context_snippet = (
            self.session_narrative_summary[:200]
            if self.session_narrative_summary
            else (self._context_buffer[-1][:200] if self._context_buffer else "")
        )
        self.called_shots.append({
            "prediction_text": prediction_text,
            "formed_at": time.time(),
            "context_snippet": context_snippet,
            "status": "open",
            "resolution_note": "",
            "resolved_at": 0.0,
        })
        print(f"   [CalledShot] OPEN: {prediction_text[:70]}")

        # Re-sweep so we never sit above the cap after appending.
        self._sweep_called_shots()
        return True

    def get_payoff_directive(self) -> str:
        """C) PAYOFF. If a freshly-resolved shot is awaiting its moment, return the
        injection directive and CONSUME it (cleared after one injection regardless
        of whether Kira uses it). Returns '' when there's nothing to surface, when a
        suppress-gate moment is active (INTENSE/CLIMACTIC), or when the ~10-min
        payoff cooldown hasn't elapsed (held for later in those two cases)."""
        if not self.pending_payoff:
            return ""
        # Never surface a payoff during a cutscene / high-intensity beat.
        if self.current_intensity in (SessionIntensity.INTENSE, SessionIntensity.CLIMACTIC):
            return ""
        now = time.time()
        if (now - self._last_payoff_wall) < self._PAYOFF_MIN_INTERVAL:
            return ""

        pp = self.pending_payoff
        self.pending_payoff = None        # consumed after ONE injection, win or skip
        self._last_payoff_wall = now
        outcome = "HIT" if pp["status"] == "hit" else "MISS"
        print(f"   [CalledShot] PAYOFF injected ({outcome}): {pp['prediction_text'][:60]}")
        directive = (
            f"[CALLED SHOT RESOLVED] You predicted: '{pp['prediction_text']}'. "
            f"Outcome: {outcome} — {pp['resolution_note']}. "
            "If it fits naturally, claim the win (smug, brief) or own the miss "
            "(begrudging, funny). If it doesn't fit the moment, skip it — never force it."
        )
        if outcome == "HIT":
            # A typed "called it." in chat while she keeps talking out loud is the
            # platonic payoff. Offered as an OPTION, never required.
            directive += (
                " If you want, you may instead drop the receipt in chat with "
                "[CHAT: called it.] (or similar) while you keep talking — optional, "
                "only if it lands."
            )
        return directive

    def get_called_shots_record(self) -> str:
        """D) VISIBILITY. Markdown block of resolved called shots for the session
        record, so lore generation can immortalize great calls. Returns '' when
        nothing resolved this session."""
        if not self.resolved_called_shots:
            return ""
        lines = []
        for s in self.resolved_called_shots:
            tag = s["status"].upper()
            note = f" — {s['resolution_note']}" if s.get("resolution_note") else ""
            lines.append(f"- [{tag}] \u201c{s['prediction_text']}\u201d{note}")
        header = (
            f"Called shots this session — {self.called_shot_hits} hit / "
            f"{self.called_shot_misses} miss:"
        )
        return header + "\n" + "\n".join(lines)

    # ── Prompt injection ───────────────────────────────────────────────────────

    def _render_entity_feeling(self, entity: str, familiarity: float) -> str:
        """Render one tracked entity for get_state_block. If a signed valence
        exists (|v| ≥ 0.15), express it as a FEELING word + trend; otherwise fall
        back to the neutral familiarity number (today's behavior). Pure renderer."""
        v = self.entity_valence.get(entity, 0.0)
        if abs(v) < 0.15:
            return f"{entity} ({familiarity:.2f})"

        if v >= 0.6:
            word = "trust"
        elif v >= 0.25:
            word = "warming to"
        elif v >= 0.15:
            word = "mild warmth"
        elif v <= -0.6:
            word = "contempt"
        elif v <= -0.25:
            word = "distrust"
        else:
            word = "wary of"

        # Trend from the last value: is the feeling deepening, softening, flipping?
        prev = self.entity_valence_prev.get(entity, v)
        delta = v - prev
        trend = ""
        if abs(delta) >= 0.08:
            if v < 0 and delta < 0:
                trend = "hardening"
            elif v < 0 < delta:
                trend = "thawing"
            elif v > 0 and delta > 0:
                trend = "warming"
            elif v > 0 > delta:
                trend = "cooling"
            elif (v <= 0) != (prev <= 0):
                trend = "souring" if v < 0 else "turning around"
        return f"{entity} — {word}" + (f", {trend}" if trend else "")

    def get_state_block(self) -> str:
        """Return a [KIRA STATE] block for injection into dynamic context.
        Called by process_and_respond. Returns empty string when state is too
        sparse to be useful (early session with no theories or attachment yet)."""
        parts = []

        # Open theories
        open_theories = [t for t in self.active_theories if t["status"] == "open"]
        if open_theories:
            theory_lines = "\n".join(
                f"  - {t['theory']}" for t in open_theories[:4]
            )
            parts.append(
                f"Active suspicions/theories (formed this session, track them):"
                f"\n{theory_lines}"
            )

        # Recently resolved — surface for callbacks
        if self.resolved_theories:
            recent = self.resolved_theories[-2:]
            res_lines = "\n".join(
                f"  - [{t['status'].upper()}] {t['theory'][:70]}"
                for t in recent
            )
            parts.append(f"Recently resolved:\n{res_lines}")

        # Top entity attachments (only entities above threshold). When a signed
        # valence exists for an entity, render the FEELING (word + trend) instead
        # of the bare familiarity number — "Isola — distrust, hardening" reads as
        # lived feeling; "Isola (0.42)" reads as telemetry. Renderer-only: the
        # familiarity gate and call site are unchanged.
        attached = [(e, v) for e, v in self.sentiment_ledger.items() if v >= 0.25]
        if attached:
            top = sorted(attached, key=lambda x: -x[1])[:5]
            att_str = ", ".join(self._render_entity_feeling(e, v) for e, v in top)
            parts.append(f"Entities you're tracking this session: {att_str}")

        # Investment note (only when meaningful)
        if self.story_investment > 0.60:
            parts.append(
                "You're genuinely invested in what's happening right now."
            )
        elif self.story_investment > 0.35:
            parts.append("You've gotten into this.")

        if not parts:
            return ""

        header = (
            "[KIRA STATE \u2014 active theories and what you're tracking; "
            "let these shape reactions naturally; don't recite them or open with them]"
        )
        return header + "\n" + "\n\n".join(parts)

    # ── Background LLM tasks ───────────────────────────────────────────────────

    async def _update_narrative_summary(self) -> None:
        """Rolling ~150-word situation summary. Background fire-and-forget.
        Ported from VN autopilot's _update_narrative_summary."""
        if not getattr(self.ai_core, "anthropic_client", None):
            return
        self._summary_task_running = True
        try:
            accumulated = "\n---\n".join(self._context_buffer[-20:])
            prev = (
                self.session_narrative_summary
                or "No previous summary \u2014 this is the start."
            )
            prompt = (
                "You are maintaining a running ~150-word summary of what's happening "
                "in a live stream session (could be a game, visual novel, media, or "
                "general conversation).\n\n"
                f"Previous summary:\n{prev}\n\n"
                f"New content:\n{accumulated}\n\n"
                "Write an updated summary in ~150 words. Track names, events, and "
                "emotional beats. Be factual and concise. No commentary or editorializing."
            )
            for attempt in range(3):
                try:
                    resp = await asyncio.wait_for(
                        self.ai_core.anthropic_client.messages.create(
                            model=CLAUDE_CHAT_MODEL,
                            max_tokens=250,
                            messages=[{"role": "user", "content": prompt}],
                        ),
                        timeout=20,
                    )
                    self.session_narrative_summary = resp.content[0].text.strip()
                    self._context_buffer = self._context_buffer[-5:]
                    self._contexts_since_summary = 0
                    self.update_emotional_trajectory()
                    print(
                        f"   [KiraState] Summary updated "
                        f"({len(self.session_narrative_summary)} chars)."
                    )
                    return
                except asyncio.TimeoutError:
                    print("   [WARN] kira_state (narrative summary) LLM call timed out after 20s — skipping")
                    return
                except Exception as e:
                    err_low = str(e).lower()
                    if any(s in err_low for s in (
                        "rate limit", "429", "529", "overloaded", "too many requests"
                    )):
                        delay = 5.0 * (2 ** attempt)
                        print(
                            f"   [KiraState] Summary rate-limited "
                            f"(attempt {attempt+1}/3) — backing off {delay:.0f}s."
                        )
                        await asyncio.sleep(delay)
                        continue
                    print(f"   [WARN] kira_state: narrative summary error: {e}")
                    return
            print("   [KiraState] Summary: all retries exhausted \u2014 skipping.")
        finally:
            self._summary_task_running = False

    async def _maybe_form_theory(self) -> None:
        """Periodically form a theory about the current situation.
        Ported from VN autopilot's _maybe_form_theory; generalized for all modes.

        Piggybacks called-shot RESOLUTION (B): the SAME single Sonnet call also
        examines open called shots against the latest situation and marks any that
        reality has resolved (hit/miss). No new LLM entry point is created."""
        if not getattr(self.ai_core, "anthropic_client", None):
            return
        if not self.session_narrative_summary:
            return

        # Skip forming a brand-new theory when already at the open cap, but still
        # run the call if there are open shots to resolve.
        open_theory_count = sum(1 for t in self.active_theories if t["status"] == "open")
        form_new_theory = open_theory_count < 5

        prior_block = ""
        if self.active_theories:
            all_t = "\n".join(
                f"  - {t['theory']}" for t in self.active_theories[-6:]
            )
            prior_block = (
                f"\n\nTheories you've already formed (don't repeat these):\n{all_t}"
            )

        attached = [(e, v) for e, v in self.sentiment_ledger.items() if v >= 0.3]
        entity_note = ""
        if attached:
            top = sorted(attached, key=lambda x: -x[1])[:4]
            entity_note = (
                f"\n\nEntities you're tracking: {', '.join(e for e, _ in top)}."
            )

        # ── Called-shot resolution block (piggybacked) ────────────────────────
        self._sweep_called_shots()
        open_shots = [s for s in self.called_shots if s["status"] == "open"]
        shots_block = ""
        if open_shots:
            shot_lines = "\n".join(
                f"  {i+1}. {s['prediction_text']}" for i, s in enumerate(open_shots[:5])
            )
            shots_block = (
                "\n\nPART 2 — RESOLVE CALLED SHOTS:\n"
                "You earlier made these concrete predictions out loud. Based ONLY on "
                "the situation so far above, has reality RESOLVED any of them yet?\n"
                f"{shot_lines}\n"
                "For each, decide HIT (clearly came true), MISS (clearly proven wrong), "
                "NOT_A_PREDICTION (this line isn't actually a concrete, falsifiable "
                "prediction — it's a scene description, a reaction, or vague musing), "
                "or OPEN (a real prediction, just not resolved yet — when in doubt between "
                "HIT/MISS, OPEN)."
            )

        theory_part = (
            "PART 1 — THEORY:\n"
            "Based on what's happened so far, do you have a GENUINE theory, suspicion, "
            "or prediction about where things are going?\n"
            "If you have a genuine, specific theory (about a character, plot thread, "
            "or foreshadowed event): write it in Kira's first-person voice, 1-2 sentences.\n"
            "Examples of good theories:\n"
            "  'I don\u2019t trust Greenway. Something about how he talks about the mission.'\n"
            "  'Nagisa\u2019s health keeps getting mentioned. I have a bad feeling about this.'\n"
            "  'This informant is going to turn on Jonny. He\u2019s too helpful.'\n"
            "If nothing specific comes to mind: write NONE."
        ) if form_new_theory else (
            "PART 1 — THEORY:\nWrite NONE (you already have plenty of open theories)."
        )

        out_format = (
            "\n\nOUTPUT FORMAT — exactly these lines, nothing else:\n"
            "THEORY: <your theory in first person, or NONE>"
        )
        if open_shots:
            out_format += (
                "\nSHOT 1: <HIT|MISS|NOT_A_PREDICTION|OPEN> | <short note on what resolved it, or 'not yet'>"
                "\n(one SHOT line per prediction listed, in order)"
            )

        prompt = (
            "You are Kira, live on stream \u2014 currently paying close attention to "
            "what's unfolding (could be a game, visual novel, or media).\n\n"
            f"SITUATION SO FAR:\n{self.session_narrative_summary}"
            f"{entity_note}{prior_block}\n\n"
            f"{theory_part}"
            f"{shots_block}"
            f"{out_format}"
        )
        try:
            resp = await asyncio.wait_for(
                self.ai_core.anthropic_client.messages.create(
                    model=CLAUDE_CHAT_MODEL,
                    max_tokens=220,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=20,
            )
            result = resp.content[0].text.strip()
            self._parse_theory_and_shots(result, open_shots, form_new_theory)
        except asyncio.TimeoutError:
            print("   [WARN] kira_state (theory formation) LLM call timed out after 20s — skipping")
        except Exception as e:
            print(f"   [WARN] kira_state: theory formation error: {e}")

    def _parse_theory_and_shots(
        self, result: str, open_shots: "list[dict]", form_new_theory: bool
    ) -> None:
        """Parse the combined THEORY/SHOT output. Forms a new theory (B never
        touches this) and resolves any called shots reality answered."""
        if not result:
            return

        # PART 1 — theory.
        m_theory = re.search(r"^THEORY:\s*(.*)$", result, re.MULTILINE | re.IGNORECASE)
        theory_text = m_theory.group(1).strip() if m_theory else ""
        if (form_new_theory and theory_text
                and theory_text.upper() != "NONE" and len(theory_text) > 15):
            self.active_theories.append({
                "theory": theory_text,
                "formed_at": time.time(),
                "context_snapshot": self.session_narrative_summary[:200],
                "status": "open",
            })
            print(f"   [KiraState] Theory formed: {theory_text[:80]}")

        # PART 2 — called-shot resolutions.
        if not open_shots:
            return
        for m in re.finditer(
            r"^SHOT\s*(\d+):\s*(HIT|MISS|NOT_A_PREDICTION|OPEN)\s*\|\s*(.*)$",
            result, re.MULTILINE | re.IGNORECASE,
        ):
            try:
                idx = int(m.group(1)) - 1
            except ValueError:
                continue
            verdict = m.group(2).strip().upper()
            note = m.group(3).strip()
            if not (0 <= idx < len(open_shots)):
                continue
            if verdict == "OPEN":
                continue
            shot = open_shots[idx]
            if shot["status"] != "open":
                continue
            # The capture heuristic is permissive and sometimes opens a shot on a
            # line that isn't really a falsifiable prediction. Discard those here
            # so they never reach payoff.
            if verdict == "NOT_A_PREDICTION":
                shot["status"] = "discarded"
                shot["resolved_at"] = time.time()
                print(f"   [CalledShot] DISCARDED: {shot['prediction_text'][:70]}")
                continue
            shot["status"] = "hit" if verdict == "HIT" else "miss"
            shot["resolution_note"] = note
            shot["resolved_at"] = time.time()
            if verdict == "HIT":
                self.called_shot_hits += 1
            else:
                self.called_shot_misses += 1
            self.resolved_called_shots.append(dict(shot))
            # Most-recent resolution wins the single payoff slot.
            self.pending_payoff = dict(shot)
            print(f"   [CalledShot] {verdict}: {shot['prediction_text'][:60]} — {note[:50]}")
