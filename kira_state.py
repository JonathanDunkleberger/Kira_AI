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
import time

try:
    from config import CLAUDE_CHAT_MODEL
except Exception:
    CLAUDE_CHAT_MODEL = "claude-sonnet-4-6"


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
        # Called shots scaffold — full resolution mechanic wired in a later batch.
        # Schema (future): {text, formed_at, context_snapshot, status}
        self.called_shots: list[dict] = []

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
                print(f"   [KiraState] Summary task error: {e}")

        # Theory check: every 50 updates (was 25) + wall-clock floor, max 5 open.
        if self._updates_since_theory_check >= 50:
            open_count = sum(1 for t in self.active_theories if t["status"] == "open")
            if (open_count < 5
                    and self.session_narrative_summary
                    and (now - self._last_theory_wall) >= self._THEORY_MIN_INTERVAL):
                try:
                    asyncio.ensure_future(self._maybe_form_theory())
                    self._last_theory_wall = now
                except Exception as e:
                    print(f"   [KiraState] Theory task error: {e}")
            # Reset counter whether or not we scheduled.
            self._updates_since_theory_check = 0
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
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
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
        except Exception as e:
            print(f"   [KiraState] Theory resolution error: {e}")
        return None

    # ── Prompt injection ───────────────────────────────────────────────────────

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

        # Top entity attachments (only entities above threshold)
        attached = [(e, v) for e, v in self.sentiment_ledger.items() if v >= 0.25]
        if attached:
            top = sorted(attached, key=lambda x: -x[1])[:5]
            att_str = ", ".join(f"{e} ({v:.2f})" for e, v in top)
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
                    resp = await self.ai_core.anthropic_client.messages.create(
                        model=CLAUDE_CHAT_MODEL,
                        max_tokens=250,
                        messages=[{"role": "user", "content": prompt}],
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
                except Exception as e:
                    err_low = str(e).lower()
                    if any(s in err_low for s in (
                        "rate limit", "429", "529", "overloaded", "too many requests"
                    )):
                        delay = 5.0 * (2 ** attempt)
                        print(
                            f"   [KiraState] Summary rate-limited "
                            f"(attempt {attempt+1}/3) \u2014 backing off {delay:.0f}s."
                        )
                        await asyncio.sleep(delay)
                        continue
                    print(f"   [KiraState] Summary error: {e}")
                    return
            print("   [KiraState] Summary: all retries exhausted \u2014 skipping.")
        finally:
            self._summary_task_running = False

    async def _maybe_form_theory(self) -> None:
        """Periodically form a theory about the current situation.
        Ported from VN autopilot's _maybe_form_theory; generalized for all modes."""
        if not getattr(self.ai_core, "anthropic_client", None):
            return
        if not self.session_narrative_summary:
            return

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

        prompt = (
            "You are Kira, live on stream \u2014 currently paying close attention to "
            "what's unfolding (could be a game, visual novel, or media).\n\n"
            "Based on what's happened so far, do you have a GENUINE theory, suspicion, "
            "or prediction about where things are going?\n\n"
            f"SITUATION SO FAR:\n{self.session_narrative_summary}"
            f"{entity_note}{prior_block}\n\n"
            "If you have a genuine, specific theory (about a character, plot thread, "
            "or foreshadowed event): output it in Kira's first-person voice, 1-2 sentences.\n"
            "Examples of good theories:\n"
            "  'I don\u2019t trust Greenway. Something about how he talks about the mission.'\n"
            "  'Nagisa\u2019s health keeps getting mentioned. I have a bad feeling about this.'\n"
            "  'That locked door keeps showing up in the background. We\u2019re going in there.'\n"
            "  'This informant is going to turn on Jonny. He\u2019s too helpful.'\n\n"
            "If nothing specific comes to mind: output exactly NONE.\n\n"
            "Just the theory text or NONE."
        )
        try:
            resp = await self.ai_core.anthropic_client.messages.create(
                model=CLAUDE_CHAT_MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.content[0].text.strip()
            if result and result.upper() != "NONE" and len(result) > 15:
                theory = {
                    "theory": result,
                    "formed_at": time.time(),
                    "context_snapshot": self.session_narrative_summary[:200],
                    "status": "open",
                }
                self.active_theories.append(theory)
                print(f"   [KiraState] Theory formed: {result[:80]}")
        except Exception as e:
            print(f"   [KiraState] Theory formation error: {e}")
