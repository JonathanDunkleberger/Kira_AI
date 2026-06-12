# salience_filter.py — label-based salience scoring for Kira's input gate.
#
# Two responsibilities:
#   1. Input gate (pre-triage): should this input enter the response pipeline?
#      Fast drop for ambient/NPC content; hard floor for JONNY voice.
#   2. Context weighting helpers: staleness notes + LOW truncation so ambient
#      context doesn't crowd out higher-priority input.
#
# Pure Python, synchronous, <1ms per call. No LLM, no network, no I/O.
# Module-level _ambient_seen tracks per-source novelty across the session.

import re
import time
import random

# ─── Tuning constants ─────────────────────────────────────────────────────────

SALIENCE_HIGH_THRESHOLD   = 70     # score ≥ 70  → HIGH  (direct address / clear question)
SALIENCE_MEDIUM_THRESHOLD = 40     # score ≥ 40  → MEDIUM (normal triage path)
SALIENCE_LOW_THRESHOLD    = 20     # score ≥ 20  → LOW   (triage, biased toward BRIEF)
                                   # score < 20  → DROP  (discarded, subject to stochastic pass)

AMBIENT_PASS_RATE         = 0.08   # 8% of DROPs are stochastically promoted to LOW
                                   # Simulates background content occasionally cutting through —
                                   # one funny NPC line per few minutes, not a firehose.

NOVELTY_WINDOW_S          = 30.0   # Seconds before same-source repeated content is allowed
NOVELTY_JACCARD_THRESHOLD = 0.65   # Word-overlap ratio that triggers the novelty penalty
NOVELTY_PENALTY           = -15    # Score penalty for near-duplicate ambient content

AMBIENT_CONTEXT_MAX_CHARS = 120    # Hard truncation for LOW tier in dynamic_context injection

DIRECT_ADDRESS_BUMP       = 20     # Bump when Kira's name appears in content
QUESTION_BUMP             = 15     # Bump for "?" in chat or game_dialogue
PLAYER_ADDRESS_BUMP       = 10     # Bump for second-person address in game_dialogue

# Recency decay — vision/ambient only. Escalating penalty with age.
# These are mutually exclusive tiers (25s penalty replaces 15s, not stacks).
STALE_PENALTY_15S         = -10    # Mild: observation is 15–24s old
STALE_PENALTY_25S         = -25    # Heavy: observation is 25–29s old
STALE_INELIGIBLE_S        = 30.0   # >30s: primary_eligible=False — reference only, never leads

# ─── Base scores by source ────────────────────────────────────────────────────

_BASE: dict = {
    "voice":         100,  # Jonny speaking — always HIGH before any bump/penalty
    "chat":          55,   # Twitch/YouTube viewer — can reach HIGH via address bump
    "game_dialogue": 20,   # Game character speech — can reach MEDIUM via address bump
    "ambient_npc":   10,   # Background NPC chatter — almost always DROP
    "audio_summary": 15,   # Mood/music summary from audio_agent
    "vision":        35,   # Scene description from vision_agent
    "system":        0,    # Internal messages — never surfaced to pipeline
}

# ─── Module-level novelty cache ───────────────────────────────────────────────

# {source_key: (last_content: str, last_ts: float)}
_ambient_seen: dict = {}

# ─── Public API ───────────────────────────────────────────────────────────────

def score(
    source: str,
    content: str,
    capture_ts: float = 0.0,
    kira_names: tuple = ("kira",),
) -> tuple:
    """Score an input or context source for salience.

    Args:
        source:      raw source string ("voice", "chat", "game_dialogue",
                     "ambient_npc", "audio_summary", "vision")
        content:     the text content to score
        capture_ts:  wall-clock time when content was captured (for recency
                     decay). 0.0 or None = no recency penalty (treat as fresh).
        kira_names:  lowercase name variants used to detect direct address

    Returns:
        (score: int, tier: str, primary_eligible: bool)

        tier:
          "HIGH"   — enters pipeline; voice triage bypass skips Groq call
          "MEDIUM" — normal triage path
          "LOW"    — triage with immersive=True hint (biases BRIEF/QUIET);
                     context is truncated to AMBIENT_CONTEXT_MAX_CHARS
          "DROP"   — discarded; subject to AMBIENT_PASS_RATE stochastic rescue

        primary_eligible:
          False when a vision/ambient observation is older than STALE_INELIGIBLE_S.
          The source can still be referenced in context, but must NOT drive a
          proactive comment ("I can see X right now" when X is 35s stale).
          Always True for voice and chat — they are live input by definition.
    """
    s = _base_score(source)
    s += _direct_address_bump(content, source, kira_names)
    s += _novelty_penalty_delta(source, content)
    recency_delta, primary_eligible = _recency_decay(source, capture_ts)
    s += recency_delta

    # ── JONNY VOICE FLOOR ────────────────────────────────────────────────────
    # No amount of short content, novelty penalty, or any other factor can
    # prevent Jonny's voice from entering the response pipeline. Her creator's
    # voice is always at minimum MEDIUM. primary_eligible is always True for
    # voice — it's a live input, freshness never applies.
    if source == "voice":
        s = max(s, SALIENCE_MEDIUM_THRESHOLD)
        primary_eligible = True

    tier = _tier(s)

    # ── Stochastic pass-through ───────────────────────────────────────────────
    # 8% of DROPs become LOW — the occasional background line that cuts through.
    # Voice is floored at MEDIUM so it can never reach this branch.
    if tier == "DROP" and random.random() < AMBIENT_PASS_RATE:
        tier = "LOW"

    return s, tier, primary_eligible


def truncate_for_low(text: str) -> str:
    """Truncate ambient LOW-tier content to AMBIENT_CONTEXT_MAX_CHARS.
    Prevents ambient context from crowding out higher-priority content.
    Word-boundary aware: never cuts mid-word."""
    if len(text) <= AMBIENT_CONTEXT_MAX_CHARS:
        return text
    return text[:AMBIENT_CONTEXT_MAX_CHARS].rsplit(" ", 1)[0] + "…"


def staleness_note(capture_ts: float = 0.0) -> str:
    """Return a parenthetical note when an observation is stale, else ''.
    Injected into [VISUAL PERCEPTION] headers so Claude knows the observation's
    age without Kira having to guess — she treats it as 'may have changed'
    rather than confidently describing it as live.

    Returns '' when capture_ts is 0/None (freshness unknown — give benefit of doubt)
    or when age < 15s (fresh enough to treat as current).
    """
    if not capture_ts:
        return ""
    age = time.time() - capture_ts
    if age < 15.0:
        return ""
    if age < 25.0:
        return f" (~{int(age)}s ago — may have changed)"
    return f" (~{int(age)}s ago — likely stale, reference only)"


# ─── Private helpers ──────────────────────────────────────────────────────────

def _base_score(source: str) -> int:
    return _BASE.get(source, 10)


def _direct_address_bump(content: str, source: str, kira_names: tuple) -> int:
    """Bump score when content is explicitly directed at Kira or the player."""
    text_lower = content.lower()

    # Any source: Kira's name mentioned → direct address
    if any(name in text_lower for name in kira_names):
        return DIRECT_ADDRESS_BUMP

    # Chat or game dialogue with a question
    if source in ("chat", "game_dialogue") and "?" in content:
        return QUESTION_BUMP

    # Game dialogue addressed to the player character (second-person / character name)
    if source == "game_dialogue" and re.search(
        r"\b(you|bond|agent|commander)\b", text_lower
    ):
        return PLAYER_ADDRESS_BUMP

    return 0


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity. Cheap: set intersection on tokenised words."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _novelty_penalty_delta(source: str, content: str) -> int:
    """Penalise near-duplicate repeated content from ambient/context sources.

    The same NPC barking the same patrol line three times gets −15 on the second
    and subsequent occurrences within NOVELTY_WINDOW_S seconds.

    Voice and chat are excluded — human speech is always novel by definition.
    """
    if source in ("voice", "chat", "system"):
        return 0
    now = time.time()
    key = source
    if key in _ambient_seen:
        prev_content, prev_ts = _ambient_seen[key]
        if now - prev_ts < NOVELTY_WINDOW_S:
            if _jaccard(content, prev_content) >= NOVELTY_JACCARD_THRESHOLD:
                # Timer resets only when fresh content arrives — don't update cache here
                return NOVELTY_PENALTY
    # Fresh content — update cache
    _ambient_seen[key] = (content, now)
    return 0


def _recency_decay(source: str, capture_ts: float) -> tuple:
    """Escalating recency penalty for vision/ambient sources.

    Returns (penalty: int, primary_eligible: bool).
    penalty is 0 or negative.

    primary_eligible=False means this source should NOT drive a proactive comment.
    Kira can reference it ("earlier I noticed…") but should not open with it
    ("I can see X right now") when the observation is >30s old.

    Voice and chat are live input — no recency penalty ever applies.

    Decay tiers (mutually exclusive — later tiers replace earlier ones):
      15s–24s:  STALE_PENALTY_15S  (-10)  — mild, possibly still current
      25s–29s:  STALE_PENALTY_25S  (-25)  — heavy, likely outdated
      ≥30s:     STALE_PENALTY_25S  (-25)  + primary_eligible=False
    """
    if source in ("voice", "chat", "system"):
        return 0, True
    if not capture_ts:
        # No timestamp → unknown age → no penalty (benefit of the doubt)
        return 0, True
    age = time.time() - capture_ts
    if age >= STALE_INELIGIBLE_S:
        return STALE_PENALTY_25S, False
    if age >= 25.0:
        return STALE_PENALTY_25S, True
    if age >= 15.0:
        return STALE_PENALTY_15S, True
    return 0, True


def _tier(s: int) -> str:
    if s >= SALIENCE_HIGH_THRESHOLD:
        return "HIGH"
    if s >= SALIENCE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    if s >= SALIENCE_LOW_THRESHOLD:
        return "LOW"
    return "DROP"
