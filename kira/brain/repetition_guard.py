# repetition_guard.py — kill the single fastest bot-tell: Kira repeating herself.
#
# A bot says the same thing the same way. A person varies, or notices they're
# repeating ("I know I keep saying this—") — which is MORE human than silent
# non-repetition, so we prefer it when natural. This module is the CORE, all-games
# guard: a fast LEXICAL similarity check against a rolling window of her own recent
# spoken lines (ai_core._recent_tts_texts), used two ways:
#   1. PROACTIVE (primary, zero added latency): inject an avoidance directive into
#      the generation prompt so she varies BEFORE the line is produced.
#   2. BACKSTOP (observability): flag a line that still came out too similar.
#
# DELIBERATELY LEXICAL, NOT EMBEDDING-BASED. Latency is a feature here (presence is
# sacred) — token-set Jaccard + difflib ratio is sub-millisecond and catches the
# near-duplicate phrasings that read as robotic. No model call, no network, no I/O.
# Same discipline as salience_filter / chat_director.

import os
import re
import difflib

_WORD = re.compile(r"[a-z0-9']+")

# Tunables (env-overridable). The threshold is "how close is too close" on a 0..1 scale.
REPEAT_THRESHOLD = float(os.getenv("KIRA_REPEAT_THRESHOLD", "0.80"))
REPEAT_WINDOW    = int(os.getenv("KIRA_REPEAT_WINDOW", "5"))   # how many recent lines to guard against


def _tokens(s: str):
    return _WORD.findall((s or "").lower())


def similarity(a: str, b: str) -> float:
    """Fast 0..1 similarity of two short utterances. max() of token-set Jaccard (catches
    'reordered same words') and difflib sequence ratio (catches 'minor rewording') — either
    signal firing means it reads as a repeat. Sub-millisecond; pure function."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    jac = len(ta & tb) / len(ta | tb)
    seq = difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
    return max(jac, seq)


def most_similar(text: str, recent):
    """(closest_prior_line, score) over the recent window, or (None, 0.0)."""
    best, score = None, 0.0
    for r in recent or ():
        if not r:
            continue
        s = similarity(text, r)
        if s > score:
            best, score = r, s
    return best, score


def is_repetitive(text: str, recent, threshold: float = REPEAT_THRESHOLD):
    """(is_repeat, closest_line, score). True when the new line is too close to one she
    just said — the backstop for logging / self-aware handling."""
    b, s = most_similar(text, recent)
    return (s >= threshold), b, s


def avoidance_block(recent, n: int = REPEAT_WINDOW) -> str:
    """The PROACTIVE directive: list her most-recent lines and tell her to vary — preferring
    self-aware acknowledgement over verbatim repetition. '' when there's nothing recent to guard.
    Injected into the generation prompt so she steers off a repeat BEFORE producing it."""
    lines = [r.strip() for r in list(recent or ())[-n:] if r and r.strip()]
    if not lines:
        return ""
    body = "\n".join(f'  - "{l[:120]}"' for l in lines)
    return (
        "[DON'T REPEAT YOURSELF — you said these in the last few moments:\n" + body + "\n"
        "Say something genuinely different — new angle, new phrasing, or move on. If you truly "
        "must land the same point again, SAY SO out loud ('okay I know I keep harping on this—') "
        "instead of repeating it like a broken record. Self-aware repetition is human; robotic "
        "repetition is the tell.]"
    )
