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


# ── F-9 VERBAL-TIC GOVERNOR (2026-07-08, the "doing a lot of heavy lifting" class) ──────────
# avoidance_block guards LINE-level near-duplicates over a 5-line window; a verbal TIC is a
# PHRASE recurring across many otherwise-different lines over a longer horizon — invisible to
# the line guard. These pure functions find distinctive 3-5-word phrases that keep coming back
# and produce a HARD ban directive. Deliberately no ban-evasion invitation: the line guard's
# "say so out loud" escape is exactly what she exploits playfully ("I almost said—"), which is
# charming once and wallpaper by the third time. Same discipline as the rest of this module:
# lexical, sub-millisecond, no model call. Callers own the window (e.g. the Pokémon seam feeds
# its own long deque); nothing here changes existing consumers.
TIC_MIN_LINES = int(os.getenv("KIRA_TIC_MIN_LINES", "3"))   # lines a phrase must recur in
TIC_MAX_BANS = int(os.getenv("KIRA_TIC_MAX_BANS", "4"))     # cap the ban list (never a lecture)

_TIC_STOP = frozenset(
    "the a an and or but so of to in on at for with is are was were be been being am i you "
    "it its it's im i'm this that these those we they he she my your our their me him her "
    "just like really very going gonna got get have has had do does did not no yes okay oh "
    "well then than as if what when where who how why there here now still too also can "
    "could would should will won't don't didn't that's let's about out up down one two "
    "little bit lot".split()
)


def overused_phrases(lines, min_lines: int = TIC_MIN_LINES, max_phrases: int = TIC_MAX_BANS):
    """Distinctive 3-5-word phrases that appear in >= min_lines DISTINCT lines of the window.
    Distinctive = at least 2 non-stopword tokens (so 'and then i just' never flags). Counted
    once per line (a single rambly line can't self-flag). Longest variant of a tic wins;
    contained sub/super-phrases are deduped. Returns [] when she's varying naturally."""
    from collections import Counter
    seen_in_lines = Counter()
    for line in lines or ():
        toks = _tokens(line)
        grams = set()
        for n in (3, 4, 5):
            for i in range(len(toks) - n + 1):
                g = toks[i:i + n]
                if sum(1 for w in g if w not in _TIC_STOP) >= 2:
                    grams.add(" ".join(g))
        for g in grams:
            seen_in_lines[g] += 1
    hits = [(g, c) for g, c in seen_in_lines.items() if c >= min_lines]
    hits.sort(key=lambda gc: (-gc[1], -len(gc[0])))          # most-recurrent, then longest form
    out = []
    for g, _c in hits:
        if any(_gram_overlap(g, o) for o in out):            # one entry per tic, not per window slice
            continue
        out.append(g)
        if len(out) >= max_phrases:
            break
    return out


def _gram_overlap(a: str, b: str, k: int = 3) -> bool:
    """Two flagged grams are the SAME tic when one contains the other or they share a >=k-token
    seam (the n-gram window slices one long tic into overlapping fragments)."""
    if a in b or b in a:
        return True
    ta, tb = a.split(), b.split()
    for n in range(min(len(ta), len(tb)), k - 1, -1):
        if ta[-n:] == tb[:n] or tb[-n:] == ta[:n]:
            return True
    return False


def tic_ban_block(lines, min_lines: int = TIC_MIN_LINES) -> str:
    """The HARD-BAN directive for detected tics — no self-aware escape hatch (that's the
    ban-evasion she plays). '' when no phrase is overused."""
    phrases = overused_phrases(lines, min_lines=min_lines)
    if not phrases:
        return ""
    body = "; ".join(f'"{p}"' for p in phrases)
    return (
        "[RETIRED PHRASES — you've leaned on these too many times recently: " + body + ". "
        "Do not use them or close variants, and do NOT joke about avoiding them ('I almost "
        "said—', 'you know what I'm not allowed to say') — no meta, no winking. Just express "
        "the thought in genuinely fresh words.]"
    )


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
