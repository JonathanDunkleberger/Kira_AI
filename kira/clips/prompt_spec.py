"""Shared clip-candidate prompt spec (Phase K debt D4).

The lore+clips artifact prompt lived in TWO places (bot.py session-end artifacts
and scripts/backfill_clips.py) as drifting copies. The format block below is a
CONTRACT: ``clip_cutter.parse_candidates`` parses exactly this candidate shape
(### Clip N — / **Timestamp:** / **Score:** / **Key exchange:**), so a change
here must keep that parser in sync — that is WHY this is one module.

Callers append their own source sections (transcript, live highlights, called
shots) between the head and the tail.
"""

CANDIDATE_PROMPT_TAIL = (
    "Begin output. Lore first, then `===CLIPS===` on its own line, then clip candidates."
)


def candidate_prompt_head(activity: str, date_str: str, duration_min,
                          clip_count: str = "8-12", wide_net: bool = False) -> str:
    """The shared head: session framing + the two-output contract + the
    candidate FORMAT spec. ``clip_count`` is a display string ("8-12" or "12");
    ``wide_net`` adds the backfill's include-all-strong-moments directive."""
    net = ("Cast a wide net — include all strong moments, not just the "
           "absolute peaks. " if wide_net else "")
    return (
        f"You are reviewing a full stream session transcript for the AI VTuber Kira. "
        f"Activity: {activity}. Duration: ~{duration_min} minutes. Date: {date_str}.\n\n"
        "You will produce TWO outputs, separated by the exact delimiter line `===CLIPS===`.\n\n"
        "OUTPUT 1 — LORE NOTES (markdown). Identify 3-7 durable canon points established or developed "
        "this session for this activity. Format as bullet points.\n\n"
        f"OUTPUT 2 — CLIP CANDIDATES (markdown). Identify {clip_count} of the funniest, sharpest, or most "
        f"emotionally landing moments. {net}For each one provide:\n"
        "  ### Clip N — Short title\n"
        "  **Timestamp:** approximate HH:MM:SS into stream\n"
        "  **Score:** X/10 (clip-worthiness: self-contained without context, has a punchline/payoff, quotable title potential, energy)\n"
        "  **Why it's good:** 1-2 sentences\n"
        "  **Suggested YouTube short title:** under 60 chars\n"
        "  **Key exchange:** 2-4 quoted lines\n\n"
        "Sort candidates best-first (highest score first).\n\n"
    )
