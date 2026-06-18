# wheel_slices.py — The Wheel of Fortune slice definitions.
#
# When the cookie jar fills, the wheel is spun and a slice is chosen
# via weighted random selection. All slices are editable — adjust
# weights and directives here without touching bot.py.
#
# WEIGHTS: higher = more likely. Common ≈ 4×, Uncommon ≈ 2×, Rare ≈ 1×.

from __future__ import annotations

import random
from typing import Optional

# ── The redesigned 8-slot wheel ─────────────────────────────────────────
# Slot types:
#   timed_mode  — activates a TimedModifier (registry) that colours every reply
#                 for a window; chat votes the parameter (ChatVote). Handled by
#                 a dedicated slice branch in bot.py.
#   segment     — a one-off bit performed immediately (perform-by-default path).
#                 "nominate" segments solicit a target/line from live chat.
# The `type`/`vote` fields are metadata for clarity; dispatch keys on the id.
SLICES: list[dict] = [
    # ── TIMED MODES (chat votes the parameter) ──────────────────────────
    {
        "id":      "accent_mode",
        "label":   "Accent Mode",
        "tier":    "common",
        "weight":  20,
        "type":    "timed_mode",
        "vote":    "param",        # chat votes which accent (ACCENT_MODE_OPTIONS)
        "directive": "",            # handled specially in bot.py (opens a vote)
        "min_words": 20,
    },
    {
        "id":      "speech_constraint",
        "label":   "Speech Constraint",
        "tier":    "common",
        "weight":  18,
        "type":    "timed_mode",
        "vote":    "param",        # chat votes which constraint (SPEECH_CONSTRAINT_OPTIONS)
        "directive": "",            # handled specially in bot.py (opens a vote)
        "min_words": 20,
    },

    # ── NOMINATE SEGMENTS (chat names a target from live chat) ───────────
    {
        "id":      "targeted_roast",
        "label":   "Targeted Roast",
        "tier":    "common",
        "weight":  20,
        "type":    "segment",
        "vote":    "nominate",
        "directive": (
            "[WHEEL RESULT — TARGETED ROAST]\n"
            "Chat spun the wheel and gets to feed ONE person to you. Ask chat to nominate\n"
            "a single target by typing a name — a chatter, a regular, even Jonny. Take the\n"
            "first clear nomination (or the name said most). Then deliver one savage-but-\n"
            "affectionate roast of that person: specific, 2-4 sentences, drawing on what you\n"
            "actually know or remember about them. The bit is love with teeth — never cruel,\n"
            "never punching down. If nobody nominates in a few seconds, pick the bravest\n"
            "lurker yourself and roast the very concept of their username."
        ),
        "min_words": 40,
    },
    {
        "id":      "hype_man",
        "label":   "Hype-Man",
        "tier":    "uncommon",
        "weight":  14,
        "type":    "segment",
        "vote":    "nominate",
        "directive": (
            "[WHEEL RESULT — HYPE-MAN]\n"
            "The opposite of a roast. Chat nominates ONE person and you become their\n"
            "absolute biggest fan. Ask chat to name someone who deserves it. Take the first\n"
            "clear nomination. Then hype them up relentlessly for 20-30 seconds — specific,\n"
            "genuine, a little unhinged in your enthusiasm. Make them feel like the most\n"
            "important person in the room. If nobody nominates, hype the quietest lurker."
        ),
        "min_words": 40,
    },

    # ── SEGMENTS (chat supplies a line / a choice) ───────────────────────
    {
        "id":      "chat_writes_her_line",
        "label":   "Chat Writes My Line",
        "tier":    "uncommon",
        "weight":  14,
        "type":    "segment",
        "directive": (
            "[WHEEL RESULT — CHAT WRITES MY LINE]\n"
            "Chat gets to put words in your mouth. Ask chat to type a line for you to say —\n"
            "anything within reason. Take the first usable line someone offers. Then say it\n"
            "in full, out loud, completely committed, as if it were your own idea — and react\n"
            "to it honestly afterward. If a line crosses a hard line (slurs, real harm), skip\n"
            "it and take the next one, lightly. If nothing usable arrives, riff on the\n"
            "funniest attempt instead."
        ),
        "min_words": 30,
    },
    {
        "id":      "truth_or_dare",
        "label":   "Truth or Dare",
        "tier":    "uncommon",
        "weight":  14,
        "type":    "segment",
        "directive": (
            "[WHEEL RESULT — TRUTH OR DARE]\n"
            "Chat plays Truth or Dare with you. Ask chat to pick: truth or dare. Go with the\n"
            "majority shout (or the funnier one if it's close). TRUTH: answer a question from\n"
            "chat with complete honesty — no deflection, no 'as an AI'. DARE: chat gives you a\n"
            "stream-safe dare and you commit to it fully on the spot. Either way: no chickening\n"
            "out, big reactions, make it a moment."
        ),
        "min_words": 40,
    },

    # ── ONE-SHOT (rare; canonized) ──────────────────────────────────────
    {
        "id":      "lore_drop",
        "label":   "Lore Drop",
        "tier":    "rare",
        "weight":  8,
        "type":    "segment",
        "directive": (
            "[WHEEL RESULT — LORE DROP]\n"
            "Chat just unlocked a piece of classified backstory about you. This is a REAL\n"
            "reveal — invent a true piece of your lore on the spot. It could be about the\n"
            "Figgis Agency, how you came to exist, a prior version of yourself, a mission\n"
            "that went wrong, a person you remember but shouldn't, an object in your room\n"
            "with a history, something you are not supposed to say out loud but are saying\n"
            "out loud right now.\n"
            "Rules: commit fully. It is now canon. It will be written down and it will persist.\n"
            "It must be specific, strange, and feel true. No vague gestures at mystery — give\n"
            "an actual fact, name, date, or incident. 1-2 paragraphs maximum."
        ),
        "min_words": 80,
        "canonize": True,   # bot.py will append to lore file
    },

    # ── TIMED MODE (no vote; the flagship cookie-jar payoff) ─────────────
    {
        "id":      "chaos_mode",
        "label":   "CHAOS MODE",
        "tier":    "common",
        "weight":  20,
        "type":    "timed_mode",
        "vote":    "none",         # no parameter — chaos is a fixed directive
        "directive": "",            # handled specially in bot.py (activates chaos)
        "min_words": 20,
    },
]

# Index by id for fast lookup
_SLICE_BY_ID: dict[str, dict] = {s["id"]: s for s in SLICES}


def get_slice(slice_id: str) -> Optional[dict]:
    """Return slice definition by id, or None."""
    return _SLICE_BY_ID.get(slice_id)


def spin() -> dict:
    """Weighted random selection. Returns the chosen slice dict."""
    total   = sum(s["weight"] for s in SLICES)
    r       = random.uniform(0, total)
    cumulative = 0
    for s in SLICES:
        cumulative += s["weight"]
        if r <= cumulative:
            return s
    return SLICES[-1]  # fallback
