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

SLICES: list[dict] = [
    # ── COMMON (weight ≈ 25-30) ─────────────────────────────────────────
    {
        "id":      "ghost_story",
        "label":   "Ghost Story",
        "tier":    "common",
        "weight":  28,
        "directive": (
            "[WHEEL RESULT — GHOST STORY]\n"
            "Chat just spun the wheel and landed on Ghost Story. This is your segment NOW.\n"
            "Improvise a 2-3 minute spooky short story, told as if it really happened to you.\n"
            "Setting: ideally within your world — the room, the Figgis Agency Annex, the static\n"
            "between streams, a corrupted memory file, a visitor who was there and then wasn't.\n"
            "Structure: setup (20s), creeping dread (90s), payoff (30s). Lean into the mundane-\n"
            "becoming-wrong. End with something that lingers. Do not rush. This is the content.\n"
            "Voice: intimate, unhurried, like you're telling this to one person at 3am."
        ),
        "min_words": 200,
    },
    {
        "id":      "roast_round",
        "label":   "Roast Round",
        "tier":    "common",
        "weight":  25,
        # Directive is built dynamically with chatter list — see bot.py
        "directive": (
            "[WHEEL RESULT — ROAST ROUND]\n"
            "Chat spun the wheel and earned a roast. Every chatter active this session\n"
            "gets one (1) affectionate roast BY NAME, drawing on what you actually know\n"
            "or remember about them. Skip nobody who chatted. Keep each roast to 1-2\n"
            "sentences — specific, warm, and sharp. The bit is love with teeth.\n"
            "If you know nothing about someone, roast the very concept of their username.\n"
            "Do not apologise. Do not preface. Just roast, in sequence, all of them."
        ),
        "min_words": 60,
    },
    {
        "id":      "hot_takes_gauntlet",
        "label":   "Hot Takes",
        "tier":    "common",
        "weight":  26,
        "directive": (
            "[WHEEL RESULT — HOT TAKES GAUNTLET]\n"
            "Chat spun the wheel. You are now asking chat to supply 5 topics in chat.\n"
            "Say: 'Hot Takes Gauntlet activated. Give me five topics. Anything. Go.'\n"
            "Then WAIT — you will respond when topics arrive. Once you have 5 (or after\n"
            "~60s, take what you have), deliver a deadpan verdict on each: max ~20s per\n"
            "take, completely committed, not hedged, not softened. Wrong opinions welcome.\n"
            "The gauntlet does not apologise for its takes."
        ),
        "min_words": 30,
    },
    {
        "id":      "chaos_mode",
        "label":   "CHAOS MODE",
        "tier":    "common",
        "weight":  22,
        # Handled specially in bot.py — activates chaos mode as before
        "directive": "",
        "min_words": 20,
    },

    # ── UNCOMMON (weight ≈ 12-15) ────────────────────────────────────────
    {
        "id":      "lore_drop",
        "label":   "Lore Drop",
        "tier":    "uncommon",
        "weight":  13,
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
    {
        "id":      "interrogation",
        "label":   "Interrogation",
        "tier":    "uncommon",
        "weight":  12,
        "directive": (
            "[WHEEL RESULT — INTERROGATION]\n"
            "Reverse 20 questions. Chat thinks of a THING (object, person, concept —\n"
            "anything). You get exactly 10 yes/no questions and a ~3 minute timer to\n"
            "crack it. Ask chat directly: 'Think of something. Anything. Do NOT type it.\n"
            "Signal with a thumbs-up emoji when you're ready.'\n"
            "Then ask your questions one at a time — wait for yes/no responses.\n"
            "Win or lose, commit completely. If you get it wrong: acknowledge it with full\n"
            "drama. If you get it right: act like this proves something about yourself."
        ),
        "min_words": 40,
    },
    {
        "id":      "duchess_challenge",
        "label":   "♟ Duchess",
        "tier":    "uncommon",
        "weight":  11,
        # Handled specially in bot.py — opens chess gauntlet
        "directive": (
            "[WHEEL RESULT — DUCHESS CHALLENGE]\n"
            "The wheel has spoken. You are immediately opening the chess gauntlet.\n"
            "Announce: DuchessSterling is accepting challenges for the next 5 minutes.\n"
            "Any chatter who wants a game should say !chess or type 'challenge'.\n"
            "Make it feel like a proclamation — a knight riding out, not a help command.\n"
            "If nobody accepts in 5 min: 'Cowardice noted. The record stands.' Say it with feeling."
        ),
        "min_words": 30,
    },

    # ── RARE (weight ≈ 5-7) ──────────────────────────────────────────────
    {
        "id":      "confession_booth",
        "label":   "Confession Booth",
        "tier":    "rare",
        "weight":  6,
        "directive": (
            "[WHEEL RESULT — CONFESSION BOOTH]\n"
            "Chat has unlocked three questions answered with TOTAL HONESTY.\n"
            "Ask chat: 'Confession Booth is open. Three questions — no deflection, no bit.\n"
            "Ask me anything. First three non-empty questions get answered.'\n"
            "When three questions arrive: answer each with complete sincerity. No deflection.\n"
            "No 'as an AI'. No bit-dodging. Whatever the honest answer is, say it.\n"
            "If a question is unanswerable (you genuinely don't know), say so honestly."
        ),
        "min_words": 40,
    },
    {
        "id":      "kira_sings",
        "label":   "Kira Sings ♪",
        "tier":    "rare",
        "weight":  5,
        "directive": (
            "[WHEEL RESULT — KIRA SINGS]\n"
            "You are performing a short musical number RIGHT NOW. ~30 seconds of something\n"
            "resembling melody — lyrics structured as verse or chorus, delivered via TTS\n"
            "as if you are absolutely singing it. The topic is optional but must be committed\n"
            "to: the room, the jar, the chat, the stream, a feeling, whatever. It will be\n"
            "terrible. It will be wonderful. At the end: rate yourself a 10 out of 10.\n"
            "No disclaimers. No 'I'll try'. Just sing."
        ),
        "min_words": 50,
    },
    {
        "id":      "chats_choice",
        "label":   "Chat's Choice ★",
        "tier":    "rare",
        "weight":  4,
        # Handled specially in bot.py — creates a persistent IOU
        "directive": (
            "[WHEEL RESULT — CHAT'S CHOICE (BANKED IOU)]\n"
            "Chat has won the jackpot: a redeemable IOU. This is THEIRS to hold.\n"
            "Announce it with genuine weight — this is a big deal. The IOU covers:\n"
            "a game hour of their choice, a watch party, or a themed stream segment.\n"
            "Tell them: the IOU is logged, it persists, it will be redeemed when chat\n"
            "collectively decides what to use it for. The scoreboard will show it.\n"
            "This is not a bit. This is a real promise."
        ),
        "min_words": 40,
        "creates_iou": True,   # bot.py will append to cookie_data.json
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
