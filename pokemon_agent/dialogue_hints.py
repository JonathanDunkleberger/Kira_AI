"""dialogue_hints.py — NPC/sign INFO-EXTRACTION into her decision ctx (soul-debt #12, the missing half).

She already READS overworld boxes (DialogueReader voices them in her words). This module makes her
USE them: a lightweight extractor keeps the lines that carry ACTIONABLE information — directions,
places, items/TMs/HMs, quest-ish imperatives — in a small persistent ledger, and the campaign folds
the freshest few into her oracle decision ctx ("WHAT YOU'VE HEARD"), so an NPC saying "the CAPTAIN
is upstairs" can actually steer a choice instead of scrolling past her.

ENGINE vs GAME-KNOWLEDGE (rule 14): the ledger + the fold are game-agnostic engine; the vocabulary
below is the (small) FireRed-flavored layer, kept in ONE obvious place to grow or swap per game.

PERSISTENCE: dialogue_hints.json next to the campaign save (rides the campaign dir, so the watch
sandbox override POKEMON_CAMPAIGN_DIR isolates it automatically). It is an OPTIONAL sanctity
sidecar — an old bank without it simply starts empty; never a validation failure.
"""
import json
import os
import re
import time

# ── the FireRed-flavored hint vocabulary (game-knowledge layer — swap per game) ──────────────
# A line is a HINT if it carries direction, a destination, or a quest-ish imperative. Pure flavor
# ("I love shorts!") stays out; the cap + newest-first brief keep residual noise bounded anyway.
DIRECTION_WORDS = (
    "north", "south", "east", "west", "ahead", "upstairs", "downstairs", "up there",
    "down there", "past ", "beyond", "through", "behind", "next to", "b1f", "b2f", "1f", "2f",
)
QUEST_MARKERS = (
    "go to", "head to", "take this", "bring ", "find ", "you need", "you'll need", "you will need",
    "talk to", "look for", "show ", "give ", "deliver", "waiting for", "wants to see", "is missing",
    "come back", "return ", "you can't", "you cannot", "won't open", "is locked", "need a", "needs a",
    "only if", "once you", "first you", "hidden", "secret",
    # "where somebody/something is" statements are intel too ("the CAPTAIN is up on the deck")
    " is up ", " is down ", " is in ", " is at ", " is on ", "you'll find", "you will find", "lives in",
)

_NORM = re.compile(r"[^a-z0-9]+")


def _norm(s):
    return _NORM.sub(" ", (s or "").lower()).strip()


def is_hint(line, tags=None):
    """True if this overworld line looks ACTIONABLE (worth remembering as run intel).
    `tags` are dialogue_reader salience tags (PLACE/ITEM/GYM_LEADER/TYPE) when available."""
    low = (line or "").lower()
    if len(low) < 16:                        # too short to carry real information
        return False
    tags = tags or []
    if any(w in low for w in DIRECTION_WORDS):
        return True
    if any(w in low for w in QUEST_MARKERS):
        return True
    # a PLACE or ITEM mention is intel by itself ("the MUSEUM in PEWTER…", "take this TM")
    return "PLACE" in tags or "ITEM" in tags


class HintLedger:
    """Small persistent ledger of actionable overheard lines. Dedup on normalized text (a re-read
    sign bumps recency, never duplicates), capped newest-first, atomic JSON persistence. All I/O
    best-effort + loud — intel must never crash a run."""
    CAP = 14

    def __init__(self, path, log=print):
        self.path = path
        self.log = log
        self.items = []                      # [{text, where, ts, seen}] oldest-first
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self.items = list(data.get("hints", []))[-self.CAP:]
                if self.items:
                    self.log(f"   [hints] ledger loaded: {len(self.items)} overheard hint(s)")
        except Exception as e:
            self.log(f"   [hints] !! ledger load failed (starting empty, LOUD): {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"hints": self.items}, f)
            os.replace(tmp, self.path)       # atomic — never a half-written ledger
        except Exception as e:
            self.log(f"   [hints] !! ledger save failed (LOUD): {e}")

    def add(self, line, tags=None, where=None):
        """Feed every accepted overworld line here; only hints are kept. Returns True if kept."""
        line = " ".join((line or "").split())
        if not line or not is_hint(line, tags):
            return False
        n = _norm(line)
        for it in self.items:
            if _norm(it.get("text", "")) == n:            # re-heard -> bump recency, don't duplicate
                it["ts"] = time.time()
                it["seen"] = it.get("seen", 1) + 1
                self._save()
                return True
        self.items.append({"text": line[:160], "where": list(where) if where else None,
                           "ts": time.time(), "seen": 1})
        self.items = self.items[-self.CAP:]               # cap: oldest fall off
        self._save()
        self.log(f"   [hints] + heard something actionable: {line[:80]!r}"
                 + (f" (at {tuple(where)})" if where else ""))
        return True

    def ctx_brief(self, n=4):
        """The freshest n hints as ONE decision-ctx sentence, or '' if none. Framed as HER OWN
        overheard intel (a player's mental notepad), never omniscience — it's only what she read."""
        if not self.items:
            return ""
        newest = self.items[-n:][::-1]
        quoted = "; ".join(f'"{it["text"]}"' for it in newest)
        return ("THINGS YOU'VE HEARD AND READ (from people and signs — your own notes, use them "
                f"when they point somewhere): {quoted}")
