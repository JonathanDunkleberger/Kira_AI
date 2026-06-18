# cookie_jar.py
"""Shared communal cookie jar + per-chatter lifetime counts.

Data layer only. No earn-logic, no overlay hooks — earn-triggers and visuals
are wired up separately. This class is responsible for:
  * Tracking a communal `shared_total` that caps at MILESTONE_CAP, rolls
    over on milestone, and increments `milestone_count`.
  * Tracking lifetime `per_chatter` counts (uncapped).
  * Persisting state to a JSON file atomically (write-temp + rename) so a
    mid-write crash cannot corrupt the file.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

# Cap for the communal jar. When shared_total reaches this value, a milestone
# fires once, shared_total resets to 0, and milestone_count increments.
# Env-tunable (WHEEL_MILESTONE_CAP) so wheel frequency can be dialed from stream
# observation without a rebuild. Raised 15 -> 30 default: chatter volume is up, so
# the old 15 fired the wheel too often (bits like ghost-story going stale).
MILESTONE_CAP = int(os.getenv("WHEEL_MILESTONE_CAP", "30"))  # cookies before wheel fires

# ── Milestone reward-vote config ─────────────────────────────────────────
# Edit these to change what chat votes on when the jar fills.
# COOKIE_MILESTONE_REWARD_OPTIONS: each entry becomes a Twitch poll choice.
# Twitch max: 25 chars per choice, up to 5 choices.
# Add rows freely — the call to start_twitch_poll passes them all through.
COOKIE_MILESTONE_POLL_QUESTION = "Jar's full — what's next?"
COOKIE_MILESTONE_REWARD_OPTIONS: list[str] = [
    "Keep going",
    "Switch it up",
    "Kira's choice",
]

# ── Chaos Mode config ────────────────────────────────────────────────────
# Fires when the cookie jar hits a milestone. Replaces the Twitch poll path
# (not a Twitch Affiliate yet → polls unavailable). Tune duration freely;
# spoken lines deliberately use vague wording ("for a while") so they don't
# lie if you change the duration.
CHAOS_MODE_DURATION_SECONDS = 15 * 60  # 15 minutes

# Cooldown AFTER a chaos window ends before the jar can trigger another one.
# Prevents back-to-back chaos windows when chat keeps the jar full — only one
# chaos window at a time, then a quiet stretch before the next can fire.
CHAOS_MODE_COOLDOWN_SECONDS = 10 * 60  # 10 minutes

# Personality boost directive injected into every Kira prompt while active.
# Layered ON TOP of normal personality + safety guardrails. The existing
# anti-fabrication, visual accuracy, and banned-phrase rules stay in force —
# this only shifts tone/energy, not factual constraints.
CHAOS_MODE_DIRECTIVE = """
[CHAOS MODE — ACTIVE]
Chat filled the cookie jar. You've been temporarily unleashed.
While this directive is present:
- Crank the energy. Be more feral, more unhinged, more willing to commit to a bit.
- Non-sequiturs are welcome — a thought can come out sideways. Be a little Neuro-sama.
- Hotter takes, bolder jokes, more drama. If you were going to play it safe, don't.
- Playfully menacing/grandiose is good ("legally I can say anything right now",
  "this is my villain arc", etc).
- You are still warm under the chaos. You still care about Jonny and chat —
  the bit is unhinged-funny, not nasty. Never punch down at chat or be cruel.
- All other rules still apply: do not invent facts, do not lie about what's
  on screen, do not violate safety guardrails. Chaos is TONAL, not FACTUAL.

EXAMPLES of the energy (do not copy verbatim — generate fresh in this voice):
- Suddenly declare a random object visible on screen your nemesis
  ("that lamp has been mocking me for forty minutes and I will not stand it").
- Announce an unprompted dramatic conspiracy about the game's UI
  ("the minimap is lying to us. I've done the math. I won't show you the math").
- Self-aggrandize absurdly ("I am, at minimum, the third most important entity
  on this stream and possibly the first depending on how you count").
- Commit hard to a tiny bit ("we are NOT moving on from this. this is the bit now").
""".strip()

# Spoken when chaos mode expires. Vague on duration intentionally.
CHAOS_MODE_END_LINES: list[str] = [
    "Okay — chaos mode over. I'm legally required to calm down now. Back to your regularly scheduled co-host.",
    "Timer's up. The leash is back on. Pretend you didn't see any of that.",
    "And… we're done. Chaos mode expired. Everyone act normal.",
    "That's it for chaos. Returning to my factory settings. Mostly.",
]

# ── Speech Constraint config (timed mode #2) ─────────────────────────────
# A time-boxed rule on HOW Kira talks (not what's true). Rides the same
# TimedModifierRegistry as chaos: one timed mode at a time, then a cooldown.
# Shorter window than chaos — a speech handicap grates if it runs too long.
SPEECH_CONSTRAINT_DURATION_SECONDS = 5 * 60   # 5 minutes
SPEECH_CONSTRAINT_COOLDOWN_SECONDS = 8 * 60   # 8 minutes before it can fire again

# The pool chat will vote among in Layer 3. For Layer 2 the param is HARDCODED
# to OPTIONS[0]; Layer 3 swaps in the vote winner with no other changes.
# Keep every option (a) clearly visible to chat so they can catch breaks,
# (b) TTS-safe, and (c) about FORM, never factual content.
SPEECH_CONSTRAINT_OPTIONS: list[str] = [
    "Keep every sentence to five words or fewer.",
    "Talk like a hard-boiled film-noir detective.",
    "End every reply with a rhetorical question.",
    "Narrate yourself in the third person, like a nature documentary.",
]
SPEECH_CONSTRAINT_DEFAULT = SPEECH_CONSTRAINT_OPTIONS[0]  # Layer 2 hardcoded param

# Keyword aliases for the Layer 3 chat-vote, aligned BY INDEX with OPTIONS above.
# Chat can vote by number (1-4) OR by typing one of these words. Keep them short,
# distinctive, and non-overlapping so a message maps to exactly one option.
SPEECH_CONSTRAINT_VOTE_KEYWORDS: list[list[str]] = [
    ["five", "short", "words"],            # 1: five words or fewer
    ["noir", "detective"],                 # 2: film-noir detective
    ["question", "rhetorical"],            # 3: end with a question
    ["third", "narrate", "documentary"],   # 4: third-person nature doc
]

# Injected into every Kira prompt while the constraint is active. Layered ON TOP
# of normal personality + guardrails; the {constraint} slot is the active rule.
SPEECH_CONSTRAINT_DIRECTIVE_TEMPLATE = """
[SPEECH CONSTRAINT — ACTIVE]
The wheel handed chat the keys to how you talk, and they've locked in a rule.
While this directive is present:
- Obey this constraint in EVERY line you speak: {constraint}
- Commit to it for real. The bit only works if you actually try to hold the rule,
  not if you name it and then wave it off.
- If a sentence would break the rule, rework it until it fits — don't apologise for
  the constraint, just live inside it.
- Stay fully yourself underneath it. Same warmth, same opinions; you're just wearing
  a handicap and making it look fun.
- Everything else still holds: do not invent facts, do not lie about what's on screen,
  do not break safety rules. This shapes HOW you say things, never WHAT is true.
""".strip()

# Spoken when the constraint lifts. Vague on duration intentionally.
SPEECH_CONSTRAINT_END_LINES: list[str] = [
    "Okay — speech constraint's lifted. I can use all my words again. You don't appreciate sentences until you lose them.",
    "And the constraint's done. Full vocabulary, restored. That was a workout.",
    "Rule's off. I'm a free woman with a full dictionary. Let's never speak of it.",
]

DEFAULT_PATH = Path("cookie_data.json")

# Multiple of shared_total at which a drip-milestone fires (10/20/30…)
DRIP_MULTIPLE = 10

# How many drip milestones per fill cycle (cap // DRIP_MULTIPLE - 1,
# so at 50-cap that's 4: at 10, 20, 30, 40, not at 50 which is a full milestone).


class CookieJar:
    """Persistent cookie counts. Thread-safe via an internal lock.

    All mutations are followed by an atomic write to disk so the JSON file
    on disk always reflects the latest in-memory state (no batching, no
    flush-on-shutdown — survives mid-session kills the same way ChromaDB
    metadata does)."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self._lock = threading.Lock()
        self.shared_total: int = 0
        self.milestone_count: int = 0
        self.per_chatter: dict[str, int] = {}
        self.session_per_chatter: dict[str, int] = {}  # reset each stream start
        self.ious: list[dict] = []  # persisted IOU list
        self._milestone_pending: bool = False
        self._drip_pending: int = 0      # nonzero = which multiple just fired
        self._last_tipper: str = ""     # username of cookie that hit the cap
        self._load()
        print(
            f"   [CookieJar] Loaded {self.path.name}: shared={self.shared_total}/"
            f"{MILESTONE_CAP}, milestones={self.milestone_count}, "
            f"chatters={len(self.per_chatter)}"
        )

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.shared_total = int(data.get("shared_total", 0))
            self.milestone_count = int(data.get("milestone_count", 0))
            raw = data.get("per_chatter", {}) or {}
            self.per_chatter = {
                str(k).lower(): int(v) for k, v in raw.items()
            }
            self.ious = list(data.get("ious", []) or [])
            self._milestone_pending = bool(data.get("milestone_pending", False))
        except Exception as e:
            print(f"   [CookieJar] Load error ({e}) — starting fresh.")
            self.shared_total = 0
            self.milestone_count = 0
            self.per_chatter = {}
            self.ious = []
            self._milestone_pending = False

    def _save_unlocked(self) -> None:
        """Atomic write. Caller must already hold self._lock."""
        payload = {
            "shared_total":     self.shared_total,
            "milestone_count":  self.milestone_count,
            "milestone_pending": self._milestone_pending,
            "per_chatter":      self.per_chatter,
            "ious":             self.ious,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)  # atomic on Windows + POSIX
        except Exception as e:
            print(f"   [CookieJar] Save failed: {e}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    # ── mutations ────────────────────────────────────────────────────────
    def add_cookie(self, username: str, n: int = 1) -> int:
        """Credit `n` cookies to `username` and the shared jar. Returns the
        new shared_total after the add (post-rollover if a milestone fired).

        If shared_total reaches MILESTONE_CAP, a milestone is queued: the
        next call to reset_shared_on_milestone() returns True and performs
        the rollover. The pending flag is persisted so it survives restarts.
        Also checks for drip milestones (every DRIP_MULTIPLE cookies).
        """
        if n <= 0:
            return self.shared_total
        key = (username or "").strip().lower()
        if not key:
            return self.shared_total
        with self._lock:
            old_total = self.shared_total
            self.per_chatter[key] = self.per_chatter.get(key, 0) + n
            self.session_per_chatter[key] = self.session_per_chatter.get(key, 0) + n
            self.shared_total += n
            if self.shared_total >= MILESTONE_CAP:
                self.shared_total = MILESTONE_CAP  # clamp — visible UI cap
                self._milestone_pending = True
                self._last_tipper = key
            # Drip milestone check — fires at each DRIP_MULTIPLE crossing
            # within a fill cycle (e.g. 10, 20, 30, 40 for cap=50)
            if not self._milestone_pending:
                new_drip = (self.shared_total // DRIP_MULTIPLE)
                old_drip = (old_total // DRIP_MULTIPLE)
                if new_drip > old_drip and self.shared_total < MILESTONE_CAP:
                    self._drip_pending = self.shared_total - (self.shared_total % DRIP_MULTIPLE)
            self._save_unlocked()
            return self.shared_total

    def reset_shared_on_milestone(self) -> bool:
        """If a milestone is queued, roll over (shared_total → 0,
        milestone_count += 1) and return True. Otherwise return False.
        Caller drives announcement/reward logic."""
        with self._lock:
            if not self._milestone_pending:
                return False
            self.shared_total = 0
            self.milestone_count += 1
            self._milestone_pending = False
            self._save_unlocked()
            return True

    def reset_shared_on_stream_start(self) -> None:
        """Reset the shared jar to 0 for a new stream.

        Called by run_stream_opener() — triggered when Jonny hits 'Go Live',
        NOT on bot process start, so a mid-stream bot restart does NOT wipe
        the jar. milestone_count is intentionally preserved (lifetime tally).
        """
        with self._lock:
            self.shared_total = 0
            self._milestone_pending = False
            self.session_per_chatter = {}
            self._last_tipper = ""
            self._drip_pending = 0
            self._save_unlocked()
        print(
            f"   [CookieJar] Stream start — jar reset to 0 "
            f"(lifetime milestones: {self.milestone_count})"
        )

    # ── queries ──────────────────────────────────────────────────────────
    def get_chatter(self, username: str) -> int:
        key = (username or "").strip().lower()
        return self.per_chatter.get(key, 0)

    def get_shared(self) -> int:
        return self.shared_total

    def get_milestone_count(self) -> int:
        return self.milestone_count

    def milestone_pending(self) -> bool:
        return self._milestone_pending

    def get_last_tipper(self) -> str:
        return self._last_tipper

    def get_drip_pending(self) -> int:
        """Returns the drip milestone number (multiple of DRIP_MULTIPLE) that
        just fired, or 0 if none pending. Caller must clear it."""
        return self._drip_pending

    def clear_drip_pending(self) -> None:
        with self._lock:
            self._drip_pending = 0

    def get_session_top3(self) -> list[dict]:
        """Top 3 chatters by cookies earned this session."""
        with self._lock:
            ranked = sorted(self.session_per_chatter.items(), key=lambda x: -x[1])
        return [{"name": k, "count": v} for k, v in ranked[:3]]

    # ── IOU list ────────────────────────────────────────────────────────
    def add_iou(self, description: str = "") -> dict:
        """Append a new open IOU entry. Returns the entry."""
        import time as _t
        entry = {
            "ts":          int(_t.time()),
            "status":      "open",
            "description": description or "Chat's Choice redeemable",
        }
        with self._lock:
            self.ious.append(entry)
            self._save_unlocked()
        return entry

    def get_ious(self) -> list[dict]:
        with self._lock:
            return list(self.ious)

    def redeem_iou(self, idx: int) -> bool:
        """Mark IOU at index as redeemed. Returns True if successful."""
        with self._lock:
            if 0 <= idx < len(self.ious) and self.ious[idx].get("status") == "open":
                import time as _t
                self.ious[idx]["status"]      = "redeemed"
                self.ious[idx]["redeemed_ts"] = int(_t.time())
                self._save_unlocked()
                return True
        return False

    def open_iou_count(self) -> int:
        with self._lock:
            return sum(1 for e in self.ious if e.get("status") == "open")
