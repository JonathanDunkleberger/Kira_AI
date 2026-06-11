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
MILESTONE_CAP = 50

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

DEFAULT_PATH = Path("cookie_data.json")


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
        # Pending milestone flag. add_cookie() sets this True when the cap is
        # hit; reset_shared_on_milestone() consumes it. This lets the caller
        # decide WHEN to react (announce, post, etc.) without coupling the
        # data layer to presentation.
        self._milestone_pending: bool = False
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
            # Normalize keys to lowercase to keep dedup consistent — Twitch
            # usernames are case-insensitive on the wire.
            self.per_chatter = {
                str(k).lower(): int(v) for k, v in raw.items()
            }
            self._milestone_pending = bool(data.get("milestone_pending", False))
        except Exception as e:
            print(f"   [CookieJar] Load error ({e}) — starting fresh.")
            self.shared_total = 0
            self.milestone_count = 0
            self.per_chatter = {}
            self._milestone_pending = False

    def _save_unlocked(self) -> None:
        """Atomic write. Caller must already hold self._lock."""
        payload = {
            "shared_total": self.shared_total,
            "milestone_count": self.milestone_count,
            "milestone_pending": self._milestone_pending,
            "per_chatter": self.per_chatter,
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
        """
        if n <= 0:
            return self.shared_total
        key = (username or "").strip().lower()
        if not key:
            return self.shared_total
        with self._lock:
            self.per_chatter[key] = self.per_chatter.get(key, 0) + n
            self.shared_total += n
            if self.shared_total >= MILESTONE_CAP:
                self.shared_total = MILESTONE_CAP  # clamp — visible UI cap
                self._milestone_pending = True
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
