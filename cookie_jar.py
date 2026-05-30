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
MILESTONE_CAP = 100

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
