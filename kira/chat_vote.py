# chat_vote.py — generalized chat parameter-vote.
#
# A ChatVote is a time-boxed poll where chat picks ONE option from a fixed list
# by typing its number ("1".."N") or a distinctive keyword. It is the reusable
# vote contract for the Wheel: Layer 3 wires it to Speech Constraint (chat votes
# which constraint runs); later layers reuse the SAME engine for other voted
# slots. It is a PURE state container — the bot owns the asyncio timer, the
# spoken announce/result, and the overlay broadcast. This holds only the rules:
#   - tally(): map a chat message to an option, one vote per user (last wins).
#   - resolve(): ALWAYS returns a winning index so the show never stalls
#     (tie OR no-vote -> random among the max-count options).
#
# Mirrors timed_modifier.py: pure, no asyncio, no I/O, unit-testable in isolation.

import re
import random
import time
from typing import Optional


class ChatVote:
    """One active parameter-vote.

    options: list of dicts, each ``{"label": str, "keywords": [str, ...]}``.
    `label` is what's spoken/displayed; `keywords` are extra lowercased words a
    message can contain to count as that option (the number is always accepted).
    """

    def __init__(self, prompt: str, options: list[dict], duration_s: float,
                 allow_keywords: bool = True):
        if not options:
            raise ValueError("ChatVote needs at least one option")
        self.prompt = prompt
        self.options = options
        self.duration_s = float(duration_s)
        self.allow_keywords = bool(allow_keywords)
        self.ends_at = time.time() + float(duration_s)
        self.votes: dict[str, int] = {}   # username -> option index (last wins)

    # ── mapping a message to an option ───────────────────────────────────
    def _map(self, message: str) -> Optional[int]:
        """Return the option index this message votes for, or None if it isn't a
        clean vote. Number-primary; keyword-fallback; ambiguous -> None (ignored)."""
        msg = (message or "").strip().lower()
        if not msg:
            return None
        n = len(self.options)
        # Number-primary: standalone digits 1..n (matches "1", "vote 2", "#3", "4!").
        nums = {int(t) for t in re.findall(r"\b([1-9])\b", msg) if 1 <= int(t) <= n}
        if len(nums) == 1:
            return next(iter(nums)) - 1
        if len(nums) > 1:
            return None  # typed multiple numbers -> ambiguous, don't count
        # Keyword-fallback: count only if EXACTLY one option's keyword matches.
        if self.allow_keywords:
            hits = set()
            for i, opt in enumerate(self.options):
                for kw in opt.get("keywords", []) or []:
                    if re.search(r"\b" + re.escape(str(kw).lower()) + r"\b", msg):
                        hits.add(i)
                        break
            if len(hits) == 1:
                return next(iter(hits))
        return None

    def tally(self, username: str, message: str) -> bool:
        """Record (or move) `username`'s vote. Returns True iff a vote was counted
        or changed (so the caller can push a fresh overlay tally)."""
        idx = self._map(message)
        if idx is None:
            return False
        if self.votes.get(username) == idx:
            return False  # same vote again — no change, no overlay churn
        self.votes[username] = idx  # last wins (one vote per user)
        return True

    # ── tallies / resolution ─────────────────────────────────────────────
    def counts(self) -> list[int]:
        c = [0] * len(self.options)
        for idx in self.votes.values():
            c[idx] += 1
        return c

    def total_votes(self) -> int:
        return len(self.votes)

    def leaders(self) -> list[int]:
        """Indices tied for the highest count. When nobody voted, every option is
        a leader at 0 — which is exactly why no-vote resolves like a full tie."""
        c = self.counts()
        mx = max(c)
        return [i for i, v in enumerate(c) if v == mx]

    def resolve(self) -> int:
        """ALWAYS return a winning index. Tie OR no-vote -> random among the
        max-count options, so a stalled/empty chat still produces a result."""
        return random.choice(self.leaders())

    def remaining_s(self) -> int:
        return max(0, int(round(self.ends_at - time.time())))
