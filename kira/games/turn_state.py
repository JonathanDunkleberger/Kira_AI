# turn_state.py — Reusable structured-state scaffold for turn-based games.
# ─────────────────────────────────────────────────────────────────────────────
# THE PROBLEM this exists to solve:
#   Kira used to read a turn-based game (Codenames, Wordle, chess on a screen)
#   by glancing at ONE vision frame each turn and reasoning from scratch. That
#   is fragile — a single stale or blurry frame and her whole turn is wrong.
#
# THE PATTERN (mirrors ChessAgent's board model):
#   Maintain a PERSISTENT in-memory model of the game that she UPDATES as moves
#   happen, then reasons over the tracked model — not a one-shot glance. Vision
#   becomes a way to *update* the model, not the sole source of truth each turn.
#
# This base class captures what every turn-based game shares so chess / Codenames
# / future games all wear the same shape:
#   - a live/idle lifecycle (active flag + reset),
#   - turn tracking (turn number + whose move),
#   - an append-only event history,
#   - a single prompt surface: get_state_block() + has_context().
#
# Subclasses implement the game-specific model and override _state_lines() (or
# get_state_block() directly) to render it as plain-language context for the
# prompt. No engine-speak, no raw internals — the block is what Kira "sees."

from __future__ import annotations

import time
from typing import Any


class TurnBasedGameState:
    """Persistent, in-memory model for one turn-based game.

    Lifecycle:
        start(...)  — subclass sets up the model and calls super().start().
        reset()     — back to idle; the state block is no longer injected.

    Prompt surface:
        has_context()      — True only while a game is live AND initialised.
        get_state_block()  — plain-language block injected into the prompt.

    Subclasses MUST override _state_lines() (preferred) or get_state_block().
    """

    #: Human-readable game name, e.g. "Codenames". Override in subclasses.
    game_name: str = "game"

    def __init__(self) -> None:
        self.active: bool = False
        self.turn_number: int = 0
        # Whose move it is, free-form: "kira" | "opponent" | "" (unknown).
        self.to_move: str = ""
        self.started_at: float = 0.0
        # Append-only log of {ts, kind, **data} dicts — a turn-by-turn trail.
        self.history: list[dict[str, Any]] = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Mark the game live. Subclasses set up their model THEN call this."""
        self.active = True
        self.turn_number = 0
        self.started_at = time.time()
        self.history = []

    def reset(self) -> None:
        """Return to idle. The state block stops being injected immediately."""
        self.active = False
        self.turn_number = 0
        self.to_move = ""
        self.started_at = 0.0
        self.history = []

    # ── Turn / history bookkeeping ───────────────────────────────────────────

    def advance_turn(self, to_move: str = "") -> None:
        """Tick the turn counter and (optionally) set whose move it now is."""
        self.turn_number += 1
        if to_move:
            self.to_move = to_move

    def log_event(self, kind: str, **data: Any) -> dict[str, Any]:
        """Append a timestamped event to the history trail and return it."""
        evt = {"ts": time.time(), "turn": self.turn_number, "kind": kind, **data}
        self.history.append(evt)
        return evt

    # ── Prompt surface ───────────────────────────────────────────────────────

    def has_context(self) -> bool:
        """True only while a game is live. Callers gate injection on this so the
        block is NEVER injected between games (mirrors ChessAgent.has_context)."""
        return bool(self.active)

    def _state_lines(self) -> list[str]:
        """Override: return the body lines of the state block (no header)."""
        raise NotImplementedError

    def get_state_block(self) -> str:
        """Plain-language model rendered for prompt injection. Empty when idle."""
        if not self.has_context():
            return ""
        lines = self._state_lines()
        if not lines:
            return ""
        return "\n".join(lines)
