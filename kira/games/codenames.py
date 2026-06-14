# codenames.py — Structured state tracker for Codenames.
# ─────────────────────────────────────────────────────────────────────────────
# Built on TurnBasedGameState (the reusable turn-based-game scaffold). Kira keeps
# a PERSISTENT model of the 25-word grid she updates as the game unfolds, and
# reasons over THAT — not a single (often stale) vision frame each turn.
#
# What it tracks:
#   - the 25-word grid,
#   - each word's confirmed identity: her-team / opponent / neutral / assassin
#     (or still unknown),
#   - her role (guesser vs spymaster / clue-giver),
#   - the clue history (clue word + number + who gave it),
#   - the running guess state (current clue she's working, guesses left,
#     what she's already guessed this turn).
#
# How she uses it:
#   - As GUESSER: she reasons over the tracked unrevealed words, not one glance.
#   - As CLUE-GIVER: before committing a clue she checks it against the tracked
#     assassin / opponent / neutral words so she never points the other team at
#     a word that loses the game.

from __future__ import annotations

from typing import Any

from .turn_state import TurnBasedGameState


# ─── Identity tags ───────────────────────────────────────────────────────────
# TEAM     = one of Kira's own team's agents (good — she wants these).
# OPPONENT = the other team's agents (bad — guessing one helps them).
# NEUTRAL  = bystander (ends her turn, otherwise harmless).
# ASSASSIN = instant loss if guessed. The single most important word to avoid.
# UNKNOWN  = identity not yet revealed / not yet known.
TEAM = "team"
OPPONENT = "opponent"
NEUTRAL = "neutral"
ASSASSIN = "assassin"
UNKNOWN = "unknown"

_VALID_IDENTITIES = {TEAM, OPPONENT, NEUTRAL, ASSASSIN, UNKNOWN}

# Plain-language labels for the state block.
_IDENTITY_LABEL = {
    TEAM: "YOUR TEAM",
    OPPONENT: "opponent",
    NEUTRAL: "neutral",
    ASSASSIN: "ASSASSIN \u2014 instant loss",
    UNKNOWN: "unknown",
}

# Roles.
GUESSER = "guesser"
SPYMASTER = "spymaster"   # the clue-giver
_VALID_ROLES = {GUESSER, SPYMASTER}


def _norm(word: str) -> str:
    """Normalise a word for matching: stripped + uppercased (cards are caps)."""
    return (word or "").strip().upper()


class CodenamesState(TurnBasedGameState):
    """Persistent tracker for a single Codenames game.

    Words are matched case-insensitively; the display form (as first entered) is
    preserved for the state block. All mutators are forgiving — an unknown word
    is added rather than rejected, so a mid-game vision correction can introduce
    a word that was previously misread.
    """

    game_name = "Codenames"

    def __init__(self) -> None:
        super().__init__()
        # norm-word -> {"display": str, "identity": str}
        self._grid: dict[str, dict[str, str]] = {}
        # Preserves board order for rendering (norm-word list).
        self._order: list[str] = []
        self.role: str = GUESSER
        self.my_team_label: str = ""   # cosmetic, e.g. "red" / "blue"
        # Clue history: [{"clue", "number", "by", "turn"}].
        self.clues: list[dict[str, Any]] = []
        # Running guess state for the clue currently in play.
        self.current_clue: dict[str, Any] | None = None
        self.guesses_left: int = 0
        self.guessed_this_turn: list[str] = []

    # ── Setup ────────────────────────────────────────────────────────────────

    def start(self, words: list[str] | None = None, *,
              role: str = GUESSER, my_team_label: str = "") -> None:
        """Begin a game. Resets any prior state, seeds the grid, sets the role."""
        self.reset()
        super().start()
        self.set_role(role)
        self.my_team_label = (my_team_label or "").strip()
        if words:
            self.set_grid(words)

    def reset(self) -> None:
        super().reset()
        self._grid = {}
        self._order = []
        self.role = GUESSER
        self.my_team_label = ""
        self.clues = []
        self.current_clue = None
        self.guesses_left = 0
        self.guessed_this_turn = []

    def set_role(self, role: str) -> str:
        """Set Kira's role: 'guesser' or 'spymaster' (clue-giver)."""
        r = (role or "").strip().lower()
        if r in ("clue", "cluegiver", "clue-giver", "spy", "spymaster"):
            r = SPYMASTER
        elif r in ("guess", "guesser", "operative"):
            r = GUESSER
        self.role = r if r in _VALID_ROLES else GUESSER
        return self.role

    def set_grid(self, words: list[str]) -> None:
        """Replace the grid with `words` (order preserved). Identities reset to
        UNKNOWN unless a word was already known, in which case it's kept."""
        prev = {k: v["identity"] for k, v in self._grid.items()}
        self._grid = {}
        self._order = []
        for w in words:
            nw = _norm(w)
            if not nw or nw in self._grid:
                continue
            self._grid[nw] = {
                "display": (w or "").strip(),
                "identity": prev.get(nw, UNKNOWN),
            }
            self._order.append(nw)

    # ── Mutators ─────────────────────────────────────────────────────────────

    def reveal(self, word: str, identity: str) -> bool:
        """Mark `word`'s confirmed identity. Adds the word if it isn't on the
        tracked grid yet (forgiving — supports a mid-game vision correction).
        Returns True if the word now exists on the grid."""
        ident = (identity or "").strip().lower()
        if ident not in _VALID_IDENTITIES:
            return False
        nw = _norm(word)
        if not nw:
            return False
        if nw not in self._grid:
            self._grid[nw] = {"display": (word or "").strip(), "identity": ident}
            self._order.append(nw)
        else:
            self._grid[nw]["identity"] = ident
        self.log_event("reveal", word=self._grid[nw]["display"], identity=ident)
        return True

    def record_clue(self, clue: str, number: int, by: str = "me") -> dict[str, Any]:
        """Log a clue and (for guessing) open the running guess state. `number`
        is how many words the clue points to; guesses_left = number + 1 per the
        standard rule (you may always make one extra guess)."""
        n = max(0, int(number))
        entry = {"clue": (clue or "").strip(), "number": n,
                 "by": (by or "me").strip(), "turn": self.turn_number}
        self.clues.append(entry)
        self.current_clue = entry
        # The +1 bonus guess only applies when the clue is for Kira's side.
        self.guesses_left = n + 1 if entry["by"] != "opponent" else 0
        self.guessed_this_turn = []
        self.log_event("clue", clue=entry["clue"], number=n, by=entry["by"])
        return entry

    def record_guess(self, word: str, identity: str | None = None) -> dict[str, Any]:
        """Record a guess against the current clue. If `identity` is given the
        word is also revealed. Decrements guesses_left; a wrong (non-team) guess
        ends the turn (guesses_left -> 0)."""
        nw = _norm(word)
        display = self._grid.get(nw, {}).get("display", (word or "").strip())
        if identity:
            self.reveal(word, identity)
        self.guessed_this_turn.append(display)
        result = self._grid.get(nw, {}).get("identity", UNKNOWN)
        if self.guesses_left > 0:
            self.guesses_left -= 1
        # Any non-team reveal ends the turn immediately.
        if result in (OPPONENT, NEUTRAL, ASSASSIN):
            self.guesses_left = 0
        self.log_event("guess", word=display, result=result)
        return {"word": display, "result": result, "guesses_left": self.guesses_left}

    # ── Queries ──────────────────────────────────────────────────────────────

    def words_with(self, identity: str) -> list[str]:
        """Display words currently tagged with `identity`."""
        return [self._grid[nw]["display"] for nw in self._order
                if self._grid[nw]["identity"] == identity]

    def unrevealed(self) -> list[str]:
        """Display words whose identity is still UNKNOWN — the live guess space."""
        return self.words_with(UNKNOWN)

    def team_words(self) -> list[str]:
        return self.words_with(TEAM)

    def assassin_words(self) -> list[str]:
        return self.words_with(ASSASSIN)

    def danger_words(self) -> dict[str, list[str]]:
        """The words a clue-giver must steer AWAY from: assassin first, then
        opponent, then neutral. (Already-revealed words can't be re-guessed, but
        they're still listed by identity for completeness elsewhere.)"""
        return {
            ASSASSIN: self.assassin_words(),
            OPPONENT: self.words_with(OPPONENT),
            NEUTRAL: self.words_with(NEUTRAL),
        }

    def identity_of(self, word: str) -> str:
        return self._grid.get(_norm(word), {}).get("identity", UNKNOWN)

    def check_clue(self, clue: str, targets: list[str]) -> dict[str, Any]:
        """Spymaster safety gate. Given a candidate `clue` and the `targets` she
        intends it to connect, verify every target is one of HER team's words
        and flag any that are assassin / opponent / neutral. Also surfaces the
        live danger words so she can sanity-check the clue won't pull them.

        Returns:
            {
              "safe": bool,                  # all targets are her team's
              "clue": str,
              "bad_targets": [ {word, identity}, ... ],
              "ok_targets":  [word, ...],
              "danger": {assassin:[...], opponent:[...], neutral:[...]},
            }
        """
        bad: list[dict[str, str]] = []
        ok: list[str] = []
        for t in targets or []:
            ident = self.identity_of(t)
            disp = self._grid.get(_norm(t), {}).get("display", (t or "").strip())
            if ident == TEAM:
                ok.append(disp)
            else:
                bad.append({"word": disp, "identity": ident})
        return {
            "safe": not bad,
            "clue": (clue or "").strip(),
            "bad_targets": bad,
            "ok_targets": ok,
            "danger": self.danger_words(),
        }

    # ── Prompt surface ───────────────────────────────────────────────────────

    def _state_lines(self) -> list[str]:
        role_word = "the SPYMASTER (clue-giver)" if self.role == SPYMASTER else "a GUESSER"
        team = f" (your team: {self.my_team_label})" if self.my_team_label else ""
        lines: list[str] = [
            "[CODENAMES BOARD STATE \u2014 this is YOUR live game. Reason over THIS "
            "tracked board, not a single glance at the screen. It is the source of truth.]",
            f"  You are {role_word}{team}.",
        ]

        unrevealed = self.unrevealed()
        if unrevealed:
            lines.append(f"  Still in play ({len(unrevealed)}): "
                         + ", ".join(unrevealed) + ".")

        # Known identities, grouped (only the interesting/confirmed ones).
        for ident in (TEAM, ASSASSIN, OPPONENT, NEUTRAL):
            ws = self.words_with(ident)
            if ws:
                lines.append(f"  {_IDENTITY_LABEL[ident]}: " + ", ".join(ws) + ".")

        # Clue history (most recent few).
        if self.clues:
            recent = self.clues[-4:]
            hist = "; ".join(
                f"\"{c['clue']}\" {c['number']} ({c['by']})" for c in recent
            )
            lines.append(f"  Clues so far: {hist}.")

        # Running guess state.
        if self.current_clue:
            cc = self.current_clue
            guessed = ", ".join(self.guessed_this_turn) if self.guessed_this_turn else "none yet"
            lines.append(
                f"  Working clue: \"{cc['clue']}\" for {cc['number']} \u2014 "
                f"{self.guesses_left} guess(es) left, guessed this turn: {guessed}."
            )

        # Role-specific directive.
        if self.role == SPYMASTER:
            danger = self.danger_words()
            flat = danger[ASSASSIN] + danger[OPPONENT] + danger[NEUTRAL]
            if flat:
                lines.append(
                    "  CLUE-GIVER RULE: before you commit a clue, make sure it can't "
                    "point the other team at any of these \u2014 "
                    + ", ".join(flat)
                    + ". The ASSASSIN is the one word that loses instantly; steer hard around it."
                )
            else:
                lines.append(
                    "  CLUE-GIVER RULE: pick a clue that links your team's words and "
                    "can't be misread onto a word you don't control."
                )
        else:
            lines.append(
                "  GUESSER RULE: choose only from the words still in play above, and "
                "reason from this tracked board \u2014 not a single screen glance."
            )

        return lines

    def snapshot(self) -> dict[str, Any]:
        """Machine-readable view (for the dashboard / debugging)."""
        return {
            "active": self.active,
            "role": self.role,
            "my_team_label": self.my_team_label,
            "grid": [{"word": self._grid[nw]["display"],
                      "identity": self._grid[nw]["identity"]} for nw in self._order],
            "clues": list(self.clues),
            "current_clue": self.current_clue,
            "guesses_left": self.guesses_left,
            "guessed_this_turn": list(self.guessed_this_turn),
        }
