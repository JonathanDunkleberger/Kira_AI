# kira.games — structured state for turn-based games.
#
# The reusable scaffold (TurnBasedGameState) lets Kira reason over a persistent,
# in-memory model of a game she updates over time, instead of re-reading the
# whole board from a single (and often stale) vision frame every turn. Chess
# already does this with its own ChessAgent; this package is the generalised
# pattern so Codenames / Wordle / future games share one shape.

from .turn_state import TurnBasedGameState
from .codenames import CodenamesState

__all__ = ["TurnBasedGameState", "CodenamesState"]
