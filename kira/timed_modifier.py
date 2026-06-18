# timed_modifier.py — generalized 5-minute "timed mode" state.
#
# A TimedModifier is a named, time-boxed behavior modifier whose `directive` is
# injected into Kira's prompt while it's active (e.g. CHAOS MODE). The registry
# holds AT MOST ONE active modifier (one-at-a-time) plus per-name cooldowns.
#
# This generalizes the old bespoke chaos-mode state: chaos is now ONE instance
# routed through the registry. The registry is a PURE state container — the bot
# still owns the asyncio timer and the side effects (overlay broadcast, end-line
# speech), so existing chaos behavior is unchanged. Later layers add more
# instances (accent, speech-constraint) that reuse the exact same rules:
#   - can_activate(): one active at a time AND the name's cooldown must have
#     elapsed (the chaos cooldown-block rule, generalized).
#   - active_directive(): the directive injected into every prompt while active.

import time
from typing import Optional


class TimedModifier:
    """One active timed modifier. `until` is the wall-clock expiry; `cooldown_s`
    is armed against the NAME when the modifier ends."""

    def __init__(self, name: str, directive: str, duration_s: float, cooldown_s: float):
        self.name = name
        self.directive = directive
        self.duration_s = float(duration_s)
        self.cooldown_s = float(cooldown_s)
        self.until = time.time() + float(duration_s)


class TimedModifierRegistry:
    """At most one active modifier + per-name cooldowns. Pure state; no asyncio."""

    def __init__(self) -> None:
        self.active: Optional[TimedModifier] = None
        self._cooldown_until: dict[str, float] = {}

    def can_activate(self, name: str) -> bool:
        """True iff nothing is active AND this name's cooldown has elapsed.
        This is the generalized chaos cooldown-block: one window at a time, then a
        quiet stretch before the same mode can fire again."""
        if self.active is not None:
            return False
        return time.time() >= self._cooldown_until.get(name, 0.0)

    def start(self, modifier: TimedModifier) -> None:
        """Make `modifier` the active one. Callers gate on can_activate() first
        (matches how chaos was only ever activated when not active / off cooldown)."""
        self.active = modifier

    def end(self) -> Optional[TimedModifier]:
        """Clear the active modifier and ARM its cooldown (now + cooldown_s against
        its name). Returns the modifier that ended, or None if nothing was active."""
        m = self.active
        if m is None:
            return None
        self.active = None
        self._cooldown_until[m.name] = time.time() + m.cooldown_s
        return m

    def is_active(self, name: Optional[str] = None) -> bool:
        if self.active is None:
            return False
        return name is None or self.active.name == name

    def active_directive(self) -> str:
        """The directive to inject into the prompt while a modifier is active
        (empty string when none) — the single injection signal for all modes."""
        return self.active.directive if self.active is not None else ""

    def until(self) -> float:
        """Wall-clock expiry of the active modifier, or 0.0 if none."""
        return self.active.until if self.active is not None else 0.0

    def cooldown_until(self, name: str) -> float:
        """Wall-clock time before which `name` may not re-activate (0.0 if clear)."""
        return self._cooldown_until.get(name, 0.0)
