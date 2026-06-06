# vts_expression_controller.py — bridges Kira's EmotionalState to VTube Studio hotkeys.
#
# Hiyori Momose model uses TriggerAnimation hotkeys (fire-and-forget one-shot animations).
# Emotion → animation mapping is in EMOTION_TO_EXPRESSION below; update it after
# checking which animation name corresponds to which visual in VTube Studio.
#
# Behavior:
#   1) Only states in SIGNIFICANT_STATES can trigger an animation. HAPPY and SASSY
#      never animate — they're Kira's deadpan baseline and a motion there undercuts
#      the dryness. Animations are punctuation for real emotional shifts only.
#   2) If the incoming state is not significant, the transition is still tracked
#      (so the next significant state fires correctly) but no hotkey is sent.
#   3) Identical consecutive states are no-ops (new_state == last_triggered).

from __future__ import annotations

import asyncio
from typing import Optional

from persona import EmotionalState
from vts_client import VTSClient
from config import ENABLE_VTS_EXPRESSIONS


# Emotion → animation hotkey NAME (case-insensitive; resolved against VTS at runtime).
# All Hiyori hotkeys are TriggerAnimation (one-shot, fire-and-forget).
# Fill in the animation name after confirming which visual it produces in VTube Studio.
# A value of None means: no animation for this state (model stays at idle).
#
# Current Hiyori hotkeys (from HotkeysInCurrentModelRequest 2026-06-05):
#   "My Animation 1"  → hiyori_m03.motion3.json  (35e965fc477342888a9c6beb2464b12d)
#   "My Animation 2"  → hiyori_m06.motion3.json  (135ec454ec524d8c8d7c90a20bbc915d)
#   "My Animation 3"  → hiyori_m10.motion3.json  (f9c5528eab6a4b1cb1189356c4d6a967)
EMOTION_TO_EXPRESSION: dict[EmotionalState, Optional[str]] = {
    EmotionalState.HAPPY:       None,               # hiyori_m03 fits, but HAPPY is her default register — fires too often
    EmotionalState.SASSY:       None,               # deadpan IS the delivery; motion undercuts dryness
    EmotionalState.EMOTIONAL:   None,               # no animation fits genuine warmth; baseline > wrong motion
    EmotionalState.MOODY:       "My Animation 3",   # hiyori_m10 — "ugh, darn" reluctant/put-out face
    EmotionalState.HYPERACTIVE: "My Animation 2",   # hiyori_m06 — hands up, leaning in, giddy "ooh this'll be good"
}

# Only real departures from her baseline get an animation beat.
# HAPPY excluded: it's her default state — animating it = constant twitching.
# SASSY/EMOTIONAL excluded: no animation fits; deadpan/warmth work better still.
SIGNIFICANT_STATES: frozenset[EmotionalState] = frozenset({
    EmotionalState.HYPERACTIVE,
    EmotionalState.MOODY,
})


class VTSExpressionController:
    """Thin async wrapper around VTSClient that maps EmotionalState → hotkey trigger.
    All public methods are fire-and-forget safe and never raise to the caller."""

    def __init__(self):
        self.enabled = ENABLE_VTS_EXPRESSIONS
        self._client: Optional[VTSClient] = VTSClient() if self.enabled else None
        self._last_triggered: Optional[EmotionalState] = None
        # Animation names confirmed absent in the current model — log once, skip silently.
        self._skip_expressions: set[str] = set()
        # Last-seen VTS modelID; used to detect model swaps and flush the skip set.
        self._known_model_id: Optional[str] = None

    async def _refresh_model_id_if_changed(self) -> bool:
        """Polls VTS for the current modelID. If it differs from the last seen
        value, clears the skip set and updates the cached modelID. Returns True
        if a change was detected."""
        if self._client is None:
            return False
        try:
            current = await self._client.get_current_model_id()
        except Exception:
            return False
        if current and current != self._known_model_id:
            if self._known_model_id is not None:
                print(f"   [VTS] Model swap detected — clearing missing-animation skip list.")
            self._known_model_id = current
            self._skip_expressions.clear()
            return True
        return False

    async def on_emotion_change(self, new_state: EmotionalState) -> None:
        """Call when Kira's EmotionalState actually transitions. Best-effort, never raises.

        Only fires an animation for states in SIGNIFICANT_STATES (EMOTIONAL,
        HYPERACTIVE, MOODY). HAPPY and SASSY are always silent — they're her
        deadpan baseline register and motion there undercuts the dryness.
        The transition is still recorded so the next significant state works."""
        if not self.enabled or self._client is None:
            return
        if new_state == self._last_triggered:
            return

        # Always advance tracking so the next transition is relative to now.
        self._last_triggered = new_state

        # Gate: only significant states get an animation.
        if new_state not in SIGNIFICANT_STATES:
            return

        new_expression = EMOTION_TO_EXPRESSION.get(new_state)
        print(f"   [VTS] Emotion transition → {new_state.name} (mapped: '{new_expression}')")

        try:
            # No animation mapped for this state — no-op (model stays at idle).
            if new_expression is None:
                return

            # Previously confirmed absent in this model — silent no-op.
            if new_expression in self._skip_expressions:
                return

            # Check for a model swap; if so, flush the skip set before resolving.
            await self._refresh_model_id_if_changed()

            # Pre-resolve to distinguish "name doesn't exist" from a trigger error.
            resolved = await self._client._resolve_name_to_id(new_expression)
            if resolved is None:
                print(f"   [VTS] Current model has no animation for {new_state.name} ('{new_expression}') — skipping until model changes.")
                self._skip_expressions.add(new_expression)
                return

            # Fire the animation. TriggerAnimation is one-shot — no untoggle needed.
            ok = await self._client.trigger_expression(new_expression)
            print(f"   [VTS] Emotion {new_state.name} → triggered '{new_expression}' ({'success' if ok else 'fail'})")
        except Exception as e:
            print(f"   [VTS] on_emotion_change suppressed error: {e}")

    def fire_and_forget(self, new_state: EmotionalState, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Schedule on_emotion_change without awaiting. Safe to call from any thread
        as long as `loop` is the bot's running event loop (or current loop if None)."""
        if not self.enabled:
            return
        try:
            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(self.on_emotion_change(new_state), loop)
            else:
                # Same-thread async context — schedule as a task.
                asyncio.create_task(self.on_emotion_change(new_state))
        except Exception as e:
            print(f"   [VTS] fire_and_forget suppressed error: {e}")

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass

    async def connect_eager(self) -> bool:
        """Connect + authenticate to VTube Studio up front, so the first emotion
        transition isn't lost to lazy-connect latency. Returns True on success,
        False on any failure. Never raises."""
        if not self.enabled or self._client is None:
            return False
        try:
            ok = await self._client.connect()
            if ok:
                print("   [VTS] Connected & authenticated — ready to drive expressions.")
                # Prime modelID so the first emotion transition can detect swaps.
                try:
                    self._known_model_id = await self._client.get_current_model_id()
                except Exception:
                    self._known_model_id = None

                # Print the active mapping so mis-spellings surface immediately.
                active = {s.name: name for s, name in EMOTION_TO_EXPRESSION.items() if name is not None}
                if active:
                    print("   [VTS] Emotion → hotkey mapping:")
                    for state_name, anim_name in active.items():
                        print(f"          {state_name} → '{anim_name}'")
                else:
                    print("   [VTS] Warning: no animations are mapped — all states will no-op.")

                # Validate: warn for any mapped name not present in the current model.
                try:
                    hotkeys = await self._client.list_hotkeys()
                    vts_names = {(hk.get("name") or "").strip().lower() for hk in hotkeys}
                    for state_name, anim_name in active.items():
                        if anim_name.strip().lower() not in vts_names:
                            print(f"   [VTS] WARNING: '{anim_name}' (mapped to {state_name}) is NOT in the current model's hotkeys — will be skipped.")
                        else:
                            print(f"   [VTS] Confirmed: '{anim_name}' found in model hotkeys.")
                except Exception as e:
                    print(f"   [VTS] Startup hotkey validation failed: {e}")
            else:
                print("   [VTS] Eager connect failed — expressions will retry lazily on next emotion change.")
            return ok
        except Exception as e:
            print(f"   [VTS] Eager connect suppressed error: {e}")
            return False
