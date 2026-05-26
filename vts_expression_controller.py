# vts_expression_controller.py — bridges Kira's EmotionalState to VTube Studio hotkeys.
#
# All expression hotkeys in this model are type ToggleExpression: firing the same
# hotkey twice turns the expression OFF. We deliberately do NOT use the model's
# RemoveAllExpressions hotkey — it resets the model to its "default" state, which
# on the Suki model includes the copyright watermark (水印) being ON. Instead, we
# track exactly the one expression we last toggled on, and toggle just that one
# off before toggling the new one on. The watermark and all outfit / accessory
# toggles are therefore never touched by this controller.
#
# Behavior:
#   1) If a previous emotion expression is currently on → fire its hotkey again to OFF.
#   2) If the new emotion has a mapped expression (non-None) → fire its hotkey to ON.
#   3) HAPPY maps to None: previous gets untoggled, nothing new applied (neutral face).
#
# Watermark safety: the watermark toggle hotkey ID is hard-blacklisted. Even if the
# mapping is ever misconfigured to point at it, the controller will refuse to fire it.

from __future__ import annotations

import asyncio
from typing import Optional

from persona import EmotionalState
from vts_client import VTSClient
from config import ENABLE_VTS_EXPRESSIONS


# Hard blacklist — these hotkey IDs will NEVER be fired by this controller, regardless
# of what's in EMOTION_TO_EXPRESSION. Add any other "do not automate" toggles here.
BLACKLISTED_HOTKEY_IDS: set[str] = {
    "564fef6591ed441cb7e07a0ac7278e1f",   # 水印 (copyright watermark) — must stay manual
}


# Emotion → expression hotkey NAME (case-insensitive; resolved against VTS at runtime).
# A value of None means: no expression overlay for that state (untoggle previous only).
EMOTION_TO_EXPRESSION: dict[EmotionalState, Optional[str]] = {
    EmotionalState.HAPPY:       None,       # baseline — untoggle previous, apply nothing
    EmotionalState.SASSY:       "星星眼",    # star eyes
    EmotionalState.EMOTIONAL:   "脸红",      # blush
    EmotionalState.MOODY:       "痴呆",      # dazed
    EmotionalState.HYPERACTIVE: "爱心眼",    # heart eyes
}


class VTSExpressionController:
    """Thin async wrapper around VTSClient that maps EmotionalState → hotkey trigger.
    All public methods are fire-and-forget safe and never raise to the caller."""

    def __init__(self):
        self.enabled = ENABLE_VTS_EXPRESSIONS
        self._client: Optional[VTSClient] = VTSClient() if self.enabled else None
        self._last_triggered: Optional[EmotionalState] = None
        # The exact hotkey name we last toggled ON (if any). Tracked separately so we
        # can untoggle precisely the same hotkey even if the mapping later changes.
        self._active_expression_name: Optional[str] = None
        # Expression NAMES we've already confirmed don't exist in the currently
        # loaded model. Kept so we log the warning ONCE per name and silently
        # no-op on subsequent transitions instead of spamming. Cleared whenever
        # we detect a model swap.
        self._skip_expressions: set[str] = set()
        # Last-seen VTS modelID; used to detect model swaps so we can flush
        # _skip_expressions and re-evaluate availability against the new model.
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
                print(f"   [VTS] Model swap detected (modelID changed) \u2014 clearing missing-expression skip list.")
            self._known_model_id = current
            self._skip_expressions.clear()
            return True
        return False

    async def _safe_trigger(self, hotkey_name_or_id: str) -> bool:
        """Wrapper around VTSClient.trigger_expression that enforces the blacklist.
        Returns False (without firing) if the resolved target is blacklisted."""
        if not hotkey_name_or_id:
            return False
        # Direct ID match against blacklist.
        if hotkey_name_or_id in BLACKLISTED_HOTKEY_IDS:
            print(f"   [VTS] Refusing to fire blacklisted hotkey ID {hotkey_name_or_id}")
            return False
        # If a NAME was passed, resolve it to an ID and check that too.
        if not VTSClient._looks_like_id(hotkey_name_or_id):
            resolved = await self._client._resolve_name_to_id(hotkey_name_or_id)
            if resolved and resolved in BLACKLISTED_HOTKEY_IDS:
                print(f"   [VTS] Refusing to fire blacklisted hotkey '{hotkey_name_or_id}' ({resolved})")
                return False
        return await self._client.trigger_expression(hotkey_name_or_id)

    async def on_emotion_change(self, new_state: EmotionalState) -> None:
        """Call when Kira's EmotionalState actually transitions. Best-effort, never raises.

        Untoggles the previously-active expression (if any), then toggles the new one
        (if mapped). Never touches outfit/accessory hotkeys, never fires the watermark.

        Model-swap survival:
          * If the previous expression name no longer exists in the current model
            (model was swapped), we don't try to untoggle it \u2014 the swap reset
            its state for us. We just clear our tracking and continue.
          * If the new expression name doesn't exist in the current model, we
            log ONCE per name and add it to a skip set, so subsequent
            transitions to that emotion no-op silently until the model changes
            again.
          * Whenever VTS reports a different modelID than we last saw, the skip
            set is flushed so newly-available expressions get re-tried."""
        if not self.enabled or self._client is None:
            return
        if new_state == self._last_triggered:
            return

        new_expression = EMOTION_TO_EXPRESSION.get(new_state)
        prev_expression = self._active_expression_name

        try:
            # Step 1: untoggle the previously-active expression, if any.
            if prev_expression:
                # If the previous name doesn't resolve in the current model, the
                # active model was swapped after we toggled it on. The previous
                # expression's state is no longer ours to manage \u2014 just clear
                # our tracking and proceed. Also reset the skip set since the
                # model has changed.
                prev_resolved = await self._client._resolve_name_to_id(prev_expression)
                if prev_resolved is None:
                    print(f"   [VTS] Previous expression '{prev_expression}' not in current model \u2014 assuming model swap; clearing tracked state.")
                    self._active_expression_name = None
                    await self._refresh_model_id_if_changed()
                else:
                    off_ok = await self._safe_trigger(prev_expression)
                    print(f"   [VTS] Untoggled previous expression '{prev_expression}' ({'success' if off_ok else 'fail'})")
                    if not off_ok:
                        # If we couldn't untoggle, bail without updating state so the next
                        # transition retries cleanly. Don't stack a second expression on top.
                        return
                    self._active_expression_name = None

            # Step 2: toggle the new expression on (skip for baseline / None).
            if new_expression is None:
                print(f"   [VTS] Emotion {new_state.name} \u2192 baseline (no overlay)")
                self._last_triggered = new_state
                return

            # Step 2a: previously-confirmed-missing in this model \u2014 silent no-op.
            if new_expression in self._skip_expressions:
                self._last_triggered = new_state
                return

            # Step 2b: pre-resolve so we can distinguish "name doesn't exist"
            # from "trigger API error". _resolve_name_to_id already refreshes
            # the hotkey cache on miss and logs the refresh.
            resolved = await self._client._resolve_name_to_id(new_expression)
            if resolved is None:
                # Maybe the model was swapped \u2014 re-check modelID, and if it
                # changed, flush the skip set and try resolving once more.
                if await self._refresh_model_id_if_changed():
                    resolved = await self._client._resolve_name_to_id(new_expression)
                if resolved is None:
                    print(f"   [VTS] Current model has no expression for {new_state.name} ('{new_expression}') \u2014 skipping until model changes.")
                    self._skip_expressions.add(new_expression)
                    # Mark as "handled" so we don't keep re-checking on every
                    # transition into this state. _skip_expressions guards the next ones.
                    self._last_triggered = new_state
                    return

            # Step 2c: actually trigger. trigger_expression handles the 202
            # stale-ID auto-refresh path internally for free.
            on_ok = await self._safe_trigger(new_expression)
            print(f"   [VTS] Emotion {new_state.name} \u2192 triggered '{new_expression}' ({'success' if on_ok else 'fail'})")
            if on_ok:
                self._active_expression_name = new_expression
                self._last_triggered = new_state
        except Exception as e:
            # Defensive: VTSClient already swallows errors, but belt + suspenders.
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
        False on any failure. Never raises.

        NOTE on the watermark (水印): the VTube Studio API does not expose current
        ToggleExpression state in a way we can rely on without per-expression file
        lookups, and the model's default-on state includes the watermark. If the
        watermark is visible at stream start, press Tab once manually in VTS to
        toggle it off. This controller will never touch it after that."""
        if not self.enabled or self._client is None:
            return False
        try:
            ok = await self._client.connect()
            if ok:
                print("   [VTS] Eager connect succeeded — ready to drive expressions.")
                print("   [VTS] Watermark note: if 水印 is visible, press Tab once in VTS to clear it. "
                      "This controller will never auto-toggle it.")
                # Prime modelID so the first emotion transition can detect swaps.
                try:
                    self._known_model_id = await self._client.get_current_model_id()
                except Exception:
                    self._known_model_id = None
            else:
                print("   [VTS] Eager connect failed — expressions will retry lazily on next emotion change.")
            return ok
        except Exception as e:
            print(f"   [VTS] Eager connect suppressed error: {e}")
            return False
