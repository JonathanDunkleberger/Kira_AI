# vts_client.py — VTube Studio WebSocket API client for Kira.
#
# Drives Live2D facial expressions via VTube Studio's plugin API.
# Protocol reference: https://github.com/DenchiSoft/VTubeStudio
#
# Lifecycle:
#   1) Connect to ws://localhost:8001 (configurable).
#   2) If no stored token: send AuthenticationTokenRequest. The user must approve
#      the plugin in the VTS UI ONCE. Token is then persisted to VTS_TOKEN_PATH.
#   3) Send AuthenticationRequest with the token. Now the session is authed.
#   4) Use HotkeysInCurrentModelRequest to enumerate hotkeys.
#   5) Use HotkeyTriggerRequest to fire an expression hotkey by ID or name.
#
# All public methods FAIL GRACEFULLY — any error returns False / [] and logs once.
# Never raises out to the caller. If VTS isn't running, Kira keeps running normally.

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

from kira.config import (
    VTS_WS_URL,
    VTS_PLUGIN_NAME,
    VTS_PLUGIN_DEVELOPER,
    VTS_TOKEN_PATH,
)

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"


def _msg(message_type: str, data: dict | None = None, request_id: str | None = None) -> str:
    return json.dumps({
        "apiName": API_NAME,
        "apiVersion": API_VERSION,
        "requestID": request_id or str(uuid.uuid4()),
        "messageType": message_type,
        "data": data or {},
    })


class VTSClient:
    """Persistent VTube Studio WebSocket client. One instance per bot."""

    def __init__(self):
        self._ws = None
        self._token: Optional[str] = self._load_token()
        self._connected = False
        self._authed = False
        self._lock = asyncio.Lock()
        self._warned_unavailable = False
        self._hotkeys_cache: list[dict] = []  # last-known hotkey list

    # ---------- token persistence ----------

    def _load_token(self) -> Optional[str]:
        try:
            if os.path.exists(VTS_TOKEN_PATH):
                with open(VTS_TOKEN_PATH, "r", encoding="utf-8") as f:
                    tok = f.read().strip()
                    return tok or None
        except Exception:
            pass
        return None

    def _save_token(self, token: str) -> None:
        try:
            with open(VTS_TOKEN_PATH, "w", encoding="utf-8") as f:
                f.write(token)
        except Exception as e:
            print(f"   [VTS] Failed to persist token: {e}")

    # ---------- low-level send/recv ----------

    async def _recv_json(self) -> dict:
        raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    async def _send(self, message_type: str, data: dict | None = None) -> dict:
        await self._ws.send(_msg(message_type, data))
        return await self._recv_json()

    # ---------- connection / auth ----------

    async def connect(self) -> bool:
        """Connect + authenticate. Returns True on success, False on any failure."""
        if not _WS_AVAILABLE:
            if not self._warned_unavailable:
                print("   [VTS] 'websockets' package not installed — expressions disabled.")
                self._warned_unavailable = True
            return False

        async with self._lock:
            if self._connected and self._authed and self._ws is not None:
                return True
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(VTS_WS_URL, max_size=2**22),
                    timeout=5.0,
                )
                self._connected = True
            except (OSError, asyncio.TimeoutError, WebSocketException) as e:
                if not self._warned_unavailable:
                    print(f"   [VTS] Could not connect to {VTS_WS_URL} ({e}). Is VTube Studio running with API enabled?")
                    self._warned_unavailable = True
                self._connected = False
                self._ws = None
                return False

            # Acquire token if we don't have one yet.
            if not self._token:
                print(f"   [VTS] Requesting authentication token — APPROVE the '{VTS_PLUGIN_NAME}' plugin in VTube Studio.")
                try:
                    resp = await asyncio.wait_for(self._send("AuthenticationTokenRequest", {
                        "pluginName": VTS_PLUGIN_NAME,
                        "pluginDeveloper": VTS_PLUGIN_DEVELOPER,
                    }), timeout=120.0)  # generous — user has to click in VTS UI
                except asyncio.TimeoutError:
                    print("   [VTS] Token request timed out (no approval in VTS).")
                    await self._safe_close()
                    return False
                token = (resp.get("data") or {}).get("authenticationToken")
                if not token:
                    print(f"   [VTS] Token request denied / invalid: {resp.get('data')}")
                    await self._safe_close()
                    return False
                self._token = token
                self._save_token(token)
                print("   [VTS] Token approved and saved.")

            # Authenticate the session.
            try:
                resp = await self._send("AuthenticationRequest", {
                    "pluginName": VTS_PLUGIN_NAME,
                    "pluginDeveloper": VTS_PLUGIN_DEVELOPER,
                    "authenticationToken": self._token,
                })
            except (ConnectionClosed, WebSocketException, asyncio.TimeoutError) as e:
                print(f"   [VTS] Auth send failed: {e}")
                await self._safe_close()
                return False
            authed = bool((resp.get("data") or {}).get("authenticated"))
            if not authed:
                reason = (resp.get("data") or {}).get("reason", "unknown")
                print(f"   [VTS] Authentication rejected: {reason}. Deleting stored token; you will be re-prompted next time.")
                # Stored token may be stale — clear it so next connect re-requests.
                try:
                    if os.path.exists(VTS_TOKEN_PATH):
                        os.remove(VTS_TOKEN_PATH)
                except Exception:
                    pass
                self._token = None
                await self._safe_close()
                return False

            self._authed = True
            self._warned_unavailable = False
            print(f"   [VTS] Connected and authenticated to VTube Studio.")
            return True

    async def _safe_close(self):
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass
        self._ws = None
        self._connected = False
        self._authed = False

    async def close(self):
        async with self._lock:
            await self._safe_close()

    async def _ensure_connected(self) -> bool:
        if self._connected and self._authed and self._ws is not None:
            return True
        return await self.connect()

    # ---------- public API ----------

    async def list_hotkeys(self) -> list[dict]:
        """Return a list of hotkey dicts for the currently loaded model.
        Each dict has at least: hotkeyID, name, type, file (expression file, if any).
        Returns [] on failure."""
        if not await self._ensure_connected():
            return []
        async with self._lock:
            try:
                resp = await self._send("HotkeysInCurrentModelRequest", {})
            except (ConnectionClosed, WebSocketException, asyncio.TimeoutError) as e:
                print(f"   [VTS] list_hotkeys failed: {e}")
                await self._safe_close()
                return []
        hotkeys = (resp.get("data") or {}).get("availableHotkeys") or []
        self._hotkeys_cache = hotkeys
        return hotkeys

    async def trigger_expression(self, hotkey_id_or_name: str) -> bool:
        """Trigger a VTS hotkey by its ID or by its display name.
        Returns True on success, False on any failure (never raises).

        Auto-recovery: if VTS returns errorID 202 ("hotkey ID or name was not
        found in model") — which happens when the active VTS model has been
        swapped and our cached hotkey IDs are stale — we clear the hotkey
        cache, re-enumerate the new model's hotkeys, re-resolve the original
        NAME to a fresh ID, and retry the trigger ONCE. If we were called with
        a raw ID instead of a name we cannot re-resolve, so we bail."""
        if not hotkey_id_or_name:
            return False
        if not await self._ensure_connected():
            return False

        # Remember whether the caller passed a name (so we can re-resolve on stale-ID).
        original_input = hotkey_id_or_name
        passed_as_name = not self._looks_like_id(hotkey_id_or_name)

        # Resolve name → hotkeyID if needed. VTS API expects the GUID-style hotkeyID.
        hotkey_id = hotkey_id_or_name
        if passed_as_name:
            hotkey_id = await self._resolve_name_to_id(hotkey_id_or_name)
            if not hotkey_id:
                print(f"   [VTS] No hotkey found matching name '{hotkey_id_or_name}'.")
                return False

        ok, err_id = await self._trigger_by_id(hotkey_id)
        if ok:
            return True

        # 202 = stale ID (model swap). Refresh cache + re-resolve + retry once,
        # but only if we have a name to re-resolve from.
        if err_id == 202 and passed_as_name:
            print("   [VTS] Hotkey IDs stale (model swapped?) — refreshing and retrying.")
            self._hotkeys_cache = []  # force re-enumeration on next resolve
            fresh_id = await self._resolve_name_to_id(original_input)
            if not fresh_id:
                print(f"   [VTS] Retry abandoned: '{original_input}' not found in new model.")
                return False
            if fresh_id == hotkey_id:
                # Same ID came back — refresh didn't help, don't loop.
                print(f"   [VTS] Retry abandoned: resolver returned same stale ID for '{original_input}'.")
                return False
            ok2, _ = await self._trigger_by_id(fresh_id)
            return ok2

        return False

    async def _trigger_by_id(self, hotkey_id: str) -> tuple[bool, Optional[int]]:
        """Low-level HotkeyTriggerRequest. Returns (success, error_id).
        error_id is None on success or when the failure wasn't a structured VTS error."""
        async with self._lock:
            try:
                resp = await self._send("HotkeyTriggerRequest", {"hotkeyID": hotkey_id})
            except (ConnectionClosed, WebSocketException, asyncio.TimeoutError) as e:
                print(f"   [VTS] trigger_expression failed: {e}")
                await self._safe_close()
                return False, None
        data = resp.get("data") or {}
        if resp.get("messageType") == "APIError" or "errorID" in data:
            print(f"   [VTS] Hotkey trigger error: {data}")
            err_id = data.get("errorID")
            try:
                err_id = int(err_id) if err_id is not None else None
            except (TypeError, ValueError):
                err_id = None
            return False, err_id
        return True, None

    @staticmethod
    def _looks_like_id(s: str) -> bool:
        # VTS hotkeyIDs are 32-char hex strings (no dashes). Display names are not.
        s = s.strip()
        return len(s) == 32 and all(c in "0123456789abcdefABCDEF" for c in s)

    async def _resolve_name_to_id(self, name: str) -> Optional[str]:
        target = name.strip().lower()
        # Try cache first.
        for hk in self._hotkeys_cache:
            if (hk.get("name") or "").strip().lower() == target:
                return hk.get("hotkeyID")
        # Cache miss — refresh from VTS and try again. Logged so a stale-cache
        # situation (e.g. user swapped models) is visible in the bot output.
        print(f"   [VTS] Hotkey '{name}' not in cache \u2014 refreshing hotkey list for current model.")
        for hk in await self.list_hotkeys():
            if (hk.get("name") or "").strip().lower() == target:
                return hk.get("hotkeyID")
        return None

    async def get_current_model_id(self) -> Optional[str]:
        """Returns the modelID of the model currently loaded in VTube Studio,
        or None on any failure (e.g. no model loaded, VTS unreachable).
        Never raises."""
        if not await self._ensure_connected():
            return None
        async with self._lock:
            try:
                resp = await self._send("CurrentModelRequest", {})
            except (ConnectionClosed, WebSocketException, asyncio.TimeoutError) as e:
                print(f"   [VTS] get_current_model_id failed: {e}")
                await self._safe_close()
                return None
        data = resp.get("data") or {}
        # VTS returns modelID when a model is loaded, and modelLoaded=False otherwise.
        if not data.get("modelLoaded", False):
            return None
        mid = data.get("modelID")
        return mid if isinstance(mid, str) and mid else None


# ---------- standalone enumeration helper ----------
# Run `python vts_client.py` to print every hotkey in the currently loaded model.
# Use this to pick which hotkey names to map to which EmotionalState.

async def _enumerate_main():
    client = VTSClient()
    if not await client.connect():
        print("Could not connect/auth. Make sure VTube Studio is running and the API is enabled "
              "(Settings → API → 'Start API' must be ON, default port 8001).")
        return
    hotkeys = await client.list_hotkeys()
    if not hotkeys:
        print("No hotkeys returned. Is a model loaded?")
        await client.close()
        return
    print(f"\nFound {len(hotkeys)} hotkey(s) in the currently loaded model:\n")
    print(f"{'NAME':<32} {'TYPE':<28} {'FILE':<32} HOTKEY-ID")
    print("-" * 130)
    for hk in hotkeys:
        name = (hk.get("name") or "")[:30]
        t = (hk.get("type") or "")[:26]
        f = (hk.get("file") or "")[:30]
        hid = hk.get("hotkeyID") or ""
        print(f"{name:<32} {t:<28} {f:<32} {hid}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(_enumerate_main())
