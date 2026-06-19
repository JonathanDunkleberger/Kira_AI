# obs_anchor.py — capture OBS's recording-start wall-clock via OBS WebSocket v5.
#
# WHY: clip alignment needs a shared clock between Kira's UTC event logs and the OBS
# video file. Today that's re-derived from the container `creation_time` tag (good
# when present) or a slow Whisper acoustic match (the fallback). Asking OBS directly
# for the record-start instant gives a guaranteed shared anchor at capture time.
#
# DESIGN: opt-in (OBS_RECORD_ANCHOR_ENABLED), and FULLY GRACEFUL — any failure
# (lib missing, OBS not running, WS disabled, auth wrong, not recording) returns None
# and the cutter keeps using today's fallback. It never raises into the caller and
# never blocks the stream opener (the caller fires it as a background task).
#
# Python 3.10-safe (uses asyncio.wait_for, not asyncio.timeout).

import asyncio
import base64
import hashlib
import json
import time


async def query_record_start(url: str, password: str = "", timeout: float = 4.0) -> "dict | None":
    """Query OBS WebSocket v5 for record status. Returns
    {"record_start_epoch": float, "obs_duration_ms": int} when recording is active,
    else None. Never raises."""
    try:
        import websockets  # asyncio client; already a transitive dep
    except Exception as e:
        print(f"   [OBSAnchor] websockets library unavailable: {e}")
        return None

    async def _do() -> "dict | None":
        async with websockets.connect(url, open_timeout=timeout) as ws:
            # ── op:0 Hello (may carry an auth challenge) ──
            hello = json.loads(await ws.recv())
            hd = hello.get("d", {})
            identify = {"op": 1, "d": {"rpcVersion": hd.get("rpcVersion", 1)}}
            auth = hd.get("authentication")
            if auth:
                if not password:
                    print("   [OBSAnchor] OBS requires a WS password but OBS_WEBSOCKET_PASSWORD is empty")
                    return None
                secret = base64.b64encode(
                    hashlib.sha256((password + auth["salt"]).encode("utf-8")).digest()
                ).decode()
                identify["d"]["authentication"] = base64.b64encode(
                    hashlib.sha256((secret + auth["challenge"]).encode("utf-8")).digest()
                ).decode()
            # ── op:1 Identify → expect op:2 Identified ──
            await ws.send(json.dumps(identify))
            ident = json.loads(await ws.recv())
            if ident.get("op") != 2:
                print(f"   [OBSAnchor] identify rejected by OBS: {ident}")
                return None
            # ── op:6 GetRecordStatus → expect op:7 RequestResponse ──
            await ws.send(json.dumps({
                "op": 6, "d": {"requestType": "GetRecordStatus", "requestId": "rec1"},
            }))
            data = None
            for _ in range(6):  # skip any interleaved events until our response
                msg = json.loads(await ws.recv())
                if msg.get("op") == 7 and msg.get("d", {}).get("requestId") == "rec1":
                    data = msg["d"].get("responseData", {})
                    break
            if not data or not data.get("outputActive"):
                print(f"   [OBSAnchor] OBS is not currently recording (status={data})")
                return None
            dur_ms = int(data.get("outputDuration", 0))
            return {
                "record_start_epoch": time.time() - dur_ms / 1000.0,
                "obs_duration_ms": dur_ms,
            }

    try:
        return await asyncio.wait_for(_do(), timeout=timeout)
    except Exception as e:
        print(f"   [OBSAnchor] could not reach OBS at {url} ({e}); using clip-time fallback.")
        return None
