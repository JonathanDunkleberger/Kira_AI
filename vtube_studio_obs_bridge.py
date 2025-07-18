# vtube_studio_obs_bridge.py
"""
Bridge Kira-sama's TTS and text output to VTube Studio (for mouth movement) and OBS (for scene/source control).
- Requires: obs-websocket-py, requests, python-osc
- VTube Studio must have API enabled (default: ws://127.0.0.1:8001)
- OBS WebSocket must be enabled (default: ws://127.0.0.1:4455)
"""
import asyncio
import json
import requests
from streamlabs_obs_websocket import StreamlabsOBS

# VTube Studio API settings
VTS_API_URL = "ws://127.0.0.1:8001"
# OBS WebSocket settings
# SLOBS WebSocket settings
SLOBS_HOST = "127.0.0.1"
SLOBS_PORT = 59650  # Default SLOBS WebSocket port
SLOBS_TOKEN = "507b2a7b3ef68deb6ae2b2437b7dddcde84db"  # Set this to your SLOBS API token


# Example: send TTS state to OBS and VTube Studio


class VTubeStudioBridge:
    def __init__(self):
        self.slobs = None

    def connect_obs(self):
        try:
            self.slobs = StreamlabsOBS(token=SLOBS_TOKEN, port=SLOBS_PORT, address=SLOBS_HOST)
            self.slobs.connect()
            print("Connected to Streamlabs OBS WebSocket.")
        except Exception as e:
            print(f"SLOBS connection failed: {e}")

    def disconnect_obs(self):
        if self.slobs:
            self.slobs.disconnect()
            self.slobs = None

    def set_obs_talking(self, talking: bool):
        # Example: Toggle a source visibility in SLOBS when Kira-sama is talking
        # Replace 'KiraModel' with your actual source name
        try:
            if not self.slobs:
                print("SLOBS not connected.")
                return
            source_name = 'KiraModel'
            self.slobs.set_source_visibility(source_name, talking)
        except Exception as e:
            print(f"SLOBS talking state error: {e}")


    async def vtube_studio_talk(self, text: str):
        # No OSC: just simulate speaking duration for OBS toggling
        await asyncio.sleep(max(0.5, min(len(text) / 20, 3)))

    async def kira_speaks(self, text: str):
        # Call this when Kira-sama is about to speak
        self.set_obs_talking(True)
        await self.vtube_studio_talk(text)
        self.set_obs_talking(False)

# Example usage
if __name__ == "__main__":
    bridge = VTubeStudioBridge()
    bridge.connect_obs()
    try:
        asyncio.run(bridge.kira_speaks("Hello, I'm Kira-sama and I'm now synced with OBS and VTube Studio!"))
    finally:
        bridge.disconnect_obs()
