import asyncio
import ctypes
import os
from config import HYTALE_LOG_PATH

# Windows API for window title
user32 = ctypes.windll.user32

def get_active_window_title():
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buff = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buff, length + 1)
    return buff.value

class GameModeController:
    def __init__(self, vision_agent, media_bridge=None):
        self.vision = vision_agent
        self.media_bridge = media_bridge 
        self.is_active = False
        self.current_game_name = "Unknown Game"

    async def get_full_context(self, user_input):
        """
        Orchestrates the context gathering:
        1. Detects Active Window -> Determines Game Name
        2. Updates Media Bridge path if needed
        3. Captures Vision
        4. Formats the System Note
        """
        
        # 1. Detect Game
        window_title = get_active_window_title()
        self.current_game_name = self._identify_game(window_title)
        
        # 2. Update Media Bridge (Dynamic Pathing)
        if self.media_bridge:
            self._update_log_path(self.current_game_name)

        # 3. Vision Capture (Heartbeat/On-Demand)
        # Vision is now async API call, allow it to run concurrently
        visual_desc = await self.vision.capture_and_describe()
        
        # 4. Log Context
        log_context = self.media_bridge.get_latest_events() if self.media_bridge else ""
        
        return visual_desc, log_context, self.current_game_name

    def _identify_game(self, title):
        """Map window titles to game names."""
        title_lower = title.lower()
        if "hytale" in title_lower:
            return "Hytale"
        if "minecraft" in title_lower:
            return "Minecraft"
        if "terraria" in title_lower:
            return "Terraria"
        # Can expand for Media Players too
        if "vlc" in title_lower or "netflix" in title_lower:
             return "Movie/Video"
        return "Unknown Game"

    def _update_log_path(self, game_name):
        """Routes the correct log file to the bridge."""
        if game_name == "Hytale":
            self.media_bridge.set_target_log(HYTALE_LOG_PATH)
        elif game_name == "Minecraft":
            # Example path
            appdata = os.getenv("APPDATA")
            mc_log = os.path.join(appdata, ".minecraft", "logs", "latest.log")
            self.media_bridge.set_target_log(mc_log)
        else:
            # If unknown game or no logs, we might just unset the log path
            pass
