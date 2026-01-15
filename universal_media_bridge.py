import asyncio
import os

class UniversalMediaBridge:
    """
    Universal Media Bridge (formerly Game Log Bridge).
    Monitors logs only when a matching game is detected.
    Silent otherwise.
    """
    def __init__(self, input_queue, character_name="DuchessSterling"):
        self.log_path = None
        self.input_queue = input_queue
        self.character_name = character_name
        self.is_active = False
        self._stop_event = asyncio.Event()
        self.recent_events_buffer = []

    def set_target_log(self, log_path):
        """Updates the log file being watched."""
        if self.log_path != log_path:
            print(f"   [MediaBridge] Switching target to: {log_path}")
            self.log_path = log_path
            # Reset buffer on game switch if desired, or keep for context
            # self.recent_events_buffer.clear() 

    async def start(self):
        """Monitors logs for chat and game events."""
        print(f"-> Universal Media Bridge active.")
        
        while not self._stop_event.is_set():
            # If no path is set or we are not active, idle
            if not self.is_active or not self.log_path or not os.path.exists(self.log_path):
                await asyncio.sleep(2)
                continue

            # We have a valid path and are active
            try:
                with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(0, os.SEEK_END)
                    
                    while not self._stop_event.is_set():
                        # Check if we should still be reading this file
                        if not self.is_active or not self.log_path: 
                            break 

                        line = f.readline()
                        if line:
                            await self._parse_line(line)
                        else:
                            await asyncio.sleep(0.1)
            except Exception as e:
                print(f"   [MediaBridge] File read error: {e}")
                await asyncio.sleep(5)

    async def _parse_line(self, line):
        """Parses logs using generic heuristics or game-specific rules."""
        
        if len(self.recent_events_buffer) > 5:
            self.recent_events_buffer.pop(0)

        # Basic Chat Detection (Universal-ish)
        # Looks for "Name: message" pattern common in logs
        if f"{self.character_name}:" in line:
            content = line.split(f"{self.character_name}:")[-1].strip()
            await self.input_queue.put(("game_chat", content))
            self.recent_events_buffer.append(f"Jonny said: {content}")

        # Death / Event Detection keywords
        keywords = ["died", "was slain", "fell from", "achievement", "advancement"]
        if any(k in line.lower() for k in keywords):
             # Cleanup timestamp if exists (simple heuristic)
             clean_line = line.strip()
             await self.input_queue.put(("game_event", clean_line))
             self.recent_events_buffer.append(f"Event: {clean_line}")

    def get_latest_events(self):
        if not self.recent_events_buffer:
            return "" # Return empty string instead of "No recent" to keep prompt clean
        return " | ".join(self.recent_events_buffer)

    def stop(self):
        self._stop_event.set()
