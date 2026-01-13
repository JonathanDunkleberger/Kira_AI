# undertale_bridge.py - Decoupled Game Agent
# Monitors Undertale specific memory or screen regions.

import time
import threading
from config import ENABLE_GAME_AGENT

class UndertaleBridge:
    def __init__(self):
        self.game_state = "Not Playing"
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        if not ENABLE_GAME_AGENT:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._game_loop, daemon=True)
        self.thread.start()
        print("   [Game] Undertale Bridge started.")

    def stop(self):
        self.running = False

    def _game_loop(self):
        while self.running:
            try:
                # Logic to check if Undertale process is running
                # Logic to read health/location via OCR or Memory reading
                # For now, placeholder:
                state = "Undertale Not Detected"
                
                with self.lock:
                    self.game_state = state
                
                time.sleep(2) # moderate poll rate
            except Exception as e:
                print(f"   [Game] Error: {e}")
                time.sleep(5)

    def get_game_state(self):
        with self.lock:
            return self.game_state

    def send_input(self, command):
        # Allow Kira to press buttons if she explicitly asks to "[PRESS: Z]"
        if not ENABLE_GAME_AGENT: return
        print(f"   [Game] Mock Pressing: {command}")
        # Implement pydirectinput or pyautogui here
