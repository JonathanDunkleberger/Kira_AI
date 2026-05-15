import base64
import time
import asyncio
import re
from collections import deque
from io import BytesIO
from PIL import ImageGrab
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, ENABLE_VISION

class ContextBuffer:
    def __init__(self, maxlen=3):
        self.buffer = deque(maxlen=maxlen)
    
    def add(self, description):
        timestamp = time.strftime("%H:%M:%S")
        self.buffer.append(f"[{timestamp}] {description}")
        
    def get_context_string(self):
        if not self.buffer:
            return "No visual history yet."
        return "\n".join(self.buffer)

class UniversalVisionAgent:
    # General-purpose screen description (used when no specific activity is set)
    DESCRIBE_PROMPT = (
        "You are looking at Jonny's computer screen. "
        "Describe what is happening in 2 clear, specific sentences.\n\n"
        "If it is a Game: Describe the action, the environment, "
        "and the current vibe (intense, chill, chaotic).\n"
        "If it is Video/Media: Describe the scene, the people, "
        "or the topic being discussed.\n"
        "If it is Desktop/Code: Summarize what he is working on.\n\n"
        "Be factual and specific. Do not editorialize, add jokes, "
        "or write commentary about what Kira might say. Just describe what you see."
    )

    # Visual Novel extraction prompt — returns structured parseable output
    VN_DESCRIBE_PROMPT = (
        "You are looking at a Visual Novel game screen. Extract the following information exactly:\n"
        "SPEAKER: [the name of the character speaking, or 'Narration' if no speaker label]\n"
        "DIALOGUE: [the exact dialogue or narration text visible on screen]\n"
        "CHOICES: [if a choice menu is visible, list each option numbered like '1. Option text'; otherwise write 'none']\n"
        "SCENE: [one sentence describing the scene mood or background]\n\n"
        "Be precise. Copy dialogue text exactly as shown. If the screen is a loading/transition screen, "
        "write DIALOGUE: [loading screen] and CHOICES: none."
    )

    # Media/watching prompt
    MEDIA_DESCRIBE_PROMPT = (
        "You are looking at something Jonny is watching (anime, movie, YouTube, etc.). "
        "Describe in 2 sentences: what is on screen, who is in it, and the emotional tone of the scene. "
        "Be specific enough that Kira could make a relevant comment or question."
    )

    def __init__(self):
        self.client = None
        self.api_key = OPENAI_API_KEY
        if self.api_key:
             self.client = AsyncOpenAI(api_key=self.api_key)
        else:
             print("   [Vision] Warning: OPENAI_API_KEY not found in config.")

        self.context_buffer = ContextBuffer(maxlen=3)
        self.last_description = "I'm just getting my bearings. One sec!"
        self.last_capture_time = 0
        self.is_active = ENABLE_VISION   # Respect .env setting on startup
        self.quality_mode = "fast"       # 'fast' or 'high'
        self.activity_type = "general"   # Set by GameModeController / bot

        # Shared Buffer (populated by dashboard to avoid double-capturing)
        self.shared_frame = None
        self.shared_frame_time = 0

    def update_shared_frame(self, frame):
        """Receives a frame from the dashboard to prevent double-capturing."""
        self.shared_frame = frame
        self.shared_frame_time = time.time()

    def get_vision_context(self):
        """Returns the latest vision context instantly."""
        if not self.last_description:
            return "Vision Initializing..."
        
        # Calculate freshness
        time_diff = int(time.time() - self.last_capture_time)
        return f"[Seen {time_diff}s ago] {self.last_description}"

    async def heartbeat_loop(self):
        print("   [System] Eco-Mode Vision Active (30s heartbeat).")
        while True:
            if self.is_active:
                # Use the appropriate prompt based on current activity
                desc = await self.capture_and_describe(is_heartbeat=True)
                if desc:
                    self.last_description = desc
            await asyncio.sleep(30)

    async def capture_and_describe(self, is_heartbeat=False, force_refresh=False):
        """Captures screen and calls Vision API. Uses activity-aware prompts."""
        if not self.client:
            return "Vision unavailable (Missing API Key)."
            return "Vision unavailable (Missing API Key)."

        try:
            # Capture and Scale based on quality mode
            # Run blocking image capture/processing in a thread
            def process_image():
                # Check for shared frame (freshness < 1.0s)
                if self.shared_frame and (time.time() - self.shared_frame_time) < 1.0:
                    img = self.shared_frame.copy() # Use the shared frame
                else:
                    img = ImageGrab.grab() # Fallback capture
                
                if self.quality_mode == "high" and not is_heartbeat:
                    img.thumbnail((1920, 1080)) # 1080p for high detail
                    quality = 85
                else:
                    img.thumbnail((1280, 720)) # 720p for fast/heartbeat
                    quality = 60
                    
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=quality)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            base64_image = await asyncio.to_thread(process_image)

            # Select prompt based on activity type
            if self.activity_type == "vn":
                prompt = self.VN_DESCRIBE_PROMPT
            elif self.activity_type == "media":
                prompt = self.MEDIA_DESCRIBE_PROMPT
            else:
                prompt = self.DESCRIBE_PROMPT

            # Auto-detect context
            if self.context_buffer.buffer and self.activity_type != "vn":
                prompt += f"\nPrevious context: {self.context_buffer.get_context_string()}"
            
            # Dynamic Detail: Forced 'high' for better cognition (as requested)
            visual_detail = "high"

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini", # Fastest vision model
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": visual_detail}}
                ]}],
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            if self.activity_type != "vn":
                self.context_buffer.add(content)
            self.last_capture_time = time.time()
            return content
        except Exception as e:
            return f"My vision is a bit glitchy: {e}"

    async def capture_vn_state(self) -> dict:
        """
        VN-specific capture: returns a structured dict with keys:
          speaker, dialogue, choices (list of strings), scene
        Returns None if capture fails or screen is transitioning.
        """
        raw = await self.capture_and_describe(is_heartbeat=False)
        if not raw or "loading screen" in raw.lower():
            return None
        try:
            result = {"speaker": "", "dialogue": "", "choices": [], "scene": ""}
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("SPEAKER:"):
                    result["speaker"] = line[len("SPEAKER:"):].strip()
                elif line.startswith("DIALOGUE:"):
                    result["dialogue"] = line[len("DIALOGUE:"):].strip()
                elif line.startswith("SCENE:"):
                    result["scene"] = line[len("SCENE:"):].strip()
                elif line.startswith("CHOICES:"):
                    choices_raw = line[len("CHOICES:"):].strip()
                    if choices_raw.lower() != "none":
                        # Parse "1. Option A, 2. Option B" or newline-separated
                        found = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', choices_raw)
                        result["choices"] = [c.strip().rstrip(',') for c in found if c.strip()]
            # Only return if we got actual dialogue
            if result["dialogue"] and result["dialogue"] != "[loading screen]":
                return result
            return None
        except Exception:
            return None
