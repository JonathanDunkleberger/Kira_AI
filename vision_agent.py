import base64
import time
import asyncio
from collections import deque
from io import BytesIO
from PIL import ImageGrab
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

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
        self.is_active = False
        self.quality_mode = "fast" # 'fast' or 'high'
        
        # Shared Buffer
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
        print("   [System] Eco-Mode Vision Active (30s).")
        while True:
            if self.is_active:
                desc = await self.capture_and_describe(is_heartbeat=True)
                if desc:
                    self.last_description = desc
            await asyncio.sleep(30)

    async def capture_and_describe(self, is_heartbeat=False, force_refresh=False):
        """Captures screen and calls API. Uses low-res detail for speed."""
        if not self.client:
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

            if is_heartbeat:
                # Universal Context Prompt
                prompt = (
                    "You are looking at Jonny's computer screen. Describe what is happening in 2 vivid sentences.\n\n"
                    "If it is a Game: Describe the action, the environment, and the current 'vibe' (Danger, Chill, Chaos).\n"
                    "If it is Video/Media: Describe the scene, the people, or the topic being discussed.\n"
                    "If it is Desktop/Code: Summarize what he is working on or point out a specific funny detail (like a weird folder name or a syntax error).\n\n"
                    "Goal: Provide enough context for Kira to ask a question or make a witty comment. Do not be generic."
                )
            else:
                prompt = (
                    "You are looking at Jonny's computer screen. Describe what is happening in 2 vivid sentences.\n\n"
                    "If it is a Game: Describe the action, the environment, and the current 'vibe' (Danger, Chill, Chaos).\n"
                    "If it is Video/Media: Describe the scene, the people, or the topic being discussed.\n"
                    "If it is Desktop/Code: Summarize what he is working on or point out a specific funny detail (like a weird folder name or a syntax error).\n\n"
                    "Goal: Provide enough context for Kira to ask a question or make a witty comment. Do not be generic."
                )
            
            # Auto-detect context
            if self.context_buffer.buffer:
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
            self.context_buffer.add(content)
            self.last_capture_time = time.time()
            return content
        except Exception as e:
            return f"My vision is a bit glitchy: {e}"
