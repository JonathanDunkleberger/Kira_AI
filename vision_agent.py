import time
import threading
import mss
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import VISION_POLL_INTERVAL, VISION_MODEL_PATH

class VisionAgent:
    def __init__(self):
        self.latest_frame = None
        self.latest_description = "Vision is initializing..."
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.model = None
        self.tokenizer = None

        print(f"   [Vision] Loading Moondream model from {VISION_MODEL_PATH}...")
        try:
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                VISION_MODEL_PATH, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cuda",
                local_files_only=True 
            )
            self.tokenizer = AutoTokenizer.from_pretrained(VISION_MODEL_PATH)
            print("   [Vision] Model loaded successfully.")
        except Exception as e:
            print(f"   [Vision] CRITICAL LOAD ERROR: {e}")
            self.latest_description = f"My eyes are broken: {e}"

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.thread.start()
        print("   [Vision] Agent Started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("   [Vision] Agent Stopped.")

    def get_latest_description(self):
        """Thread-safe getter for bot.py to use."""
        with self.lock:
            return self.latest_description

    def _vision_loop(self):
        while self.running:
            try:
                # 1. Capture
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    sct_img = sct.grab(monitor)
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    img_resized = img.resize((512, 512))
                    
                    with self.lock:
                        self.latest_frame = img_resized

                # 2. Analyze (Only if model exists)
                if self.model:
                    enc_image = self.model.encode_image(img_resized)
                    description = self.model.answer_question(enc_image, "Describe this image briefly.", self.tokenizer)
                    
                    with self.lock:
                        self.latest_description = description
                
                time.sleep(VISION_POLL_INTERVAL)

            except Exception as e:
                # Silent fail in loop to keep bot alive
                time.sleep(5)
