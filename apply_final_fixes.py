
import os

files = {
    "memory.py": r'''# memory.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import time
import uuid
from config import MEMORY_PATH

class MemoryManager:
    def __init__(self):
        if not os.path.exists(MEMORY_PATH): os.makedirs(MEMORY_PATH)
        
        # Initialize Database
        self.client = chromadb.PersistentClient(path=MEMORY_PATH)
        self.turns = self.client.get_or_create_collection(name="turns")
        self.facts = self.client.get_or_create_collection(name="facts")
        
        # Initialize Embedding Model (CPU optimized)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("   [System] Memory Manager Initialized.")

    def add_turn(self, user_text, ai_text, source="voice"):
        """Saves the interaction to the database for short-term recall."""
        try:
            timestamp = time.time()
            # Embed both user and AI text
            emb_user = self.embedding_model.encode(user_text).tolist()
            emb_ai = self.embedding_model.encode(ai_text).tolist()

            self.turns.add(
                embeddings=[emb_user, emb_ai],
                documents=[user_text, ai_text],
                metadatas=[
                    {"role": "user", "timestamp": timestamp, "source": source},
                    {"role": "assistant", "timestamp": timestamp, "source": "ai"}
                ],
                ids=[str(uuid.uuid4()), str(uuid.uuid4())]
            )
        except Exception as e:
            print(f"   [Memory Error] Failed to save turn: {e}")

    def get_semantic_context(self, query_text):
        """Retrieves relevant past conversation or facts."""
        try:
            if self.facts.count() == 0: return ""
            
            # Semantic Search
            results = self.facts.query(
                query_embeddings=[self.embedding_model.encode(query_text).tolist()],
                n_results=3
            )
            
            context = []
            if results and results['documents']:
                for doc in results['documents'][0]:
                    context.append(f"- {doc}")
            
            return "\n".join(context) if context else ""
        except Exception as e:
            # Fallback for index errors (common in ChromaDB on reload)
            print(f"   [Memory Error] Retrieval failed: {e}")
            return ""

    def add_fact(self, text):
        """Stores a permanent fact (from Summarizer)."""
        try:
            self.facts.add(
                embeddings=[self.embedding_model.encode(text).tolist()],
                documents=[text],
                ids=[str(uuid.uuid4())]
            )
            print(f"   [Memory] Stored Fact: {text}")
        except Exception as e:
            print(f"   [Memory Error]: {e}")
''',
    "vision_agent.py": r'''# vision_agent.py
import base64
import time
import asyncio
from io import BytesIO
from PIL import ImageGrab
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

class UniversalVisionAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.last_description = ""
        self.last_capture_time = 0
        self.is_active = False
        self.shared_frame = None # From dashboard

    def update_shared_frame(self, frame):
        self.shared_frame = frame

    def get_vision_context(self):
        if not self.last_description: return ""
        age = int(time.time() - self.last_capture_time)
        return f"[Visuals ({age}s ago)]: {self.last_description}"

    async def heartbeat_loop(self):
        print("   [System] Vision Heartbeat Active.")
        while True:
            if self.is_active:
                await self.capture_and_describe()
            await asyncio.sleep(15) 

    async def capture_and_describe(self, is_heartbeat=True):
        if not self.client: return "Vision Disabled."
        try:
            # Capture Frame
            if self.shared_frame: img = self.shared_frame.copy()
            else: img = ImageGrab.grab()
            
            # Resize for Speed (720p is plenty for context)
            img.thumbnail((1280, 720))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=60)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Prompt: Opinionated but Simple
            prompt = (
                "Identify the most interesting or notable thing on this screen in 1 sentence. "
                "If it's a game, comment on the action, health, or environment. "
                "If it's text, summarize the topic. Be concise."
            )

            resp = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}],
                max_tokens=100
            )
            self.last_description = resp.choices[0].message.content
            self.last_capture_time = time.time()
            return self.last_description
        except Exception as e:
            return f"Vision Glitch: {e}"
'''
}

for filename, content in files.items():
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully wrote {filename}")
