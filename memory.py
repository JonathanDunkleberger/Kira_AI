# memory.py - Handles long-term memory using a persistent vector database.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import time
import uuid
from config import (
    MEMORY_PATH, MEMORY_INCLUDE_EPISODIC, MEMORY_MAX_AGE_DAYS, 
    MEMORY_MAX_ITEMS, MEMORY_MAX_TOKENS
)
from utils_logger import logger

class MemoryManager:
    def __init__(self, collection_name="conversation_memory"):
        logger.info("-> Initializing Memory Manager...")
        if not os.path.exists(MEMORY_PATH):
            os.makedirs(MEMORY_PATH)
            
        self.client = chromadb.PersistentClient(
            path=MEMORY_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
        logger.info("   Memory Manager initialized.")

    def add_memory(self, user_text: str, ai_text: str):
        """Adds a new raw conversation turn to the memory."""
        try:
            current_time = time.time()
            self.collection.add(
                embeddings=[
                    self.embedding_model.encode(user_text).tolist(),
                    self.embedding_model.encode(ai_text).tolist()
                ],
                documents=[user_text, ai_text],
                metadatas=[
                    {"role": "user", "timestamp": current_time, "type": "turn"}, 
                    {"role": "assistant", "timestamp": current_time, "type": "turn"}
                ],
                ids=[str(uuid.uuid4()), str(uuid.uuid4())]
            )
        except Exception as e:
            logger.error(f"   ERROR: Failed to add raw memory turn: {e}")

    def add_summarized_memory(self, summary_text: str):
        """Adds a new high-level, summarized memory to the database."""
        try:
            current_time = time.time()
            self.collection.add(
                embeddings=[self.embedding_model.encode(summary_text).tolist()],
                documents=[summary_text],
                metadatas=[{"role": "summary", "timestamp": current_time, "type": "summary"}],
                ids=[str(uuid.uuid4())]
            )
            logger.info(f"   ✅ Consolidated Memory Added: '{summary_text}'")
        except Exception as e:
            logger.error(f"   ERROR: Failed to add summarized memory: {e}")

    def search_memories(self, query_text: str, n_results: int = 5) -> str:
        """Searches for memories semantically similar to the query text."""
        if self.collection.count() == 0:
            return "No memories yet."
        try:
            # Enforce max items via config
            n_results = min(n_results, MEMORY_MAX_ITEMS, self.collection.count())
            
            # Use where clause to filter by type if episodic memory is disabled
            where_filter = {}
            if not MEMORY_INCLUDE_EPISODIC:
                where_filter = {"type": "summary"}

            results = self.collection.query(
                query_embeddings=[self.embedding_model.encode(query_text).tolist()],
                n_results=n_results,
                where=where_filter,
                include=['documents', 'metadatas']
            )
            
            formatted_results = []
            total_chars = 0
            
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    
                    # Recency check (optional, but good for filtering old noise)
                    # timestamp = meta.get('timestamp', 0)
                    # if (time.time() - timestamp) > (MEMORY_MAX_AGE_DAYS * 86400):
                    #     continue

                    # Soft Token Limit Check (approx 4 chars per token)
                    if (total_chars / 4) > MEMORY_MAX_TOKENS:
                        break

                    if meta.get('type') == 'summary':
                        formatted_results.append(f"- {doc}")
                        total_chars += len(doc)
                    elif MEMORY_INCLUDE_EPISODIC:
                        role = meta.get('role', 'unknown').capitalize()
                        formatted_results.append(f"-[Log: {role} says]: {doc}")
                        total_chars += len(doc)
            
            return "\n".join(formatted_results) if formatted_results else "No highly relevant memories found."
        except Exception as e:
            logger.error(f"   ERROR: Failed to search memories: {e}")
            return "I had a problem searching my memory."