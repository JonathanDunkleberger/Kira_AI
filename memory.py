# memory.py - Handles long-term memory using a persistent vector database.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer # Requires sentence-transformers
import os
import time # Added for timestamps
from config import MEMORY_PATH

class MemoryManager:
    def __init__(self, collection_name="conversation_memory"):
        print("-> Initializing Memory Manager...")
        
        # Ensure the memory directory exists
        if not os.path.exists(MEMORY_PATH):
            os.makedirs(MEMORY_PATH)
            
        # Use PersistentClient to save DB to disk
        self.client = chromadb.PersistentClient(
            path=MEMORY_PATH,
            settings=Settings(anonymized_telemetry=False) # Disables telemetry
        )
        
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Using a local sentence transformer model to create embeddings.
        # Run on CPU to save VRAM on the main GPU. This model will be downloaded on first run.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
        print("   Memory Manager initialized.")

    def add_memory(self, user_text: str, ai_text: str):
        """Adds a new conversation turn to the memory."""
        try:
            # Create a unique ID for each memory turn based on count
            turn_id = str(self.collection.count()) 
            user_embedding = self.embedding_model.encode(user_text).tolist()
            ai_embedding = self.embedding_model.encode(ai_text).tolist()

            self.collection.add(
                embeddings=[user_embedding, ai_embedding],
                documents=[user_text, ai_text],
                metadatas=[{"role": "user", "timestamp": time.time()}, {"role": "assistant", "timestamp": time.time()}], # Added timestamp
                ids=[f"{turn_id}_user", f"{turn_id}_ai"]
            )
            print(f"   Memory added (Turn {turn_id}).")
        except Exception as e:
            print(f"   ERROR: Failed to add memory: {e}")

    def search_memories(self, query_text: str, n_results: int = 5) -> str:
        """Searches for memories semantically similar to the query text."""
        if self.collection.count() == 0:
            return "No memories yet."
        try:
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()), # Ensure we don't request more results than exist
                include=['documents', 'metadatas'] # Ensure documents are included in results
            )
            
            # Format the results into a readable string for the AI's context
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    role = results['metadatas'][0][i].get('role', 'unknown')
                    formatted_results.append(f"[{role.capitalize()}]: {doc}") # Capitalize role for readability
            
            if formatted_results:
                # This changed from "Here are some relevant memories:" to a more neutral header
                # to prevent the AI from repeating it.
                return f"[Memory Context]:\n- " + "\n- ".join(formatted_results)
            else:
                return "No highly relevant memories found."
        except Exception as e:
            print(f"   ERROR: Failed to search memories: {e}")
            return "I had a problem searching my memory."