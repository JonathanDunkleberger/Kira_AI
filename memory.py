# memory.py - Handles long-term memory using a persistent vector database.
# This version supports storing both raw dialogue and consolidated summaries.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer  # Requires sentence-transformers
import os
import time
from config import MEMORY_PATH  # Assuming you have a config.py for your memory path

class MemoryManager:
    """
    Manages loading, adding, and searching for memories in a persistent
    ChromaDB vector database.
    """
    def __init__(self, collection_name="conversation_memory"):
        """
        Initializes the MemoryManager, setting up the database client,
        collection, and the embedding model.
        """
        print("-> Initializing Memory Manager...")

        # Ensure the persistent storage directory exists
        if not os.path.exists(MEMORY_PATH):
            os.makedirs(MEMORY_PATH)

        # Initialize the persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=MEMORY_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create the collection where memories will be stored
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Load the sentence transformer model for creating embeddings.
        # This runs on the CPU to save VRAM on the main GPU for the LLM.
        # This model will be downloaded automatically on the first run.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("   Memory Manager initialized.")

    def add_memory(self, user_text: str, ai_text: str):
        """
        Adds a raw conversation turn (user input and AI response) to the memory.
        This is for capturing the immediate, turn-by-turn context.
        """
        try:
            # Create a unique ID for the conversation turn
            turn_id = str(self.collection.count())

            # Encode both the user's and the AI's text into vector embeddings
            user_embedding = self.embedding_model.encode(user_text).tolist()
            ai_embedding = self.embedding_model.encode(ai_text).tolist()

            # Add the embeddings, documents (text), and metadata to the collection
            self.collection.add(
                embeddings=[user_embedding, ai_embedding],
                documents=[user_text, ai_text],
                metadatas=[{"role": "user", "timestamp": time.time()}, {"role": "assistant", "timestamp": time.time()}],
                ids=[f"{turn_id}_user", f"{turn_id}_ai"]
            )
            print(f"   Memory added (Turn {turn_id}).")
        except Exception as e:
            print(f"   ERROR: Failed to add raw memory: {e}")

    def add_summarized_memory(self, summary_text: str):
        """
        Adds a consolidated, third-person summary to the memory database.
        This is for storing key facts and takeaways from a conversation.
        """
        try:
            summary_id = f"summary_{self.collection.count()}"
            summary_embedding = self.embedding_model.encode(summary_text).tolist()

            self.collection.add(
                embeddings=[summary_embedding],
                documents=[summary_text],
                metadatas=[{"role": "summary", "timestamp": time.time()}],
                ids=[summary_id]
            )
            print(f"   Consolidated memory added ({summary_id}).")
        except Exception as e:
            print(f"   ERROR: Failed to add summarized memory: {e}")

    def search_memories(self, query_text: str, n_results: int = 5) -> str:
        """
        Searches for memories semantically similar to the query text and formats
        them for injection into the AI's context prompt.
        """
        if self.collection.count() == 0:
            return ""  # Return an empty string if no memories exist

        try:
            # Encode the search query into a vector embedding
            query_embedding = self.embedding_model.encode(query_text).tolist()

            # Query the collection for the most similar memories
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),  # Ensure we don't request more results than exist
                include=['documents', 'metadatas']
            )

            # Format the retrieved memories into a string for the AI's context prompt
            formatted_results = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    role = results['metadatas'][0][i].get('role', 'unknown')
                    # This format is easy for the LLM to parse
                    formatted_results.append(f"[{role.capitalize()}]: {doc}")
            
            if formatted_results:
                # *** THIS IS THE REVISED LINE ***
                # Using symbolic, non-verbal separators to prevent the AI from speaking them.
                return f"--- RELEVANT MEMORY START ---\n" + "\n- ".join(formatted_results) + "\n--- RELEVANT MEMORY END ---\n\n"
            else:
                return ""  # Return empty string if no relevant memories are found
        except Exception as e:
            print(f"   ERROR: Failed to search memories: {e}")
            return "[System Error: Could not access memory.]\n\n"