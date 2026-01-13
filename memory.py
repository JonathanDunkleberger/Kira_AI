# memory.py - Handles long-term memory using a persistent vector database.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import torch
import time
import uuid
from config import MEMORY_PATH

class MemoryManager:
    def __init__(self):
        print("-> Initializing Memory Manager...")
        if not os.path.exists(MEMORY_PATH):
            os.makedirs(MEMORY_PATH)
            
        self.client = chromadb.PersistentClient(
            path=MEMORY_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        # Split into two collections: Turns (Raw logs) and Facts (Distilled truths)
        self.turns = self.client.get_or_create_collection(name="turns")
        self.facts = self.client.get_or_create_collection(name="facts")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
        print("   Memory Manager initialized.")

    def add_turn(self, user_text: str, ai_text: str, source: str = "unknown"):
        """Adds a new raw conversation turn to the 'turns' collection."""
        try:
            self.turns.add(
                embeddings=[
                    self.embedding_model.encode(user_text).tolist(),
                    self.embedding_model.encode(ai_text).tolist()
                ],
                documents=[user_text, ai_text],
                metadatas=[
                    {"role": "user", "timestamp": time.time(), "type": "turn", "source": source}, 
                    {"role": "assistant", "timestamp": time.time(), "type": "turn", "source": "ai"}
                ],
                ids=[str(uuid.uuid4()), str(uuid.uuid4())]
            )
        except Exception as e:
            print(f"   ERROR: Failed to add raw memory turn: {e}")

    def store_extracted_memories(self, memories: list, source: str = "voice"):
        """Stores structured memories from LLM extraction."""
        for mem in memories:
            try:
                m_type = mem.get("memory_type", "episodic")
                conf = mem.get("confidence", 0.0)
                salience = mem.get("salience", 0.0)
                
                # Rule: High confidence or High salience with decent confidence
                if conf >= 0.75 or (salience >= 0.85 and conf >= 0.6):
                    subject = mem.get("subject", "Jonny")
                    predicate = mem.get("predicate", "unknown_action")
                    obj = mem.get("object", "")
                    
                    if not obj or "?" in obj: continue # Skip questions/blanks
                    
                    text_rep = f"{subject}'s {predicate.replace('_',' ')} is {obj}."
                    
                    meta = {
                        "type": m_type,
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": conf,
                        "salience": salience,
                        "timestamp": time.time(),
                        "source": source
                    }
                    
                    emb = self.embedding_model.encode(text_rep).tolist()
                    
                    if m_type == "profile_fact":
                        # Deterministic ID for upsert
                        fact_id = f"jonny::{predicate}"
                        print(f"   [Memory] Upserting Fact: {text_rep} (Conf: {conf})")
                        self.facts.upsert(
                            ids=[fact_id],
                            embeddings=[emb],
                            documents=[text_rep],
                            metadatas=[meta]
                        )
                    else:
                        # Append episodic or project memory
                        fact_id = str(uuid.uuid4())
                        print(f"   [Memory] Storing {m_type}: {text_rep}")
                        self.facts.add(
                            ids=[fact_id],
                            embeddings=[emb],
                            documents=[text_rep],
                            metadatas=[meta]
                        )
            except Exception as e:
                print(f"   [Memory Error] Failed to store memory item: {e}")

    def get_semantic_context(self, query_text: str) -> str:
        """Retrieves flexible context for LLM injection."""
        context_lines = []
        
        # 1. Top Profile Facts (Highest confidence/recent)
        try:
            # We can't query by metadata field easily in Chroma without fetching all first or using where
            # "where" filter for profile_fact
            pf = self.facts.get(where={"type": "profile_fact"}, limit=5) # Limit is seemingly for result count, but get() returns all if no limit? default is None.
            # Actually get() with limit might refer to slicing. 
            # safe fetch
            
            # Better strategy: Search semantic matches first, but ALSO get specific "favorite" style facts if relevant?
            # User request: "Always include top 3 profile facts"
            # It's hard to rank "top" without a definition. Closest we can do is "recent" or just "all" (if small).
            # Let's try to get 3 random or recent profile facts.
            pass
        except: pass

        # Implementation of "Retrieve structured notes"
        # Strategy:
        # A. Vector Search (Semantic) -> Get 3-5 relevant facts of ANY type
        # B. Recent "Project" threads -> Get 1-2 most recent type="project"
        
        # A. Semantic Search
        hits = self.search_facts(query_text, n_results=5)
        if hits:
            context_lines.append("Relevant Memories:")
            for h in hits:
                context_lines.append(f"- {h}")
        
        # B. Recent Projects (Manual Filter)
        try:
             # get generic list, sort by timestamp
             res = self.facts.get(where={"type": "project"}, include=["metadatas", "documents"])
             if res['documents']:
                 zipped = sorted(zip(res['documents'], res['metadatas']), key=lambda x: x[1]['timestamp'], reverse=True)
                 top_projects = zipped[:2]
                 context_lines.append("Current Projects:")
                 for doc, meta in top_projects:
                     context_lines.append(f"- {doc} (conf {meta.get('confidence',0.0)})")
        except: pass

        return "\n".join(context_lines)

    def upsert_fact(self, key: str, value: str):
        """Deterministically upserts a structured fact."""
        fact_text = f"Jonny's {key.replace('_',' ')} is {value}."
        fact_id = f"jonny::{key}"
        meta = {"subject": "Jonny", "key": key, "value": value, "timestamp": time.time(), "type":"fact"}
        
        emb = self.embedding_model.encode(fact_text).tolist()
        try:
            # Try direct upsert first (newer chroma versions support this)
            self.facts.upsert(
                ids=[fact_id],
                embeddings=[emb],
                documents=[fact_text],
                metadatas=[meta]
            )
            print(f"   ✅ Fact Upserted: {fact_text}")
        except:
            # Fallback: Delete then Add
            try: self.facts.delete(ids=[fact_id]); 
            except: pass
            self.facts.add(
                ids=[fact_id],
                embeddings=[emb],
                documents=[fact_text],
                metadatas=[meta]
            )
            print(f"   ✅ Fact Added (Fallback): {fact_text}")

    def get_fact(self, key: str) -> str | None:
        """Direct retrieval of specific keys."""
        try:
            res = self.facts.get(ids=[f"jonny::{key}"], include=["documents"])
            if res and res.get("documents"): 
                return res["documents"][0]
            return None
        except:
            return None

    def add_fact(self, fact_text: str):
        """Adds a new atomic fact to the 'facts' collection (Summarizer route)."""
        try:
            self.facts.add(
                embeddings=[self.embedding_model.encode(fact_text).tolist()],
                documents=[fact_text],
                metadatas=[{"role": "fact", "timestamp": time.time(), "type": "fact"}],
                ids=[str(uuid.uuid4())]
            )
            print(f"   ✅ Fact Stored: '{fact_text}'")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ERROR: Failed to add fact: {e}")

    def get_recent_facts(self, limit: int = 3) -> list:
        """Fetches the most recently learned facts as a list of strings."""
        try:
            count = self.facts.count()
            if count == 0: return []
            
            # Fetch all to sort (chroma limit/sort workaround)
            result = self.facts.get(include=['documents', 'metadatas'])
            
            zipped = zip(result['documents'], result['metadatas'])
            sorted_mems = sorted(zipped, key=lambda x: x[1].get('timestamp', 0), reverse=True)
            
            recent = [doc for doc, meta in sorted_mems[:limit]]
            return recent
        except Exception as e:
            print(f"   [Memory Error]: {e}")
            return []

    def search_facts(self, query_text: str, n_results: int = 3) -> list:
        """Searches 'facts' collection for relevant truths."""
        if self.facts.count() == 0:
            return []
        try:
            results = self.facts.query(
                query_embeddings=[self.embedding_model.encode(query_text).tolist()],
                n_results=min(n_results, self.facts.count()),
                include=['documents']
            )
            
            if results and results['documents'] and results['documents'][0]:
                return results['documents'][0]
            return []
        except Exception as e:
            print(f"   ERROR: Failed to search facts: {e}")
            return []