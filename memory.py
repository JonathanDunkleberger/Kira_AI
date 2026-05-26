# memory.py - Handles long-term memory using a persistent vector database.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import torch
import time
import uuid
import hashlib
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
        self.chatters = self.client.get_or_create_collection(name="chatters")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self._validate_collections()
        print("   Memory Manager initialized.")

    def _validate_collections(self):
        """Startup health-check: exercise BOTH the metadata segment (.get) AND
        the HNSW vector segment (.query) on each non-empty collection. The
        vector segment is what actually breaks when index files don't get
        persisted, and a plain .get() will not catch that — so we issue a real
        vector query with a zero-vector probe. We don't care about the hit
        contents, only whether the query succeeds.
        Loud warnings here so a broken-index state is caught at launch instead
        of failing silently mid-stream."""
        # 384 matches the all-MiniLM-L6-v2 embedding dim used elsewhere.
        probe_dim = 384
        try:
            probe = self.embedding_model.encode("healthcheck").tolist()
            probe_dim = len(probe)
        except Exception:
            probe = [0.0] * probe_dim

        for name, coll in [
            ("turns",    self.turns),
            ("facts",    self.facts),
            ("chatters", self.chatters),
        ]:
            # 1) Metadata segment check
            try:
                coll.get(limit=1)
            except Exception as e:
                print(f"   [Memory] WARNING: ChromaDB '{name}' metadata segment unreadable: {e}")
                print(f"   [Memory] WARNING: Memory may be corrupted. "
                      f"Restore from backup or delete: {MEMORY_PATH}")
                continue
            # 2) Vector segment check (only meaningful if there's data to index)
            try:
                count = coll.count()
            except Exception:
                count = 0
            if count == 0:
                continue
            try:
                coll.query(query_embeddings=[probe], n_results=1)
            except Exception as e:
                print(f"   [Memory] WARNING: ChromaDB '{name}' HNSW vector index unreadable: {e}")
                print(f"   [Memory] WARNING: '{name}' has {count} rows in sqlite but the vector "
                      f"index is broken/missing. Run: python repair_memory_db.py")

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
                    
                    # New extractor provides a ready-to-store natural-language fact; old extractor used subject/predicate/object
                    if "fact" in mem and isinstance(mem["fact"], str) and mem["fact"].strip():
                        text_rep = mem["fact"].strip()
                    else:
                        text_rep = f"{subject}'s {predicate.replace('_', ' ').strip()} is {obj.replace('_', ' ').strip()}."
                    
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
        
        # 1. FACTS: Strict "Truths" about Jonny
        # Search specifically for high-confidence facts
        hits = self.search_facts(query_text, n_results=3)
        if hits:
            # Remove duplicates and format
            unique_hits = list(set(hits)) 
            context_lines.append(f"[KNOWN FACTS]: {'; '.join(unique_hits)}")
        
        # 2. PROJECTS: What is he working on?
        # Only fetch if the user mentions "project", "code", "work", etc.
        if any(w in query_text.lower() for w in ["project", "code", "working", "build"]):
            try:
                # Optimized query for project type
                proj_res = self.facts.get(where={"type": "project"}, limit=1)
                if proj_res['documents']:
                    context_lines.append(f"[CURRENT PROJECT]: {proj_res['documents'][0]}")
            except Exception: pass

        return "\n".join(context_lines)

    def upsert_fact(self, key: str, value: str):
        """Deterministically upserts a structured fact."""
        fact_text = f"Jonny's {key.replace('_', ' ').strip()} is {value.replace('_', ' ').strip()}."
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
        except Exception:
            # Fallback: Delete then Add
            try: self.facts.delete(ids=[fact_id]); 
            except Exception: pass
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
        except Exception:
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
            print(f"   [Memory] ChromaDB read failed "
                  f"(collection='facts', path='{MEMORY_PATH}'): {e}")
            return []

    def add_highlight(self, activity: str, highlight: str, kira_take: str = ""):
        """Stores a memorable moment from a media session. Specific, with character names
        and plot detail — not vague vibes."""
        try:
            text = f"During {activity}: {highlight}"
            if kira_take:
                text += f" (Kira's take: {kira_take})"
            meta = {
                "type": "highlight",
                "activity": activity,
                "timestamp": time.time(),
                "source": "scene_extractor",
            }
            emb = self.embedding_model.encode(text).tolist()
            self.facts.add(
                ids=[str(uuid.uuid4())],
                embeddings=[emb],
                documents=[text],
                metadatas=[meta],
            )
            print(f"   [Memory] Highlight stored: {text[:80]}...")
        except Exception as e:
            print(f"   [Memory Error] Highlight storage failed: {e}")

    def add_session_summary(self, activity: str, summary: str):
        """Stores a session-end summary as a durable memory."""
        try:
            text = f"Session recap ({activity}): {summary}"
            meta = {
                "type": "session_summary",
                "activity": activity,
                "timestamp": time.time(),
                "source": "session_summarizer",
            }
            emb = self.embedding_model.encode(text).tolist()
            self.facts.add(
                ids=[str(uuid.uuid4())],
                embeddings=[emb],
                documents=[text],
                metadatas=[meta],
            )
            print(f"   [Memory] Session summary stored ({activity}).")
        except Exception as e:
            print(f"   [Memory Error] Session summary storage failed: {e}")

    # ── Per-Chatter Memory ────────────────────────────────────────────────────

    def record_chatter_message(self, username: str, platform: str, message: str):
        """Logs a single chat message from a viewer. Lightweight — fires every message."""
        try:
            text = f"{username} ({platform}): {message}"
            meta = {
                "type": "chatter_message",
                "username": username,
                "platform": platform,
                "timestamp": time.time(),
            }
            emb = self.embedding_model.encode(message).tolist()
            self.chatters.add(
                ids=[str(uuid.uuid4())],
                embeddings=[emb],
                documents=[text],
                metadatas=[meta],
            )
        except Exception as e:
            print(f"   [Memory] chatter message log failed: {e}")

    def store_chatter_fact(self, username: str, platform: str, fact: str, tone: str = ""):
        """Stores a durable fact about a chatter (opinions, preferences, callbacks).
        Upserts on a deterministic ID so we don't get duplicates of similar facts.
        Optional tone tag describes how this chatter tends to interact."""
        try:
            text = f"{username}: {fact}"
            fact_id = f"chatter::{username.lower()}::{hashlib.md5(fact.encode()).hexdigest()[:12]}"
            meta = {
                "type": "chatter_fact",
                "username": username,
                "platform": platform,
                "timestamp": time.time(),
            }
            if tone:
                meta["tone"] = tone
            emb = self.embedding_model.encode(text).tolist()
            self.chatters.upsert(
                ids=[fact_id],
                embeddings=[emb],
                documents=[text],
                metadatas=[meta],
            )
            tone_str = f" [{tone}]" if tone else ""
            print(f"   [ChatterMem] Stored: {username} — {fact[:60]}{tone_str}")
        except Exception as e:
            print(f"   [Memory] chatter fact storage failed: {e}")

    def get_chatter_context(self, username: str, n_results: int = 5) -> str:
        """Returns a formatted string of what Kira knows about this chatter.
        Used when a chatter speaks — gives Kira recall."""
        try:
            results = self.chatters.get(
                where={"username": username},
                limit=50,
                include=["documents", "metadatas"],
            )
            if not results or not results.get("documents"):
                return ""

            facts = []
            recent_msgs = []
            tones = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                if meta.get("type") == "chatter_fact":
                    facts.append(doc)
                    if meta.get("tone"):
                        tones.append(meta["tone"])
                else:
                    recent_msgs.append((meta.get("timestamp", 0), doc))

            recent_msgs.sort(reverse=True)
            recent_msgs = [doc for ts, doc in recent_msgs[:n_results]]

            lines = []
            if facts:
                lines.append(f"What you know about {username}: " + " | ".join(facts[:5]))
            if tones:
                # Most recently stored tone wins
                lines.append(f"General vibe: {tones[-1]}")
            if recent_msgs:
                lines.append(f"Recent {username} messages: " + " | ".join(reversed(recent_msgs)))
            return "\n".join(lines)
        except Exception as e:
            print(f"   [Memory] chatter context failed: {e}")
            return ""

    def is_first_time_chatter(self, username: str) -> bool:
        """Returns True if this username has never been seen before."""
        try:
            results = self.chatters.get(
                where={"username": username},
                limit=1,
            )
            return not results or not results.get("documents")
        except Exception:
            return False

    def count_chatter_messages(self, username: str) -> int:
        """Return total historical message count for this chatter across all sessions."""
        try:
            results = self.chatters.get(
                where={"username": username},
            )
            return len(results.get("ids", []))
        except Exception as e:
            print(f"   [Memory] count_chatter_messages failed: {e}")
            return 0

    def get_recent_chatters(self, days: int = 14, limit: int = 20) -> list:
        """Returns usernames seen in chat within the last N days, ranked by activity."""
        try:
            cutoff = time.time() - (days * 86400)
            results = self.chatters.get(
                where={"type": "chatter_message"},
                limit=2000,
                include=["metadatas"],
            )
            if not results or not results.get("metadatas"):
                return []
            counts = {}
            for meta in results["metadatas"]:
                ts = meta.get("timestamp", 0)
                if ts < cutoff:
                    continue
                uname = meta.get("username")
                if not uname:
                    continue
                counts[uname] = counts.get(uname, 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            return [name for name, _ in ranked[:limit]]
        except Exception as e:
            print(f"   [Memory] recent chatters lookup failed: {e}")
            return []

    def get_last_session_summary(self) -> str | None:
        """Returns the most recent session summary stored in memory."""
        try:
            results = self.facts.get(
                where={"type": "session_summary"},
                limit=50,
                include=["documents", "metadatas"],
            )
            if not results or not results.get("documents"):
                return None
            zipped = list(zip(results["documents"], results["metadatas"]))
            zipped.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
            return zipped[0][0]
        except Exception:
            return None
