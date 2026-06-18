# memory.py - Handles long-term memory using a persistent vector database.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import re
import torch
import time
import uuid
import hashlib
from kira.config import MEMORY_PATH
from kira.memory.identity_manager import normalize_chatter_key

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
        # Short-lived cache for the full facts.get() document scan used by
        # _direct_fact_lookup(). Facts never change mid-session (writes are rare
        # background events), so re-fetching the entire table on every voice turn
        # is pure waste. Cache expires after 30s and is immediately invalidated
        # whenever a new fact is written so freshness is always guaranteed.
        self._facts_cache: list = []
        self._facts_cache_ts: float = 0.0
        self._FACTS_CACHE_TTL: float = 30.0
        self._validate_collections()
        self._audit_chatter_key_normalization()
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
                    self._invalidate_facts_cache()
            except Exception as e:
                print(f"   [Memory Error] Failed to store memory item: {e}")

    # Topic-keyword → substring probes for deterministic recall of high-signal
    # facts (cats, family, location, etc.). When the user query contains any
    # of these triggers, scan the facts collection for documents containing
    # the corresponding substring(s) and surface them as [KNOWN FACTS (direct)]
    # ABOVE the vector hits. This guarantees that name-recall queries ("what
    # are my cats' names") never miss because of embedding similarity ranking.
    _DIRECT_FACT_TRIGGERS = [
        # (regex matched against lowercased query, list of substrings to scan documents for)
        (r"\bcats?\b|\bkitt(?:y|ies|en)\b|\bpets?\b",
            ["Cartofell", "Madoka", " cats", "two cats"]),
        (r"\bcartofell\b",            ["Cartofell"]),
        (r"\b(?:madoka)\b",            ["Madoka"]),
        (r"\b(?:girlfriend|partner|wife|fianc)\b",
            ["girlfriend", "partner", "fianc"]),
        (r"\blondon\b|\bmov(?:e|ing)\b",
            ["London", "move", "moving"]),
        (r"\bfavou?rite\s+(?:anime|character|game|show|vn|visual novel)\b",
            ["favorite"]),
        (r"\bjob\b|\bwork(?:ing)?\b|\bcareer\b",
            ["job", "work"]),
        (r"\bname[sd]?\b",
            []),  # bare "names?" alone is too noisy — leave list empty; here as a marker
    ]

    def _invalidate_facts_cache(self):
        """Drop the cached facts document list. Called whenever a fact is written."""
        self._facts_cache = []
        self._facts_cache_ts = 0.0

    def _get_all_fact_docs(self) -> list:
        """Return all documents from the facts collection, using a 30s in-memory
        cache. The full table scan + ChromaDB/SQLite I/O happens at most once per
        30 seconds regardless of how many concurrent voice turns fire."""
        now = time.time()
        if self._facts_cache and (now - self._facts_cache_ts) < self._FACTS_CACHE_TTL:
            return self._facts_cache
        try:
            data = self.facts.get(include=["documents"])
            self._facts_cache = data.get("documents") or []
            self._facts_cache_ts = now
        except Exception as e:
            print(f"   [Memory] facts.get() cache refresh failed: {e}")
            self._facts_cache = []
        return self._facts_cache

    def _direct_fact_lookup(self, query_text: str, limit: int = 6) -> list:
        """Substring-scan the facts collection for documents matching topic-keyword
        triggers in the query. Bypasses vector similarity for deterministic recall
        of high-confidence stored facts (Fix 2). Uses cached docs to avoid a full
        SQLite scan on every call."""
        import re as _re
        ql = (query_text or "").lower()
        substrings: list[str] = []
        for pattern, subs in self._DIRECT_FACT_TRIGGERS:
            if subs and _re.search(pattern, ql):
                substrings.extend(subs)
        if not substrings:
            return []
        # Dedupe substrings preserving order
        seen_sub = set()
        substrings = [s for s in substrings if not (s.lower() in seen_sub or seen_sub.add(s.lower()))]
        docs = self._get_all_fact_docs()
        out: list[str] = []
        seen_docs = set()
        for doc in docs:
            if not doc:
                continue
            dl = doc.lower()
            if any(s.lower() in dl for s in substrings):
                if doc not in seen_docs:
                    out.append(doc)
                    seen_docs.add(doc)
                    if len(out) >= limit:
                        break
        return out

    def get_semantic_context(self, query_text: str) -> str:
        """Retrieves flexible context for LLM injection.

        Fix 1: hits are presented as individual bullets, NOT joined with '; '
               (which mangled multi-line compound documents into run-on text and
               obscured the fact boundaries). Order is preserved (no set()).
        Fix 2: a direct keyword-triggered substring probe runs FIRST for
               high-signal topics (cats, family, location, favorites) so
               name-recall is deterministic instead of probabilistic.
        """
        context_lines = []

        # 0. DIRECT FACTS — deterministic keyword probe (Fix 2)
        direct = self._direct_fact_lookup(query_text)
        if direct:
            block = "[KNOWN FACTS (direct)]:\n" + "\n".join(f"- {d}" for d in direct)
            context_lines.append(block)

        # 1. FACTS: Strict "Truths" about Jonny (vector search)
        hits = self.search_facts(query_text, n_results=3)
        if hits:
            # Preserve ranking order; only drop exact dupes against direct block.
            already = {d for d in direct}
            ordered_unique = []
            seen = set()
            for h in hits:
                if h in already or h in seen:
                    continue
                ordered_unique.append(h)
                seen.add(h)
            if ordered_unique:
                block = "[KNOWN FACTS]:\n" + "\n".join(f"- {h}" for h in ordered_unique)
                context_lines.append(block)

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
            self._invalidate_facts_cache()
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
            self._invalidate_facts_cache()
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
            self._invalidate_facts_cache()
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
            self._invalidate_facts_cache()
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
            self._invalidate_facts_cache()
            print(f"   [Memory] Session summary stored ({activity}).")
        except Exception as e:
            print(f"   [Memory Error] Session summary storage failed: {e}")

    # ── Per-Chatter Memory ────────────────────────────────────────────────────

    def _audit_chatter_key_normalization(self) -> None:
        """Constraint #3: loudly flag stored chatter keys NOT already in normalized
        form (pre-migration rows). Read-only — never mutates. While such rows exist,
        recall for those keys will MISS until the one-time chatter-key migration runs;
        surfacing it at boot means that degraded state is never silent."""
        try:
            rows = self.chatters.get(include=["metadatas"])
            metas = rows.get("metadatas", []) or []
            bad = set()
            for m in metas:
                u = m.get("username")
                if u is not None and u != normalize_chatter_key(u):
                    bad.add(u)
            if bad:
                _sample = sorted(bad)[:8]
                print(f"   [Memory] ⚠ chatter-key audit: {len(bad)} stored key(s) are "
                      f"un-normalized (pre-migration) — recall will MISS these until the "
                      f"chatter-key migration runs. Examples: {_sample}")
            else:
                print(f"   [Memory] chatter-key audit: all {len(metas)} chatter rows normalized.")
        except Exception as e:
            print(f"   [Memory] chatter-key audit skipped (non-fatal): {e}")

    def record_chatter_message(self, username: str, platform: str, message: str):
        """Logs a single chat message from a viewer. Lightweight — fires every message."""
        try:
            text = f"{username} ({platform}): {message}"
            meta = {
                "type": "chatter_message",
                # Normalized key for cross-platform recall (strip @, lowercase, trim).
                # Raw handle stays in the `documents` text above for display.
                "username": normalize_chatter_key(username),
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
            fact_id = f"chatter::{normalize_chatter_key(username)}::{hashlib.md5(fact.encode()).hexdigest()[:12]}"
            meta = {
                "type": "chatter_fact",
                # Normalized key (strip @, lowercase, trim); raw handle kept in text.
                "username": normalize_chatter_key(username),
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

    def store_chatter_preferred_name(self, username: str, platform: str, preferred_name: str):
        """Persist the name a chatter wants to be called ('just call me TOOT').

        Additive: a single deterministic-ID row (type='chatter_preferred_name')
        upserted per chatter — never duplicates, never alters the normalized
        'username' lookup key, no migration (Chroma metadata is schemaless per
        record; this mirrors the optional 'tone' key). A later genuine declaration
        overwrites it. Purely the DISPLAY/spoken name; storage keys stay the handle."""
        try:
            pname = (preferred_name or "").strip()
            key = normalize_chatter_key(username)
            if not key or not pname:
                return
            text = f"{username} prefers to be called {pname}"
            pref_id = f"chatter::{key}::preferred_name"
            meta = {
                "type": "chatter_preferred_name",
                "username": key,                 # SAME normalized key — never a new lookup field
                "platform": platform,
                "preferred_name": pname,
                "timestamp": time.time(),
            }
            emb = self.embedding_model.encode(text).tolist()
            self.chatters.upsert(
                ids=[pref_id],
                embeddings=[emb],
                documents=[text],
                metadatas=[meta],
            )
            print(f"   [ChatterMem] Preferred name set: {key} → '{pname}'")
        except Exception as e:
            print(f"   [Memory] chatter preferred-name storage failed: {e}")

    def get_chatter_preferred_name(self, username: str):
        """Return the chatter's stated preferred name, or None. Direct O(1) lookup
        on the deterministic preferred-name row; never raises into the caller."""
        try:
            res = self.chatters.get(
                ids=[f"chatter::{normalize_chatter_key(username)}::preferred_name"],
                include=["metadatas"],
            )
            metas = res.get("metadatas") if res else None
            if metas:
                return (metas[0] or {}).get("preferred_name") or None
        except Exception as e:
            print(f"   [Memory] chatter preferred-name lookup failed: {e}")
        return None

    # ── Diversity-first fact surfacing (READ-PATH ONLY — zero writes) ───────────
    # Stopwords stripped before topic-clustering facts. Includes fact-boilerplate
    # ("known", "fact", "referred") so clustering keys on the actual subject.
    _FACT_STOPWORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
        "been", "being", "to", "of", "in", "on", "at", "for", "with", "by", "as",
        "that", "this", "it", "its", "they", "them", "their", "he", "she", "his",
        "her", "you", "your", "i", "me", "my", "we", "our", "kira", "known",
        "has", "have", "had", "who", "which", "about", "into", "from", "fact",
        "referred", "refers", "calls", "called", "call", "said", "says", "say",
        "like", "likes", "liked", "very", "really", "just", "also", "still",
        "always", "often", "tends", "tend", "thing", "things", "themselves",
        # Generic identity-category nouns — they bridge UNRELATED topics ("banana
        # person" vs "earth crater person"), so excluding them lets clustering key
        # on the real subject ("banana") instead of fragmenting/over-merging on them.
        "person", "people", "guy", "guys", "dude", "dudes", "man", "men", "woman",
        "women", "girl", "boy", "one", "ones", "someone", "somebody", "kind",
        "type", "sort", "stuff", "name", "named",
    })

    def _fact_tokens(self, doc: str, username: str) -> set:
        """Content tokens of a fact doc minus the 'username:' prefix and stopwords.
        Read-only helper for topic clustering — no storage interaction."""
        body = doc or ""
        # Strip the leading "{handle}: " prefix robustly. The stored doc uses the
        # RAW handle ("@RunTimeRiot: …"), NOT the normalized key, so a fixed-string
        # strip misses and the name contaminates every fact (→ all facts collapse to
        # one cluster). Split on the first ": " instead.
        if ": " in body:
            body = body.split(": ", 1)[1]
        _key = normalize_chatter_key(username)
        out = set()
        for t in re.findall(r"[a-z0-9']+", body.lower()):
            t = t.strip("'")  # drop surrounding apostrophes/quotes ('banana ≠ banana)
            if len(t) <= 2 or t in self._FACT_STOPWORDS or t == _key:
                continue
            # Light singular/plural stem so morphological variants of the SAME topic
            # cluster together (banana/bananas, joke/jokes) instead of fragmenting.
            if len(t) > 4 and t.endswith("es"):
                t = t[:-2]
            elif len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]
            if len(t) > 2:
                out.add(t)
        return out

    def _diverse_facts(self, fact_items: list, username: str, limit: int) -> list:
        """Pick up to `limit` DISTINCT facts from (timestamp, doc) tuples.

        Collapses near-duplicate paraphrases of one trait (chatter-local
        dominant-token clustering) to a single representative, then ranks clusters
        by DURABILITY (how many rows reinforce the trait) so an enduring fact
        surfaces even if old, with RECENCY as the tiebreak. Each surviving cluster
        contributes one line; the representative is the cluster's freshest phrasing.

        ZERO WRITES — pure in-memory read-time selection. The stored rows (incl. the
        143 'banana' variants) are untouched; only what gets SURFACED changes."""
        if not fact_items:
            return []
        toks = [self._fact_tokens(doc, username) for _ts, doc in fact_items]
        # Chatter-local token frequency: a token recurring across many of THIS
        # chatter's facts marks a calcified topic (e.g. 'banana' in 143 rows).
        freq: dict = {}
        for tset in toks:
            for t in tset:
                freq[t] = freq.get(t, 0) + 1
        # Topic key per fact = its most-recurring content token (ties → longer, then
        # alphabetical, for determinism). Facts sharing a key fold into one cluster.
        clusters: dict = {}
        for (ts, doc), tset in zip(fact_items, toks):
            if tset:
                key = max(tset, key=lambda t: (freq[t], len(t), t))
            else:
                key = (doc or "").strip().lower()   # tokenless fact → its own bucket
            c = clusters.setdefault(key, {"items": [], "max_ts": 0.0})
            c["items"].append((ts, doc))
            if (ts or 0) > c["max_ts"]:
                c["max_ts"] = ts or 0
        # Fold clusters whose keys are prefix-related (banana ⊂ bananas ⊂ bananana)
        # so morphological / repetition / misspelling variants of ONE topic collapse
        # to a single line. Requires a true prefix ≥5 chars, which keeps unrelated
        # words apart ("modern" is not a prefix of "moderator"). Shortest key wins.
        _keys = sorted(clusters.keys(), key=len)
        _canon: dict = {}
        for _k in _keys:
            _tgt = _k
            for _shorter in _keys:
                if _shorter == _k:
                    break
                if len(_shorter) >= 5 and _k.startswith(_shorter):
                    _tgt = _canon.get(_shorter, _shorter)
                    break
            _canon[_k] = _tgt
        if any(v != k for k, v in _canon.items()):
            _merged: dict = {}
            for _k, _c in clusters.items():
                _m = _merged.setdefault(_canon[_k], {"items": [], "max_ts": 0.0})
                _m["items"].extend(_c["items"])
                _m["max_ts"] = max(_m["max_ts"], _c["max_ts"])
            clusters = _merged
        # Durability first (cluster size), recency (max_ts) breaks ties. One line each.
        ranked = sorted(
            clusters.values(),
            key=lambda c: (len(c["items"]), c["max_ts"]),
            reverse=True,
        )
        return [max(c["items"], key=lambda it: it[0] or 0)[1] for c in ranked[:limit]]

    def get_chatter_context(self, username: str, n_results: int = 5) -> str:
        """Returns a formatted string of what Kira knows about this chatter.
        Used when a chatter speaks — gives Kira recall."""
        try:
            results = self.chatters.get(
                where={"username": normalize_chatter_key(username)},
                limit=50,
                include=["documents", "metadatas"],
            )
            if not results or not results.get("documents"):
                return ""

            facts = []          # (timestamp, doc) for chatter_fact rows
            recent_msgs = []
            tones = []
            preferred = None
            for doc, meta in zip(results["documents"], results["metadatas"]):
                _mtype = meta.get("type")
                if _mtype == "chatter_fact":
                    facts.append((meta.get("timestamp", 0), doc))
                    if meta.get("tone"):
                        tones.append(meta["tone"])
                elif _mtype == "chatter_preferred_name":
                    preferred = meta.get("preferred_name") or preferred
                else:
                    recent_msgs.append((meta.get("timestamp", 0), doc))

            recent_msgs.sort(reverse=True)
            recent_msgs = [doc for ts, doc in recent_msgs[:n_results]]

            # Diversity-first fact surfacing (READ-ONLY; see _diverse_facts). A
            # running bit calcifies into dozens of paraphrased rows that flood the
            # recall slots and freeze a chatter's identity ("banana person"). We do
            # NOT delete those rows — we just SELECT better at read time: collapse
            # near-duplicates to one line each, then fill remaining slots with
            # DISTINCT topics. Pure in-memory; no write of any kind.
            surfaced_facts = self._diverse_facts(facts, username, 5)

            lines = []
            if preferred:
                lines.append(f"Prefers to be called: {preferred} (use this name, not the raw handle)")
            if surfaced_facts:
                lines.append(f"What you know about {username}: " + " | ".join(surfaced_facts))
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
                where={"username": normalize_chatter_key(username)},
                limit=1,
            )
            return not results or not results.get("documents")
        except Exception as e:
            print(f"   [WARN] memory: is_first_time_chatter lookup failed for {username!r}: {e}")
            return False

    def count_chatter_messages(self, username: str) -> int:
        """Return total historical message count for this chatter across all sessions."""
        try:
            results = self.chatters.get(
                where={"username": normalize_chatter_key(username)},
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
