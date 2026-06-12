"""Diagnostic: inspect chroma.sqlite3 directly (bypasses HNSW index) and
also exercise the runtime retrieval path. Read-only — copies the live DB
to TEMP first so the running dashboard isn't disturbed."""
import os, sqlite3, shutil, sys, tempfile, traceback
from pathlib import Path

MEM = Path(__file__).parent.parent / "memory_db"
src = MEM / "chroma.sqlite3"
dst = Path(tempfile.gettempdir()) / "chroma_inspect.sqlite3"
shutil.copy2(src, dst)
print(f"copied -> {dst}")

db = sqlite3.connect(dst)
c = db.cursor()

print("\n=== COLLECTIONS ===")
for r in c.execute("SELECT id, name FROM collections").fetchall():
    print(" ", r)

print("\n=== TABLES ===")
for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
    print(" ", r[0])

# Schema for embedding_metadata
print("\n=== embedding_metadata columns ===")
for r in c.execute("PRAGMA table_info(embedding_metadata)").fetchall():
    print(" ", r)

# Raw search for 'cat'/Cartofell/Madoka across all documents
print("\n=== DOCUMENTS containing Cartofell / Madoka / cat ===")
rows = c.execute("""
    SELECT em.string_value, e.embedding_id, e.segment_id
    FROM embedding_metadata em
    JOIN embeddings e ON em.id = e.id
    WHERE em.key = 'chroma:document'
      AND (em.string_value LIKE '%Cartofell%'
        OR em.string_value LIKE '%Madoka%'
        OR em.string_value LIKE '%cat%')
""").fetchall()
print(f"hits: {len(rows)}")
for doc, eid, seg in rows[:50]:
    print(f"  seg={seg[:8]} id={eid[:8]} :: {doc[:200]}")

# Per-collection counts via segments
print("\n=== Per-collection row counts (from embeddings table) ===")
seg_to_coll = {}
for r in c.execute("SELECT id, collection FROM segments").fetchall():
    seg_to_coll.setdefault(r[1], []).append(r[0])
coll_names = {cid: name for cid, name in c.execute("SELECT id,name FROM collections").fetchall()}
for cid, segs in seg_to_coll.items():
    placeholders = ",".join("?" * len(segs))
    n = c.execute(f"SELECT COUNT(*) FROM embeddings WHERE segment_id IN ({placeholders})", segs).fetchone()[0]
    print(f"  {coll_names.get(cid,'?'):10}  segs={len(segs)}  rows={n}")

db.close()

# Now run the actual retrieval path the bot uses
print("\n=== RUNTIME RETRIEVAL PATH ===")
print("(this opens the live DB read-only — may fail if dashboard locks it)")
try:
    # Use ":memory:"-style probe by talking to memory.py directly
    from kira.memory.memory import MemoryManager
    mm = MemoryManager()
    for q in [
        "what are my cats' names",
        "cats",
        "Cartofell",
        "Madoka",
        "my pets",
    ]:
        print(f"\n  query: {q!r}")
        try:
            hits = mm.search_facts(q, n_results=5)
        except Exception as e:
            print(f"    search_facts FAILED: {e}")
            continue
        for h in hits:
            print(f"    - {h}")

        # Also probe raw distances to see what the threshold would need to be
        try:
            emb = mm.embedding_model.encode(q).tolist()
            raw = mm.facts.query(query_embeddings=[emb], n_results=5,
                                  include=['documents','distances'])
            docs = (raw.get('documents') or [[]])[0]
            dists = (raw.get('distances') or [[]])[0]
            for d, dist in zip(docs, dists):
                print(f"      dist={dist:.4f}  :: {d[:120]}")
        except Exception as e:
            print(f"    raw query FAILED: {e}")
except Exception:
    traceback.print_exc()
