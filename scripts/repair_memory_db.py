"""
repair_memory_db.py

Repairs Chroma collections whose HNSW vector index files were never persisted
to disk (symptom: '.query()' fails with
"Error creating hnsw segment reader: Nothing found on disk"
while '.get()' and '.count()' still work because the embeddings are intact
in chroma.sqlite3).

Strategy (non-destructive, Option A):
  1. Back up the entire memory_db/ directory to memory_db.bak/ FIRST.
     If a backup already exists, refuse to overwrite — user must move it.
  2. For each target collection, read all rows via
     .get(include=['embeddings','documents','metadatas']).
  3. Delete by IDs in the same collection, then re-add — this forces Chroma
     to rebuild and persist the HNSW index.
  4. Verify with a real vector .query() on each repaired collection.
  5. If verification fails for any collection, STOP. The backup is intact
     and the user can restore by:  rm -r memory_db && mv memory_db.bak memory_db

By default repairs 'facts' and 'chatters' and leaves 'turns' alone (it has a
healthy index). Pass --include turns to repair it too.

Usage:
    # Stop dashboard.py first — Chroma uses an exclusive sqlite handle.
    python repair_memory_db.py
    python repair_memory_db.py --dry-run         # diagnose only, no writes
    python repair_memory_db.py --include turns   # also repair turns
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

# Repo root on path so first-party imports resolve when run as scripts/repair_memory_db.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MEMORY_PATH


DEFAULT_TARGETS = ("facts", "chatters")
LEFT_ALONE_BY_DEFAULT = ("turns",)


def log(msg: str) -> None:
    print(f"[repair] {msg}", flush=True)


def backup(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        raise SystemExit(
            f"[repair] ABORT: backup target already exists: {dst_dir}\n"
            f"[repair] Move or delete it before running this script again."
        )
    if not src_dir.exists():
        raise SystemExit(f"[repair] ABORT: memory_db dir not found: {src_dir}")
    log(f"Backing up {src_dir} -> {dst_dir} ...")
    shutil.copytree(src_dir, dst_dir)
    src_bytes = sum(p.stat().st_size for p in src_dir.rglob("*") if p.is_file())
    dst_bytes = sum(p.stat().st_size for p in dst_dir.rglob("*") if p.is_file())
    if src_bytes != dst_bytes:
        raise SystemExit(
            f"[repair] ABORT: backup size mismatch (src={src_bytes}, dst={dst_bytes})"
        )
    log(f"Backup OK ({src_bytes:,} bytes mirrored).")


def repair_collection(client, name: str, dry_run: bool) -> bool:
    """Read all rows from collection 'name', then delete-and-re-add to force
    HNSW index rebuild. Returns True on success."""
    log(f"--- Collection '{name}' ---")
    try:
        coll = client.get_collection(name=name)
    except Exception as e:
        log(f"  Collection '{name}' not found ({e}); skipping.")
        return True

    count = coll.count()
    log(f"  Row count (sqlite): {count}")
    if count == 0:
        log(f"  Empty collection, nothing to rebuild.")
        return True

    try:
        data = coll.get(include=["embeddings", "documents", "metadatas"])
    except Exception as e:
        log(f"  FAILED reading rows: {e}")
        return False

    # Chroma returns embeddings as a numpy ndarray; `or []` would raise
    # "truth value of an array is ambiguous". Use explicit None checks and
    # normalize embeddings to a list of plain Python lists for re-add.
    def _none_to_empty(v):
        return [] if v is None else v
    ids        = _none_to_empty(data.get("ids"))
    raw_embs   = _none_to_empty(data.get("embeddings"))
    documents  = _none_to_empty(data.get("documents"))
    metadatas  = _none_to_empty(data.get("metadatas"))
    # Normalize each embedding to a plain list so chroma.add() is happy and
    # so our length checks below work uniformly.
    embeddings = [list(e) for e in raw_embs]

    log(f"  Read {len(ids)} ids, {len(embeddings)} embeddings, "
        f"{len(documents)} documents, {len(metadatas)} metadatas.")
    if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
        log("  FAILED: id/embedding/document/metadata length mismatch.")
        return False
    if len(ids) != count:
        log(f"  WARNING: count() reported {count} but got {len(ids)} rows; continuing.")

    # Sanity check that embeddings are non-empty vectors
    if embeddings and (not embeddings[0] or len(embeddings[0]) == 0):
        log("  FAILED: embeddings appear empty — refusing to wipe.")
        return False
    dim = len(embeddings[0]) if embeddings else None
    log(f"  Embedding dim: {dim}")

    if dry_run:
        log("  [dry-run] Would delete and re-add all rows here.")
        return True

    # Delete by IDs in batches to keep memory/IO sane
    BATCH = 500
    log(f"  Deleting {len(ids)} rows (batch size {BATCH}) ...")
    try:
        for i in range(0, len(ids), BATCH):
            coll.delete(ids=ids[i : i + BATCH])
    except Exception as e:
        log(f"  FAILED during delete: {e}")
        return False

    after_delete = coll.count()
    log(f"  Row count after delete: {after_delete}")
    if after_delete != 0:
        log(f"  FAILED: expected 0 rows after delete, got {after_delete}.")
        return False

    log(f"  Re-adding {len(ids)} rows (batch size {BATCH}) ...")
    try:
        for i in range(0, len(ids), BATCH):
            j = i + BATCH
            coll.add(
                ids=ids[i:j],
                embeddings=embeddings[i:j],
                documents=documents[i:j],
                metadatas=metadatas[i:j],
            )
    except Exception as e:
        log(f"  FAILED during re-add: {e}")
        return False

    after_add = coll.count()
    log(f"  Row count after re-add: {after_add}")
    if after_add != len(ids):
        log(f"  FAILED: expected {len(ids)} rows after re-add, got {after_add}.")
        return False

    # Verify with a real vector query (this is what was broken before)
    try:
        probe = embeddings[0]
        results = coll.query(query_embeddings=[probe], n_results=min(3, after_add))
        got = len((results.get("ids") or [[]])[0])
        log(f"  Vector query verification: returned {got} hit(s).")
        if got == 0:
            log(f"  FAILED: vector query returned 0 results on probe.")
            return False
    except Exception as e:
        log(f"  FAILED vector query verification: {e}")
        return False

    log(f"  '{name}' repaired and verified.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Diagnose and back up only; make no writes to the DB.")
    ap.add_argument("--include", action="append", default=[],
                    help="Additional collections to repair (e.g. 'turns'). Repeatable.")
    ap.add_argument("--only", action="append", default=[],
                    help="Repair ONLY these collections (overrides defaults).")
    args = ap.parse_args()

    if args.only:
        targets = tuple(args.only)
    else:
        targets = tuple(DEFAULT_TARGETS) + tuple(args.include)

    log(f"MEMORY_PATH = {MEMORY_PATH}")
    log(f"Targets    = {targets}")
    log(f"Dry-run    = {args.dry_run}")

    src = Path(MEMORY_PATH).resolve()
    bak = src.parent / (src.name + ".bak")

    # Always back up, even on dry-run — costs nothing and protects the user.
    backup(src, bak)

    client = chromadb.PersistentClient(
        path=str(src),
        settings=Settings(anonymized_telemetry=False),
    )

    failed = []
    for name in targets:
        ok = repair_collection(client, name, dry_run=args.dry_run)
        if not ok:
            failed.append(name)
            log(f"STOPPING: '{name}' failed. Backup intact at: {bak}")
            log(f"To restore:  rmdir /S /Q \"{src}\" && move \"{bak}\" \"{src}\"")
            return 2

    if args.dry_run:
        log("Dry-run complete. No changes written.")
    else:
        log("All targeted collections repaired and verified.")
        log(f"Backup remains at: {bak}")
        log(f"If everything looks healthy after a normal launch, you can delete it.")

    if failed:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
