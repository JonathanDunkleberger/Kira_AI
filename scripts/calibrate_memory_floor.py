#!/usr/bin/env python
"""Calibrate MEMORY_SIM_FLOOR from the REAL stored facts -- READ-ONLY.

Why this exists: the smart-retrieval rework (Piece 1a) drops vector hits whose
distance is worse than a floor. ChromaDB distances are embedding/metric-specific,
so the floor must be MEASURED against the actual DB, not guessed. This tool prints
the number; it writes NOTHING (no DB writes, no files). Run it after a few real
sessions have populated the facts collection, then set MEMORY_SIM_FLOOR from the
recommended value before Commit 2 is built.

What it measures:
  1. Distance metric (hnsw:space -- L2 vs cosine). The floor's meaning depends on it.
  2. True-match baseline: for a sample of stored facts, query with a light
     paraphrase (the fact's content words only) and record the distance the
     CORRECT fact lands at (and whether it surfaces at all).
  3. Noise baseline: query with generic/unrelated strings and record the SMALLEST
     distance each still pulls (how close garbage gets).
  4. Recommends MEMORY_SIM_FLOOR sitting between true-p90 and noise-p10 -- or says
     plainly that there's no clean separation (overlap / DB too sparse) instead of
     emitting a garbage threshold.

Usage:
    python scripts/calibrate_memory_floor.py
    python scripts/calibrate_memory_floor.py --sample 60   # more true-match probes

Requires the same deps the bot uses (chromadb, sentence-transformers) and a
populated memory DB at MEMORY_PATH. Read-only: safe to run anytime.
"""
import os
import sys
import re
import math
import random
import argparse

# Repo root on path + as CWD so MEMORY_PATH (relative) resolves like the bot.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Minimum real facts needed before a calibration is meaningful. Below this we
# refuse to emit a threshold (a floor from 3 facts is noise, not signal).
MIN_FACTS_TO_CALIBRATE = 15

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your", "his", "her",
    "are", "was", "were", "has", "have", "had", "his", "she", "him", "they",
    "what", "who", "whom", "when", "where", "which", "into", "from", "about",
    "jonny", "jonnys", "kira", "kiras", "during", "session", "recap", "take",
    "is", "of", "a", "an", "to", "in", "on", "it", "its", "be", "as", "at",
}


# -- Pure helpers (no heavy imports -- unit-testable in isolation) --------------
def content_words(text):
    """Light, deterministic 'paraphrase': the fact's content words only (len>3,
    non-stopword, lowercased). Simulates a realistic keyword-style query that
    SHOULD retrieve the fact without being its verbatim text."""
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS]


def pctl(sorted_vals, q):
    """Nearest-rank-ish percentile with linear interpolation. `sorted_vals` must
    be sorted ascending. q in [0,1]. Returns None for empty input."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(vals):
    """(n, min, median, p90) for a list of distances; Nones if empty."""
    s = sorted(vals)
    if not s:
        return (0, None, None, None)
    return (len(s), s[0], pctl(s, 0.5), pctl(s, 0.9))


def recommend_floor(true_p90, noise_p10):
    """Recommend a floor between true-p90 and noise-p10 (smaller-is-closer metric).
    Returns (floor_or_None, status). If the distributions overlap (true matches are
    as far as noise), there is NO clean threshold -- say so, don't emit garbage."""
    if true_p90 is None or noise_p10 is None:
        return None, "INSUFFICIENT DATA -- not enough true-match and/or noise samples."
    if true_p90 < noise_p10:
        return round((true_p90 + noise_p10) / 2.0, 4), "CLEAN SEPARATION"
    return None, ("OVERLAP -- true matches land as far as noise (true-p90 >= noise-p10). "
                  "No clean floor; semantic signal is weak or the DB is too sparse. "
                  "Do NOT set a floor from this run.")


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser(description="Read-only MEMORY_SIM_FLOOR calibration.")
    ap.add_argument("--sample", type=int, default=40,
                    help="Max stored facts to use as true-match probes (default 40).")
    ap.add_argument("--seed", type=int, default=1234, help="Sampling seed (reproducible).")
    args = ap.parse_args()
    random.seed(args.seed)

    # Heavy imports live here so the pure helpers above stay importable for tests.
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    from kira.config import MEMORY_PATH

    print("=" * 70)
    print("MEMORY_SIM_FLOOR CALIBRATION  (read-only -- writes nothing)")
    print("=" * 70)
    print(f"MEMORY_PATH: {MEMORY_PATH}")

    if not os.path.isdir(MEMORY_PATH):
        print(f"\nNo memory DB at {MEMORY_PATH}. Run the bot for a few real sessions "
              f"first so the facts collection gets populated, then re-run this.")
        return 1

    client = chromadb.PersistentClient(path=MEMORY_PATH,
                                       settings=Settings(anonymized_telemetry=False))
    try:
        facts = client.get_collection(name="facts")  # get_collection (NOT create) -- never writes
    except Exception as e:
        print(f"\nNo 'facts' collection found ({e}). Nothing to calibrate yet.")
        return 1

    count = facts.count()
    print(f"facts collection: {count} rows")

    # -- 1. Distance metric ---------------------------------------------------
    meta = getattr(facts, "metadata", None) or {}
    space = meta.get("hnsw:space")
    if space:
        metric = space
        print(f"\n[1] Distance metric (hnsw:space): '{metric}'")
    else:
        metric = "l2"
        print("\n[1] Distance metric: not explicitly set -> ChromaDB default = 'l2' "
              "(squared L2 / euclidean).")
    if metric in ("l2", "cosine"):
        print(f"    -> For '{metric}', SMALLER distance = MORE similar. "
              f"The floor KEEPS hits with distance <= floor.")
    else:
        print(f"    -> WARNING: metric '{metric}' may be larger-is-closer (e.g. 'ip'); "
              f"the recommendation logic below assumes smaller-is-closer. Interpret with care.")

    # -- tiny-DB guard --------------------------------------------------------
    if count < MIN_FACTS_TO_CALIBRATE:
        print(f"\n[!] Only {count} stored fact(s) -- need >= {MIN_FACTS_TO_CALIBRATE} to "
              f"calibrate meaningfully.\n    Run the bot for a few more real sessions to "
              f"populate the facts collection, then re-run.\n    NO threshold emitted "
              f"(a floor from this little data would be garbage).")
        return 0

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    allrows = facts.get(include=["documents", "metadatas"])
    docs = allrows.get("documents") or []
    metas = allrows.get("metadatas") or []
    # Type breakdown (informational -- highlights/summaries are Kira-generated).
    from collections import Counter
    types = Counter((m or {}).get("type", "?") for m in metas)
    print(f"    type breakdown: " + ", ".join(f"{k}={v}" for k, v in types.most_common()))

    # -- 2. True-match baseline (light-paraphrase probes) ---------------------
    idxs = list(range(len(docs)))
    random.shuffle(idxs)
    idxs = idxs[: args.sample]
    true_dists = []
    recall_found = 0
    recall_total = 0
    for i in idxs:
        original = docs[i]
        cw = content_words(original)
        if len(cw) < 2:
            continue  # too little signal to form a fair paraphrase probe
        recall_total += 1
        q = " ".join(cw)
        res = facts.query(query_embeddings=[model.encode(q).tolist()],
                          n_results=min(5, count),
                          include=["documents", "distances"])
        rdocs = (res.get("documents") or [[]])[0]
        rdist = (res.get("distances") or [[]])[0]
        # Distance at which the ORIGINAL fact surfaces (if it does, in top-5).
        found = None
        for j, d in enumerate(rdocs):
            if d == original:
                found = rdist[j]
                break
        if found is not None:
            true_dists.append(found)
            recall_found += 1

    # -- 3. Noise baseline ----------------------------------------------------
    NOISE = ["hey", "what's up", "good morning", "lol okay", "hmm interesting",
             "let's go then", "what's the weather", "can you do that for me",
             "qwerty zxcvbn", "frobnicate the plimth wuggle", "asdf jkl semicolon",
             "random nonsense tokens here please"]
    noise_min = []
    for nq in NOISE:
        res = facts.query(query_embeddings=[model.encode(nq).tolist()],
                          n_results=min(5, count),
                          include=["distances"])
        rdist = (res.get("distances") or [[]])[0]
        if rdist:
            noise_min.append(min(rdist))  # the CLOSEST this garbage got

    tn, tmin, tmed, tp90 = summarize(true_dists)
    nn, nmin, nmed, np90 = summarize(noise_min)
    np10 = pctl(sorted(noise_min), 0.10) if noise_min else None

    print("\n[2] TRUE-MATCH baseline (paraphrase = content-word query -> original's distance)")
    print(f"    probes with usable paraphrase: {recall_total}")
    print(f"    recall (original found in top-5): {recall_found}/{recall_total}"
          + (f"  ({100*recall_found/recall_total:.0f}%)" if recall_total else ""))
    print(f"    distances -> n={tn}  min={_fmt(tmin)}  median={_fmt(tmed)}  p90={_fmt(tp90)}")
    if recall_total and recall_found / recall_total < 0.5:
        print("    [!] LOW recall -- many facts don't surface even for their own keywords; "
              "the retrieval signal itself is weak (relevant to the forgetting bug).")

    print("\n[3] NOISE baseline (generic/garbage queries -> smallest distance each pulls)")
    print(f"    queries: {len(NOISE)}")
    print(f"    closest-hit distances -> n={nn}  min={_fmt(nmin)}  median={_fmt(nmed)}  "
          f"p90={_fmt(np90)}  p10={_fmt(np10)}")

    # -- 4. Recommendation ----------------------------------------------------
    floor, status = recommend_floor(tp90, np10)
    print("\n[4] RECOMMENDATION")
    print(f"    status: {status}")
    if floor is not None:
        keep_true = sum(1 for d in true_dists if d <= floor)
        pass_noise = sum(1 for d in noise_min if d <= floor)
        print(f"    >>> MEMORY_SIM_FLOOR = {floor}")
        print(f"        (keep hits with distance <= {floor})")
        print(f"        sanity: true matches kept {keep_true}/{tn}; "
              f"noise that would slip through {pass_noise}/{nn}")
        print(f"\n    Set it via .env:  MEMORY_SIM_FLOOR={floor}")
    else:
        print("    No MEMORY_SIM_FLOOR emitted. See status above. Re-run after more real "
              "sessions populate the facts collection, or report these numbers.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
