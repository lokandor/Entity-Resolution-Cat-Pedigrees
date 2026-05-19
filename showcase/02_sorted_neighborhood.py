"""
Sorted Neighborhood  (Blocking — Dok. 2)
=========================================
Goal   : Generate candidate pairs robust to prefix-preserving typos in names.

Method : Using the `recordlinkage` library's SortedNeighbourhood indexer.
         Records are sorted by a key; a sliding window of size W generates
         candidates from adjacent records.

Install: pip install recordlinkage pandas
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import re
import pandas as pd
import recordlinkage
from data import records_a, records_b, evaluate


def normalize_key(s: str) -> str:
    """Lowercase, remove non-alpha, keep first 8 chars of name."""
    return re.sub(r"[^a-z]", "", str(s).lower())[:8]


def sorted_neighborhood(df_a, df_b, window: int = 3):
    """
    Use recordlinkage.Index.sortedneighbourhood().
    The sort key is the normalised 'name' field.
    Returns set of (a_id, b_id) candidate pairs.
    """
    a = df_a.copy()
    b = df_b.copy()

    # recordlinkage needs index = record id
    a.index = a["id"]
    b.index = b["id"]

    # Add normalised sort key
    a["_key"] = a["name"].apply(normalize_key)
    b["_key"] = b["name"].apply(normalize_key)

    indexer = recordlinkage.Index()
    indexer.sortedneighbourhood(left_on="_key", right_on="_key", window=window)

    pairs_idx = indexer.index(a, b)

    # pairs_idx is a MultiIndex of (a_id, b_id) since we set index = id
    return {(str(a_id), str(b_id)) for a_id, b_id in pairs_idx}


if __name__ == "__main__":
    print("=" * 60)
    print("Sorted Neighborhood  (Blocking / Dok. 2)")
    print("Using: recordlinkage library")
    print("=" * 60)

    print(f"\n{'Window':>6} | {'Candidates':>10} | {'Reduction':>9} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("  " + "-" * 58)
    for w in [3, 5, 7, 9, 11]:
        cands = sorted_neighborhood(records_a, records_b, window=w)
        total = len(records_a) * len(records_b)
        m = evaluate(cands)
        print(f"  {w:>6} | {len(cands):>10} | {1-len(cands)/total:>8.1%} | "
              f"{m['precision']:>9} | {m['recall']:>6} | {m['f1']:>6}")

    print("\n--- Sorted order for window=4 ---")
    import pandas as pd
    a = records_a.copy(); a.index = a["id"]; a["_key"] = a["name"].apply(normalize_key)
    b = records_b.copy(); b.index = b["id"]; b["_key"] = b["name"].apply(normalize_key)
    combined = pd.concat([
        a[["id","name","_key"]].assign(src="A"),
        b[["id","name","_key"]].assign(src="B"),
    ]).sort_values("_key")
    print(combined[["src","id","name","_key"]].head(16).to_string(index=False))
