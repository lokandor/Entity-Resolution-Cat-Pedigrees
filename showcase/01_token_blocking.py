"""
Token Blocking  (Blocking — Dok. 2)
=====================================
Goal   : Reduce O(n²) candidate pairs to a high-recall subset.

Method : Works on the full unified dataset (all sources together).
         Each record emits tokens from name + breed.
         All records sharing a token land in the same block.
         Every pair within a block becomes a candidate — regardless of source,
         so within-source duplicates are found just like cross-source ones.

         recordlinkage's self-join Index is used to avoid self-pairs and
         duplicate (i,j)/(j,i) pairs automatically.

Install: pip install recordlinkage pandas
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import re
from collections import defaultdict
import pandas as pd
import recordlinkage
from data import records, evaluate, print_matched_pairs, TRUE_MATCHES

STOPWORDS  = {"of", "the", "a", "an", "de", "von", "vom", "v", "vd", "la", "le", "van"}
BLOCK_COLS = ["name", "breed"]


# ── token explosion ───────────────────────────────────────────────────────────

def explode_tokens(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One row per (record, token) — input for recordlinkage block indexer."""
    rows = []
    for _, row in df.iterrows():
        tokens = set()
        for col in cols:
            for tok in re.findall(r"[a-z]+", str(row[col]).lower()):
                if tok not in STOPWORDS and len(tok) > 1:
                    tokens.add(tok)
        for tok in tokens:
            rows.append({"record_id": row["id"], "_token": tok})
    return pd.DataFrame(rows)


# ── blocking (full dataset self-join) ─────────────────────────────────────────

def token_blocking(df: pd.DataFrame) -> set:
    """
    Self-join token blocking on the full dataset.
    Returns set of (id_1, id_2) candidate pairs where id_1 < id_2,
    covering both within-source and cross-source pairs.
    """
    exp = explode_tokens(df, BLOCK_COLS).reset_index(drop=True)

    indexer = recordlinkage.Index()
    indexer.block("_token")
    pairs_idx = indexer.index(exp)          # self-join — no self-pairs, no duplicates

    candidates = set()
    for i, j in pairs_idx:
        id_i = exp.loc[i, "record_id"]
        id_j = exp.loc[j, "record_id"]
        if id_i != id_j:
            candidates.add(tuple(sorted((id_i, id_j))))
    return candidates


# ── block index (for display) ─────────────────────────────────────────────────

def build_block_index(df: pd.DataFrame) -> dict:
    """Return {token: {source: [record_ids]}} for all records."""
    index = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        for col in BLOCK_COLS:
            for tok in re.findall(r"[a-z]+", str(row[col]).lower()):
                if tok not in STOPWORDS and len(tok) > 1:
                    index[tok][row["source"]].append(row["id"])
    return index


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Token Blocking  (Blocking / Dok. 2)")
    print("Using: recordlinkage library  —  full dataset self-join")
    print("=" * 60)

    n = len(records)
    total_pairs = n * (n - 1) // 2
    candidates  = token_blocking(records)

    print(f"\nDataset       : {n} records across {records['source'].nunique()} sources")
    print(f"Total possible: {total_pairs} pairs")
    print(f"Candidates    : {len(candidates)} pairs")
    print(f"Reduction     : {1 - len(candidates)/total_pairs:.2%}")

    # Split candidates into cross-source and within-source for info
    rec_src = records.set_index("id")["source"]
    cross   = {(a, b) for a, b in candidates if rec_src[a] != rec_src[b]}
    within  = candidates - cross
    print(f"  Cross-source : {len(cross)}  |  Within-source : {len(within)}")

    m = evaluate(cross)     # ground truth only covers cross-source pairs
    print(f"\nCross-source evaluation vs ground truth:")
    print(f"  Precision : {m['precision']}")
    print(f"  Recall    : {m['recall']}  (must be high — blocking is a filter)")
    print(f"  F1        : {m['f1']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    # ── show blocks ───────────────────────────────────────────────────────
    block_index = build_block_index(records)
    rec_name    = records.set_index("id")["name"]
    sources     = sorted(records["source"].unique())

    # Only show blocks that have ≥2 records (potential matches)
    useful = {tok: v for tok, v in block_index.items()
              if sum(len(ids) for ids in v.values()) >= 2}
    sorted_blocks = sorted(
        useful.items(),
        key=lambda x: sum(len(v) for v in x[1].values()),
        reverse=True,
    )

    print(f"\nBlocks with ≥2 records (top 20 by size):")
    src_header = "  ".join(f"{s}" for s in sources)
    print(f"  {'Token':<16}  {src_header}")
    print("  " + "─" * 70)
    for tok, src_map in sorted_blocks[:20]:
        src_parts = []
        for s in sources:
            ids = src_map.get(s, [])
            if ids:
                names = ", ".join(f"{i}({rec_name[i].split()[0]})" for i in ids)
                src_parts.append(f"{names}")
            else:
                src_parts.append("—")
        print(f"  {tok!r:<16}  {'  |  '.join(src_parts)}")

    print(f"\nCross-source candidate pairs vs ground truth:")
    print_matched_pairs(cross)
