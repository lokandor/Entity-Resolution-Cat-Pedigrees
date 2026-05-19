"""
Meta-Blocking / BLAST  (Block Processing — Dok. 2)
===================================================
Goal   : Prune the candidate pairs produced by blocking without losing
         true matches.  "Precision++ without significant recall loss."

Method : Each candidate pair (a, b) gets a weight = number of shared
         blocks they co-occur in.  Pairs below a weight threshold are
         pruned.  Optionally use WNP (Weighted Node Pruning) or CBS
         (Cardinality Node Pruning) — here we implement WNP.

Input  : Raw candidate pairs from Token Blocking (01_token_blocking.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
from data import records_a, records_b, evaluate, print_matched_pairs
# ── helpers ───────────────────────────────────────────────────────────────────

def build_blocks(df_a, df_b):
    """Return list of blocks, each block = list of (src, id) tuples."""
    import re
    STOPWORDS = {"of", "the", "a", "an", "de", "von", "vom", "v", "la", "le", "van"}
    INDEX_COLS = ["name", "breed"]
    index = defaultdict(list)
    for df, src in [(df_a, "A"), (df_b, "B")]:
        for _, row in df.iterrows():
            for col in INDEX_COLS:
                for tok in re.findall(r"[a-z]+", str(row[col]).lower()):
                    if tok not in STOPWORDS and len(tok) > 1:
                        index[tok].append((src, row["id"]))
    return list(index.values())


def meta_blocking(df_a, df_b, threshold: float = 1.0):
    """
    Weight each candidate pair by the number of blocks they share.
    Keep only pairs with weight >= threshold.
    Returns pruned candidate set and the weight dict.
    """
    blocks = build_blocks(df_a, df_b)

    # Count how many blocks each cross-source pair co-occurs in
    pair_weight = defaultdict(int)
    for block in blocks:
        a_ids = [r[1] for r in block if r[0] == "A"]
        b_ids = [r[1] for r in block if r[0] == "B"]
        for a in a_ids:
            for b in b_ids:
                pair_weight[(a, b)] += 1

    pruned = {pair for pair, w in pair_weight.items() if w >= threshold}
    return pruned, pair_weight


if __name__ == "__main__":
    print("=" * 60)
    print("Meta-Blocking  (Block Processing / Dok. 2)")
    print("=" * 60)

    # Raw Token Blocking stats for comparison
    raw, weights = meta_blocking(records_a, records_b, threshold=1)
    print(f"\nRaw candidate pairs (threshold=1) : {len(raw)}")
    m = evaluate(raw)
    print(f"  P={m['precision']}  R={m['recall']}  F1={m['f1']}")

    print("\nEffect of pruning threshold:")
    print(f"  {'Threshold':>9} | {'Candidates':>10} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("  " + "-" * 52)
    for thr in [1, 2, 3, 4, 5]:
        cands = {p for p, w in weights.items() if w >= thr}
        m = evaluate(cands)
        print(f"  {thr:>9} | {len(cands):>10} | {m['precision']:>9} | {m['recall']:>6} | {m['f1']:>6}")

    # ── weighted pair table ───────────────────────────────────────────────
    from data import TRUE_MATCHES
    true_set = {(a, b) for a, b in TRUE_MATCHES}
    rec_a_idx = records_a.set_index("id")
    rec_b_idx = records_b.set_index("id")

    print("\nAll candidate pairs ranked by block co-occurrence weight:")
    print(f"  {'A id':>4}  {'Name (A)':<22}  {'B id':>4}  {'Name (B)':<22}  {'Wt':>3}  Result")
    print("  " + "─" * 76)
    for (a_id, b_id), w in sorted(weights.items(), key=lambda x: -x[1]):
        name_a = rec_a_idx.loc[a_id, "name"]
        name_b = rec_b_idx.loc[b_id, "name"]
        mark   = "✓ TRUE MATCH" if (a_id, b_id) in true_set else ""
        print(f"  {a_id:>4}  {name_a:<22}  {b_id:>4}  {name_b:<22}  {w:>3}  {mark}")

    # ── pruned set at best threshold ──────────────────────────────────────
    best_thr = 3
    pruned = {p for p, w in weights.items() if w >= best_thr}
    print(f"\nPruned candidate pairs (threshold={best_thr}):")
    print_matched_pairs(pruned)
