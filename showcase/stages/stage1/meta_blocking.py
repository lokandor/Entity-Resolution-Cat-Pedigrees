"""Meta-Blocking (WNP) — prune candidate pairs by block co-occurrence weight."""
import re
from collections import defaultdict


def prune_candidates(df, candidates, threshold: int = 2) -> set:
    """
    Weight each candidate pair by the number of token blocks they share.
    Keep only pairs with weight >= threshold (Weighted Node Pruning).

    Higher threshold = fewer candidates, higher precision, lower recall.
    threshold=1 is equivalent to no pruning (same as raw token blocking output).
    """
    STOPWORDS = {"of", "the", "a", "an", "de", "von", "vom", "v", "vd",
                 "la", "le", "van", "retr", "germ"}
    INDEX_COLS = [c for c in ["name", "breed"] if c in df.columns]

    rec_idx = df.set_index("id")
    src     = rec_idx["source"]

    index = defaultdict(list)
    for rid, row in rec_idx.iterrows():
        for col in INDEX_COLS:
            for tok in re.findall(r"[a-z]+", str(row[col]).lower()):
                if tok not in STOPWORDS and len(tok) > 1:
                    index[tok].append(rid)

    pair_weight = defaultdict(int)
    for ids in index.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if src.get(a) != src.get(b):
                    key = tuple(sorted((a, b)))
                    pair_weight[key] += 1

    candidate_set = {tuple(sorted(p)) for p in candidates}
    return {p for p in candidate_set if pair_weight.get(p, 0) >= threshold}
