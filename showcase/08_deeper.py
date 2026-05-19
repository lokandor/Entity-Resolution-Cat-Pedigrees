"""
DeepER  (Matching — Dok. 4)
============================
Real implementation using sentence-transformers (bi-encoder).

Original DeepER: average pre-trained word embeddings (word2vec/GloVe)
per record, then cosine similarity.

Here we use sentence-transformers (all-MiniLM-L6-v2) which is the
modern successor — it produces better record embeddings than averaged
GloVe vectors while following the exact same bi-encoder pattern:
  1. Encode each record independently into a dense vector.
  2. Compute cosine similarity between vectors.
  3. Pairs above a threshold are predicted matches.

This is dramatically faster than cross-encoders (Ditto) at inference time.

Install: pip install sentence-transformers
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from data import records, evaluate, print_matched_pairs, inference_pairs, TRUE_MATCHES

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, ~80 MB
FIELDS     = ["name", "breed", "sire", "dam"]   # fields used for record repr.


def record_text(row) -> str:
    """Concatenate field values — simulates DeepER's input to the encoder."""
    return " ".join(str(row[f]) for f in FIELDS)


if __name__ == "__main__":
    print("=" * 60)
    print(f"DeepER  (Matching / Dok. 4)  —  bi-encoder: {MODEL_NAME}")
    print("=" * 60)

    print("\nLoading sentence-transformers model ...")
    model = SentenceTransformer(MODEL_NAME)

    # ── encode all records (full dataset) ────────────────────────────────
    all_texts = [record_text(r) for _, r in records.iterrows()]
    all_ids   = list(records["id"])

    print("Encoding records ...")
    emb = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    print(f"Embedding shape: {emb.shape}  ({len(all_ids)} records)")

    # ── cosine similarity for all unique pairs ────────────────────────────
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sim_matrix = cos_sim(emb)   # n×n matrix

    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}
    pairs      = inference_pairs()

    print(f"\n{'Threshold':>9} | {'Predicted':>9} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("  " + "-" * 50)
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        predicted = set()
        for id_1, id_2 in pairs:
            if sim_matrix[id_to_idx[id_1], id_to_idx[id_2]] >= thr:
                predicted.add((id_1, id_2))
        m = evaluate(predicted)
        print(f"  {thr:>9.2f} | {len(predicted):>9} | {m['precision']:>9} | "
              f"{m['recall']:>6} | {m['f1']:>6}")

    # ── best threshold detail ─────────────────────────────────────────────
    best_thr  = 0.65
    predicted = set()
    scores    = []
    for id_1, id_2 in pairs:
        s = sim_matrix[id_to_idx[id_1], id_to_idx[id_2]]
        scores.append((id_1, id_2, s))
        if s >= best_thr:
            predicted.add((id_1, id_2))

    m = evaluate(predicted)
    print(f"\nBest threshold={best_thr}: P={m['precision']}  R={m['recall']}  F1={m['f1']}")
    print("\nMatched pairs:")
    print_matched_pairs(predicted)

    print("\nTop-15 pairs by embedding similarity:")
    true_set = {tuple(sorted(p)) for p in TRUE_MATCHES}
    for id_1, id_2, s in sorted(scores, key=lambda x: -x[2])[:15]:
        mark = "<-- TRUE MATCH" if (id_1, id_2) in true_set else ""
        print(f"  {id_1} — {id_2}   sim={s:.4f}  {mark}")

    print(f"\nModel: {MODEL_NAME} ({emb.shape[1]}-dim embeddings)")
    print("DeepER advantage: O(n) encoding, O(n²) dot products — very fast.")
