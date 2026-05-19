"""
Correlation Clustering  (Clustering — Dok. 2)
==============================================
Real implementation using pyjedai.

pyjedai provides a production-quality implementation of Correlation Clustering
(and other clustering algorithms: CC, Markov, etc.) for entity resolution.

We build a graph of candidate pairs with similarity weights from
sentence-transformers, then pass it to pyjedai's CorrelationClustering.

Install: pip install pyjedai sentence-transformers pandas numpy
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from itertools import product

from data import records_a, records_b, evaluate, TRUE_MATCHES


# ── similarity graph via sentence-transformers ────────────────────────────────

def build_similarity_pairs(df_a, df_b, threshold: float = 0.50):
    """
    Encode records with sentence-transformers and return
    a list of (a_id, b_id, similarity) tuples above threshold.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    FIELDS = ["name", "breed", "sire", "dam"]
    def text(row): return " ".join(str(row[f]) for f in FIELDS)

    model   = SentenceTransformer("all-MiniLM-L6-v2")
    emb_a   = model.encode([text(r) for _, r in df_a.iterrows()], show_progress_bar=False)
    emb_b   = model.encode([text(r) for _, r in df_b.iterrows()], show_progress_bar=False)
    sim_mat = cos_sim(emb_a, emb_b)

    pairs = []
    for i, a_id in enumerate(df_a["id"]):
        for j, b_id in enumerate(df_b["id"]):
            s = float(sim_mat[i, j])
            if s >= threshold:
                pairs.append((a_id, b_id, s))
    return pairs


# ── pyjedai Correlation Clustering ───────────────────────────────────────────

def run_pyjedai_correlation_clustering(df_a, df_b, threshold: float = 0.60):
    """
    Use pyjedai's CorrelationClustering on the similarity graph.
    pyjedai expects a Data object and similarity scores.
    """
    from pyjedai.datamodel import Data
    from pyjedai.clustering import CorrelationClustering

    data = Data(
        dataset_1         = df_a,
        id_column_name_1  = "id",
        dataset_2         = df_b,
        id_column_name_2  = "id",
        attributes_1      = ["name", "breed", "sire", "dam"],
        attributes_2      = ["name", "breed", "sire", "dam"],
    )

    # Build similarity graph from sentence-transformers
    sim_pairs = build_similarity_pairs(df_a, df_b, threshold=threshold)
    print(f"  Similarity pairs above {threshold}: {len(sim_pairs)}")

    # pyjedai CorrelationClustering needs a graph with similarity weights
    cc = CorrelationClustering()

    # Convert pairs to pyjedai format: dict {(id1, id2): weight}
    graph = {(a, b): s for a, b, s in sim_pairs}
    clusters = cc.process(graph, data)

    return clusters


def clusters_to_pairs(clusters):
    pairs = set()
    for cluster in clusters:
        a_ids = [x for x in cluster if str(x).startswith("A")]
        b_ids = [x for x in cluster if str(x).startswith("B")]
        for a in a_ids:
            for b in b_ids:
                pairs.add((str(a), str(b)))
    return pairs


# ── fallback: greedy pivot (if pyjedai not available or API changed) ──────────

def greedy_pivot_clustering(sim_pairs, all_ids, seed: int = 42):
    """
    Greedy pivot-based Correlation Clustering approximation.
    Used as fallback if pyjedai is unavailable.
    """
    import random
    random.seed(seed)

    graph = {}
    for a, b, s in sim_pairs:
        graph.setdefault(a, {})[b] = s
        graph.setdefault(b, {})[a] = s

    unassigned = set(all_ids)
    clusters   = []
    while unassigned:
        pivot   = random.choice(sorted(unassigned))
        cluster = {pivot}
        for neighbour in graph.get(pivot, {}):
            if neighbour in unassigned:
                cluster.add(neighbour)
        clusters.append(frozenset(cluster))
        unassigned -= cluster
    return clusters


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Correlation Clustering  (Clustering / Dok. 2)")
    print("Using: pyjedai  (fallback: greedy pivot)")
    print("=" * 60)

    all_ids = list(records_a["id"]) + list(records_b["id"])

    print(f"\n{'Threshold':>9} | {'Clusters':>8} | {'Multi':>5} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("  " + "-" * 56)

    for thr in [0.45, 0.55, 0.65, 0.70]:
        sim_pairs = build_similarity_pairs(records_a, records_b, threshold=thr)

        try:
            clusters = run_pyjedai_correlation_clustering(records_a, records_b, thr)
        except Exception as e:
            print(f"  [pyjedai unavailable: {e}] — using greedy pivot")
            clusters = greedy_pivot_clustering(sim_pairs, all_ids)

        multi = sum(1 for c in clusters if len(c) > 1)
        pairs = clusters_to_pairs(clusters)
        m     = evaluate(pairs)
        print(f"  {thr:>9.2f} | {len(clusters):>8} | {multi:>5} | "
              f"{m['precision']:>9} | {m['recall']:>6} | {m['f1']:>6}")

    # ── detail at best threshold ──────────────────────────────────────────
    print("\n--- Detail at threshold=0.65 ---")
    sim_pairs = build_similarity_pairs(records_a, records_b, threshold=0.65)
    try:
        clusters = run_pyjedai_correlation_clustering(records_a, records_b, 0.65)
    except Exception:
        clusters = greedy_pivot_clustering(sim_pairs, all_ids)

    true_set  = {(a, b) for a, b in TRUE_MATCHES}
    rec_a_idx = records_a.set_index("id")
    rec_b_idx = records_b.set_index("id")

    for cluster in sorted([c for c in clusters if len(c) > 1], key=lambda c: min(str(x) for x in c)):
        members = sorted(str(x) for x in cluster)
        a_ids   = [x for x in members if x.startswith("A")]
        b_ids   = [x for x in members if x.startswith("B")]
        is_tp   = any((a, b) in true_set for a in a_ids for b in b_ids)
        names   = ([rec_a_idx.loc[a, "name"] for a in a_ids if a in rec_a_idx.index] +
                   [rec_b_idx.loc[b, "name"] for b in b_ids if b in rec_b_idx.index])
        mark    = "TRUE MATCH" if is_tp else "FALSE POSITIVE"
        print(f"  {members}  {names}  [{mark}]")
