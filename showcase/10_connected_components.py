"""
Connected Components  (Clustering — Dok. 2)
============================================
Goal   : Group matched pairs into clusters where each cluster represents
         one real-world entity.

Method : Build a graph where nodes are record IDs and edges are predicted
         matches.  Each connected component = one entity cluster.

Pros   : Simple, fast, O(n+m) with union-find.
Cons   : One false-positive edge can chain unrelated records together
         ("chaining problem").
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx
from data import records, evaluate, inference_pairs, TRUE_MATCHES


def connected_components_clustering(match_pairs):
    """
    Build undirected graph from match pairs.
    Returns list of clusters (each cluster = frozenset of record IDs).
    """
    G = nx.Graph()
    G.add_nodes_from(records["id"])   # all records, regardless of source
    G.add_edges_from(match_pairs)
    return [frozenset(c) for c in nx.connected_components(G)]


def clusters_to_pairs(clusters):
    """Convert clusters to cross-source pairs for evaluation against TRUE_MATCHES."""
    rec_src = records.set_index("id")["source"]
    pairs   = set()
    for cluster in clusters:
        members = list(cluster)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if rec_src.get(a) != rec_src.get(b):
                    pairs.add(tuple(sorted((a, b))))
    return pairs


if __name__ == "__main__":
    print("=" * 60)
    print("Connected Components Clustering  (Clustering / Dok. 2)")
    print("=" * 60)

    # ── scenario 1 : perfect matches (no noise) ──────────────────────────
    print("\n--- Scenario 1: perfect match pairs (no false positives) ---")
    clusters = connected_components_clustering(TRUE_MATCHES)
    multi = [c for c in clusters if len(c) > 1]
    print(f"  Clusters formed : {len(clusters)}")
    print(f"  Multi-record    : {len(multi)}")
    for c in sorted(multi, key=lambda x: sorted(x)[0]):
        print(f"    {sorted(c)}")

    # ── scenario 2 : one false positive edge (chaining demo) ─────────────
    print("\n--- Scenario 2: one false-positive edge (chaining effect) ---")
    noisy_pairs = set(TRUE_MATCHES) | {("A01", "B11")}  # FP: A01 erroneously linked to B11
    clusters_noisy = connected_components_clustering(noisy_pairs)
    multi_noisy = [c for c in clusters_noisy if len(c) > 1]
    for c in sorted(multi_noisy, key=lambda x: sorted(x)[0]):
        flag = " <-- CHAINED" if len(c) > 2 else ""
        print(f"    {sorted(c)}{flag}")

    preds = clusters_to_pairs(clusters_noisy)
    m = evaluate(preds)
    print(f"\n  Impact of 1 FP edge: P={m['precision']}  R={m['recall']}  F1={m['f1']}")

    # ── scenario 3 : realistic output from Ditto (some FP/FN) ────────────
    print("\n--- Scenario 3: realistic predicted pairs from 04_ditto.py ---")
    from itertools import product as iproduct
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    FIELDS = ["name", "breed", "dob", "sire", "dam", "country"]
    def serialize(row):
        return " ".join(f"[COL] {f} [VAL] {row[f]}" for f in FIELDS)

    texts = {r["id"]: serialize(r) for _, r in records.iterrows()}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    vec.fit(list(texts.values()))

    predicted_pairs = set()
    for id_1, id_2 in inference_pairs():
        sim = cosine_similarity(vec.transform([texts[id_1]]),
                                vec.transform([texts[id_2]]))[0, 0]
        if sim >= 0.55:
            predicted_pairs.add((id_1, id_2))

    clusters_real = connected_components_clustering(predicted_pairs)
    multi_real    = [c for c in clusters_real if len(c) > 1]
    print(f"  Matched pairs used : {len(predicted_pairs)}")
    print(f"  Clusters formed    : {len(clusters_real)}")
    print(f"  Multi-record       : {len(multi_real)}")
    preds_real = clusters_to_pairs(clusters_real)
    m2 = evaluate(preds_real)
    print(f"  P={m2['precision']}  R={m2['recall']}  F1={m2['f1']}")
