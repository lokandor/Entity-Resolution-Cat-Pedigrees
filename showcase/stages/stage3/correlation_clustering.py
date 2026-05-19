"""Correlation Clustering — optimises global consistency. Falls back to greedy pivot."""
import random

_FIELDS = ["name", "breed", "sire", "dam"]


def cluster_corr(df, match_pairs, fields=None):
    """
    Groups matched pairs into entity clusters via Correlation Clustering.
    Falls back to greedy pivot approximation if pyjedai is unavailable.
    Returns list[frozenset] — every record in exactly one cluster.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    if fields is None:
        fields = _FIELDS

    model  = SentenceTransformer("all-MiniLM-L6-v2")
    texts  = {r["id"]: " ".join(str(r[f]) for f in fields if f in r.index)
              for _, r in df.iterrows()}
    ids    = list(texts)
    embs   = model.encode([texts[i] for i in ids], show_progress_bar=False)
    id2i   = {rid: k for k, rid in enumerate(ids)}

    sim_pairs = [
        (a, b, float(cos_sim([embs[id2i[a]]], [embs[id2i[b]]])[0, 0]))
        for a, b in match_pairs
    ]

    all_ids = list(df["id"])

    try:
        from pyjedai.datamodel import Data
        from pyjedai.clustering import CorrelationClustering
        df_a = df[df["source"] == df["source"].iloc[0]].reset_index(drop=True)
        df_b = df[df["source"] != df["source"].iloc[0]].reset_index(drop=True)
        data = Data(dataset_1=df_a, id_column_name_1="id",
                    dataset_2=df_b, id_column_name_2="id",
                    attributes_1=fields, attributes_2=fields)
        graph    = {(a, b): s for a, b, s in sim_pairs}
        clusters = CorrelationClustering().process(graph, data)
        return [frozenset(c) for c in clusters]
    except Exception:
        graph = {}
        for a, b, s in sim_pairs:
            graph.setdefault(a, {})[b] = s
            graph.setdefault(b, {})[a] = s
        random.seed(42)
        unassigned = set(all_ids)
        clusters   = []
        while unassigned:
            pivot   = min(unassigned)
            cluster = {pivot} | {n for n in graph.get(pivot, {}) if n in unassigned}
            clusters.append(frozenset(cluster))
            unassigned -= cluster
        return clusters
