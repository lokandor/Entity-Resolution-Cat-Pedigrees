"""DeepER — bi-encoder cosine similarity, no training needed."""

_DEFAULT_FIELDS = ["name", "breed", "sire", "dam"]


def match_deeper(df, candidates, threshold=0.65,
                 model_name="all-MiniLM-L6-v2", fields=None):
    """Returns matched pairs whose embedding cosine similarity >= threshold."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from ._device import device_str

    if fields is None:
        fields = _DEFAULT_FIELDS

    import numpy as np

    model = SentenceTransformer(model_name, device=device_str())

    def _val(row, f):
        v = row.get(f, "") if hasattr(row, "get") else (row[f] if f in row.index else "")
        s = str(v).strip()
        return s if s and s.lower() not in ("nan", "none", "") else None

    rec_idx = df.set_index("id")

    # encode each field separately so we can skip empty ones per pair
    field_embs = {}
    for f in fields:
        vals = {r["id"]: (_val(r, f) or "") for _, r in df.iterrows()}
        field_embs[f] = {rid: emb for rid, emb in zip(
            vals.keys(),
            model.encode(list(vals.values()), show_progress_bar=False)
        )}

    matched = set()
    for a, b in candidates:
        ra, rb = rec_idx.loc[a], rec_idx.loc[b]
        sims = []
        for f in fields:
            va, vb = _val(ra, f), _val(rb, f)
            if va and vb:  # only compare when both sides have data
                sims.append(cos_sim([field_embs[f][a]], [field_embs[f][b]])[0, 0])
        if sims and float(np.mean(sims)) >= threshold:
            matched.add((a, b))
    return matched
