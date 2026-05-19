import re


def block_snm(df, window=5):
    """Sorted Neighbourhood on normalised name prefix. Returns cross-source candidate pairs."""
    src = df.set_index("id")["source"]
    d = df.copy()
    d["_key"] = d["name"].apply(lambda s: re.sub(r"[^a-z]", "", str(s).lower())[:8])
    d = d.sort_values("_key").reset_index(drop=True)
    cands = set()
    n = len(d)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            cands.add(tuple(sorted((d.loc[i, "id"], d.loc[j, "id"]))))
    return {(a, b) for a, b in cands if src.get(a) != src.get(b)}
