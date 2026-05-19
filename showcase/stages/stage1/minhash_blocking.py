"""MinHash LSH blocking — fuzzy matching on character n-grams of name."""


def block_minhash(df, threshold: float = 0.4, num_perm: int = 128, ngram: int = 3) -> set:
    """
    MinHash LSH on character n-grams of the cat name.
    Catches typos and abbreviations that token/SNM blocking misses.
    Returns cross-source candidate pairs.

    Requires: pip install datasketch
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        raise ImportError("datasketch not installed — run: pip install datasketch")

    src = df.set_index("id")["source"]

    def _shingles(text: str, n: int) -> set:
        s = str(text).lower().replace(" ", "")
        return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

    lsh  = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mhs  = {}

    for _, row in df.iterrows():
        mh = MinHash(num_perm=num_perm)
        for shingle in _shingles(row["name"], ngram):
            mh.update(shingle.encode("utf-8"))
        lsh.insert(row["id"], mh)
        mhs[row["id"]] = mh

    candidates = set()
    for _, row in df.iterrows():
        rid = row["id"]
        for match_id in lsh.query(mhs[rid]):
            if match_id != rid and src.get(rid) != src.get(match_id):
                candidates.add(tuple(sorted((rid, match_id))))

    return candidates
