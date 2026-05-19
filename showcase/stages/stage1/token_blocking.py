import re
from collections import defaultdict

_DEFAULT_STOP = {
    "of", "the", "a", "an", "de", "von", "vom", "v", "vd",
    "la", "le", "van", "retr", "germ",
}


def block_token(df, stop_words=None, min_tok_len=2):
    """Inverted index on name+breed tokens. Returns cross-source candidate pairs."""
    if stop_words is None:
        stop_words = _DEFAULT_STOP
    src = df.set_index("id")["source"]
    idx = defaultdict(list)
    for _, row in df.iterrows():
        for col in ["name", "breed"]:
            for tok in re.findall(r"[a-z]+", str(row[col]).lower()):
                if tok not in stop_words and len(tok) >= min_tok_len:
                    idx[tok].append(row["id"])
    cands = set()
    for ids in idx.values():
        seen = list(dict.fromkeys(ids))
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                cands.add(tuple(sorted((seen[i], seen[j]))))
    return {(a, b) for a, b in cands if src.get(a) != src.get(b)}
