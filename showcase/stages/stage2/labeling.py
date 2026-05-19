"""Weak labeling for supervised matchers."""
import os
import re


_PROMPT = (
    "Are these two cat records the same cat?\n"
    "Record 1: {a}\nRecord 2: {b}\n"
    "Answer only YES or NO."
)


def _llm_labels(texts, cand_list, call_fn, max_pairs=200):
    """Generic LLM labeling loop. call_fn(prompt) -> 'YES'/'NO' string."""
    pairs, labels = [], []
    for a, b in cand_list[:max_pairs]:
        ans = call_fn(_PROMPT.format(a=texts[a], b=texts[b])).strip().upper()
        pairs.append((a, b))
        labels.append(1 if "YES" in ans else 0)
    return pairs, labels


def weak_labels(df, candidates, labeler="snorkel", lm_model=None, lm_url=None,
                max_pairs=None):
    """
    Apply labeling functions to candidate pairs.
    Returns (pairs, labels) suitable for training supervised matchers.

    max_pairs: cap on how many candidate pairs to label (None = all).
               For LLM labelers this controls API cost directly.

    labeler options:
      "snorkel"    — hand-crafted LFs via Snorkel (free, no GPU)
      "openai"     — OpenAI API (requires OPENAI_API_KEY env var)
      "lmstudio"   — LM Studio local server (requires LM Studio running on lm_url)
    """
    rec_idx = df.set_index("id")

    _FIELDS_ALL = ["name", "breed", "dob", "sire", "dam", "country"]

    def serialize(row):
        return " | ".join(f"{f}: {row[f]}" for f in _FIELDS_ALL if f in row.index)

    texts     = {r["id"]: serialize(r) for _, r in df.iterrows()}
    cand_list = list(candidates)
    if max_pairs is not None:
        import random as _random
        cand_list = _random.sample(cand_list, min(max_pairs, len(cand_list)))

    # ── OpenAI labeling ───────────────────────────────────────────────────────
    if labeler == "openai" and os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            print(f"  Generating OpenAI labels for {len(cand_list)} pairs ...")
            def _call(prompt):
                resp = client.chat.completions.create(
                    model=lm_model or "gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5, temperature=0,
                )
                return resp.choices[0].message.content
            pairs, labels = _llm_labels(texts, cand_list, _call, max_pairs=len(cand_list))
            print(f"  OpenAI labeled {len(pairs)} pairs")
            return pairs, labels
        except Exception as e:
            print(f"  [OpenAI labeling error: {e}] -- falling back to Snorkel")

    # ── LM Studio labeling ────────────────────────────────────────────────────
    if labeler == "lmstudio":
        _url = lm_url or os.environ.get("LM_STUDIO_URL", "http://localhost:1234")
        try:
            from openai import OpenAI
            client = OpenAI(base_url=f"{_url}/v1", api_key="lm-studio")
            model  = lm_model or "local-model"
            print(f"  Generating LM Studio labels ({model}) for {len(cand_list)} pairs ...")
            def _call(prompt):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5, temperature=0,
                )
                return resp.choices[0].message.content
            pairs, labels = _llm_labels(texts, cand_list, _call, max_pairs=len(cand_list))
            print(f"  LM Studio labeled {len(pairs)} pairs")
            return pairs, labels
        except Exception as e:
            print(f"  [LM Studio labeling error: {e}] -- falling back to Snorkel")

    def jac(a, b):
        sa = set(re.findall(r"[a-z]+", str(a).lower()))
        sb = set(re.findall(r"[a-z]+", str(b).lower()))
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

    def yr(d):
        m = re.search(r"(19|20)\d{2}", str(d))
        return m.group(0) if m else ""

    try:
        from snorkel.labeling import labeling_function, PandasLFApplier
        from snorkel.labeling.model import LabelModel
        import pandas as pd

        M, NM, AB = 1, 0, -1

        @labeling_function()
        def lf_nm_hi(x): return M  if jac(x.name_a,  x.name_b)  >= 0.50 else AB
        @labeling_function()
        def lf_nm_lo(x): return NM if jac(x.name_a,  x.name_b)  <  0.15 else AB
        @labeling_function()
        def lf_br_ok(x): return M  if jac(x.breed_a, x.breed_b) >= 0.50 else AB
        @labeling_function()
        def lf_br_no(x): return NM if jac(x.breed_a, x.breed_b) == 0.00 else AB
        @labeling_function()
        def lf_dob(x):
            ya, yb = yr(x.dob_a), yr(x.dob_b)
            return (M if ya == yb else NM) if ya and yb else AB
        @labeling_function()
        def lf_sr(x): return M if jac(x.sire_a, x.sire_b) >= 0.35 else AB
        @labeling_function()
        def lf_dm(x): return M if jac(x.dam_a,  x.dam_b)  >= 0.35 else AB

        LFS = [lf_nm_hi, lf_nm_lo, lf_br_ok, lf_br_no, lf_dob, lf_sr, lf_dm]

        rows = []
        for a, b in cand_list:
            r1, r2 = rec_idx.loc[a], rec_idx.loc[b]
            rows.append({
                "id_a": a, "id_b": b,
                "name_a":  r1["name"],  "name_b":  r2["name"],
                "breed_a": r1["breed"], "breed_b": r2["breed"],
                "dob_a":   r1["dob"],   "dob_b":   r2["dob"],
                "sire_a":  r1["sire"],  "sire_b":  r2["sire"],
                "dam_a":   r1["dam"],   "dam_b":   r2["dam"],
            })
        lf_df = pd.DataFrame(rows).reset_index(drop=True)
        L  = PandasLFApplier(lfs=LFS).apply(lf_df)
        lm = LabelModel(cardinality=2, verbose=False)
        lm.fit(L_train=L, n_epochs=300, lr=0.01, seed=42)
        hard = lm.predict(L, tie_break_policy="random")
        return list(zip(lf_df["id_a"], lf_df["id_b"])), list(hard)

    except ImportError:
        print("  [snorkel not installed — using Jaccard heuristic labels]")
        pairs, labels = [], []
        for a, b in candidates:
            r1, r2 = rec_idx.loc[a], rec_idx.loc[b]
            score = (jac(r1["name"], r2["name"]) * 2 +
                     jac(r1["breed"], r2["breed"])) / 3
            pairs.append((a, b))
            labels.append(1 if score >= 0.40 else 0)
        return pairs, labels
