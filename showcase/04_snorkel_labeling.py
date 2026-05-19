"""
Snorkel Labeling  (Labeling — Dok. 3)
======================================
Generates training labels for supervised matching methods using
programmatic weak supervision — no hand-labeling required.

Pipeline:
  1. Candidate pairs from token blocking (cross-source A × B)
  2. Labeling functions (LFs) vote MATCH / NON_MATCH / ABSTAIN per pair
  3. Snorkel LabelModel combines noisy LF votes → probabilistic labels
  4. Labels saved to snorkel_labels.csv

Output: snorkel_labels.csv  (id_a, id_b, label, prob_match)
  → consumed by 05_adapter_em.py, 06_deepmatcher.py, 08_cot_distillation.py

Real-world note: at 4M records the same LFs run on the output of blocking
  (tens of millions of candidate pairs), making hand-labeling unnecessary.

Install: pip install snorkel recordlinkage pandas
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import re
import numpy as np
import pandas as pd

from data import records, evaluate, TRUE_MATCHES

SNORKEL_LABELS_PATH    = "snorkel_labels.csv"
CANDIDATE_PAIRS_PATH   = "candidate_pairs.csv"

MATCH     =  1
NON_MATCH =  0
ABSTAIN   = -1

# Country code normalisation — maps abbreviations to canonical names
COUNTRY_MAP = {
    "de":      "germany",
    "ger":     "germany",
    "uk":      "united kingdom",
    "gb":      "united kingdom",
    "england": "united kingdom",
    "scot.":   "scotland",
    "nl":      "netherlands",
    "no":      "norway",
    "cz":      "czech republic",
}


# ── similarity helpers ────────────────────────────────────────────────────────

def word_jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z]+", str(a).lower()))
    sb = set(re.findall(r"[a-z]+", str(b).lower()))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def extract_year(dob: str) -> str:
    m = re.search(r"(19|20)\d{2}", str(dob))
    return m.group(0) if m else ""


def canon_country(c: str) -> str:
    c = str(c).lower().strip().rstrip(".")
    return COUNTRY_MAP.get(c, c)


# ── token blocking (candidate pair generation) ────────────────────────────────

def token_blocking(df: pd.DataFrame) -> set:
    """
    Self-join token blocking on the full dataset.
    Returns all candidate pairs (within-source and cross-source)
    where both records share at least one token from name or breed.
    """
    STOPWORDS = {"of", "the", "a", "an", "de", "von", "vom", "v", "vd",
                 "la", "le", "van", "retr", "germ"}
    BLOCK_COLS = ["name", "breed"]

    from collections import defaultdict
    index = defaultdict(list)

    for _, row in df.iterrows():
        for col in BLOCK_COLS:
            for tok in re.findall(r"[a-z]+", str(row[col]).lower()):
                if tok not in STOPWORDS and len(tok) > 1:
                    index[tok].append(row["id"])

    candidates = set()
    for ids in index.values():
        unique_ids = list(dict.fromkeys(ids))
        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                candidates.add(tuple(sorted((unique_ids[i], unique_ids[j]))))
    return candidates


# ── build pairs DataFrame for Snorkel ────────────────────────────────────────

def build_pairs_df(candidates: set, rec_idx: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate pair with all field values for both records."""
    rows = []
    for id_1, id_2 in candidates:
        r1 = rec_idx.loc[id_1]
        r2 = rec_idx.loc[id_2]
        rows.append({
            "id_a":      id_1,        "id_b":      id_2,
            "name_a":    r1["name"],  "name_b":    r2["name"],
            "breed_a":   r1["breed"], "breed_b":   r2["breed"],
            "dob_a":     r1["dob"],   "dob_b":     r2["dob"],
            "sire_a":    r1["sire"],  "sire_b":    r2["sire"],
            "dam_a":     r1["dam"],   "dam_b":     r2["dam"],
            "country_a": r1["country"],"country_b": r2["country"],
        })
    return pd.DataFrame(rows)


# ── labeling functions ────────────────────────────────────────────────────────

try:
    from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
    from snorkel.labeling.model import LabelModel
except ImportError:
    print("Snorkel is not installed. Run: pip install snorkel")
    sys.exit(1)


@labeling_function()
def lf_name_high(x):
    """High name token overlap → likely match."""
    return MATCH if word_jaccard(x.name_a, x.name_b) >= 0.5 else ABSTAIN


@labeling_function()
def lf_name_low(x):
    """Very low name overlap → likely non-match."""
    return NON_MATCH if word_jaccard(x.name_a, x.name_b) < 0.15 else ABSTAIN


@labeling_function()
def lf_breed_mismatch(x):
    """No shared breed tokens at all → non-match."""
    return NON_MATCH if word_jaccard(x.breed_a, x.breed_b) == 0.0 else ABSTAIN


@labeling_function()
def lf_breed_match(x):
    """Strong breed token overlap → supporting evidence for match."""
    return MATCH if word_jaccard(x.breed_a, x.breed_b) >= 0.5 else ABSTAIN


@labeling_function()
def lf_dob_year(x):
    """Birth year agreement or disagreement."""
    ya, yb = extract_year(x.dob_a), extract_year(x.dob_b)
    if ya and yb:
        return MATCH if ya == yb else NON_MATCH
    return ABSTAIN


@labeling_function()
def lf_sire_match(x):
    """Similar sire names → strong evidence for match (parents are shared)."""
    return MATCH if word_jaccard(x.sire_a, x.sire_b) >= 0.35 else ABSTAIN


@labeling_function()
def lf_dam_match(x):
    """Similar dam names → strong evidence for match."""
    return MATCH if word_jaccard(x.dam_a, x.dam_b) >= 0.35 else ABSTAIN


@labeling_function()
def lf_country_conflict(x):
    """Clearly different countries after normalisation → non-match."""
    ca = canon_country(x.country_a)
    cb = canon_country(x.country_b)
    # Allow partial prefix match (e.g. "united kingdom" vs "united states")
    if ca == cb:
        return ABSTAIN
    if ca[:5] == cb[:5]:   # same prefix → could be abbreviation variant
        return ABSTAIN
    return NON_MATCH


LFS = [
    lf_name_high, lf_name_low,
    lf_breed_mismatch, lf_breed_match,
    lf_dob_year,
    lf_sire_match, lf_dam_match,
    lf_country_conflict,
]



# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Snorkel Labeling  (Labeling / Dok. 3)")
    print("=" * 60)

    rec_idx = records.set_index("id")

    # ── 1. Candidate pairs via token blocking ─────────────────────────────
    candidates = token_blocking(records)
    n = len(records)
    total_possible = n * (n - 1) // 2
    print(f"\nBlocking")
    print(f"  Total possible pairs   : {total_possible}")
    print(f"  Candidate pairs        : {len(candidates)}")
    print(f"  Reduction ratio        : {1 - len(candidates)/total_possible:.1%}")

    pairs_df = build_pairs_df(candidates, rec_idx)
    pairs_df = pairs_df.reset_index(drop=True)

    # ── 2. Apply labeling functions ───────────────────────────────────────
    print(f"\nApplying {len(LFS)} labeling functions ...")
    L = PandasLFApplier(lfs=LFS).apply(pairs_df)

    print("\nLF Analysis:")
    print(LFAnalysis(L=L, lfs=LFS).lf_summary().to_string())

    # ── 3. LabelModel ────────────────────────────────────────────────────
    print("\nTraining LabelModel ...")
    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=L, n_epochs=500, lr=0.01, seed=42)

    probs       = label_model.predict_proba(L)[:, 1]   # P(MATCH)
    hard_labels = label_model.predict(L, tie_break_policy="random")

    # ── 4. Save candidate pairs (full blocking output for matcher inference)
    pairs_df[["id_a", "id_b"]].to_csv(CANDIDATE_PAIRS_PATH, index=False)
    print(f"\nSaved {len(pairs_df)} candidate pairs → {CANDIDATE_PAIRS_PATH}")

    # ── 5. Save labels (training signal for supervised matchers) ──────────
    out = pairs_df[["id_a", "id_b"]].copy()
    out["label"]      = hard_labels
    out["prob_match"] = probs

    # Drop ABSTAIN rows (tie-break policy may still leave some at -1)
    out = out[out["label"] != ABSTAIN].reset_index(drop=True)

    out.to_csv(SNORKEL_LABELS_PATH, index=False)
    print(f"Saved {len(out)} labeled pairs → {SNORKEL_LABELS_PATH}")
    print(f"  MATCH labels     : {(out['label'] == 1).sum()}")
    print(f"  NON_MATCH labels : {(out['label'] == 0).sum()}")

    # ── 5. Evaluate against ground truth ─────────────────────────────────
    predicted = {(r.id_a, r.id_b) for _, r in out.iterrows() if r.label == 1}
    m = evaluate(predicted)
    print(f"\nLabel quality vs ground truth:")
    print(f"  Precision : {m['precision']}")
    print(f"  Recall    : {m['recall']}")
    print(f"  F1        : {m['f1']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    # ── 6. Sample output ──────────────────────────────────────────────────
    true_set = {(a, b) for a, b in TRUE_MATCHES}
    print("\nSample labeled pairs (highest prob_match):")
    print(f"  {'id_a':>4}  {'id_b':>4}  {'label':>5}  {'prob':>5}  {'ground truth':>12}")
    print("  " + "-" * 42)
    for _, row in out.sort_values("prob_match", ascending=False).head(15).iterrows():
        gt = "MATCH" if (row.id_a, row.id_b) in true_set else "non-match"
        print(f"  {row.id_a:>4}  {row.id_b:>4}  "
              f"{'MATCH' if row.label == 1 else 'NO':>5}  "
              f"{row.prob_match:>5.3f}  {gt:>12}")

    print("\nNext step: run a supervised matcher (05–09) — it will load")
    print(f"  snorkel_labels.csv  (training labels)")
    print(f"  candidate_pairs.csv (inference scope — only blocked pairs)")
