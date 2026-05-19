"""
Near-Duplicate Dataset Detection  (Preprocessing — Dok. 3)
===========================================================
Goal   : Before running ER, detect whether two *entire* datasets are
         near-duplicate variants scraped from the same source website.

Method : Extract dataset-level similarity features → CatBoost classifier
         (CatBoost is explicitly mentioned in Dok. 3 for this task).

Install: pip install catboost pandas numpy scikit-learn
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data import records_a, records_b


# ── feature extraction ────────────────────────────────────────────────────────

def _tokens(df: pd.DataFrame, cols: list) -> set:
    tokens = set()
    for col in cols:
        for v in df[col].dropna():
            tokens.update(str(v).lower().split())
    return tokens


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def dataset_features(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    """
    Dataset-level similarity features used by the near-dup classifier.
    All features are numeric — CatBoost handles them natively.
    """
    text_cols = ["name", "breed", "country"]
    tok1, tok2 = _tokens(df1, text_cols), _tokens(df2, text_cols)

    # Unique values per key attribute
    names1  = set(df1["name"].str.lower())
    names2  = set(df2["name"].str.lower())
    breeds1 = set(df1["breed"].str.lower())
    breeds2 = set(df2["breed"].str.lower())

    avg_len1 = df1["name"].str.len().mean()
    avg_len2 = df2["name"].str.len().mean()

    return [
        _jaccard(tok1, tok2),                                   # 1. token Jaccard (name+breed+country)
        _jaccard(names1, names2),                               # 2. name-value Jaccard
        _jaccard(breeds1, breeds2),                             # 3. breed Jaccard
        min(len(df1), len(df2)) / max(len(df1), len(df2)),      # 4. dataset size ratio
        1 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2), # 5. avg name-length similarity
        len(df1),                                               # 6. size of dataset 1
        len(df2),                                               # 7. size of dataset 2
    ]


FEATURE_NAMES = [
    "token_jaccard", "name_jaccard", "breed_jaccard",
    "size_ratio", "name_len_sim", "size_a", "size_b",
]


# ── simulation of dataset pairs for training ──────────────────────────────────

def _add_noise(df: pd.DataFrame, rng: np.random.Generator, frac: float = 0.4) -> pd.DataFrame:
    """Corrupt a fraction of name values (simulates re-scrape typos)."""
    df = df.copy()
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    mask  = rng.random(len(df)) < frac
    def corrupt(s):
        s = list(str(s))
        if len(s) > 3:
            i = rng.integers(1, len(s))
            s[i] = rng.choice(chars)
        return "".join(s)
    df.loc[mask, "name"] = df.loc[mask, "name"].apply(corrupt)
    return df


def build_training_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y = [], []

    # ── positive: noisy variants of the same source ──────────────────────
    for base_df in [records_a, records_b]:
        for _ in range(40):
            variant = _add_noise(base_df, rng, frac=rng.uniform(0.1, 0.5))
            sample  = variant.sample(frac=rng.uniform(0.6, 1.0),
                                     random_state=int(rng.integers(9999)))
            X.append(dataset_features(base_df, sample))
            y.append(1)

    # ── negative: shuffled/permuted columns → clearly different source ───
    for _ in range(80):
        shuffled = records_b.copy()
        shuffled["breed"]   = rng.permutation(shuffled["breed"].values)
        shuffled["country"] = rng.permutation(shuffled["country"].values)
        shuffled["name"]    = rng.permutation(shuffled["name"].values)
        X.append(dataset_features(records_a, shuffled))
        y.append(0)

    return np.array(X, dtype=np.float32), np.array(y)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Near-Duplicate Dataset Detection  (Preprocessing / Dok. 3)")
    print("Using: CatBoostClassifier")
    print("=" * 60)

    X, y = build_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    train_pool = Pool(X_train, y_train, feature_names=FEATURE_NAMES)
    test_pool  = Pool(X_test,  y_test,  feature_names=FEATURE_NAMES)

    clf = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=4,
        loss_function="Logloss",
        verbose=50,
        random_seed=42,
    )
    clf.fit(train_pool, eval_set=test_pool, early_stopping_rounds=20)

    y_pred = clf.predict(test_pool)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["different-source", "near-duplicate"]))

    print("Feature importances (CatBoost):")
    for name, imp in zip(FEATURE_NAMES, clf.get_feature_importance()):
        print(f"  {name:<20} {imp:.2f}")

    # ── predict on actual A vs B ──────────────────────────────────────────
    feat_ab = dataset_features(records_a, records_b)
    label   = clf.predict([feat_ab])[0]
    prob    = clf.predict_proba([feat_ab])[0]
    print(f"\nPrediction for records_a vs records_b:")
    print(f"  Features    : {[round(f, 3) for f in feat_ab]}")
    print(f"  Prediction  : {'near-duplicate' if label == 1 else 'different-source'}")
    print(f"  P(near-dup) : {prob[1]:.3f}")
    print("\n=> Proceed with full cross-source ER pipeline.")
