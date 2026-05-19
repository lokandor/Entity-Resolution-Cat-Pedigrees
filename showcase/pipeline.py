"""
Configurable ER Pipeline
========================
Single entry-point: pick one method per stage and run the full pipeline.

Usage (single run):
    python pipeline.py --blocking token --matcher deeper --cluster cc
    python pipeline.py --blocking both  --matcher ditto  --cluster corr --preprocessing

Usage (batch — all 30 combinations):
    python pipeline.py --run-all          # all 30, results table (slow if GPU missing)
    python pipeline.py --run-all --fast   # only DeepER matcher, 6 combos

Stages:
    Blocking  : token | snm | both (union of token + sorted-neighbourhood)
    Matching  : deeper | ditto | adapter | deepmatcher | cot
    Clustering: cc (Connected Components) | corr (Correlation Clustering)

Supervised matchers (ditto / adapter / deepmatcher / cot) generate their own
training labels via Snorkel weak supervision — no hand-labelled data needed.
"""
import argparse, sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from data import records as _SYNTHETIC_RECORDS, evaluate as _SYNTHETIC_EVALUATE, TRUE_MATCHES

# ── active dataset (replaced by _init_pipeline when using real data) ──────────
records   = _SYNTHETIC_RECORDS
_SRC      = records.set_index("id")["source"]
_TRUTH    = TRUE_MATCHES        # None when no ground truth
_evaluate = _SYNTHETIC_EVALUATE


def _init_pipeline(df: pd.DataFrame, true_matches=None):
    """Switch the active dataset to `df` (e.g. loaded from JSON files)."""
    global records, _SRC, _TRUTH, _evaluate
    records   = df
    _SRC      = df.set_index("id")["source"]
    _TRUTH    = true_matches or set()
    if true_matches:
        _evaluate = lambda pairs: _SYNTHETIC_EVALUATE(pairs, true_matches)
    else:
        _evaluate = lambda pairs: {"precision": "n/a", "recall": "n/a", "f1": "n/a",
                                    "tp": "n/a", "fp": "n/a", "fn": "n/a"}


def _cross(pairs):
    return {(a, b) for a, b in pairs if _SRC.get(a) != _SRC.get(b)}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — Preprocessing  (stub — will be CatBoost near-dup detection)
# ═══════════════════════════════════════════════════════════════════════════════

def run_preprocessing():
    print("\n── Preprocessing ──────────────────────────────────────────────")
    print("  Near-duplicate dataset detection (CatBoost) — see 00_preprocessing.py")
    print("  [stub: full implementation coming]")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Blocking
# ═══════════════════════════════════════════════════════════════════════════════

_STOP    = {"of","the","a","an","de","von","vom","v","vd","la","le","van","retr","germ"}
_BLK_COL = ["name", "breed"]


def block_token():
    """Token blocking: records sharing any name/breed token become candidates."""
    idx = defaultdict(list)
    for _, row in records.iterrows():
        for col in _BLK_COL:
            for tok in re.findall(r"[a-z]+", str(row[col]).lower()):
                if tok not in _STOP and len(tok) > 1:
                    idx[tok].append(row["id"])
    cands = set()
    for ids in idx.values():
        seen = list(dict.fromkeys(ids))
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                cands.add(tuple(sorted((seen[i], seen[j]))))
    return _cross(cands)


def block_snm(window=5):
    """Sorted Neighbourhood: sort by normalised name prefix, slide window."""
    df = records.copy()
    df["_key"] = df["name"].apply(lambda s: re.sub(r"[^a-z]", "", str(s).lower())[:8])
    df = df.sort_values("_key").reset_index(drop=True)
    cands = set()
    n = len(df)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            cands.add(tuple(sorted((df.loc[i, "id"], df.loc[j, "id"]))))
    return _cross(cands)


def run_blocking(method):
    """Returns set of cross-source candidate (id_a, id_b) pairs."""
    if method == "token": return block_token()
    if method == "snm":   return block_snm()
    if method == "both":  return block_token() | block_snm()
    raise ValueError(f"Unknown blocking method: {method!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL — Snorkel weak labeling  (called automatically by supervised matchers)
# ═══════════════════════════════════════════════════════════════════════════════

def _weak_labels(candidates):
    """
    Apply Snorkel LFs to candidate pairs.
    Returns (pairs, labels) for use as training data.
    Falls back to the simple Jaccard heuristic if snorkel is not installed.
    """
    rec_idx = records.set_index("id")

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

        M = 1; NM = 0; AB = -1

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
        for a, b in candidates:
            r1, r2 = rec_idx.loc[a], rec_idx.loc[b]
            rows.append({
                "id_a": a, "id_b": b,
                "name_a":  r1["name"],  "name_b":  r2["name"],
                "breed_a": r1["breed"], "breed_b": r2["breed"],
                "dob_a":   r1["dob"],   "dob_b":   r2["dob"],
                "sire_a":  r1["sire"],  "sire_b":  r2["sire"],
                "dam_a":   r1["dam"],   "dam_b":   r2["dam"],
            })
        df  = pd.DataFrame(rows).reset_index(drop=True)
        L   = PandasLFApplier(lfs=LFS).apply(df)
        lm  = LabelModel(cardinality=2, verbose=False)
        lm.fit(L_train=L, n_epochs=300, lr=0.01, seed=42)
        hard = lm.predict(L, tie_break_policy="random")
        return list(zip(df["id_a"], df["id_b"])), list(hard)

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


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Matching
# ═══════════════════════════════════════════════════════════════════════════════

_FIELDS     = ["name", "breed", "sire", "dam"]
_FIELDS_ALL = ["name", "breed", "dob", "sire", "dam", "country"]


def match_deeper(candidates):
    """Bi-encoder: sentence-transformers cosine similarity (no training needed)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    THR   = 0.65
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = {r["id"]: " ".join(str(r[f]) for f in _FIELDS) for _, r in records.iterrows()}
    ids   = list(texts)
    embs  = model.encode([texts[i] for i in ids], show_progress_bar=False)
    id2i  = {rid: k for k, rid in enumerate(ids)}

    return {(a, b) for a, b in candidates
            if cos_sim([embs[id2i[a]]], [embs[id2i[b]]])[0, 0] >= THR}


def _bert_match(candidates, model_name, lora=False):
    """Shared BERT fine-tuning loop for Ditto and AdapterEM."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    BATCH = 8; EPOCHS = 5; LR = 2e-5; MAX_LEN = 128

    def serialize(row):
        return " ".join(f"[COL] {f} [VAL] {row[f]}" for f in _FIELDS_ALL)

    texts  = {r["id"]: serialize(r) for _, r in records.iterrows()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("  Generating weak labels (Snorkel) ...")
    train_pairs, train_labels = _weak_labels(candidates)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if lora:
        from peft import get_peft_model, LoraConfig, TaskType
        cfg   = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                           target_modules=["query", "value"], lora_dropout=0.1)
        model = get_peft_model(base, cfg).to(device)
    else:
        model = base.to(device)

    class PairDS(Dataset):
        def __init__(self, pairs, labels):
            self.pairs = pairs; self.labels = labels
        def __len__(self): return len(self.pairs)
        def __getitem__(self, i):
            a, b = self.pairs[i]
            enc = tokenizer(texts[a], texts[b], truncation=True, max_length=MAX_LEN,
                            padding="max_length", return_tensors="pt")
            return ({k: v.squeeze(0) for k, v in enc.items()},
                    torch.tensor(self.labels[i], dtype=torch.long))

    loader = DataLoader(PairDS(train_pairs, train_labels), batch_size=BATCH, shuffle=True)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    tag    = "LoRA" if lora else "full fine-tuning"

    print(f"  Training {model_name} ({tag}) — {EPOCHS} epochs ...")
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for batch, lbl in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch, labels=lbl.to(device))
            out.loss.backward(); opt.step(); opt.zero_grad()
            tot += out.loss.item()
        print(f"    epoch {ep+1}/{EPOCHS}  loss={tot/len(loader):.4f}")

    cands = list(candidates)
    model.eval()
    predicted = set()
    with torch.no_grad():
        for i, (batch, _) in enumerate(DataLoader(PairDS(cands, [0]*len(cands)), batch_size=BATCH)):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            for j, p in enumerate(preds):
                if p == 1:
                    predicted.add(cands[i * BATCH + j])
    return predicted


def match_ditto(candidates):
    """Full fine-tuning of BERT on Snorkel-labeled record pairs."""
    return _bert_match(candidates, "bert-base-uncased", lora=False)


def match_adapter(candidates):
    """Parameter-efficient LoRA fine-tuning (~13% of weights trained)."""
    return _bert_match(candidates, "bert-base-uncased", lora=True)


def match_deepmatcher(candidates):
    """MLP on per-attribute TF-IDF difference vectors (DeepMatcher spirit)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.feature_extraction.text import TfidfVectorizer

    HIDDEN = 64; EPOCHS = 25; LR = 1e-3; BATCH = 16; F_NAME = 64; F_BREED = 32

    rec_idx   = records.set_index("id")
    vec_name  = TfidfVectorizer(max_features=F_NAME).fit(records["name"].astype(str))
    vec_breed = TfidfVectorizer(max_features=F_BREED).fit(records["breed"].astype(str))

    def feat(r1, r2):
        vn = np.abs(vec_name.transform([r1["name"]]).toarray()   -
                    vec_name.transform([r2["name"]]).toarray())[0]
        vb = np.abs(vec_breed.transform([r1["breed"]]).toarray() -
                    vec_breed.transform([r2["breed"]]).toarray())[0]
        return np.concatenate([vn, vb]).astype(np.float32)

    IN = F_NAME + F_BREED

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(IN, HIDDEN), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(HIDDEN, 32), nn.ReLU(),
                nn.Linear(32, 2),
            )
        def forward(self, x): return self.net(x)

    print("  Generating weak labels (Snorkel) ...")
    train_pairs, train_labels = _weak_labels(candidates)

    class PairDS(Dataset):
        def __init__(self, pairs, labels):
            self.X = [feat(rec_idx.loc[a], rec_idx.loc[b]) for a, b in pairs]
            self.Y = labels
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            return torch.tensor(self.X[i]), torch.tensor(self.Y[i], dtype=torch.long)

    model  = MLP()
    loader = DataLoader(PairDS(train_pairs, train_labels), batch_size=BATCH, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Training DeepMatcher MLP — {EPOCHS} epochs ...")
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for x, y in loader:
            out = model(x); loss = loss_fn(out, y)
            loss.backward(); opt.step(); opt.zero_grad()
            tot += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1}/{EPOCHS}  loss={tot/len(loader):.4f}")

    cands = list(candidates)
    model.eval()
    with torch.no_grad():
        X     = torch.tensor(np.array([feat(rec_idx.loc[a], rec_idx.loc[b]) for a, b in cands]))
        preds = model(X).argmax(dim=1).numpy()
    return {cands[i] for i, p in enumerate(preds) if p == 1}


def match_cot(candidates):
    """
    CoT Distillation: OpenAI generates labels via chain-of-thought;
    DistilBERT distilled on those labels. Falls back to Snorkel if no API key.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    STUDENT = "distilbert-base-uncased"; BATCH = 8; EPOCHS = 5; LR = 2e-5; MAX_LEN = 128

    def serialize(row):
        return " | ".join(f"{f}: {row[f]}" for f in _FIELDS_ALL)

    texts  = {r["id"]: serialize(r) for _, r in records.iterrows()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── label generation ──────────────────────────────────────────────────────
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            print("  Generating labels via OpenAI CoT ...")
            train_pairs, train_labels = [], []
            for a, b in list(candidates)[:60]:      # cap API calls
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content":
                        f"Are these two dog records the same dog?\n"
                        f"Record 1: {texts[a]}\nRecord 2: {texts[b]}\n"
                        f"Answer only YES or NO."}],
                    max_tokens=5, temperature=0,
                )
                ans = resp.choices[0].message.content.strip().upper()
                train_pairs.append((a, b))
                train_labels.append(1 if "YES" in ans else 0)
            print(f"  OpenAI labeled {len(train_pairs)} pairs")
        except Exception as e:
            print(f"  [OpenAI error: {e}] — falling back to Snorkel labels")
            train_pairs, train_labels = _weak_labels(candidates)
    else:
        print("  [OPENAI_API_KEY not set] — using Snorkel labels for distillation")
        train_pairs, train_labels = _weak_labels(candidates)

    # ── distil into student model ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(STUDENT)
    model     = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=2).to(device)

    class PairDS(Dataset):
        def __init__(self, pairs, labels):
            self.pairs = pairs; self.labels = labels
        def __len__(self): return len(self.pairs)
        def __getitem__(self, i):
            a, b = self.pairs[i]
            enc = tokenizer(texts[a], texts[b], truncation=True, max_length=MAX_LEN,
                            padding="max_length", return_tensors="pt")
            return ({k: v.squeeze(0) for k, v in enc.items()},
                    torch.tensor(self.labels[i], dtype=torch.long))

    loader = DataLoader(PairDS(train_pairs, train_labels), batch_size=BATCH, shuffle=True)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"  Distilling into {STUDENT} — {EPOCHS} epochs ...")
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for batch, lbl in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch, labels=lbl.to(device))
            out.loss.backward(); opt.step(); opt.zero_grad()
            tot += out.loss.item()
        print(f"    epoch {ep+1}/{EPOCHS}  loss={tot/len(loader):.4f}")

    cands = list(candidates)
    model.eval()
    predicted = set()
    with torch.no_grad():
        for i, (batch, _) in enumerate(DataLoader(PairDS(cands, [0]*len(cands)), batch_size=BATCH)):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            for j, p in enumerate(preds):
                if p == 1:
                    predicted.add(cands[i * BATCH + j])
    return predicted


_MATCHERS = {
    "deeper":      match_deeper,
    "ditto":       match_ditto,
    "adapter":     match_adapter,
    "deepmatcher": match_deepmatcher,
    "cot":         match_cot,
}


def run_matching(method, candidates):
    if method not in _MATCHERS:
        raise ValueError(f"Unknown matcher: {method!r}. Choose: {list(_MATCHERS)}")
    return _MATCHERS[method](candidates)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def _clusters_to_pairs(clusters):
    """Extract cross-source pairs from entity clusters for evaluation."""
    final = set()
    for cluster in clusters:
        members = list(cluster)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = str(members[i]), str(members[j])
                if _SRC.get(a) != _SRC.get(b):
                    final.add(tuple(sorted((a, b))))
    return final


def cluster_cc(match_pairs):
    """Connected Components: O(n+m), simple but prone to chaining."""
    G = nx.Graph()
    G.add_nodes_from(records["id"])
    G.add_edges_from(match_pairs)
    return _clusters_to_pairs(nx.connected_components(G))


def cluster_corr(match_pairs):
    """
    Correlation Clustering via pyjedai.
    Falls back to greedy pivot approximation if pyjedai is unavailable.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    FIELDS = ["name", "breed", "sire", "dam"]
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    texts  = {r["id"]: " ".join(str(r[f]) for f in FIELDS) for _, r in records.iterrows()}
    ids    = list(texts)
    embs   = model.encode([texts[i] for i in ids], show_progress_bar=False)
    id2i   = {rid: k for k, rid in enumerate(ids)}

    sim_pairs = [(a, b, float(cos_sim([embs[id2i[a]]], [embs[id2i[b]]])[0, 0]))
                 for a, b in match_pairs]

    all_ids = list(records["id"])

    try:
        from pyjedai.datamodel import Data
        from pyjedai.clustering import CorrelationClustering
        df_a = records[records["source"] == "A"].reset_index(drop=True)
        df_b = records[records["source"] == "B"].reset_index(drop=True)
        data = Data(dataset_1=df_a, id_column_name_1="id",
                    dataset_2=df_b, id_column_name_2="id",
                    attributes_1=FIELDS, attributes_2=FIELDS)
        graph    = {(a, b): s for a, b, s in sim_pairs}
        clusters = CorrelationClustering().process(graph, data)
    except Exception:
        # Greedy pivot fallback
        graph = {}
        for a, b, s in sim_pairs:
            graph.setdefault(a, {})[b] = s
            graph.setdefault(b, {})[a] = s
        import random; random.seed(42)
        unassigned = set(all_ids); clusters = []
        while unassigned:
            pivot   = min(unassigned)
            cluster = {pivot} | {n for n in graph.get(pivot, {}) if n in unassigned}
            clusters.append(frozenset(cluster))
            unassigned -= cluster

    return _clusters_to_pairs(clusters)


_CLUSTERERS = {"cc": cluster_cc, "corr": cluster_corr}


def run_clustering(method, match_pairs):
    if method not in _CLUSTERERS:
        raise ValueError(f"Unknown clusterer: {method!r}. Choose: {list(_CLUSTERERS)}")
    return _CLUSTERERS[method](match_pairs)


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(blocking, matcher, cluster, preprocessing=False, verbose=True):
    """Run one pipeline combination. Returns metrics dict (or counts if no ground truth)."""
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  {len(records):,} records  |  "
              f"blocking={blocking}   matcher={matcher}   cluster={cluster}")
        print(f"{'═'*60}")

    if preprocessing:
        run_preprocessing()

    if verbose: print(f"\n── Blocking ({blocking}) ──────────────────────────────────────")
    candidates = run_blocking(blocking)
    if verbose:
        m = _evaluate(candidates)
        recall_str = f"  R={m['recall']}" if m["recall"] != "n/a" else ""
        print(f"  Candidates : {len(candidates):,}{recall_str}")

    if verbose: print(f"\n── Matching ({matcher}) ──────────────────────────────────────")
    match_pairs = run_matching(matcher, candidates)
    if verbose:
        m = _evaluate(match_pairs)
        if m["f1"] != "n/a":
            print(f"  Predicted  : {len(match_pairs):,}  "
                  f"P={m['precision']}  R={m['recall']}  F1={m['f1']}")
        else:
            print(f"  Predicted  : {len(match_pairs):,} match pairs")

    if verbose: print(f"\n── Clustering ({cluster}) ────────────────────────────────────")
    final_pairs = run_clustering(cluster, match_pairs)
    m = _evaluate(final_pairs)
    if verbose:
        if m["f1"] != "n/a":
            print(f"  P={m['precision']}  R={m['recall']}  F1={m['f1']}  "
                  f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
        else:
            print(f"  Resolved   : {len(final_pairs):,} cross-source entity pairs")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL 30 COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(fast=False):
    from itertools import product as iprod

    blockings = ["token", "snm", "both"]
    matchers  = ["deeper"] if fast else ["deeper", "ditto", "adapter", "deepmatcher", "cot"]
    clusters  = ["cc", "corr"]

    combos = list(iprod(blockings, matchers, clusters))

    print(f"\nRunning {len(combos)} pipeline combination(s) ...")
    if not fast:
        print("Note: ditto / adapter / cot require BERT (slow without GPU).")
        print("      Use --fast to only run DeepER (6 fast combos).\n")

    results = []
    for i, (bl, ma, cl) in enumerate(combos, 1):
        print(f"\n[{i:>2}/{len(combos)}]  blocking={bl:<5}  matcher={ma:<12}  cluster={cl}")
        try:
            m = run_pipeline(bl, ma, cl, verbose=False)
        except Exception as e:
            m = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                 "tp": 0, "fp": 0, "fn": 0, "_error": str(e)[:40]}
        results.append({"blocking": bl, "matcher": ma, "cluster": cl, **m})
        print(f"       P={m['precision']}  R={m['recall']}  F1={m['f1']}")

    # ── summary table sorted by F1 desc ───────────────────────────────────────
    results.sort(key=lambda r: -r["f1"])
    W = 74
    print(f"\n{'═'*W}")
    print(f" {'#':>2}  {'Blocking':<7}  {'Matcher':<12}  {'Cluster':<6}  "
          f"{'P':>6}  {'R':>6}  {'F1':>6}  Note")
    print(f"{'─'*W}")
    for rank, r in enumerate(results, 1):
        note = f"ERROR: {r.get('_error','')}" if "_error" in r else ""
        print(f" {rank:>2}  {r['blocking']:<7}  {r['matcher']:<12}  {r['cluster']:<6}  "
              f"{r['precision']:>6}  {r['recall']:>6}  {r['f1']:>6}  {note}")
    print(f"{'═'*W}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Configurable ER Pipeline — Pedigree Cat Records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # synthetic showcase data (default)
  python pipeline.py --blocking token --matcher deeper --cluster cc
  python pipeline.py --run-all --fast

  # real JSON data
  python pipeline.py --data-dir data/ --sample 50000 --blocking both --matcher deeper --cluster cc
  python pipeline.py --data-dir data/ --blocking both --matcher deeper --cluster cc
  python pipeline.py --data-dir data/ --run-all --fast
""",
    )
    parser.add_argument("--blocking",      choices=["token", "snm", "both"])
    parser.add_argument("--matcher",       choices=["deeper", "ditto", "adapter", "deepmatcher", "cot"])
    parser.add_argument("--cluster",       choices=["cc", "corr"])
    parser.add_argument("--preprocessing", action="store_true",
                        help="Run preprocessing stage before blocking")
    parser.add_argument("--run-all",       action="store_true",
                        help="Run all combinations and print a ranked results table")
    parser.add_argument("--fast",          action="store_true",
                        help="With --run-all: only use DeepER matcher (6 fast combos)")
    parser.add_argument("--data-dir",      default=None,
                        help="Load real JSON data from this directory instead of synthetic data")
    parser.add_argument("--sample",        type=int, default=None,
                        help="With --data-dir: load only the first N records (for development)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="With --data-dir: force rebuild of the Parquet cache")

    args = parser.parse_args()

    # ── load real data if requested ───────────────────────────────────────────
    if args.data_dir:
        from load_data import load_records
        df = load_records(
            data_dir=args.data_dir,
            sample=args.sample,
            rebuild_cache=args.rebuild_cache,
        )
        _init_pipeline(df)      # no ground truth for real data
        print(f"\nUsing real data: {len(records):,} records from "
              f"{records['source'].nunique()} sources")
    else:
        print(f"Using synthetic showcase data: {len(records)} records")

    if args.run_all:
        run_all(fast=args.fast)
    elif args.blocking and args.matcher and args.cluster:
        run_pipeline(args.blocking, args.matcher, args.cluster,
                     preprocessing=args.preprocessing)
    else:
        parser.print_help()
        print("\nError: supply --blocking, --matcher, --cluster  OR  --run-all")
        sys.exit(1)


if __name__ == "__main__":
    main()