"""
DeepMatcher  (Matching — Dok. 4 & 5)
======================================
Real PyTorch implementation of the DeepMatcher architecture.

Architecture (from the original paper / GitHub: anhaidgroup/deepmatcher):
  1. Attribute Embedding  : embed each attribute's tokens with a GRU
  2. Attribute Comparison : compute abs-difference of attribute representations
  3. Aggregation          : attention-weighted sum over attributes
  4. Classification       : MLP → match / no-match

Original deepmatcher pip package has PyTorch/torchtext compatibility issues
with newer versions, so we implement the architecture directly in PyTorch.

Install: pip install torch numpy pandas scikit-learn
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import re
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import product

from data import records, evaluate, print_matched_pairs, all_record_pairs, inference_pairs, TRUE_MATCHES

SNORKEL_LABELS_PATH = "snorkel_labels.csv"

ATTRIBUTES  = ["name", "breed", "dob", "sire", "dam", "country"]
VOCAB_SIZE  = 256        # character-level (ord values)
EMBED_DIM   = 32
HIDDEN_DIM  = 64
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 1e-3


# ── character tokenisation ────────────────────────────────────────────────────

def char_ids(text: str, max_len: int = 30) -> list:
    """Convert string to list of character ASCII ids, padded/truncated."""
    ids = [min(ord(c), VOCAB_SIZE - 1) for c in str(text).lower()[:max_len]]
    return ids + [0] * (max_len - len(ids))


def record_to_tensors(row) -> list:
    """Return list of tensors — one per attribute."""
    return [torch.tensor(char_ids(row[a]), dtype=torch.long) for a in ATTRIBUTES]


# ── DeepMatcher model ─────────────────────────────────────────────────────────

class AttributeEncoder(nn.Module):
    """GRU over character embeddings — encodes one attribute into a vector."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.gru   = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embed(x)                     # [batch, seq_len, embed_dim]
        _, h = self.gru(emb)                    # h: [2, batch, hidden_dim]
        return torch.cat([h[0], h[1]], dim=-1)  # [batch, hidden_dim*2]


class DeepMatcher(nn.Module):
    def __init__(self, n_attrs: int = len(ATTRIBUTES)):
        super().__init__()
        self.encoder   = AttributeEncoder()
        self.attention = nn.Linear(HIDDEN_DIM * 2, 1)   # attribute importance
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )
        self.n_attrs = n_attrs

    def forward(self, attrs_a: list, attrs_b: list):
        """
        attrs_a, attrs_b: each is a list of n_attrs tensors [batch, seq_len]
        """
        comparisons = []
        for a_attr, b_attr in zip(attrs_a, attrs_b):
            ea = self.encoder(a_attr)                   # [batch, hidden*2]
            eb = self.encoder(b_attr)
            comparisons.append(torch.abs(ea - eb))      # element-wise abs diff

        # Stack: [batch, n_attrs, hidden*2]
        stacked = torch.stack(comparisons, dim=1)

        # Attention over attributes
        attn_scores  = self.attention(stacked).squeeze(-1)  # [batch, n_attrs]
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        aggregated   = (stacked * attn_weights).sum(dim=1)  # [batch, hidden*2]

        return self.classifier(aggregated)


# ── dataset ───────────────────────────────────────────────────────────────────

class DeepMatcherDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs  = pairs
        self.labels = labels
        self.rec_idx = records.set_index("id")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        a_id, b_id = self.pairs[idx]
        ta = record_to_tensors(self.rec_idx.loc[a_id])
        tb = record_to_tensors(self.rec_idx.loc[b_id])
        return ta, tb, torch.tensor(self.labels[idx], dtype=torch.long)


def collate(batch):
    attrs_a = [[] for _ in ATTRIBUTES]
    attrs_b = [[] for _ in ATTRIBUTES]
    labels  = []
    for ta, tb, lbl in batch:
        for i, (a, b) in enumerate(zip(ta, tb)):
            attrs_a[i].append(a)
            attrs_b[i].append(b)
        labels.append(lbl)
    attrs_a = [torch.stack(x) for x in attrs_a]
    attrs_b = [torch.stack(x) for x in attrs_b]
    return attrs_a, attrs_b, torch.stack(labels)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DeepMatcher  (Matching / Dok. 4, 5)  —  PyTorch GRU architecture")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if os.path.exists(SNORKEL_LABELS_PATH):
        df     = pd.read_csv(SNORKEL_LABELS_PATH)
        pairs  = list(zip(df["id_a"], df["id_b"]))
        labels = list(df["label"].astype(int))
        print(f"Loaded {len(pairs)} labeled pairs from snorkel_labels.csv "
              f"({sum(labels)} positives)")
    else:
        print("[snorkel_labels.csv not found — run 04_snorkel_labeling.py first]")
        print("Falling back to TRUE_MATCHES for training labels.")
        all_pairs   = all_record_pairs()
        true_list   = list(TRUE_MATCHES)
        non_matches = [p for p in all_pairs if p not in set(map(tuple, TRUE_MATCHES))]
        rng         = np.random.default_rng(42)
        neg_idx     = rng.choice(len(non_matches), len(true_list) * 3, replace=False)
        pairs  = true_list + [non_matches[i] for i in neg_idx]
        labels = [1] * len(true_list) + [0] * len(neg_idx)

    infer_pairs = inference_pairs()

    rng = np.random.default_rng(42)
    idx    = rng.permutation(len(pairs))
    pairs  = [pairs[i] for i in idx]
    labels = [labels[i] for i in idx]

    split       = int(0.75 * len(pairs))
    train_ds    = DeepMatcherDataset(pairs[:split], labels[:split])
    test_ds     = DeepMatcherDataset(pairs[split:], labels[split:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model     = DeepMatcher().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining for {EPOCHS} epochs ...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for attrs_a, attrs_b, lbl in train_loader:
            attrs_a = [a.to(device) for a in attrs_a]
            attrs_b = [b.to(device) for b in attrs_b]
            lbl     = lbl.to(device)
            out     = model(attrs_a, attrs_b)
            loss    = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}")

    # ── inference on all A×B pairs ────────────────────────────────────────
    infer_ds     = DeepMatcherDataset(infer_pairs, [0]*len(infer_pairs))
    infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE, collate_fn=collate)

    model.eval()
    predicted = set()
    with torch.no_grad():
        for i, (attrs_a, attrs_b, _) in enumerate(infer_loader):
            attrs_a = [a.to(device) for a in attrs_a]
            attrs_b = [b.to(device) for b in attrs_b]
            preds   = model(attrs_a, attrs_b).argmax(dim=-1).cpu().numpy()
            for j, p in enumerate(preds):
                if p == 1:
                    predicted.add(infer_pairs[i * BATCH_SIZE + j])

    m = evaluate(predicted)
    print(f"\nResults: P={m['precision']}  R={m['recall']}  F1={m['f1']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print("\nMatched pairs:")
    print_matched_pairs(predicted)
    print("\nArchitecture: GRU attribute encoder + attention aggregation + MLP")
    print("Original repo: https://github.com/anhaidgroup/deepmatcher")
