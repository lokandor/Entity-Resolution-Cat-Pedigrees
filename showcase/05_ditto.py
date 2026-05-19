"""
Ditto  (Matching — Dok. 4)
===========================
Fine-tunes a pre-trained language model on serialised record pairs.

Serialisation format (from the paper):
    [COL] name [VAL] Luna Sunshine [COL] breed [VAL] Labrador ...

The model sees both records concatenated with [SEP] and predicts
MATCH / NON_MATCH as a binary classification task.

Key difference from AdapterEM (06): Ditto does full fine-tuning of all
model weights, while AdapterEM only trains small adapter modules (~13%).
Ditto is typically more accurate given enough data; AdapterEM is more
parameter-efficient in low-resource scenarios.

Training labels come from Snorkel (04_snorkel_labeling.py).
Run Snorkel first to generate snorkel_labels.csv.

Install: pip install transformers torch pandas
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from itertools import product

from data import records, evaluate, print_matched_pairs, all_record_pairs, inference_pairs, TRUE_MATCHES

MODEL_NAME          = "bert-base-uncased"   # use roberta-base for best results
MAX_LEN             = 128
BATCH_SIZE          = 8
EPOCHS              = 5
LR                  = 2e-5
FIELDS              = ["name", "breed", "dob", "sire", "dam", "country"]
SNORKEL_LABELS_PATH = "snorkel_labels.csv"


# ── serialisation ─────────────────────────────────────────────────────────────

def serialize(row) -> str:
    return " ".join(f"[COL] {f} [VAL] {row[f]}" for f in FIELDS)


# ── dataset ───────────────────────────────────────────────────────────────────

class DittoDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer, texts):
        self.pairs     = pairs
        self.labels    = labels
        self.tokenizer = tokenizer
        self.texts     = texts

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        a_id, b_id = self.pairs[idx]
        enc = self.tokenizer(
            self.texts[a_id], self.texts[b_id],
            truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt",
        )
        return (
            {k: v.squeeze(0) for k, v in enc.items()},
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── label loading ─────────────────────────────────────────────────────────────

def load_labeled_data():
    if os.path.exists(SNORKEL_LABELS_PATH):
        df           = pd.read_csv(SNORKEL_LABELS_PATH)
        train_pairs  = list(zip(df["id_a"], df["id_b"]))
        train_labels = list(df["label"].astype(int))
        print(f"  Loaded {len(train_pairs)} labeled pairs from snorkel_labels.csv "
              f"({sum(train_labels)} positives)")
    else:
        print("  [snorkel_labels.csv not found — run 04_snorkel_labeling.py first]")
        print("  Falling back to TRUE_MATCHES for training labels.")
        all_pairs   = all_record_pairs()
        true_list   = list(TRUE_MATCHES)
        non_matches = [p for p in all_pairs if p not in set(map(tuple, TRUE_MATCHES))]
        rng         = np.random.default_rng(42)
        neg_idx     = rng.choice(len(non_matches), len(true_list) * 2, replace=False)
        train_pairs  = true_list + [non_matches[i] for i in neg_idx]
        train_labels = [1] * len(true_list) + [0] * len(neg_idx)

    return train_pairs, train_labels


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"Ditto  (Matching / Dok. 4)  —  full fine-tuning on {MODEL_NAME}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    print("\nLoading training labels ...")
    train_pairs, train_labels = load_labeled_data()
    infer_pairs = inference_pairs()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)
    model.to(device)

    texts = {r["id"]: serialize(r) for _, r in records.iterrows()}

    train_ds     = DittoDataset(train_pairs, train_labels, tokenizer, texts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"\nFine-tuning all weights for {EPOCHS} epochs ...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch, lbl in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lbl   = lbl.to(device)
            out   = model(**batch, labels=lbl)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}")

    # ── inference on all A×B pairs ────────────────────────────────────────
    print("\nRunning inference ...")
    infer_ds     = DittoDataset(infer_pairs, [0]*len(infer_pairs), tokenizer, texts)
    infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE)

    model.eval()
    predicted = set()
    with torch.no_grad():
        for i, (batch, _) in enumerate(infer_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            for j, p in enumerate(preds):
                if p == 1:
                    predicted.add(infer_pairs[i * BATCH_SIZE + j])

    m = evaluate(predicted)
    print(f"\nResults: P={m['precision']}  R={m['recall']}  F1={m['f1']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print("\nMatched pairs:")
    print_matched_pairs(predicted)
    print("\nNote: Ditto = full fine-tuning. AdapterEM (06) trains only ~13% of params.")
    print("      Use Ditto when data is abundant; AdapterEM for low-resource domains.")
