"""
AdapterEM  (Matching — Dok. 5)
================================
Real implementation using PEFT (Parameter-Efficient Fine-Tuning) with LoRA.

AdapterEM inserts small adapter modules into a frozen pre-trained language
model.  Only the adapter weights (~13% of params) are updated, making it:
  - GPU-memory efficient
  - Fast to adapt to a new domain / pedigree website
  - Strong in low-resource scenarios

We use LoRA (PEFT library) on bert-base-uncased.
LoRA is the modern, well-supported equivalent of the original adapter modules.

Install: pip install peft transformers torch
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from itertools import product

import pandas as pd
from data import records, evaluate, print_matched_pairs, all_record_pairs, inference_pairs, TRUE_MATCHES

SNORKEL_LABELS_PATH = "snorkel_labels.csv"

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 8
EPOCHS      = 5
LR          = 3e-4   # higher LR than full fine-tuning — only adapter weights
FIELDS      = ["name", "breed", "dob", "sire", "dam", "country"]

LORA_CONFIG = LoraConfig(
    task_type    = TaskType.SEQ_CLS,
    r            = 8,           # rank — controls adapter capacity
    lora_alpha   = 16,          # scaling factor
    lora_dropout = 0.1,
    target_modules = ["query", "value"],   # inject into attention Q and V
)


# ── helpers ───────────────────────────────────────────────────────────────────

def serialize(row) -> str:
    return " ".join(f"[COL] {f} [VAL] {row[f]}" for f in FIELDS)


class ERPairDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer, texts):
        self.pairs = pairs; self.labels = labels
        self.tokenizer = tokenizer; self.texts = texts

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        enc = self.tokenizer(self.texts[a], self.texts[b],
                              truncation=True, max_length=MAX_LEN,
                              padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, \
               torch.tensor(self.labels[idx], dtype=torch.long)


def build_labeled_data():
    """
    Load labels from Snorkel (snorkel_labels.csv) if available.
    Falls back to TRUE_MATCHES with a warning if Snorkel hasn't been run yet.
    """
    if os.path.exists(SNORKEL_LABELS_PATH):
        df = pd.read_csv(SNORKEL_LABELS_PATH)
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
        rng         = np.random.default_rng(0)
        neg_idx     = rng.choice(len(non_matches), len(true_list) * 2, replace=False)
        train_pairs  = true_list + [non_matches[i] for i in neg_idx]
        train_labels = [1] * len(true_list) + [0] * len(neg_idx)

    return train_pairs, train_labels


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"AdapterEM  (Matching / Dok. 5)  —  LoRA on {MODEL_NAME}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print("Loading base model and applying LoRA adapters ...")

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)

    # ── apply LoRA — only adapter weights will be trained ─────────────────
    model = get_peft_model(base_model, LORA_CONFIG)
    model.print_trainable_parameters()
    model.to(device)

    texts = {r["id"]: serialize(r) for _, r in records.iterrows()}

    print("\nLoading training labels ...")
    train_pairs, train_labels = build_labeled_data()
    infer_pairs = inference_pairs()
    print(f"  Training set: {len(train_pairs)} pairs ({sum(train_labels)} positives)")

    train_ds     = ERPairDataset(train_pairs, train_labels, tokenizer, texts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)

    # ── fine-tuning (only adapter weights update) ──────────────────────────
    print(f"\nFine-tuning adapters for {EPOCHS} epochs ...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch, batch_labels in train_loader:
            batch        = {k: v.to(device) for k, v in batch.items()}
            batch_labels = batch_labels.to(device)
            out          = model(**batch, labels=batch_labels)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}")

    # ── inference ─────────────────────────────────────────────────────────
    print("\nRunning inference ...")
    infer_ds     = ERPairDataset(infer_pairs, [0]*len(infer_pairs), tokenizer, texts)
    infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE)

    model.eval()
    predicted = set()
    with torch.no_grad():
        for i, (batch, _) in enumerate(infer_loader):
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            for j, p in enumerate(preds):
                if p == 1:
                    predicted.add(infer_pairs[i * BATCH_SIZE + j])

    m = evaluate(predicted)
    print(f"\nResults:")
    print(f"  Precision : {m['precision']}")
    print(f"  Recall    : {m['recall']}")
    print(f"  F1        : {m['f1']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print("\nMatched pairs:")
    print_matched_pairs(predicted)
    print("\nNote: AdapterEM shines when adapting across multiple domains —")
    print("      each pedigree website gets its own adapter while sharing")
    print("      the frozen base model.")
