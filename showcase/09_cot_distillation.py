"""
CoT Distillation  (Training / Augmentation — Dok. 1)
=====================================================
Real implementation following the paper's pipeline:
  1. Use an LLM to generate Chain-of-Thought (CoT) explanations for pairs.
  2. Fine-tune a smaller model (distilbert) on (serialised pair + CoT) → label.

This produces +10–23% cross-domain F1 vs training without CoT (Dok. 1).

LLM call:
  - Set OPENAI_API_KEY env var to use real OpenAI GPT-4o-mini.
  - Without the key the script falls back to a rule-based CoT generator
    so the distillation training still runs.

Install: pip install openai transformers torch datasets
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from itertools import product

from data import records, evaluate, print_matched_pairs, all_record_pairs, inference_pairs, TRUE_MATCHES

SNORKEL_LABELS_PATH = "snorkel_labels.csv"

MODEL_NAME = "distilbert-base-uncased"
FIELDS     = ["name", "breed", "dob", "sire", "dam", "country"]
MAX_LEN    = 192    # longer — fits pair + CoT reasoning
BATCH_SIZE = 4
EPOCHS     = 5
LR         = 2e-5


# ── LLM CoT generation ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert at matching pedigree animal records. "
    "Given two records, reason step by step about whether they refer to the "
    "same animal. Consider: name similarity, breed, birth date, sire and dam names. "
    "End with 'MATCH' or 'NO MATCH'."
)

PAIR_TEMPLATE = (
    "Record A: name={name_a}, breed={breed_a}, dob={dob_a}, "
    "sire={sire_a}, dam={dam_a}\n"
    "Record B: name={name_b}, breed={breed_b}, dob={dob_b}, "
    "sire={sire_b}, dam={dam_b}\n"
    "Reasoning:"
)


def llm_cot(row_a, row_b) -> str:
    """
    Call OpenAI API to generate a CoT explanation.
    Falls back to rule-based reasoning if OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            from openai import OpenAI
            client  = OpenAI(api_key=api_key)
            prompt  = PAIR_TEMPLATE.format(
                name_a=row_a["name"],  breed_a=row_a["breed"],
                dob_a=row_a["dob"],    sire_a=row_a["sire"],  dam_a=row_a["dam"],
                name_b=row_b["name"],  breed_b=row_b["breed"],
                dob_b=row_b["dob"],    sire_b=row_b["sire"],  dam_b=row_b["dam"],
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",  "content": SYSTEM_PROMPT},
                    {"role": "user",    "content": prompt},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [OpenAI error: {e}] — using rule-based fallback")

    # ── rule-based fallback ───────────────────────────────────────────────
    def overlap(a, b):
        sa, sb = set(str(a).lower()), set(str(b).lower())
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

    notes = []
    name_sim = overlap(row_a["name"], row_b["name"])
    notes.append(f"Name similarity is {name_sim:.2f} "
                 f"('{row_a['name']}' vs '{row_b['name']}').")

    dob_match = str(row_a["dob"])[:4] == str(row_b["dob"])[:4]
    notes.append("Birth year matches." if dob_match
                 else "Birth years differ — likely different entities.")

    breed_sim = overlap(row_a["breed"], row_b["breed"])
    notes.append(f"Breed overlap {breed_sim:.2f} "
                 f"('{row_a['breed']}' vs '{row_b['breed']}').")

    sire_sim = overlap(row_a["sire"], row_b["sire"])
    if sire_sim > 0.4:
        notes.append("Sire names are similar — strong evidence for a match.")

    verdict = "MATCH" if (name_sim > 0.55 and dob_match) else "NO MATCH"
    notes.append(f"Conclusion: {verdict}.")
    return " ".join(notes)


# ── generate CoT for all labeled pairs ───────────────────────────────────────

def generate_cot_dataset():
    all_pairs = all_record_pairs()

    if os.path.exists(SNORKEL_LABELS_PATH):
        df      = pd.read_csv(SNORKEL_LABELS_PATH)
        labeled = [(( r["id_a"], r["id_b"]), int(r["label"]))
                   for _, r in df.iterrows()]
        print(f"  Loaded {len(labeled)} labeled pairs from snorkel_labels.csv")
    else:
        print("  [snorkel_labels.csv not found — run 04_snorkel_labeling.py first]")
        print("  Falling back to TRUE_MATCHES for CoT pairs.")
        true_list   = list(TRUE_MATCHES)
        non_matches = [p for p in all_pairs if p not in TRUE_MATCHES]
        rng         = np.random.default_rng(42)
        neg_idx     = rng.choice(len(non_matches), len(true_list) * 2, replace=False)
        labeled     = ([(p, 1) for p in true_list]
                       + [(non_matches[i], 0) for i in neg_idx])

    rng = np.random.default_rng(42)
    rng.shuffle(labeled)

    rec_idx = records.set_index("id")

    print("Generating CoT explanations ...")
    dataset = []
    for (a_id, b_id), label in labeled:
        cot = llm_cot(rec_idx.loc[a_id], rec_idx.loc[b_id])
        dataset.append({"a_id": a_id, "b_id": b_id, "label": label, "cot": cot})

    return dataset


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

def serialize(row) -> str:
    return " ".join(f"[COL] {f} [VAL] {row[f]}" for f in FIELDS)


class CoTDataset(Dataset):
    def __init__(self, items, tokenizer, texts):
        self.items = items; self.tokenizer = tokenizer; self.texts = texts

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item    = self.items[idx]
        pair_text = self.texts[item["a_id"]] + " [SEP] " + self.texts[item["b_id"]]
        full_text = pair_text + " [REASON] " + item["cot"]
        enc = self.tokenizer(full_text, truncation=True, max_length=MAX_LEN,
                              padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, \
               torch.tensor(item["label"], dtype=torch.long)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CoT Distillation  (Training/Augmentation / Dok. 1)")
    print(f"Student model: {MODEL_NAME}")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    print(f"\nOpenAI API key: {'SET — using GPT-4o-mini' if api_key else 'NOT SET — using rule-based fallback'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cot_data = generate_cot_dataset()
    print(f"\nGenerated {len(cot_data)} CoT-augmented training examples.")
    print("\nSample CoT for first item:")
    print(f"  {cot_data[0]['a_id']} vs {cot_data[0]['b_id']}  label={cot_data[0]['label']}")
    print(f"  {cot_data[0]['cot'][:200]}...")

    # ── load tokenizer and model ──────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["[REASON]"], special_tokens=True)

    texts = {r["id"]: serialize(r) for _, r in records.iterrows()}

    split = int(0.75 * len(cot_data))
    train_items = cot_data[:split]
    test_items  = cot_data[split:]

    train_ds     = CoTDataset(train_items, tokenizer, texts)
    test_ds      = CoTDataset(test_items,  tokenizer, texts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_loader) * EPOCHS)

    print(f"\nDistilling into student model for {EPOCHS} epochs ...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch, lbl in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lbl   = lbl.to(device)
            out   = model(**batch, labels=lbl)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += out.loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}")

    # ── evaluate on all A×B pairs ─────────────────────────────────────────
    rec_idx   = records.set_index("id")
    infer_items = [{"a_id": a, "b_id": b, "label": 0,
                    "cot": llm_cot(rec_idx.loc[a], rec_idx.loc[b])}
                   for a, b in inference_pairs()]

    infer_ds     = CoTDataset(infer_items, tokenizer, texts)
    infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE)

    model.eval()
    predicted = set()
    with torch.no_grad():
        for i, (batch, _) in enumerate(infer_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            for j, p in enumerate(preds):
                if p == 1:
                    predicted.add(all_pairs[i * BATCH_SIZE + j])

    m = evaluate(predicted)
    print(f"\nResults: P={m['precision']}  R={m['recall']}  F1={m['f1']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print("\nMatched pairs:")
    print_matched_pairs(predicted)
    print("\nSet OPENAI_API_KEY to use real GPT-4o-mini for CoT generation.")
