"""CoT Distillation — DistilBERT distilled from LLM or Snorkel labels."""

_FIELDS_ALL = ["name", "breed", "dob", "sire", "dam", "country"]
_BATCH = 8
_MAX_LEN = 128


def match_cot(df, candidates, student_model="distilbert-base-uncased", epochs=5, lr=2e-5,
              labeler="snorkel", lm_model=None, lm_url=None, max_label_pairs=None):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from .labeling import weak_labels
    from ._device import get_device

    def serialize(row):
        return " | ".join(f"{f}: {row[f]}" for f in _FIELDS_ALL if f in row.index)

    texts  = {r["id"]: serialize(r) for _, r in df.iterrows()}
    device = get_device()

    print(f"  Generating labels ({labeler}) ...")
    train_pairs, train_labels = weak_labels(df, candidates, labeler=labeler,
                                            lm_model=lm_model, lm_url=lm_url,
                                            max_pairs=max_label_pairs)

    tokenizer = AutoTokenizer.from_pretrained(student_model)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    student_model, num_labels=2).to(device)

    class PairDS(Dataset):
        def __init__(self, pairs, labels):
            self.pairs  = pairs
            self.labels = labels
        def __len__(self): return len(self.pairs)
        def __getitem__(self, i):
            a, b = self.pairs[i]
            enc = tokenizer(texts[a], texts[b], truncation=True, max_length=_MAX_LEN,
                            padding="max_length", return_tensors="pt")
            return ({k: v.squeeze(0) for k, v in enc.items()},
                    torch.tensor(self.labels[i], dtype=torch.long))

    loader = DataLoader(PairDS(train_pairs, train_labels), batch_size=_BATCH, shuffle=True)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"  Distilling into {student_model} — {epochs} epochs ...")
    for ep in range(epochs):
        model.train(); tot = 0.0
        for batch, lbl in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch, labels=lbl.to(device))
            out.loss.backward(); opt.step(); opt.zero_grad()
            tot += out.loss.item()
        print(f"    epoch {ep+1}/{epochs}  loss={tot/len(loader):.4f}")

    cands = list(candidates)
    model.eval()
    predicted = set()
    with torch.no_grad():
        for i, (batch, _) in enumerate(
                DataLoader(PairDS(cands, [0] * len(cands)), batch_size=_BATCH)):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            for j, p in enumerate(preds):
                if p == 1:
                    predicted.add(cands[i * _BATCH + j])
    return predicted
