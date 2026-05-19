"""BERT-based matchers: Ditto (full fine-tune) and AdapterEM (LoRA)."""

_FIELDS_ALL = ["name", "breed", "dob", "sire", "dam", "country"]
_BATCH = 8
_MAX_LEN = 128


def _serialize(row):
    return " ".join(f"[COL] {f} [VAL] {row[f]}" for f in _FIELDS_ALL if f in row.index)


def _bert_match(df, candidates, model_name, lora=False,
                epochs=5, lr=2e-5, batch_size=_BATCH, max_len=_MAX_LEN,
                lora_r=8, lora_alpha=16, lora_dropout=0.1,
                labeler="snorkel", lm_model=None, lm_url=None, max_label_pairs=None):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from .labeling import weak_labels
    from ._device import get_device

    texts  = {r["id"]: _serialize(r) for _, r in df.iterrows()}
    device = get_device()

    print(f"  Generating labels ({labeler}) ...")
    train_pairs, train_labels = weak_labels(df, candidates, labeler=labeler,
                                            lm_model=lm_model, lm_url=lm_url,
                                            max_pairs=max_label_pairs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if lora:
        from peft import get_peft_model, LoraConfig, TaskType
        cfg   = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha,
                           target_modules=["query", "value"], lora_dropout=lora_dropout)
        model = get_peft_model(base, cfg).to(device)
    else:
        model = base.to(device)

    class PairDS(Dataset):
        def __init__(self, pairs, labels):
            self.pairs = pairs
            self.labels = labels
        def __len__(self): return len(self.pairs)
        def __getitem__(self, i):
            a, b = self.pairs[i]
            enc = tokenizer(texts[a], texts[b], truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            return ({k: v.squeeze(0) for k, v in enc.items()},
                    torch.tensor(self.labels[i], dtype=torch.long))

    loader = DataLoader(PairDS(train_pairs, train_labels), batch_size=batch_size, shuffle=True)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr)
    tag    = "LoRA" if lora else "full fine-tune"
    print(f"  Training {model_name} ({tag}) — {epochs} epochs ...")
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
                DataLoader(PairDS(cands, [0] * len(cands)), batch_size=batch_size)):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            for j, p in enumerate(preds):
                if p == 1:
                    predicted.add(cands[i * batch_size + j])
    return predicted


def match_ditto(df, candidates, epochs=5, lr=2e-5,
                labeler="snorkel", lm_model=None, lm_url=None, max_label_pairs=None):
    """Full BERT fine-tuning on labeled pairs."""
    return _bert_match(df, candidates, "bert-base-uncased", lora=False,
                       epochs=epochs, lr=lr,
                       labeler=labeler, lm_model=lm_model, lm_url=lm_url,
                       max_label_pairs=max_label_pairs)


def match_adapter(df, candidates, lora_r=8, lora_alpha=16, lora_dropout=0.1,
                  epochs=5, lr=2e-5, labeler="snorkel", lm_model=None, lm_url=None,
                  max_label_pairs=None):
    """Parameter-efficient LoRA fine-tuning (~13% of weights trained)."""
    return _bert_match(df, candidates, "bert-base-uncased", lora=True,
                       epochs=epochs, lr=lr,
                       lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                       labeler=labeler, lm_model=lm_model, lm_url=lm_url,
                       max_label_pairs=max_label_pairs)
