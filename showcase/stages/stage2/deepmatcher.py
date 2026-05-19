"""DeepMatcher — MLP on per-attribute TF-IDF difference vectors."""


def match_deepmatcher(df, candidates, hidden=64, epochs=25, lr=1e-3,
                      f_name=64, f_breed=32, f_sire=32, f_dam=32,
                      labeler="snorkel", lm_model=None, lm_url=None,
                      max_label_pairs=None):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .labeling import weak_labels
    from ._device import get_device

    device = get_device()

    rec_idx   = df.set_index("id")
    vec_name  = TfidfVectorizer(max_features=f_name).fit(df["name"].astype(str))
    vec_breed = TfidfVectorizer(max_features=f_breed).fit(df["breed"].astype(str))
    vec_sire  = TfidfVectorizer(max_features=f_sire).fit(df["sire"].astype(str))
    vec_dam   = TfidfVectorizer(max_features=f_dam).fit(df["dam"].astype(str))

    def feat(r1, r2):
        vn = np.abs(vec_name.transform([r1["name"]]).toarray()   -
                    vec_name.transform([r2["name"]]).toarray())[0]
        vb = np.abs(vec_breed.transform([r1["breed"]]).toarray() -
                    vec_breed.transform([r2["breed"]]).toarray())[0]
        vs = np.abs(vec_sire.transform([r1["sire"]]).toarray()   -
                    vec_sire.transform([r2["sire"]]).toarray())[0]
        vd = np.abs(vec_dam.transform([r1["dam"]]).toarray()     -
                    vec_dam.transform([r2["dam"]]).toarray())[0]
        return np.concatenate([vn, vb, vs, vd]).astype(np.float32)

    in_dim = (len(vec_name.vocabulary_) + len(vec_breed.vocabulary_) +
              len(vec_sire.vocabulary_) + len(vec_dam.vocabulary_))
    batch_size = 16

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, 2),
            )
        def forward(self, x): return self.net(x)

    print(f"  Generating labels ({labeler}) ...")
    train_pairs, train_labels = weak_labels(df, candidates, labeler=labeler,
                                            lm_model=lm_model, lm_url=lm_url,
                                            max_pairs=max_label_pairs)

    class PairDS(Dataset):
        def __init__(self, pairs, labels):
            self.X = [feat(rec_idx.loc[a], rec_idx.loc[b]) for a, b in pairs]
            self.Y = labels
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            return torch.tensor(self.X[i]), torch.tensor(self.Y[i], dtype=torch.long)

    model   = MLP().to(device)
    loader  = DataLoader(PairDS(train_pairs, train_labels), batch_size=batch_size, shuffle=True)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Training DeepMatcher MLP on {device} — {epochs} epochs ...")
    for ep in range(epochs):
        model.train(); tot = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = loss_fn(out, y)
            loss.backward(); opt.step(); opt.zero_grad()
            tot += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1}/{epochs}  loss={tot/len(loader):.4f}")

    cands = list(candidates)
    model.eval()
    with torch.no_grad():
        X     = torch.tensor(np.array([feat(rec_idx.loc[a], rec_idx.loc[b]) for a, b in cands])).to(device)
        preds = model(X).argmax(dim=1).cpu().numpy()
    return {cands[i] for i, p in enumerate(preds) if p == 1}
