# Entity Resolution Pipeline — Pedigree Cat Records

## Problem Statement

~447k cat pedigree records scraped from multiple registries. The same cat can appear
multiple times (up to ~10 records per entity) with dirty, inconsistent data — different
date formats, typos, country code variants, abbreviated names, etc.

The goal is to assign every record to an entity cluster:
- Records that match others → grouped into a shared cluster
- Records that don't match anything → singleton cluster (entity of their own)

No ground truth labels exist for real data. Supervised matchers are trained using
weak or LLM-generated labels.

---

## Running the App

```bash
cd showcase
streamlit run app/main.py
```

Open **http://localhost:8501** (or the remote IP on port 8501).

---

## Pipeline Architecture

```
Parquet (447k records, multiple sources)
      │
      ▼
┌─────────────────┐
│  Stage 0        │  Remove obviously invalid records (opt-in)
│  Preprocessing  │  Missing fields, short names, impossible DOBs,
└────────┬────────┘  exact within-source duplicates
         │
         ▼
┌─────────────────┐
│  Stage 1        │  Reduces O(n²) pairs to a manageable candidate set
│  Blocking       │  Only pairs within the same block go forward
└────────┬────────┘
         │
         ▼  (optional) Meta-Blocking: prune low-weight candidates
         │
         ▼
┌─────────────────┐
│  Stage 2        │  Classifies candidate pairs as match / no-match
│  Labeling       │  Snorkel LFs  |  OpenAI API  |  LM Studio (local)
│  Matching       │  Supervised matchers trained on generated labels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 3        │  Groups matched pairs into entity clusters
│  Clustering     │  Every record ends up in exactly one cluster
└────────┬────────┘
         │
         ▼
Output: N clusters — each cluster = one real-world cat entity
```

---

## Stage 0 — Preprocessing

Opt-in toggle in the app. Filters applied in order:

| Filter | Condition |
|--------|-----------|
| Missing critical fields | `name` or `breed` is null / empty / placeholder (`"N/A"`, `"unknown"`, etc.) |
| Name too short | Fewer than 3 non-whitespace characters |
| Impossible DOB | Year outside 1950–2030 |
| Exact within-source duplicates | Same `name+breed+dob+sire+dam` within one source — keep first |

---

## Stage 1 — Blocking

Blocking must have near-perfect **recall** — any pair missed here is permanently lost.

| Method | Strategy | Best for |
|--------|----------|---------|
| `token` | Inverted index on name + breed tokens | Exact / near token matches |
| `snm` | Sorted Neighbourhood on name prefix | Alphabetically close names |
| `both` | Union of token + snm | Broadest recall |
| `minhash` | MinHash LSH on character n-grams | Typos, abbreviations, fuzzy matches |

### Hyperparameters

| Method | Parameter | Default | Notes |
|--------|-----------|---------|-------|
| `token` | Stop words | `of, the, a, …` | Add domain-specific stops |
| `token` | Min token length | 2 | Increase to reduce noise |
| `snm` | Window size | 5 | Larger = higher recall, more candidates |
| `minhash` | Similarity threshold | 0.4 | Lower = higher recall, more candidates |
| `minhash` | Num permutations | 128 | More = more accurate, slower |
| `minhash` | N-gram size | 3 | Smaller = catches more typos |

### Meta-Blocking (optional)

Weighted Node Pruning: keep only candidate pairs that co-occur in ≥ N token blocks.
Raises precision without much recall loss. Threshold 0 = off, 2 = recommended starting point.

---

## Stage 2 — Labeling

Labels are generated automatically from candidate pairs — no manual annotation needed.
Only used by supervised matchers (`ditto`, `adapter`, `deepmatcher`, `cot`). Ignored by `deeper`.

| Strategy | How | Quality | Cost |
|----------|-----|---------|------|
| **Snorkel** | Hand-crafted labeling functions (Jaccard on name/breed/dob/sire/dam) | Medium | Free |
| **OpenAI API** | GPT prompt per pair — "same cat?" YES/NO | High | Pay-per-token |
| **LM Studio** | Local LLM via OpenAI-compatible server | High | Free (local GPU) |

### LM Studio setup

1. Download [LM Studio](https://lmstudio.ai) and load a model
2. Enable the local server (default: `http://localhost:1234`)
3. In the app: select **LM Studio**, enter the URL and model name

Recommended models for 16 GB VRAM:

| Model | VRAM (Q4) | Quality |
|-------|-----------|---------|
| `qwen2.5-14b-instruct` | ~8 GB | Best quality/speed |
| `phi-4` | ~8 GB | Strong reasoning |
| `qwen2.5-32b-instruct` | ~17 GB | Better, tight fit |

### Known Snorkel weaknesses

- DOB formats vary (`2020-03-15` vs `15.03.2020`) — Jaccard scores 0 despite equal dates
- Country codes not normalised (`DE` vs `Germany`)
- No edit distance — single-character typos score low

---

## Stage 2 — Matching

| Method | Type | Model | Notes |
|--------|------|-------|-------|
| `deeper` | Unsupervised | `all-MiniLM-L6-v2` | Bi-encoder cosine similarity, no labels needed |
| `ditto` | Supervised | `bert-base-uncased` | Full BERT fine-tuning |
| `adapter` | Supervised | `bert-base-uncased` + LoRA | ~13% of weights trained |
| `deepmatcher` | Supervised | MLP on TF-IDF diff vectors | Lightweight, no transformer |
| `cot` | Supervised | `distilbert-base-uncased` | Distilled from LLM labels |

### Hyperparameters

| Matcher | Parameter | Default |
|---------|-----------|---------|
| `deeper` | Similarity threshold | 0.65 |
| `deeper` | Sentence model | `all-MiniLM-L6-v2` |
| `ditto` | Epochs / LR | 5 / 2e-5 |
| `adapter` | LoRA r / alpha / dropout | 8 / 16 / 0.1 |
| `adapter` | Epochs / LR | 5 / 2e-5 |
| `deepmatcher` | Hidden size / Epochs / LR | 64 / 25 / 1e-3 |
| `deepmatcher` | TF-IDF features (name/breed) | 64 / 32 |
| `cot` | Student model / Epochs | `distilbert-base-uncased` / 5 |

---

## Stage 3 — Clustering

| Method | Description | Watch out for |
|--------|-------------|--------------|
| `cc` | Connected Components — O(n+m), deterministic | Chaining: one wrong edge merges whole clusters |
| `corr` | Correlation Clustering — optimises global consistency | Slower; needs edge weights |

---

## Evaluation

Metrics are computed automatically when **synthetic data** is loaded (ground truth known via `entity_id` column).
On real data the metrics section is hidden — manually label ~300 pairs to get real numbers.

| Metric | Formula |
|--------|---------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2·P·R / (P+R) |

### Synthetic dataset

Generated by `generate_synthetic.py` → `DATA/synthetic_cats.parquet`

- 412 rows: 120 multi-source entities (2–5 records each across 3 sources) + 40 singletons
- Same 27 raw columns as real data + `entity_id` ground-truth column
- Realistic noise: date format variants, abbreviated cattery names, case changes, missing fields

```bash
python generate_synthetic.py   # regenerate if needed
```

---

## Experimental Plan

Work through stages in order. A bad blocker makes matcher comparison meaningless;
bad labels make matcher comparison misleading.

### Step 1 — Validate setup
Load synthetic data in the app, run `deeper + cc`, confirm non-zero F1.

### Step 2 — Compare blocking methods
Run each blocking method with `deeper + cc` (fastest). Record candidate count and recall.
Pick the blocking with highest recall as your baseline.

### Step 3 — Compare labeling strategies
With best blocking fixed, compare Snorkel vs LM Studio using `ditto`.
If LM Studio gives noticeably better F1, use it for all supervised matchers.

### Step 4 — Compare matchers
With best blocking + best labeling, run all 5 matchers. Note that `deeper` is unaffected
by labeling choice. `deepmatcher` is fastest to train — good sanity check before BERT matchers.

### Step 5 — Compare clustering
Check cluster size distribution. Any cluster > 10 records is likely a chaining error (CC).
Switch to `corr` if CC produces large spurious clusters.

### Step 6 — Scale to real data
Start with a 50k sample, check candidate count and cluster size distribution.
Manually spot-check 20–30 clusters. Scale to full 447k once satisfied.

### Step 7 — Hyperparameter tuning
Tune only the winning combination.

---

## Project Structure

```
showcase/
  app/
    main.py                    Streamlit app (Data / Pipeline / Results tabs)
  stages/
    stage0/
      preprocessing.py         Record-level filters
    stage1/
      token_blocking.py        Inverted index blocking
      snm_blocking.py          Sorted Neighbourhood blocking
      minhash_blocking.py      MinHash LSH blocking
      meta_blocking.py         WNP candidate pruning
    stage2/
      labeling.py              Snorkel / OpenAI / LM Studio label generation
      deeper.py                Bi-encoder matcher
      bert_matcher.py          Ditto (full fine-tune) + AdapterEM (LoRA)
      deepmatcher.py           MLP on TF-IDF diff vectors
      cot.py                   CoT distillation
    stage3/
      connected_components.py  CC clustering
      correlation_clustering.py Correlation clustering
      __init__.py              clusters_to_pairs() helper
  DATA/
    all_scrapers_merged.parquet  Real data (447k records)
    synthetic_cats.parquet       Synthetic data with ground truth
  generate_synthetic.py        Synthetic dataset generator
  load_data.py                 Real data loader (JSON -> pipeline schema)
  requirements.txt
```
