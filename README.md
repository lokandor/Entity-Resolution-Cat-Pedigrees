# Entity Resolution Pipeline — Pedigree Cat Records

## Problem Statement

We have ~2 million cat pedigree records in a single mixed dataset from multiple registries.
The same cat can appear multiple times (up to ~10 records per entity) with dirty, inconsistent,
or abbreviated data — different date formats, typos, country code variants, abbreviated names, etc.

The goal is to assign every record to an entity cluster:
- Records that match others → grouped into a shared cluster
- Records that don't match anything → singleton cluster (entity of their own)

Every record appears in exactly one cluster in the output. A later (out-of-scope) step will
determine which record in each cluster is the most authoritative.

No ground truth labels exist. Supervised matchers are trained using weak/LLM-generated labels.

---

## Pipeline Architecture

```
Mixed Dataset (2M records, multiple sources)
      │
      ▼
┌─────────────────┐
│  Stage 0        │  Remove obviously invalid records (runs by default)
│  Preprocessing  │  Missing fields, garbage values, impossible DOBs,
└────────┬────────┘  exact within-source duplicates
         │
         ▼
┌─────────────────┐
│  Stage 1        │  Reduces O(n²) pairs to a manageable candidate set
│  Blocking       │  Only pairs within the same block go forward
└────────┬────────┘
         │                   records that share no block
         ▼                   → singleton cluster (entity of their own)
┌─────────────────┐
│  Stage 2        │  Classifies candidate pairs as match / no-match
│  Matching       │  Supervised matchers trained on generated labels
└────────┬────────┘
         │                   pairs rejected by matcher
         ▼                   → records become singletons (if no other matches)
┌─────────────────┐
│  Stage 3        │  Groups matched pairs into entity clusters
│  Clustering     │  Every record ends up in exactly one cluster
└────────┬────────┘
         │
         ▼
Output: N clusters (matched groups + singletons)
        each cluster = one real-world cat entity
```

---

## Stage 0 — Preprocessing

Runs automatically before blocking. Filters out records that would only add noise.

| Filter | Condition | Reason |
|--------|-----------|--------|
| Missing critical fields | `name` or `breed` is null/empty/placeholder | Can't block or match without them |
| Placeholder values | `"N/A"`, `"unknown"`, `"null"`, `"-"`, `"?"`, etc. | Garbage data |
| Name too short | Fewer than 3 non-whitespace characters | Almost certainly invalid |
| Impossible DOB | Year outside 1950–2030 | Data entry error |
| Exact within-source duplicates | Same name+breed+dob+sire+dam+source | True duplicates, keep one |

Skip with: `python showcase/pipeline.py --no-preprocessing` (not recommended)

---

## Stage 1 — Blocking Methods

Blocking must have near-perfect **recall** — any pair missed here is permanently lost.

| Method | Strategy | Handles |
|--------|----------|---------|
| `token` | Inverted index on name + breed tokens | Exact/near token matches |
| `snm` | Sorted Neighbourhood on name prefix (window=5) | Alphabetically close names |
| `both` | Union of token + snm | Both of the above |
| `minhash` | MinHash LSH on character n-grams of name | Typos, abbreviations, fuzzy matches |

### Hyperparameters to tune — Blocking

| Method | Parameter | Current Value | Notes |
|--------|-----------|---------------|-------|
| `token` | stop words list | `{"of","the","a",...}` | Add domain-specific stops |
| `token` | min token length | `> 1` | Increase to reduce noise |
| `snm` | window size | `5` | Larger = higher recall, more candidates |
| `minhash` | similarity threshold | `0.4` | Lower = higher recall, more candidates |
| `minhash` | num_perm | `128` | More = more accurate estimate, slower |
| `minhash` | n-gram size | `3` | Smaller = catches more typos, larger buckets |

---

## Stage 2 — Labeling Strategy

Labels are generated automatically — labeling runs **after blocking**, only on candidate pairs.

### Snorkel (default)
Hand-crafted labeling functions (LFs) vote on each pair; a LabelModel aggregates them.
No human annotation or API needed.

Known weaknesses for this dataset:
- DOB formats vary (`2020-03-15` vs `15.03.2020`) — raw Jaccard scores 0 despite being equal
- Country codes not normalised (`DE` vs `Germany` → 0 similarity)
- No edit distance — single-character typos score low

### LLM Labeling
Send candidate pairs to an LLM: "are these the same cat?". Reasons over all fields
holistically — much more reliable than hand-crafted rules.

**LM Studio** (recommended — local, free, no cap):
```bash
# Load a model in LM Studio, enable the local server, then:
python showcase/pipeline.py --blocking both --matcher ditto --cluster cc \
    --labeler lmstudio --lm-model "qwen2.5-14b-instruct"
```

Recommended models for RX 9070 XT (16GB VRAM):

| Model | VRAM (Q4) | Quality |
|-------|-----------|---------|
| `qwen2.5-14b-instruct` | ~8GB | Best quality/speed |
| `phi-4` | ~8GB | Strong reasoning |
| `qwen2.5-32b-instruct` | ~17GB | Better, tight fit |

**OpenAI / Anthropic API** (capped, pay-per-token):
```bash
python showcase/pipeline.py --blocking both --matcher ditto --cluster cc \
    --labeler openai --lm-model "gpt-4o-mini"
```

> Claude API (`console.anthropic.com`) is separate from Claude.ai subscription — billed per token.
> ~1000 pairs with `claude-haiku-4-5` costs cents.

### Labeling Strategy Comparison

| Strategy | Human effort | Quality | Scale | Cost |
|----------|-------------|---------|-------|------|
| Snorkel | Write LFs | Medium | Unlimited | Free |
| LM Studio | None | High | Unlimited | Free (local GPU) |
| OpenAI / Claude API | None | High | Cost-capped | Low |

---

## Stage 2 — Matching Methods

| Method | Type | Model | Description |
|--------|------|-------|-------------|
| `deeper` | Unsupervised | `all-MiniLM-L6-v2` | Bi-encoder cosine similarity, no training needed |
| `adapter` | Supervised | `bert-base-uncased` + LoRA | Parameter-efficient BERT fine-tuning (~13% weights) |
| `deepmatcher` | Supervised | MLP on TF-IDF diff vectors | Lightweight, no transformer |
| `cot` | Supervised | `distilbert-base-uncased` | Distilled from LLM labels |

### Hyperparameters to tune — Matching

**DeepER (`deeper`)**

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| Similarity threshold | `0.65` | Lower = higher recall, more FP |
| Sentence model | `all-MiniLM-L6-v2` | Try `all-mpnet-base-v2` |
| Fields used | `name, breed, sire, dam` | Add/remove fields |

**AdapterEM (`adapter`)**

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| LoRA rank (r) | `8` | Higher = more expressive, more params |
| LoRA alpha | `16` | Controls scaling: alpha/r ratio |
| LoRA dropout | `0.1` | Regularisation |
| Target modules | `query, value` | Can add `key`, `dense` |
| (+ all Ditto params) | | |

**DeepMatcher (`deepmatcher`)**

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| TF-IDF features (name) | `64` | Increase for richer representation |
| TF-IDF features (breed) | `32` | |
| Hidden size | `64` | |
| Epochs | `25` | |
| Learning rate | `1e-3` | |
| Batch size | `16` | |

**CoT Distillation (`cot`)**

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| Student model | `distilbert-base-uncased` | Try `bert-base-uncased` |
| Teacher / labeler | set via `--labeler` | lmstudio recommended |
| (+ Ditto training params) | | |

---

## Stage 3 — Clustering Methods

| Method | Description | Weakness |
|--------|-------------|----------|
| `cc` | Connected Components — O(n+m), deterministic | Chaining: one wrong edge merges whole clusters |
| `corr` | Correlation Clustering — optimises global consistency | Slower; needs edge weights |

### Hyperparameters to tune — Clustering

**Correlation Clustering (`corr`)**

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| Embedding model | `all-MiniLM-L6-v2` | Used for edge weights |
| Fields for embeddings | `name, breed, sire, dam` | |
| Similarity threshold | implicit via pyjedai | Tune pivot threshold in greedy fallback |

---

## Results Table — 16 Combinations

`token` and `snm` dropped (dominated by `both`). `ditto` dropped (superseded by `adapter`).

Run with: `python showcase/pipeline.py --run-all`
Fast mode (DeepER only, 4 combos): `python showcase/pipeline.py --run-all --fast`

| # | Blocking | Matcher | Cluster | Precision | Recall | F1 | TP | FP | FN | Notes |
|---|----------|---------|---------|-----------|--------|----|----|----|----|-------|
| 1 | both | deeper | cc | | | | | | | |
| 2 | both | deeper | corr | | | | | | | |
| 3 | both | adapter | cc | | | | | | | |
| 4 | both | adapter | corr | | | | | | | |
| 5 | both | deepmatcher | cc | | | | | | | |
| 6 | both | deepmatcher | corr | | | | | | | |
| 7 | both | cot | cc | | | | | | | |
| 8 | both | cot | corr | | | | | | | |
| 9 | minhash | deeper | cc | | | | | | | |
| 10 | minhash | deeper | corr | | | | | | | |
| 11 | minhash | adapter | cc | | | | | | | |
| 12 | minhash | adapter | corr | | | | | | | |
| 13 | minhash | deepmatcher | cc | | | | | | | |
| 14 | minhash | deepmatcher | corr | | | | | | | |
| 15 | minhash | cot | cc | | | | | | | |
| 16 | minhash | cot | corr | | | | | | | |

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Precision | TP / (TP + FP) | Of predicted matches, how many are correct |
| Recall | TP / (TP + FN) | Of true matches, how many were found |
| F1 | 2·P·R / (P+R) | Harmonic mean — main ranking metric |
| TP | — | Correctly predicted matches |
| FP | — | Predicted matches that are wrong |
| FN | — | True matches that were missed |

**Note:** Metrics work on synthetic data (ground truth known). On real data they show `n/a` —
manually label ~300 pairs as a fixed eval set to get real numbers.

---

## TODO — Experimental Plan

Work through stages in order. Fix the current stage before moving to the next —
a bad blocker makes matcher comparison meaningless, bad labels make matcher comparison misleading.

### Step 1 — Validate setup
- [ ] Install dependencies: `pip install -r showcase/requirements.txt`
- [ ] Run fast mode on synthetic data, confirm all 8 combos complete without errors:
      `python showcase/pipeline.py --run-all --fast`
- [ ] Check that results table prints with non-zero F1

### Step 2 — Compare blocking methods (do this first)
Blocking recall is the ceiling for everything downstream. Compare how many true matches
each method catches and how many candidates it generates.

- [ ] Run each blocking method with DeepER + CC (fastest matcher/cluster):
      ```
      python showcase/pipeline.py --blocking both    --matcher deeper --cluster cc
      python showcase/pipeline.py --blocking minhash --matcher deeper --cluster cc
      ```
- [ ] Record candidate count and recall for each (recall is more important than candidate count)
- [ ] Pick the blocking with highest recall — this is your baseline for all further experiments
- [ ] If two blockings have equal recall, prefer fewer candidates (faster downstream)

### Step 3 — Compare labeling strategies
With the best blocking fixed, check whether better labels improve matcher quality.
Use `ditto` as the test matcher (most sensitive to label quality).

- [ ] Run with Snorkel labels (default):
      `python showcase/pipeline.py --blocking <best> --matcher ditto --cluster cc`
- [ ] Run with LM Studio labels (load a model in LM Studio first):
      `python showcase/pipeline.py --blocking <best> --matcher ditto --cluster cc --labeler lmstudio --lm-model "qwen2.5-14b-instruct"`
- [ ] Compare F1 — if LM Studio labels give noticeably better F1, use them for all supervised matchers
- [ ] If Snorkel and LM Studio are close, stick with Snorkel (no GPU cost, no model to manage)

### Step 4 — Compare matchers
With best blocking + best labeling fixed, run all 5 matchers.

- [ ] Run all 40 combinations: `python showcase/pipeline.py --run-all`
      (or filter to best blocking only to save time)
- [ ] Fill in the results table above
- [ ] Note: `deeper` needs no labels, so it is unaffected by Step 3
- [ ] Note: `deepmatcher` is fastest to train — good sanity check before BERT matchers

### Step 5 — Compare clustering
With best blocking + best labeling + best matcher, compare CC vs correlation clustering.

- [ ] Check cluster size distribution — any cluster > 10 records is likely a chaining error (CC)
- [ ] If CC produces large spurious clusters, switch to `corr`
- [ ] `corr` is slower but more conservative about merging

### Step 6 — Scale to real data
- [ ] Start with a sample: `--data-dir data/ --sample 50000`
- [ ] Check candidate count from blocking — should be manageable (not 100M+)
- [ ] Check cluster size distribution: median, 90th percentile, max
- [ ] Manually spot-check 20–30 clusters: are the records actually the same cat?
- [ ] Scale up to full 2M records once satisfied with sample results

### Step 7 — Hyperparameter tuning (winner only)
Tune only the winning pipeline combination, not all 40.

- [ ] Blocking: adjust `minhash` threshold or SNM window if recall is not high enough
- [ ] Matcher: try different learning rates / epochs / base model
- [ ] Clustering: if CC chaining is a problem, tune `corr` pivot threshold

---

## Running the Pipeline

```bash
# Single combination (synthetic data)
python showcase/pipeline.py --blocking token --matcher deeper --cluster cc

# All 40 combinations
python showcase/pipeline.py --run-all

# Fast mode — DeepER only, 8 combos
python showcase/pipeline.py --run-all --fast

# With LM Studio labeling
python showcase/pipeline.py --blocking both --matcher ditto --cluster cc \
    --labeler lmstudio --lm-model "qwen2.5-14b-instruct"

# Real data — sample
python showcase/pipeline.py --data-dir data/ --sample 50000 \
    --blocking both --matcher deeper --cluster cc

# Real data — full
python showcase/pipeline.py --data-dir data/ --blocking both --matcher deeper --cluster cc
```
