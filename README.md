# Entity Resolution — Cat Pedigree Records

Diploma project showcasing a full entity resolution (ER) pipeline applied to real-world cat pedigree data scraped from multiple registries (EasyPedigree, SibCats, WCF-BestCat, Katt, Kissat). The goal is to deduplicate and merge ~450 000 records into a single canonical dataset.

## Pipeline stages

| Stage | Methods |
|---|---|
| Preprocessing | Near-duplicate dataset detection (CatBoost) |
| Blocking | Token blocking, Sorted Neighbourhood, Meta-blocking / BLAST |
| Labeling | Snorkel weak supervision (no hand-labelled data needed) |
| Matching | DeepER, Ditto, AdapterEM (LoRA), DeepMatcher, CoT Distillation |
| Clustering | Connected Components, Correlation Clustering (pyjedai) |

## Quickstart

```bash
pip install -r showcase/requirements.txt

# interactive menu — browse methods by phase
python showcase/menu.py

# run a full pipeline in one command
python showcase/pipeline.py --blocking token --matcher deeper --cluster cc

# run all 30 combinations (slow without GPU)
python showcase/pipeline.py --run-all
```

## Real data setup

The showcase runs on a small synthetic dataset out of the box. To switch to real data:

1. Place all JSON batch files in `showcase/DATA/` (or point `merge_batches.py` at the source folder)
2. Run the merge script to produce a single Parquet file:
   ```bash
   python showcase/merge_batches.py
   ```
3. Pass the Parquet to the pipeline:
   ```python
   import pandas as pd
   from showcase.pipeline import _init_pipeline
   df = pd.read_parquet("showcase/DATA/all_scrapers_merged.parquet")
   _init_pipeline(df)
   ```

> `showcase/DATA/*.parquet` and `*.jsonl` are gitignored — generate them locally from the raw batch files.

## Structure

```
DP_METHODS.pdf          — method documentation
showcase/
  00_preprocessing.py   — near-duplicate dataset detection
  01_token_blocking.py  — token blocking
  02_sorted_neighborhood.py
  03_meta_blocking.py   — BLAST pruning
  04_snorkel_labeling.py
  05_ditto.py
  06_adapter_em.py      — LoRA fine-tuning
  07_deepmatcher.py
  08_deeper.py          — sentence-transformer bi-encoder
  09_cot_distillation.py
  10_connected_components.py
  11_correlation_clustering.py
  pipeline.py           — single configurable entry-point
  menu.py               — interactive TUI
  data.py               — synthetic dataset (30 records)
  load_data.py          — real data loader
  merge_batches.py      — merge raw JSON batches → Parquet / JSONL
  requirements.txt
```
