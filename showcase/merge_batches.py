import json
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path("D:/User Stuff/Dowloand/all_scrapers_data")
OUTPUT_PARQUET = Path("D:/User Stuff/Dowloand/all_scrapers_merged.parquet")
OUTPUT_JSONL = Path("D:/User Stuff/Dowloand/all_scrapers_merged.jsonl")

files = sorted(DATA_DIR.glob("*.json"))
total = len(files)
print(f"Found {total} files to merge")

CHUNK_SIZE = 500

# Discover full column set from a sample of files spread across the dataset
sample_indices = list(range(0, total, max(1, total // 50)))
all_keys = set()
for idx in sample_indices:
    try:
        with open(files[idx], encoding="utf-8") as fh:
            data = json.load(fh)
        if data:
            rec = data[0] if isinstance(data, list) else data
            all_keys.update(rec.keys())
    except Exception:
        pass

print(f"Discovered {len(all_keys)} columns: {sorted(all_keys)}")

# Build a fixed pyarrow schema — everything as string (nulls included)
pa_schema = pa.schema([(col, pa.large_utf8()) for col in sorted(all_keys)])

parquet_writer = None
jsonl_out = open(OUTPUT_JSONL, "w", encoding="utf-8")

records_total = 0
errors = 0

for i in range(0, total, CHUNK_SIZE):
    chunk_files = files[i : i + CHUNK_SIZE]
    chunk_records = []

    for f in chunk_files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                chunk_records.extend(data)
            elif isinstance(data, dict):
                chunk_records.append(data)
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")
            errors += 1
            continue

    if not chunk_records:
        continue

    # Write JSONL
    for rec in chunk_records:
        jsonl_out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # Build DataFrame with all known columns, cast everything to str (preserving None)
    df = pd.DataFrame(chunk_records, columns=sorted(all_keys))
    # Convert each column to nullable string
    for col in df.columns:
        df[col] = df[col].where(df[col].isna(), df[col].astype(str))

    table = pa.Table.from_pandas(df, schema=pa_schema, preserve_index=False)

    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(OUTPUT_PARQUET, pa_schema, compression="snappy")
    parquet_writer.write_table(table)

    records_total += len(chunk_records)
    pct = min((i + CHUNK_SIZE) / total * 100, 100)
    print(f"  {pct:5.1f}%  processed {records_total:,} records ({i + len(chunk_files)}/{total} files)")

jsonl_out.close()
if parquet_writer:
    parquet_writer.close()

print(f"\nDone. {records_total:,} records merged, {errors} errors.")
print(f"  Parquet : {OUTPUT_PARQUET}  ({OUTPUT_PARQUET.stat().st_size / 1_048_576:.1f} MB)")
print(f"  JSONL   : {OUTPUT_JSONL}  ({OUTPUT_JSONL.stat().st_size / 1_048_576:.1f} MB)")
print("\nLoad with:")
print("  import pandas as pd")
print(f"  df = pd.read_parquet(r'{OUTPUT_PARQUET}')")
