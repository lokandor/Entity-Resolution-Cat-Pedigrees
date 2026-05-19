"""
Real Data Loader  —  Pedigree Cat Records
==========================================
Reads all JSON batch files from the data/ directory and returns a unified
pandas DataFrame compatible with the ER pipeline.

Schema (matches pipeline expectations):
    id        — unique record ID: {SOURCE_ABBREV}_{global_index}
    source    — source database name (full)
    name      — cat name
    breed     — breed code (breed_code or ems_code)
    dob       — date of birth (date_of_birth)
    sex       — sex
    sire      — father name (father_name)
    dam       — mother name (mother_name)
    country   — country of origin (country_origin ?? country_current)
    cattery   — cattery name
    microchip — microchip number
    reg_no    — registration number (current ?? origin)
    titles    — titles before/after concatenated
    foreign_id — original ID in source database (for traceability)

Caching:
    First call builds a Parquet cache (data_cache.parquet) — takes several
    minutes for 2M+ records.  Subsequent calls load from cache in seconds.

Usage:
    from load_data import load_records
    records = load_records()                      # full dataset
    records = load_records(sample=10_000)         # first 10k rows (for dev)
    records = load_records(rebuild_cache=True)    # force rebuild
"""

import os
import glob
import json
import sys

import pandas as pd

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, "data")
CACHE_PATH = os.path.join(HERE, "data_cache.parquet")

SOURCE_ABBREV = {
    "WCF-BestCat":          "WCF",
    "HimalayanCatsOnline":  "HCO",
    "CATPEDIGREES":         "CPD",
    "FDCat":                "FDC",
    "FelisPolonia":         "FPL",
    "Bengal-Data":          "BNG",
    "BengalPedigrees":      "BPD",
}


# ── field mapping ─────────────────────────────────────────────────────────────

def _map_record(r: dict, global_idx: int) -> dict:
    src    = r.get("source_database_name") or "UNK"
    abbrev = SOURCE_ABBREV.get(src, src[:3].upper())
    return {
        "id":         f"{abbrev}_{global_idx:08d}",
        "source":     src,
        "foreign_id": r.get("foreign_id") or "",
        "name":       r.get("name") or "",
        "breed":      r.get("breed_code") or r.get("ems_code") or "",
        "dob":        r.get("date_of_birth") or "",
        "sex":        r.get("sex") or "",
        "sire":       r.get("father_name") or "",
        "dam":        r.get("mother_name") or "",
        "country":    r.get("country_origin") or r.get("country_current") or "",
        "cattery":    r.get("cattery_name") or "",
        "microchip":  r.get("microchip_number") or "",
        "reg_no":     (r.get("registration_number_current")
                       or r.get("registration_number_origin") or ""),
        "titles":     " ".join(filter(None, [
                          r.get("titles_before"), r.get("titles_after")])),
    }


# ── loading ───────────────────────────────────────────────────────────────────

def _load_from_json(data_dir: str, sample: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    rows       = []
    global_idx = 0
    n_files    = len(files)

    for file_no, fp in enumerate(files, 1):
        if sample is not None and global_idx >= sample:
            break

        try:
            with open(fp, encoding="utf-8") as f:
                batch = json.load(f)
        except Exception as e:
            print(f"\n  [skip] {os.path.basename(fp)}: {e}", file=sys.stderr)
            continue

        for r in batch:
            if sample is not None and global_idx >= sample:
                break
            rows.append(_map_record(r, global_idx))
            global_idx += 1

        if file_no % 1000 == 0 or file_no == n_files:
            pct = 100 * file_no / n_files
            print(f"\r  Loading JSON files … {file_no:>6}/{n_files}  "
                  f"({pct:>5.1f}%)  {global_idx:>9,} records", end="", flush=True)

    print()
    return pd.DataFrame(rows)


# ── public API ────────────────────────────────────────────────────────────────

def load_records(
    data_dir: str = DATA_DIR,
    cache_path: str = CACHE_PATH,
    sample: int | None = None,
    rebuild_cache: bool = False,
) -> pd.DataFrame:
    """
    Return the unified records DataFrame.

    Parameters
    ----------
    data_dir      : directory containing the JSON batch files
    cache_path    : path to the Parquet cache file
    sample        : if set, load at most this many records (skips cache)
    rebuild_cache : if True, rebuild the cache even if it exists
    """
    use_cache = (
        sample is None
        and not rebuild_cache
        and os.path.exists(cache_path)
    )

    if use_cache:
        print(f"Loading from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"  {len(df):,} records from {df['source'].nunique()} sources")
        return df

    print(f"Reading JSON files from: {data_dir}")
    if sample:
        print(f"  (sample mode: first {sample:,} records only)")

    df = _load_from_json(data_dir, sample=sample)

    print(f"\nLoaded {len(df):,} records")
    _print_summary(df)

    if sample is None:
        print(f"\nSaving cache → {cache_path}")
        df.to_parquet(cache_path, index=False)
        print("  Cache saved.")

    return df


def _print_summary(df: pd.DataFrame) -> None:
    print("\nSource distribution:")
    for src, cnt in df["source"].value_counts().items():
        pct = 100 * cnt / len(df)
        print(f"  {cnt:>10,}  ({pct:>5.1f}%)  {src}")

    print("\nField population rates:")
    for col in ["name", "breed", "dob", "sex", "sire", "dam", "country",
                 "cattery", "microchip", "reg_no"]:
        filled = (df[col] != "").sum()
        print(f"  {100*filled/len(df):>5.1f}%  {col}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load pedigree cat records from JSON batch files")
    parser.add_argument("--sample",        type=int, default=None,
                        help="Load only the first N records (for development)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild of the Parquet cache")
    parser.add_argument("--data-dir",      default=DATA_DIR,
                        help=f"Directory with JSON files (default: {DATA_DIR})")
    args = parser.parse_args()

    df = load_records(
        data_dir=args.data_dir,
        sample=args.sample,
        rebuild_cache=args.rebuild_cache,
    )

    print("\nSample records:")
    print(df[["id", "source", "name", "breed", "dob", "sire", "dam"]].head(10).to_string(index=False))