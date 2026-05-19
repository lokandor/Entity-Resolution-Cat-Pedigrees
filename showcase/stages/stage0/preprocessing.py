"""Record-level preprocessing filters for the ER pipeline."""
import re
import pandas as pd

_PLACEHOLDERS = {"n/a", "na", "unknown", "null", "-", "?", "none", "", "nan"}
_MIN_NAME_LEN  = 3
_DOB_YEAR_RE   = re.compile(r"(19|20)\d{2}")
_DOB_MIN_YEAR  = 1950
_DOB_MAX_YEAR  = 2030


def _is_placeholder(val: str) -> bool:
    return str(val).strip().lower() in _PLACEHOLDERS


def preprocess(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply record-level filters. Returns a cleaned copy of df.

    Filters applied (in order):
      1. Missing critical fields  — name or breed null/empty/placeholder
      2. Name too short           — fewer than 3 non-whitespace characters
      3. Impossible DOB           — year outside 1950-2030
      4. Exact within-source dups — same name+breed+dob+sire+dam within one source
    """
    n0 = len(df)

    # 1 — missing critical fields
    name_bad  = df["name"].isna()  | df["name"].apply(_is_placeholder)
    breed_bad = df["breed"].isna() | df["breed"].apply(_is_placeholder)
    df = df[~(name_bad | breed_bad)].copy()

    # 2 — name too short
    df = df[df["name"].str.replace(r"\s", "", regex=True).str.len() >= _MIN_NAME_LEN]

    # 3 — impossible DOB year
    def _bad_dob(val):
        if pd.isna(val) or _is_placeholder(str(val)):
            return False
        m = _DOB_YEAR_RE.search(str(val))
        if not m:
            return False
        yr = int(m.group(0))
        return yr < _DOB_MIN_YEAR or yr > _DOB_MAX_YEAR

    df = df[~df["dob"].apply(_bad_dob)]

    # 4 — exact within-source duplicates (keep first)
    dedup_cols = [c for c in ["source", "name", "breed", "dob", "sire", "dam"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")

    df = df.reset_index(drop=True)

    if verbose:
        print(f"  Preprocessing: {n0:,} -> {len(df):,} records "
              f"(removed {n0 - len(df):,})")

    return df
