"""
Synthetic pedigree dog records for Entity Resolution showcase.

Single unified dataset of 30 records from two sources (A and B):
  - Source A: 15 records — clean / canonical
  - Source B: 15 records — 10 match A (with noise), 5 unique to B

The `source` column indicates which source each record came from.
TRUE_MATCHES is kept for evaluation only — not used as a training signal.
Supervised methods get their labels from Snorkel (04_snorkel_labeling.py).

Noise types used: typos, abbreviations, date format differences,
                  country code variants, hyphenation, case changes.
"""
import pandas as pd

COLUMNS = ["id", "name", "breed", "dob", "sire", "dam", "country", "source"]

records = pd.DataFrame([
    # ── Source A — clean / canonical ──────────────────────────────────────────
    ["A01", "Luna Sunshine",     "Labrador Retriever",   "2020-03-15", "Max von Sonnenhügel",   "Bella of Greenfield",   "Germany",       "A"],
    ["A02", "Charlie Brown",     "Golden Retriever",     "2019-07-22", "Duke of Westminster",   "Molly Jane",            "United Kingdom","A"],
    ["A03", "Bella Rose",        "German Shepherd",      "2021-01-10", "Rex vom Waldpark",      "Greta Blümchen",        "Germany",       "A"],
    ["A04", "Max Thunder",       "Rottweiler",           "2018-11-05", "Titan of the North",    "Hera Black Star",       "Netherlands",   "A"],
    ["A05", "Daisy May",         "Beagle",               "2020-08-30", "Biscuit McGregor",      "Fern Wildrose",         "United Kingdom","A"],
    ["A06", "Rocky Mountains",   "Siberian Husky",       "2019-05-14", "Storm Rider",           "Arctic Moonbeam",       "Czech Republic","A"],
    ["A07", "Zoe Starlight",     "Border Collie",        "2022-02-28", "Flash McBriar",         "Shadow Dancer",         "Scotland",      "A"],
    ["A08", "Bruno Hercules",    "Dobermann",            "2017-09-03", "Kaiser vom Rhein",      "Freya Eisenherz",       "Germany",       "A"],
    ["A09", "Coco Chanel",       "Poodle",               "2021-06-18", "Pierre de la Fontaine", "Mimi Beaumont",         "France",        "A"],
    ["A10", "Thor Mjolnir",      "Norwegian Elkhound",   "2020-12-01", "Viking Stormbreaker",   "Sigrid Fjordlily",      "Norway",        "A"],
    ["A11", "Misty Blue",        "Weimaraner",           "2019-04-17", "Ghost von See",         "Silver Moonrise",       "Germany",       "A"],
    ["A12", "Pepper Jack",       "Dalmatian",            "2018-06-25", "Spot Fireman",          "Dixie Dotsworth",       "United States", "A"],
    ["A13", "Amber Sunrise",     "Irish Setter",         "2022-09-09", "Brendan O'Mahony",      "Colleen Firedancer",    "Ireland",       "A"],
    ["A14", "Ringo Star",        "Boxer",                "2020-03-21", "Punch von Feldmann",    "Zelda Kampfgeist",      "Germany",       "A"],
    ["A15", "Nala Pride",        "Rhodesian Ridgeback",  "2021-11-14", "Simba Sunridge",        "Zara Savannah",         "South Africa",  "A"],
    # ── Source B — noisy variants (B01–B10 match A) + 5 unique (B11–B15) ──────
    ["B01", "Luna Sunshne",      "Labrador Retr.",       "15.03.2020", "Max v. Sonnenhügel",    "Bella Greenfield",      "DE",            "B"],  # A01
    ["B02", "Charlie Brown",     "Golden Retr.",         "22/07/2019", "Duke Westminster",      "Molly Jane",            "UK",            "B"],  # A02
    ["B03", "Bella-Rose",        "Germ. Shepherd",       "2021-01-10", "Rex v.d. Waldpark",     "Greta Blümchen",        "Germany",       "B"],  # A03
    ["B04", "Max Thunder",       "Rottweiler",           "05-11-2018", "Titan North",           "Hera Black-Star",       "NL",            "B"],  # A04
    ["B05", "Daisy-May",         "Beagle",               "30.08.2020", "Biscuit Mcgregor",      "Fern Wildrose",         "England",       "B"],  # A05
    ["B06", "Rocky Mountn.",     "Siberian Husky",       "14/05/2019", "Storm-Rider",           "Arctic Moonbeam",       "CZ",            "B"],  # A06
    ["B07", "Zoe Starlight",     "Border Collie",        "28.02.2022", "Flash Mc Brian",        "Shadow Dancer",         "Scot.",         "B"],  # A07
    ["B08", "Bruno Hercules",    "Doberman",             "2017-09-03", "Kaiser v. Rhein",       "Freya Eisenherz",       "GER",           "B"],  # A08
    ["B09", "CoCo Chanel",       "Poodle",               "2021-06-18", "Pierre de La Fontaine", "Mimi Beaumont",         "France",        "B"],  # A09
    ["B10", "Thor Mjolnir",      "Norwegian Elkhound",   "01.12.2020", "Viking Stormbreaker",   "Sigrid Fjordlilly",     "NO",            "B"],  # A10
    ["B11", "Shadow Knight",     "Black Labrador",       "2020-07-11", "Phantom Darkwood",      "Night Sky Bella",       "Canada",        "B"],
    ["B12", "Milo Sunshine",     "Jack Russell Terrier", "2021-04-02", "Sparky McDougal",       "Penny Lanesong",        "Australia",     "B"],
    ["B13", "Freya Bergdal",     "Swedish Vallhund",     "2019-08-19", "Erik Stormvind",        "Astrid Dalblomma",      "Sweden",        "B"],
    ["B14", "Atlas Strongbow",   "Great Dane",           "2018-02-14", "Goliath von Würzburg",  "Hilda Riesendame",      "Austria",       "B"],
    ["B15", "Sable Windchaser",  "Afghan Hound",         "2022-10-31", "Zephyr al-Rashid",      "Desert Moonflower",     "UAE",           "B"],
], columns=COLUMNS)

# Backward-compatible views for existing scripts (without source column)
records_a = records[records["source"] == "A"].drop(columns="source").reset_index(drop=True)
records_b = records[records["source"] == "B"].drop(columns="source").reset_index(drop=True)
# Ground truth: frozenset pairs so order doesn't matter during evaluation
TRUE_MATCHES = {
    ("A01", "B01"), ("A02", "B02"), ("A03", "B03"),
    ("A04", "B04"), ("A05", "B05"), ("A06", "B06"),
    ("A07", "B07"), ("A08", "B08"), ("A09", "B09"),
    ("A10", "B10"),
}


def all_record_pairs():
    """All unique (id_1, id_2) pairs from the full dataset (self-join, id_1 < id_2).
    Covers both within-source and cross-source pairs."""
    ids = list(records["id"])
    return [tuple(sorted((ids[i], ids[j])))
            for i in range(len(ids)) for j in range(i + 1, len(ids))]


def inference_pairs(candidate_pairs_path="candidate_pairs.csv"):
    """
    Load blocked candidate pairs for matcher inference.
    Falls back to all_record_pairs() with a warning if the file doesn't exist
    (i.e. Snorkel hasn't been run yet).
    """
    import os
    import pandas as pd
    if os.path.exists(candidate_pairs_path):
        df = pd.read_csv(candidate_pairs_path)
        return list(zip(df["id_a"], df["id_b"]))
    print(f"  [candidate_pairs.csv not found — run 04_snorkel_labeling.py first]")
    print(f"  Falling back to all pairs (slower, not how the real pipeline works).")
    return all_record_pairs()


def print_matched_pairs(predicted_pairs, true_matches=TRUE_MATCHES):
    """Print a formatted table of predicted pairs annotated with TP/FP/FN."""
    rec_idx  = records.set_index("id")
    true_set = {tuple(sorted(p)) for p in true_matches}
    pred_set = {tuple(sorted(p)) for p in predicted_pairs}

    header = f"  {'id_a':>4}  {'Name (A)':<24}  {'id_b':>4}  {'Name (B)':<24}  Result"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for a_id, b_id in sorted(pred_set):
        name_a = rec_idx.loc[a_id, "name"] if a_id in rec_idx.index else "?"
        name_b = rec_idx.loc[b_id, "name"] if b_id in rec_idx.index else "?"
        mark   = "✓ TRUE MATCH" if (a_id, b_id) in true_set else "✗ FALSE POSITIVE"
        print(f"  {a_id:>4}  {name_a:<24}  {b_id:>4}  {name_b:<24}  {mark}")

    missed = true_set - pred_set
    if missed:
        print()
        for a_id, b_id in sorted(missed):
            name_a = rec_idx.loc[a_id, "name"] if a_id in rec_idx.index else "?"
            name_b = rec_idx.loc[b_id, "name"] if b_id in rec_idx.index else "?"
            print(f"  {a_id:>4}  {name_a:<24}  {b_id:>4}  {name_b:<24}  ✗ MISSED")


def evaluate(predicted_pairs, true_matches=TRUE_MATCHES):
    """Return Precision, Recall, F1 for predicted match pairs."""
    predicted = {tuple(sorted(p)) for p in predicted_pairs}
    truth     = {tuple(sorted(p)) for p in true_matches}
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    print(f"Source A : {len(records_a)} records")
    print(f"Source B : {len(records_b)} records")
    print(f"True matches : {len(TRUE_MATCHES)}\n")
    print(records_a[["id", "name", "breed"]].to_string(index=False))
    print()
    print(records_b[["id", "name", "breed"]].to_string(index=False))
