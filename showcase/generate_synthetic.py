"""
Generate a synthetic cat pedigree dataset with known ground-truth entity clusters.

Output: showcase/DATA/synthetic_cats.parquet
  - All 27 raw columns matching all_scrapers_merged.parquet format
  - Extra column: entity_id  (same entity_id = same real-world cat)
  - ~120 entities, 2-4 records each across 3 simulated sources + ~40 singletons
  - Realistic noise: typos, date format variants, country variants, title abbreviations,
    hyphenation, case changes, missing fields, cattery suffix/prefix styles
"""

import random
import string
import re
import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import date, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Sources ───────────────────────────────────────────────────────────────────
SOURCES = {
    "EasyPedigree": "https://www.easypedigree.com",
    "WCF-BestCat":  "https://www.wcf-bestcat.com",
    "SibCats":      "https://www.sibcats.ru",
}

# ── Breed pool (breed_code, ems_code) ─────────────────────────────────────────
BREEDS = [
    ("SIB", "SIB"),
    ("BRI", "BRI n 22"),
    ("MCO", "MCO n 09 22"),
    ("PER", "PER n 03"),
    ("RAG", "RAG n 04"),
    ("BSH", "BSH a 21"),
    ("NOR", "NFO n 09"),
    ("ABY", "ABY n"),
    ("SIAMESE", "SIA n 21"),
    ("BUR", "BUR n"),
    ("RUS", "RUS a"),
    ("TUR", "TUR w 62"),
    ("ORI", "ORI n 21"),
    ("MAN", "MAN n"),
]

SEXES = ["M", "F"]

COUNTRIES = {
    "Russia":     ("RU", "RUS", "Russia"),
    "Finland":    ("FI", "FIN", "Finland"),
    "Germany":    ("DE", "DEU", "Germany"),
    "USA":        ("US", "USA", "United States"),
    "France":     ("FR", "FRA", "France"),
    "Poland":     ("PL", "POL", "Poland"),
}

TITLE_VARIANTS = {
    "CH":   ["CH", "Ch.", "Ch", "Champion", "CH."],
    "GR":   ["GR CH", "Grand Champion", "GrCH", "GR.CH.", "Grand Ch."],
    "IC":   ["IC", "International Champion", "Int.Ch.", "IC "],
    None:   [None],
}

CATTERIES = [
    "WINDOVER", "CRIMSON TIDE", "STARLITE", "IRONWOOD", "SILVER RIDGE",
    "MISTY MEADOW", "OAKENSHIELD", "SUNRIDGE", "BLUE HOLLOW", "THORNBURY",
    "GOLDENROD", "CRYSTAL STREAM", "MORNING GLORY", "RIVER OAKS", "CEDAR GROVE",
    "AMBER FIELDS", "NIGHTSHADE", "STARFALL", "FOXGLOVE", "WINTERBOURNE",
    "RAVENSWOOD", "COPPER HILL", "WILDROSE", "ASHGROVE", "BROOKSIDE",
    "STONEGATE", "WILLOWMERE", "HARROWGATE", "ELMWOOD", "BIRCHWOOD",
    "FERNDALE", "HEATHERWOOD", "IVYWOOD", "JUNIPER RIDGE", "KESTREL",
]

GIVEN_NAMES = [
    "AURORA", "BARON", "CASSIA", "DANTE", "EMBER", "FALCON", "GAIA", "HUNTER",
    "ISOLDE", "JADE", "KODIAK", "LUNA", "MERLIN", "NOVA", "ORION", "PEARL",
    "QUEST", "RAVEN", "SIENNA", "TITAN", "UMA", "VESPER", "WREN", "XENA",
    "YUKI", "ZENITH", "ASHA", "BLAZE", "CAIRO", "DRIFT", "ECHO", "FURY",
    "GRACE", "HALO", "ICON", "JUNO", "KIRA", "LYRIC", "MIST", "NIRO",
    "ONYX", "PYRO", "QUEEN", "RHINO", "SAGE", "TOPAZ", "ULRIC", "VEGA",
    "WOLF", "XERO", "YARA", "ZEUS", "ARIA", "BRIX", "CLEO", "DUKE",
    "EVE", "FINN", "GLOW", "HART", "IVY", "JACK", "KLEO", "LOCH",
    "MOBY", "NELL", "OAK", "PIKE", "QUIN", "REEF", "SILO", "TUSK",
    "URO", "VALE", "WIST", "XULA", "YOLO", "ZONE", "ACE", "BARD",
    "CREST", "DUSK", "EDEL", "FAWN", "GUST", "HAZE", "IRIS", "JEST",
    "KELD", "LARK", "MACE", "NOEL", "OPAL", "PINE", "QUILL", "ROSE",
    "SNOW", "TIDE", "URAL", "VOLT", "WICK", "YALE", "ZINC", "ALBA",
    "BALE", "CORD", "DRAY", "ELMS", "FERN", "GALE", "HELM", "ISLE",
]

SCRAPED_AT = "2024-11-15 00:00:00"


# ── Noise helpers ─────────────────────────────────────────────────────────────

def _typo(s: str) -> str:
    """Introduce a single-character typo: swap, drop, or substitute."""
    if len(s) < 3:
        return s
    ops = ["swap", "drop", "sub"]
    op  = random.choice(ops)
    i   = random.randint(0, len(s) - 1)
    if op == "swap" and i < len(s) - 1:
        lst = list(s)
        lst[i], lst[i + 1] = lst[i + 1], lst[i]
        return "".join(lst)
    elif op == "drop":
        return s[:i] + s[i + 1:]
    else:
        repl = random.choice(string.ascii_uppercase)
        return s[:i] + repl + s[i + 1:]


def _apply_name_noise(name: str, src: str, level: int) -> str:
    """Apply source-appropriate name noise."""
    # Level 0 = clean, 1 = light, 2 = heavy
    if level == 0:
        return name

    variants = [name]

    # Case change
    variants.append(name.title())
    variants.append(name.lower().title())

    # Hyphenation of spaces
    parts = name.split()
    if len(parts) >= 2 and random.random() < 0.3:
        i = random.randint(0, len(parts) - 2)
        new_parts = parts[:i] + [parts[i] + "-" + parts[i + 1]] + parts[i + 2:]
        variants.append(" ".join(new_parts))

    # Abbreviation of cattery (first word to initials)
    if len(parts) >= 2 and random.random() < 0.25:
        abbr = parts[0][0] + "."
        variants.append(abbr + " " + " ".join(parts[1:]))

    # EasyPedigree-style: add "OF <cattery>" suffix
    if src == "EasyPedigree" and random.random() < 0.2 and len(parts) >= 2:
        cattery = random.choice(CATTERIES)
        variants.append(name + " OF " + cattery)

    # SibCats-style: no cattery prefix, just given name
    if src == "SibCats" and len(parts) >= 2 and random.random() < 0.4:
        variants.append(parts[-1])

    # Typo on level 2
    if level >= 2 and random.random() < 0.6:
        chosen = random.choice(variants)
        variants.append(_typo(chosen))

    result = random.choice(variants)
    return result.strip()


def _dob_variants(dob_iso: str, src: str) -> str:
    """Return date in source-appropriate format."""
    if not dob_iso:
        return None
    try:
        d = date.fromisoformat(dob_iso)
    except Exception:
        return dob_iso
    if src == "EasyPedigree":
        # Often just year or YYYY-MM-DD
        if random.random() < 0.3:
            return str(d.year)
        return dob_iso
    elif src == "WCF-BestCat":
        formats = [
            dob_iso,
            f"{d.day:02d}.{d.month:02d}.{d.year}",
            f"{d.day:02d}/{d.month:02d}/{d.year}",
        ]
        return random.choice(formats)
    else:  # SibCats
        formats = [
            dob_iso,
            f"{d.day:02d}.{d.month:02d}.{d.year}",
        ]
        return random.choice(formats)


def _country_variant(country_name: str, src: str) -> str:
    """Return country in source-appropriate format."""
    if not country_name:
        return None
    variants = COUNTRIES.get(country_name, (country_name, country_name, country_name))
    if src == "EasyPedigree":
        # Usually full name or absent
        return random.choice([variants[2], None])
    elif src == "WCF-BestCat":
        return random.choice([variants[0], variants[2]])
    else:  # SibCats
        return random.choice([variants[0], variants[1]])


def _pick_title(base_title: str, src: str) -> str:
    """Pick a title variant appropriate for the source."""
    if not base_title:
        return None
    variants = TITLE_VARIANTS.get(base_title, [base_title])
    title = random.choice(variants)
    if src == "SibCats" and title:
        # SibCats wraps like Ch.(WCF)
        return f"{title}.(WCF)" if random.random() < 0.4 else title
    return title


def _reg_number(src: str, entity_id: int, rec_idx: int) -> str:
    """Generate a registration number in source-appropriate format."""
    base = f"{entity_id:04d}{rec_idx:02d}"
    if src == "EasyPedigree":
        prefix = random.randint(1000, 9999)
        suffix = random.randint(1000000, 9999999)
        return f"{prefix}-{suffix}"
    elif src == "WCF-BestCat":
        return f"WCF-{base}-{random.randint(10,99)}"
    else:
        return f"SIB-{base}"


# ── Entity generation ─────────────────────────────────────────────────────────

def _generate_entity(eid: int):
    """Generate a canonical cat entity."""
    cattery    = random.choice(CATTERIES)
    given      = random.choice(GIVEN_NAMES)
    breed_code, ems_code = random.choice(BREEDS)
    sex        = random.choice(SEXES)
    country_name = random.choice(list(COUNTRIES.keys()))
    base_title = random.choice(["CH", "GR", "IC", None, None, None])

    # DOB: some entities have one, some don't
    has_dob = random.random() < 0.65
    if has_dob:
        start  = date(2010, 1, 1)
        offset = timedelta(days=random.randint(0, 14 * 365))
        dob    = (start + offset).isoformat()
    else:
        dob = None

    sire_cattery = random.choice(CATTERIES)
    sire_given   = random.choice(GIVEN_NAMES)
    dam_cattery  = random.choice(CATTERIES)
    dam_given    = random.choice(GIVEN_NAMES)

    return {
        "entity_id":    eid,
        "canonical_name": f"{cattery} {given}",
        "cattery":      cattery,
        "given":        given,
        "breed_code":   breed_code,
        "ems_code":     ems_code,
        "sex":          sex,
        "dob":          dob,
        "country_name": country_name,
        "base_title":   base_title,
        "sire_name":    f"{sire_cattery} {sire_given}",
        "dam_name":     f"{dam_cattery} {dam_given}",
    }


def _entity_to_row(entity: dict, src: str, rec_idx: int, noise_level: int) -> dict:
    """Convert a canonical entity into a noisy raw-format row for the given source."""
    eid = entity["entity_id"]

    name  = _apply_name_noise(entity["canonical_name"], src, noise_level)
    dob   = _dob_variants(entity["dob"], src) if entity["dob"] else None
    # Occasionally drop dob even if entity has one
    if dob and random.random() < 0.2:
        dob = None

    co    = _country_variant(entity["country_name"], src)
    cc    = _country_variant(entity["country_name"], src)
    title = _pick_title(entity["base_title"], src)

    # Father/mother: use entity's sire/dam with noise; occasionally missing
    father = _apply_name_noise(entity["sire_name"], src, noise_level) if random.random() > 0.3 else None
    mother = _apply_name_noise(entity["dam_name"],  src, noise_level) if random.random() > 0.3 else None

    reg_cur = _reg_number(src, eid, rec_idx)
    reg_ori = _reg_number(src, eid, rec_idx + 10) if random.random() < 0.4 else None

    cattery_field = entity["cattery"] if src != "SibCats" or random.random() > 0.5 else None

    # EMS code: sometimes absent or just the breed code
    ems = entity["ems_code"]
    if random.random() < 0.25:
        ems = entity["breed_code"]
    if random.random() < 0.15:
        ems = None

    src_url = SOURCES[src]
    foreign_url = f"{src_url}/cat/{eid}-{rec_idx}" if random.random() < 0.5 else None

    return {
        "name":                        name,
        "breed_code":                  entity["breed_code"],
        "ems_code":                    ems,
        "cattery_name":                cattery_field,
        "sex":                         entity["sex"],
        "date_of_birth":               dob,
        "country_origin":              co,
        "country_current":             cc,
        "titles_before":               title,
        "titles_after":                None,
        "father_name":                 father,
        "father_breed_code":           entity["breed_code"] if father and random.random() > 0.5 else None,
        "father_ems_code":             None,
        "father_reg_number":           None,
        "mother_name":                 mother,
        "mother_breed_code":           entity["breed_code"] if mother and random.random() > 0.5 else None,
        "mother_ems_code":             None,
        "mother_reg_number":           None,
        "registration_number_current": reg_cur,
        "registration_number_origin":  reg_ori,
        "microchip_number":            None,
        "foreign_id":                  None,
        "foreign_name":                None,
        "foreign_url":                 foreign_url,
        "source_database_name":        src,
        "source_database_url":         src_url,
        "scraped_at":                  SCRAPED_AT,
        "entity_id":                   eid,
    }


# ── Main generation logic ─────────────────────────────────────────────────────

def generate(
    n_entities: int = 120,
    n_singletons: int = 40,
    out_path: str = None,
):
    """
    Generate synthetic dataset.

    n_entities:   number of entities that appear in multiple sources (true clusters)
    n_singletons: number of entities that appear in exactly one source
    """
    if out_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, "DATA", "synthetic_cats.parquet")

    src_list = list(SOURCES.keys())
    rows = []

    print(f"Generating {n_entities} multi-source entities ...")
    for eid in range(1, n_entities + 1):
        entity = _generate_entity(eid)
        # 2-5 copies; same source may appear more than once (mirrors real scrapers)
        n_copies = random.choices([2, 3, 4, 5], weights=[30, 40, 20, 10])[0]
        # Start with one copy per source (shuffled), then fill remaining with repeats
        shuffled = src_list[:]
        random.shuffle(shuffled)
        chosen_srcs = shuffled[:min(n_copies, len(shuffled))]
        while len(chosen_srcs) < n_copies:
            chosen_srcs.append(random.choice(src_list))
        for ridx, src in enumerate(chosen_srcs):
            noise = random.choice([0, 1, 1, 2])
            rows.append(_entity_to_row(entity, src, ridx, noise))

    print(f"Generating {n_singletons} singleton entities ...")
    for eid in range(n_entities + 1, n_entities + n_singletons + 1):
        entity = _generate_entity(eid)
        src = random.choice(src_list)
        rows.append(_entity_to_row(entity, src, 0, noise_level=0))

    df = pd.DataFrame(rows)

    # Ensure column order matches real parquet (alphabetical, entity_id at end)
    REAL_COLS = [
        "breed_code", "cattery_name", "country_current", "country_origin",
        "date_of_birth", "ems_code", "father_breed_code", "father_ems_code",
        "father_name", "father_reg_number", "foreign_id", "foreign_name",
        "foreign_url", "microchip_number", "mother_breed_code", "mother_ems_code",
        "mother_name", "mother_reg_number", "name",
        "registration_number_current", "registration_number_origin",
        "scraped_at", "sex", "source_database_name", "source_database_url",
        "titles_after", "titles_before",
    ]
    final_cols = REAL_COLS + ["entity_id"]
    df = df.reindex(columns=final_cols)

    # Convert to str (all object columns, matching real parquet dtype)
    for col in REAL_COLS:
        df[col] = df[col].where(df[col].notna(), other=None)
        df[col] = df[col].astype("object")

    df["entity_id"] = df["entity_id"].astype(int)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # ── Summary ───────────────────────────────────────────────────────────────
    multi = df[df["entity_id"] <= n_entities]
    solo  = df[df["entity_id"] >  n_entities]
    cluster_sizes = multi.groupby("entity_id").size()
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(f"  Multi-source entities : {n_entities}  ({len(multi)} rows)")
    print(f"  Singleton entities    : {n_singletons} ({len(solo)} rows)")
    print(f"  Cluster size dist     : {cluster_sizes.value_counts().sort_index().to_dict()}")
    print(f"  Sources               : {df['source_database_name'].value_counts().to_dict()}")
    print(f"  Columns               : {list(df.columns)}")
    return df


if __name__ == "__main__":
    generate()
