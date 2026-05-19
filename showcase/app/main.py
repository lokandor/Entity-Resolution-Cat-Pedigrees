"""
Entity Resolution Pipeline — Streamlit App
==========================================
Run: streamlit run showcase/app/main.py
Remote access is enabled via .streamlit/config.toml (0.0.0.0:8501).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from stages.stage0.preprocessing import preprocess
from stages.stage1.token_blocking import block_token
from stages.stage1.snm_blocking import block_snm
from stages.stage1.minhash_blocking import block_minhash
from stages.stage1.meta_blocking import prune_candidates
from stages.stage2.deeper import match_deeper
from stages.stage2.bert_matcher import match_ditto, match_adapter
from stages.stage2.deepmatcher import match_deepmatcher
from stages.stage2.cot import match_cot
from stages.stage3.connected_components import cluster_cc
from stages.stage3.correlation_clustering import cluster_corr
from stages.stage3 import clusters_to_pairs

_HERE = os.path.dirname(__file__)
_PARQUET       = os.path.normpath(os.path.join(_HERE, "..", "DATA", "all_scrapers_merged.parquet"))
_PARQUET_SYNTH = os.path.normpath(os.path.join(_HERE, "..", "DATA", "synthetic_cats.parquet"))

_SOURCE_ABBREV = {
    "WCF-BestCat": "WCF", "HimalayanCatsOnline": "HCO", "CATPEDIGREES": "CPD",
    "FDCat": "FDC", "FelisPolonia": "FPL", "Bengal-Data": "BNG",
    "BengalPedigrees": "BPD", "EasyPedigree": "EZP", "SibCats": "SIB",
    "Katt": "KAT", "Kissat": "KIS",
}


def _apply_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw parquet columns → pipeline schema (id, source, name, breed, …)."""
    src = raw["source_database_name"].fillna("UNK")
    abbrev = src.map(lambda s: _SOURCE_ABBREV.get(s, str(s)[:3].upper()))
    idx_str = pd.RangeIndex(len(raw)).astype(str).str.zfill(8)

    def col(primary, fallback=None):
        s = raw[primary].fillna("") if primary in raw.columns else pd.Series([""] * len(raw))
        if fallback and fallback in raw.columns:
            s = s.where(s != "", raw[fallback].fillna(""))
        return s.astype(str).str.strip()

    titles = (
        raw.get("titles_before", pd.Series([""] * len(raw))).fillna("").astype(str)
        + " "
        + raw.get("titles_after", pd.Series([""] * len(raw))).fillna("").astype(str)
    ).str.strip()

    mapped = pd.DataFrame({
        "id":         abbrev + "_" + idx_str,
        "source":     src,
        "foreign_id": col("foreign_id"),
        "name":       col("name"),
        "breed":      col("breed_code", "ems_code"),
        "dob":        col("date_of_birth"),
        "sex":        col("sex"),
        "sire":       col("father_name"),
        "dam":        col("mother_name"),
        "country":    col("country_origin", "country_current"),
        "cattery":    col("cattery_name"),
        "microchip":  col("microchip_number"),
        "reg_no":     col("registration_number_current", "registration_number_origin"),
        "titles":     titles,
    })
    if "entity_id" in raw.columns:
        mapped["entity_id"] = raw["entity_id"].values
    return mapped


def _compute_gt_pairs(df: pd.DataFrame):
    """Return cross-source ground-truth pairs from entity_id column, or None."""
    if "entity_id" not in df.columns:
        return None
    gt = set()
    for _eid, grp in df.groupby("entity_id"):
        ids  = list(grp["id"])
        srcs = list(grp["source"])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if srcs[i] != srcs[j]:
                    gt.add(tuple(sorted((ids[i], ids[j]))))
    return gt

st.set_page_config(
    page_title="ER Pipeline — Pedigree Cats",
    page_icon="🐱",
    layout="wide",
)

for _k, _v in {
    "df": None, "pipeline_df": None, "candidates": None, "match_pairs": None,
    "clusters": None, "run_log": [], "gt_pairs": None,
}.items():
    st.session_state.setdefault(_k, _v)

st.title("🐱 Entity Resolution Pipeline")
tab_data, tab_pipeline, tab_results = st.tabs(["📁 Data", "⚙️ Pipeline", "📊 Results"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.header("Data Explorer")

    data_src = st.radio(
        "Data source",
        ["Real data", "Synthetic dataset (with ground truth)"],
        horizontal=True,
    )
    _active_parquet = _PARQUET if data_src == "Real data" else _PARQUET_SYNTH
    st.caption(f"File: `{_active_parquet}`")

    _total_rows = None
    if os.path.exists(_active_parquet):
        try:
            import pyarrow.parquet as pq
            _total_rows = pq.read_metadata(_active_parquet).num_rows
            st.caption(f"Total rows in file: {_total_rows:,}")
        except Exception:
            pass
    else:
        st.error(f"File not found: `{_active_parquet}`")
        if data_src == "Synthetic dataset (with ground truth)":
            st.info(
                "Generate the synthetic dataset first:\n"
                "```\npython showcase/generate_synthetic.py\n```"
            )

    _max_slider = min(_total_rows or 500_000, 500_000)
    _default_n  = min(10_000, _total_rows or 10_000) if data_src == "Real data" else (_total_rows or 500)
    sample_n = st.slider(
        "Sample size", 1_000 if data_src == "Real data" else 100,
        _max_slider, _default_n,
        step=1_000 if data_src == "Real data" else 100,
        format="%d rows",
    )

    col_a, col_b = st.columns([1, 2])
    load_sample = col_a.button("Load sample", type="primary")
    load_full   = col_b.button("Load full dataset")

    if load_sample or load_full:
        if not os.path.exists(_active_parquet):
            st.error(f"File not found: `{_active_parquet}`")
        else:
            with st.spinner("Reading parquet and mapping schema…"):
                raw = pd.read_parquet(_active_parquet)
                if load_sample and sample_n < len(raw):
                    raw = raw.sample(sample_n, random_state=42).reset_index(drop=True)
                df = _apply_schema(raw)
                gt = _compute_gt_pairs(df)
            st.session_state.df         = df
            st.session_state.gt_pairs   = gt
            st.session_state.candidates = None
            st.session_state.clusters   = None
            msg = f"Loaded {len(df):,} records from {df['source'].nunique()} sources."
            if gt is not None:
                msg += f" Ground truth: {len(gt):,} cross-source pairs."
            st.success(msg)

    df = st.session_state.df
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records",      f"{len(df):,}")
        c2.metric("Sources",      df["source"].nunique())
        c3.metric("Unique names", df["name"].nunique())
        c4.metric("Breeds",       df["breed"].nunique() if "breed" in df.columns else "—")

        st.subheader("Source distribution")
        sd = df["source"].value_counts().reset_index()
        sd.columns = ["Source", "Count"]
        sd["Pct %"] = (100 * sd["Count"] / len(df)).round(1)
        st.dataframe(sd, use_container_width=True, hide_index=True)

        st.subheader("Field population")
        _fields = [c for c in ["name","breed","dob","sex","sire","dam",
                                "country","cattery","microchip","reg_no"]
                   if c in df.columns]
        _pop = {f: round(100 * (df[f].astype(str).str.strip() != "").sum() / len(df), 1)
                for f in _fields}
        st.bar_chart(
            pd.DataFrame({"Field": list(_pop), "Populated %": list(_pop.values())})
            .set_index("Field")
        )

        st.subheader("Preview")
        _cols = [c for c in ["id","source","name","breed","dob","sire","dam","country"]
                 if c in df.columns]
        st.dataframe(df[_cols].head(200), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    st.header("Pipeline")

    if st.session_state.df is None:
        st.warning("Load data first in the **Data** tab.")
        st.stop()

    df = st.session_state.df

    # Stage 0
    with st.expander("Stage 0 — Preprocessing"):
        do_preproc = st.toggle("Enable (opt-in)", value=False)
        st.caption(
            "Removes records with missing/placeholder name or breed, names shorter than "
            "3 characters, impossible DOB years (outside 1950-2030), and exact "
            "within-source duplicates (same name+breed+dob+sire+dam)."
        )

    # Stage 1 — Blocking
    with st.expander("Stage 1 — Blocking", expanded=True):
        blk = st.selectbox("Method", ["both", "token", "snm", "minhash"], key="blk")
        c1, c2 = st.columns(2)
        _stop_default = "of\nthe\na\nan\nde\nvon\nvom\nv\nvd\nla\nle\nvan\nretr\ngerm"
        stop_raw = c1.text_area("Stop words (one per line)", value=_stop_default, height=110)
        min_tok  = c2.number_input("Min token length", 1, 10, 2)
        snm_win  = st.slider("SNM window size", 2, 30, 5,
                             help="Only used when method is 'snm' or 'both'")
        if blk == "minhash":
            c3, c4, c5 = st.columns(3)
            mh_threshold = c3.slider("MinHash threshold", 0.1, 0.9, 0.4, 0.05)
            mh_num_perm  = c4.select_slider("Num permutations", [64, 128, 256], value=128)
            mh_ngram     = c5.number_input("N-gram size", 2, 5, 3)
        else:
            mh_threshold, mh_num_perm, mh_ngram = 0.4, 128, 3
        stop_words = {w.strip() for w in stop_raw.splitlines() if w.strip()}

        st.divider()
        meta_thr = st.slider(
            "Meta-blocking threshold (0 = off)", 0, 5, 0,
            help="WNP: keep only candidate pairs that share >= N token blocks. "
                 "Reduces candidates, raises precision, may lower recall."
        )

    # Stage 2 — Labeling  (only for supervised matchers)
    _supervised = ["ditto", "adapter", "deepmatcher", "cot"]

    # peek at matcher choice before the expander so labeling section can read it
    _matcher_preview = st.session_state.get("matcher_sel", "deeper")

    with st.expander("Stage 2 — Labeling Strategy", expanded=True):
        st.caption(
            "Supervised matchers need training labels. "
            "Labels are generated automatically from candidate pairs — no manual annotation needed."
        )
        labeler = st.selectbox(
            "Labeling strategy",
            ["snorkel", "openai", "lmstudio"],
            format_func=lambda x: {
                "snorkel":  "Snorkel (hand-crafted LFs — free, no GPU)",
                "openai":   "OpenAI API (GPT — high quality, pay-per-token)",
                "lmstudio": "LM Studio (local LLM — free, requires running server)",
            }[x],
            key="labeler_sel",
            help="Only used by supervised matchers (ditto, adapter, deepmatcher, cot). Ignored by deeper.",
        )
        api_key  = ""
        lm_model = None
        lm_url   = None
        n_cands_hint = len(st.session_state.candidates) if st.session_state.candidates else 1000
        max_label_pairs = st.slider(
            "Max pairs to label",
            min_value=50, max_value=max(n_cands_hint, 50), value=min(200, n_cands_hint),
            step=50,
            help="How many candidate pairs to label for training. "
                 "More = better model, higher cost (LLM) or slower (Snorkel). "
                 "Pairs are sampled randomly from the candidate set."
        )
        if labeler == "openai":
            c1, c2 = st.columns(2)
            api_key  = c1.text_input("OpenAI API key", type="password",
                                     help="Leave blank to fall back to Snorkel automatically")
            lm_model = c2.text_input("Model", value="gpt-3.5-turbo")
        elif labeler == "lmstudio":
            c1, c2 = st.columns(2)
            lm_url   = c1.text_input("LM Studio URL", value="http://localhost:1234")
            lm_model = c2.text_input("Model name", value="local-model",
                                     help="Exact model name shown in LM Studio")
            st.caption("Start LM Studio, load a model, and enable the local server before running.")

    # Stage 2 — Matching
    with st.expander("Stage 2 — Matching", expanded=True):
        matcher = st.selectbox(
            "Matcher", ["deeper", "ditto", "adapter", "deepmatcher", "cot"],
            key="matcher_sel",
        )
        mp = {}

        if matcher != "deeper":
            st.info(
                f"**{matcher}** is a supervised matcher — it will use "
                f"**{labeler}** labels generated from the blocked candidate pairs."
            )

        if matcher == "deeper":
            c1, c2 = st.columns(2)
            mp["threshold"]  = c1.slider("Similarity threshold", 0.3, 0.99, 0.65, 0.01)
            mp["model_name"] = c2.selectbox(
                "Sentence model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2",
                 "paraphrase-multilingual-MiniLM-L12-v2"],
            )

        elif matcher == "ditto":
            c1, c2 = st.columns(2)
            mp["epochs"] = c1.number_input("Epochs", 1, 20, 5)
            mp["lr"]     = float(c2.select_slider(
                "Learning rate", ["1e-5", "2e-5", "5e-5", "1e-4"], value="2e-5"))

        elif matcher == "adapter":
            c1, c2, c3 = st.columns(3)
            mp["lora_r"]       = c1.number_input("LoRA rank (r)", 1, 64, 8)
            mp["lora_alpha"]   = c2.number_input("LoRA alpha", 1, 128, 16)
            mp["lora_dropout"] = c3.slider("LoRA dropout", 0.0, 0.5, 0.1, 0.05)
            c4, c5 = st.columns(2)
            mp["epochs"] = c4.number_input("Epochs", 1, 20, 5)
            mp["lr"]     = float(c5.select_slider(
                "Learning rate", ["1e-5", "2e-5", "5e-5", "1e-4"], value="2e-5"))

        elif matcher == "deepmatcher":
            c1, c2, c3 = st.columns(3)
            mp["hidden"]  = c1.number_input("Hidden size", 16, 256, 64)
            mp["epochs"]  = c2.number_input("Epochs", 5, 100, 25)
            mp["lr"]      = float(c3.select_slider(
                "Learning rate", ["1e-4", "1e-3", "1e-2"], value="1e-3"))
            c4, c5, c6, c7 = st.columns(4)
            mp["f_name"]  = c4.number_input("TF-IDF name", 8, 256, 64)
            mp["f_breed"] = c5.number_input("TF-IDF breed", 8, 128, 32)
            mp["f_sire"]  = c6.number_input("TF-IDF sire", 8, 128, 32)
            mp["f_dam"]   = c7.number_input("TF-IDF dam", 8, 128, 32)

        elif matcher == "cot":
            c1, c2 = st.columns(2)
            mp["student_model"] = c1.selectbox(
                "Student model", ["distilbert-base-uncased", "bert-base-uncased"])
            mp["epochs"] = c2.number_input("Epochs", 1, 20, 5)
            mp["lr"]     = 2e-5

    # Stage 3 — Clustering
    with st.expander("Stage 3 — Clustering", expanded=True):
        clust = st.selectbox("Method", ["cc", "corr"])
        if clust == "cc":
            st.caption("Connected Components — O(n+m), deterministic. Watch for chaining in large clusters.")
        else:
            st.caption("Correlation Clustering — more conservative merges, slower. Falls back to greedy pivot if pyjedai unavailable.")

    st.divider()

    if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
        log = []

        with st.status("Running pipeline…", expanded=True) as status:
            _df = df.copy()

            if do_preproc:
                st.write("Stage 0: Preprocessing…")
                before = len(_df)
                _df = preprocess(_df, verbose=False)
                removed = before - len(_df)
                msg = f"Preprocessing: {before:,} -> {len(_df):,} records (removed {removed:,})"
                st.write(f"  -> {msg}")
                log.append(msg)

            # Stage 1
            st.write(f"Stage 1: Blocking ({blk})…")
            if blk == "minhash":
                candidates = block_minhash(_df, threshold=mh_threshold,
                                           num_perm=mh_num_perm, ngram=mh_ngram)
            else:
                cands_tok = block_token(_df, stop_words=stop_words, min_tok_len=min_tok) \
                            if blk in ("token", "both") else set()
                cands_snm = block_snm(_df, window=snm_win) \
                            if blk in ("snm", "both") else set()
                candidates = cands_tok | cands_snm

            if meta_thr > 0:
                before_meta = len(candidates)
                candidates = prune_candidates(_df, candidates, threshold=meta_thr)
                st.write(f"  Meta-blocking: {before_meta:,} -> {len(candidates):,} pairs")
                log.append(f"Meta-blocking (thr={meta_thr}): {before_meta:,} -> {len(candidates):,}")
            msg = f"{len(candidates):,} candidate pairs"
            st.write(f"  → {msg}")
            log.append(f"Blocking ({blk}): {msg}")

            # Stage 2
            st.write(f"Stage 2: Labeling ({labeler}) + Matching ({matcher})…")
            if labeler == "openai" and api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            elif labeler == "openai" and not api_key:
                st.warning("No API key — falling back to Snorkel labels.")

            if matcher != "deeper":
                mp["labeler"]         = labeler
                mp["lm_model"]        = lm_model
                mp["lm_url"]          = lm_url
                mp["max_label_pairs"] = max_label_pairs

            _runners = {
                "deeper":      lambda: match_deeper(_df, candidates, **mp),
                "ditto":       lambda: match_ditto(_df, candidates, **mp),
                "adapter":     lambda: match_adapter(_df, candidates, **mp),
                "deepmatcher": lambda: match_deepmatcher(_df, candidates, **mp),
                "cot":         lambda: match_cot(_df, candidates, **mp),
            }
            match_pairs = _runners[matcher]()
            msg = f"{len(match_pairs):,} matched pairs"
            st.write(f"  → {msg}")
            log.append(f"Matching ({matcher}): {msg}")

            # Stage 3
            st.write(f"Stage 3: Clustering ({clust})…")
            _clustered = cluster_cc(_df, match_pairs) if clust == "cc" \
                         else cluster_corr(_df, match_pairs)
            multi_n = sum(1 for c in _clustered if len(c) > 1)
            msg = f"{len(_clustered):,} clusters, {multi_n:,} multi-record"
            st.write(f"  → {msg}")
            log.append(f"Clustering ({clust}): {msg}")

            st.session_state.candidates  = candidates
            st.session_state.match_pairs = match_pairs
            st.session_state.clusters    = _clustered
            st.session_state.run_log     = log
            st.session_state.pipeline_df = _df
            status.update(label="Done!", state="complete")

        st.info("Switch to the **Results** tab to see the output.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_results:
    st.header("Results")

    if not st.session_state.clusters:
        st.info("Run the pipeline first in the **Pipeline** tab.")
        st.stop()

    clusters    = st.session_state.clusters
    match_pairs = st.session_state.match_pairs
    _pdf = st.session_state.get("pipeline_df")
    df   = _pdf if _pdf is not None else st.session_state.df

    multi   = [c for c in clusters if len(c) > 1]
    singles = [c for c in clusters if len(c) == 1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total clusters",  f"{len(clusters):,}")
    c2.metric("Multi-record",    f"{len(multi):,}")
    c3.metric("Singletons",      f"{len(singles):,}")
    c4.metric("Largest cluster", max((len(c) for c in clusters), default=0))

    gt_pairs = st.session_state.gt_pairs
    if gt_pairs is not None:
        st.subheader("Evaluation vs. Ground Truth")
        predicted = clusters_to_pairs(clusters, df)
        tp  = len(predicted & gt_pairs)
        prec = tp / len(predicted) if predicted else 0.0
        rec  = tp / len(gt_pairs)  if gt_pairs  else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Precision",       f"{prec:.3f}")
        e2.metric("Recall",          f"{rec:.3f}")
        e3.metric("F1",              f"{f1:.3f}")
        e4.metric("True Positives",  f"{tp:,}")
        e5.metric("GT pairs",        f"{len(gt_pairs):,}")
        st.caption(
            "Ground truth from `entity_id` column — cross-source pairs within the same entity. "
            "Precision = fraction of predicted pairs that are correct. "
            "Recall = fraction of true pairs found."
        )

    with st.expander("Run log"):
        for line in st.session_state.run_log:
            st.text(line)

    st.divider()

    # ── Network graph ─────────────────────────────────────────────────────────
    st.subheader("Network Graph")
    max_viz = st.slider("Clusters to visualise (largest first)", 5, 100, 20)

    top = sorted(multi, key=lambda c: -len(c))[:max_viz]

    if not top:
        st.info("No multi-record clusters to display.")
    else:
        try:
            from pyvis.network import Network as PyvisNet

            src_series = df.set_index("id")["source"]
            rec_idx    = df.set_index("id")
            sources    = sorted(df["source"].unique())
            _palette   = ["#e94560", "#0f3460", "#a8dadc", "#e94f37",
                           "#457b9d", "#00b4d8", "#f72585", "#7209b7"]
            src_color  = {s: _palette[i % len(_palette)] for i, s in enumerate(sources)}

            net = PyvisNet(height="580px", width="100%",
                           bgcolor="#0e1117", font_color="#fafafa",
                           notebook=False)
            net.barnes_hut(spring_length=120, spring_strength=0.03,
                           overlap=0.2, damping=0.9)

            for ci, cluster in enumerate(top):
                for nid in cluster:
                    row  = rec_idx.loc[nid] if nid in rec_idx.index else None
                    src  = src_series.get(nid, "?")
                    name = str(row["name"])[:22] if row is not None else nid
                    breed = str(row.get("breed", ""))[:15] if row is not None else ""
                    tip  = (f"<b>{nid}</b><br>Source: {src}<br>"
                            f"Name: {name}<br>Breed: {breed}<br>Cluster #{ci+1}")
                    net.add_node(nid, label=nid, title=tip,
                                 color=src_color.get(src, "#888888"),
                                 group=ci, size=18)
                members = list(cluster)
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = members[i], members[j]
                        is_match = tuple(sorted((a, b))) in match_pairs
                        net.add_edge(a, b,
                                     color="#4a9eff" if is_match else "#333333",
                                     width=3 if is_match else 1)

            html = net.generate_html()
            components.html(html, height=600, scrolling=False)

            legend = "  ".join(
                f'<span style="color:{src_color[s]}">●</span> {s}' for s in sources
            )
            st.markdown(f"**Sources:** {legend}", unsafe_allow_html=True)

        except ImportError:
            st.warning(
                "`pyvis` not installed. Install it with:\n```\npip install pyvis\n```"
            )

    st.divider()

    # ── Cluster table ─────────────────────────────────────────────────────────
    st.subheader("Cluster Table")
    show_singles = st.checkbox("Include singletons", value=False)
    to_show = sorted(
        clusters if show_singles else multi,
        key=lambda c: -len(c),
    )[:500]

    _cols = [c for c in ["id", "source", "name", "breed", "dob", "sire", "dam", "country"]
             if c in df.columns]
    rec_idx = df.set_index("id")

    for ci, cluster in enumerate(to_show, 1):
        label = (f"Cluster {ci} — {len(cluster)} record{'s' if len(cluster)>1 else ''}"
                 + ("  ✦" if len(cluster) > 1 else ""))
        with st.expander(label, expanded=False):
            rows = []
            for mid in sorted(cluster):
                if mid in rec_idx.index:
                    r = rec_idx.loc[mid]
                    rows.append({col: r.get(col, "") for col in _cols})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
