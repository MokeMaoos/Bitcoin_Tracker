"""
build_section3_inventory.py — consolidate every numeric result currently
produced by the pipeline into a single markdown report at
outputs/section3_inventory.md, intended as a data source for Section 3
of the dissertation.

Reads (where present):
    outputs/transaction_features.csv
    outputs/enriched_features.csv
    outputs/labeled_features.csv
    outputs/cluster_assignments.csv
    outputs/mechanism_eval.json
    outputs/elliptic_eval.json
    outputs/temporal_eval.json
    outputs/model.pkl  (joblib bundle: model, features, km_model, classifier, label_mode)
    data/Address_Info.csv, data/Wallet_Info.csv, data/Block_Header.csv (for unique counts)

Does not invoke any pipeline step itself — purely a read-and-format pass.
"""

import datetime as dt
import io
import json
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

OUT_MD = os.path.join("outputs", "section3_inventory.md")

# Files we read.
FILES = {
    "transaction_features": "outputs/transaction_features.csv",
    "enriched_features":    "outputs/enriched_features.csv",
    "labeled_features":     "outputs/labeled_features.csv",
    "cluster_assignments":  "outputs/cluster_assignments.csv",
    "mechanism_eval":       "outputs/mechanism_eval.json",
    "elliptic_eval":        "outputs/elliptic_eval.json",
    "temporal_eval":        "outputs/temporal_eval.json",
    "model":                "outputs/model.pkl",
    "address_info":         "data/Address_Info.csv",
    "wallet_info":          "data/Wallet_Info.csv",
    "block_header":         "data/Block_Header.csv",
    "trans_detail":         "data/Trans_Detail.csv",
}

# Documented feature groups from 02_feature.py docstring.
FEATURE_GROUPS = [
    ("A. Output statistics",     ["total_output", "mean_output", "max_output",
                                   "min_output", "std_output", "value_entropy"]),
    ("B. Input statistics",      ["total_input"]),
    ("C. Address counts",        ["num_inputs", "num_outputs", "io_ratio", "value_ratio"]),
    ("D. Graph topology",        ["n_distinct_input_wallets", "n_distinct_output_wallets"]),
    ("E. FATF rule signals",     ["equal_value_flag", "peeling_flag", "coinjoin_strength"]),
    ("F. Fresh address ratio",   ["fresh_output_ratio"]),
    ("G. Wallet context",        ["wallet_size", "wallet_total_tx", "wallet_total_sent",
                                   "wallet_age", "wallet_sent_ratio",
                                   "wallet_final_balance", "wallet_balance_ratio"]),
    ("H. Wallet temporal",       ["wallet_balance_velocity", "wallet_balance_range",
                                   "wallet_balance_std"]),
    ("I. Tx metadata",           ["tx_fee", "tx_fee_normalized", "lock_time_flag",
                                   "tx_hour", "tx_day_of_week", "is_night"]),
    ("J. Output-address stats",  ["mean_out_addr_tx_count", "mean_out_addr_balance",
                                   "frac_out_addr_zero_balance", "mean_out_addr_age",
                                   "mean_out_addr_sent_ratio", "mean_out_addr_bal_velocity"]),
]
RULE_COLS = ["rule_equal_value", "rule_peeling", "rule_high_fan_out",
             "rule_fresh_outputs", "rule_mixing", "rule_high_entropy"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt(v, prec: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        # Show very small numbers with extra precision.
        if 0 < abs(v) < 1e-3:
            return f"{v:.6g}"
        return f"{v:.{prec}f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def mtime(path: str) -> str:
    if not os.path.exists(path):
        return "(missing)"
    ts = dt.datetime.fromtimestamp(os.path.getmtime(path))
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def safe_load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def md_code(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


# ── Section builders ──────────────────────────────────────────────────────────

def build_header() -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bundle_meta = {}
    if os.path.exists(FILES["model"]):
        try:
            b = joblib.load(FILES["model"])
            bundle_meta["classifier"] = b.get("classifier", "?")
            bundle_meta["label_mode"] = b.get("label_mode", "?")
            bundle_meta["n_features"] = len(b.get("features", []))
        except Exception as exc:
            bundle_meta["error"] = str(exc)

    rows = [[k, FILES[k], mtime(FILES[k])] for k in FILES]
    src_table = md_table(rows, ["key", "path", "mtime"])

    producers = [
        ("01_load.py",            "data/* (validation only)"),
        ("02_feature.py",         "outputs/transaction_features.csv"),
        ("02b_enrich_labels.py",  "outputs/enriched_features.csv"),
        ("03_label.py",           "outputs/labeled_features.csv"),
        ("04_train.py",           "outputs/model.pkl, outputs/model_<mode>.pkl, outputs/cluster_assignments.csv"),
        ("06_evaluate.py",        "outputs/elliptic_eval.json"),
        ("07_temporal_eval.py",   "outputs/temporal_eval.json"),
        ("08_mechanism_eval.py",  "outputs/mechanism_eval.json"),
    ]
    prod_table = md_table([[a, b] for a, b in producers], ["script", "produces"])

    return (
        f"# Section 3 — Results Inventory\n\n"
        f"**Generated**: {now}\n\n"
        f"**Pipeline configuration** (from `outputs/model.pkl`)\n\n"
        f"- Classifier: `{bundle_meta.get('classifier', 'n/a')}`\n"
        f"- Training label mode: `{bundle_meta.get('label_mode', 'n/a')}`\n"
        f"- Features used by the classifier: {bundle_meta.get('n_features', 'n/a')}\n\n"
        f"## Source files (paths relative to repository working directory)\n\n"
        f"{src_table}\n\n"
        f"## Producing scripts\n\n"
        f"{prod_table}\n"
    )


def section_1_dataset() -> str:
    if not os.path.exists(FILES["transaction_features"]):
        return ("## 1. Dataset overview\n\n"
                "Not available — `outputs/transaction_features.csv` is missing; "
                "run `python 02_feature.py` first.\n")

    feat = pd.read_csv(FILES["transaction_features"])
    n_tx = len(feat)

    # Datetime range — prefer block-confirmation date (unambiguous) over
    # `tx_datetime`, which on the multi-period CUBiD dataset reflects an
    # earlier dayfirst-parsing bug in 02_feature.py.
    block_dt_map = {}
    if os.path.exists(FILES["block_header"]):
        bh = pd.read_csv(FILES["block_header"],
                         usecols=lambda c: c.strip().lower().lstrip("﻿")
                                            in ("block_no", "dt"))
        bh.columns = bh.columns.str.strip().str.replace("﻿", "")
        if {"Block_No", "DT"}.issubset(bh.columns):
            bh["block_dt"] = pd.to_datetime(bh["DT"], errors="coerce",
                                             format="mixed")
            block_dt_map = dict(zip(bh["Block_No"].astype(int),
                                     bh["block_dt"]))

    if block_dt_map:
        ts = pd.to_numeric(feat["Block_No"], errors="coerce").map(block_dt_map)
        ts = pd.to_datetime(ts, errors="coerce")
        date_min = ts.min()
        date_max = ts.max()
    elif "tx_datetime" in feat.columns:
        ts = pd.to_datetime(feat["tx_datetime"], errors="coerce")
        date_min = ts.min()
        date_max = ts.max()
    else:
        date_min = date_max = None
        ts = None

    # Block / wallet / address uniques (from data CSVs).
    n_blocks_feat = feat["Block_No"].nunique()
    n_blocks_hdr  = None
    n_addr_in     = n_addr_out = n_wallets = None
    n_input_wallets_distinct = n_output_wallets_distinct = None

    if os.path.exists(FILES["block_header"]):
        bh = pd.read_csv(FILES["block_header"], usecols=lambda c: c.strip().lower() in ("﻿block_no","block_no"))
        bh.columns = bh.columns.str.strip().str.replace("﻿", "")
        n_blocks_hdr = bh["Block_No"].nunique() if "Block_No" in bh.columns else None

    # Address / wallet uniques: read Trans_Detail directly so we get the
    # actual addresses observed in the sampled transactions.
    if os.path.exists(FILES["trans_detail"]):
        td = pd.read_csv(FILES["trans_detail"],
                         usecols=["Input_Address", "Output_Address",
                                  "Input_Wallet", "Output_Wallet"])
        n_addr_in  = td["Input_Address"].nunique(dropna=True)
        n_addr_out = td["Output_Address"].nunique(dropna=True)
        wallets = pd.unique(pd.concat([td["Input_Wallet"], td["Output_Wallet"]],
                                       ignore_index=True).dropna())
        n_wallets = len(wallets)

    # Per-tx descriptive stats.
    desc_cols = ["total_input", "total_output", "num_inputs", "num_outputs", "tx_fee"]
    desc = feat[desc_cols].describe().T[["mean", "50%", "min", "max", "std"]]
    desc.columns = ["mean", "median", "min", "max", "std"]
    desc_rows = [[c, fmt(desc.loc[c, "mean"]), fmt(desc.loc[c, "median"]),
                  fmt(desc.loc[c, "min"]), fmt(desc.loc[c, "max"]),
                  fmt(desc.loc[c, "std"])] for c in desc_cols]
    desc_table = md_table(desc_rows,
                          ["column", "mean", "median", "min", "max", "std"])

    # Total volume — feature matrix stores values in satoshis (1 BTC = 1e8).
    total_satoshi = float(feat["total_input"].sum())
    total_btc     = total_satoshi / 1e8

    # Hour-of-day distribution. tx_hour comes from the TM field which is
    # unambiguous (HH:MM:SS), so it is correct even on the buggy parse.
    hour_counts = feat["tx_hour"].value_counts().sort_index()
    hour_rows = [[int(h), int(hour_counts.get(h, 0))] for h in range(24)]
    hour_table = md_table(hour_rows, ["hour", "count"])

    # Day-of-week distribution — recompute from the Block_Header-derived
    # date so the multi-period dataset reports the correct snapshot DOW.
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if block_dt_map and ts is not None:
        dow_series = ts.dt.dayofweek
        dow_counts = dow_series.value_counts().sort_index()
    else:
        dow_counts = feat["tx_day_of_week"].value_counts().sort_index()
    dow_rows = [[i, dow_names[i] if i < len(dow_names) else f"day_{i}",
                 int(dow_counts.get(i, 0))] for i in range(7)]
    dow_table = md_table(dow_rows, ["dow_index", "name", "count"])

    # Number of features actually in the matrix (excluding key/datetime cols).
    non_feature = {"Block_No", "Trans_No", "tx_datetime"}
    n_features = sum(1 for c in feat.columns if c not in non_feature)

    # Feature group breakdown (documented vs present).
    present = set(feat.columns)
    grp_rows = []
    grp_total = 0
    for name, cols in FEATURE_GROUPS:
        in_matrix = [c for c in cols if c in present]
        missing   = [c for c in cols if c not in present]
        grp_total += len(in_matrix)
        grp_rows.append([
            name,
            len(cols),
            len(in_matrix),
            ", ".join(missing) if missing else "—",
        ])
    grp_table = md_table(grp_rows,
                         ["group", "documented", "present in matrix", "missing"])

    # Sanity discrepancy note.
    feature_groups_total = sum(len(cols) for _, cols in FEATURE_GROUPS)
    note = (f"Documented feature count across the 10 groups in 02_feature.py: "
            f"**{feature_groups_total}**. "
            f"Number of non-key columns in `transaction_features.csv`: "
            f"**{n_features}**. "
            f"Number of features used by the trained model: "
            f"**{len(joblib.load(FILES['model'])['features']) if os.path.exists(FILES['model']) else 'n/a'}**. "
            f"The dissertation brief states 38 features — flagged for reconciliation.")

    out = ["## 1. Dataset overview\n"]
    out.append("### 1.1 Transaction sample\n")
    summary_rows = [
        ["Total transactions in feature matrix", fmt(n_tx)],
        ["Earliest tx_datetime",
            date_min.strftime('%Y-%m-%d %H:%M:%S') if date_min is not None else "—"],
        ["Latest tx_datetime",
            date_max.strftime('%Y-%m-%d %H:%M:%S') if date_max is not None else "—"],
        ["Unique blocks (from feature matrix)", fmt(n_blocks_feat)],
        ["Unique blocks (from Block_Header.csv)", fmt(n_blocks_hdr)],
        ["Unique input addresses (Trans_Detail.csv)",  fmt(n_addr_in)],
        ["Unique output addresses (Trans_Detail.csv)", fmt(n_addr_out)],
        ["Unique wallets (input ∪ output, Trans_Detail.csv)", fmt(n_wallets)],
        ["Total volume (sum of total_input, satoshi)",     fmt(total_satoshi, prec=0)],
        ["Total volume (sum of total_input, BTC)",         fmt(total_btc, prec=4)],
        ["Number of feature columns in matrix (excl. keys/datetime)", fmt(n_features)],
    ]
    out.append(md_table(summary_rows, ["metric", "value"]))
    out.append("\n### 1.2 Per-transaction descriptive statistics\n")
    out.append(desc_table)
    out.append("\n### 1.3 Hour-of-day distribution (UTC, 24 buckets)\n")
    out.append(hour_table)
    out.append("\n### 1.4 Day-of-week distribution (0=Monday)\n")
    out.append(dow_table)
    out.append("\n### 1.5 Feature group breakdown\n")
    out.append(grp_table)
    out.append(f"\n*Note*: {note}\n")
    return "\n".join(out)


def section_2_pseudo_labels() -> str:
    if not os.path.exists(FILES["labeled_features"]):
        return ("## 2. Pseudo-label distribution and rule firing\n\n"
                "Not available — `outputs/labeled_features.csv` is missing; "
                "run `python 03_label.py` first.\n")

    df = pd.read_csv(FILES["labeled_features"])
    n = len(df)

    # Label totals.
    label_type = df.get("label_type", pd.Series(["pseudo"] * n))
    pseudo = df["pseudo_label"]
    rows = []
    for cat, mask in [
        ("Confirmed illicit", (label_type == "confirmed") & (pseudo == 1)),
        ("Confirmed licit",   (label_type == "confirmed") & (pseudo == 0)),
        ("Pseudo illicit",    (label_type == "pseudo")    & (pseudo == 1)),
        ("Pseudo licit",      (label_type == "pseudo")    & (pseudo == 0)),
        ("Uncertain (-1)",    pseudo == -1),
    ]:
        cnt = int(mask.sum())
        rows.append([cat, fmt(cnt), f"{cnt/n:.4%}"])
    label_table = md_table(rows, ["bucket", "count", "share"])

    # Suspicion score distribution (0 through 6).
    score_counts = df["suspicion_score"].value_counts().sort_index()
    score_rows = [[s, fmt(int(score_counts.get(s, 0))),
                   f"{score_counts.get(s, 0)/n:.4%}"]
                  for s in range(7)]
    score_table = md_table(score_rows, ["suspicion_score", "count", "share"])

    # Per-rule firing counts.
    rule_rows = []
    for r in RULE_COLS:
        if r in df.columns:
            cnt = int(df[r].sum())
            rule_rows.append([r, fmt(cnt), f"{cnt/n:.4%}"])
        else:
            rule_rows.append([r, "missing", "—"])
    rule_table = md_table(rule_rows, ["rule", "n_fired", "fire rate"])

    # Co-occurrence matrix (counts of tx where both rules fired).
    rule_present = [r for r in RULE_COLS if r in df.columns]
    R = df[rule_present].astype(int).to_numpy()
    co = R.T @ R  # diagonal = single-rule counts; off-diag = joint counts
    short = [r.replace("rule_", "") for r in rule_present]
    header = ["rule"] + short
    co_rows = [[short[i]] + [int(co[i, j]) for j in range(len(short))]
               for i in range(len(short))]
    co_table = md_table(co_rows, header)

    # Rule firing on confirmed-illicit vs confirmed-licit subsets.
    if "confirmed_label" in df.columns:
        ci = df[df["confirmed_label"] == 1]
        cl = df[df["confirmed_label"] == 0]
        n_ci = max(len(ci), 1)
        n_cl = max(len(cl), 1)
        sub_rows = []
        for r in RULE_COLS:
            if r not in df.columns:
                continue
            f_ci = int(ci[r].sum()) if len(ci) else 0
            f_cl = int(cl[r].sum()) if len(cl) else 0
            sub_rows.append([
                r,
                fmt(f_ci), f"{f_ci/n_ci:.4%}",
                fmt(f_cl), f"{f_cl/n_cl:.4%}",
            ])
        sub_table = md_table(
            sub_rows,
            ["rule", "fired (illicit)", "rate (illicit)",
             "fired (licit)",  "rate (licit)"])
        sub_caption = (f"Confirmed illicit n = {len(ci):,}, "
                       f"confirmed licit n = {len(cl):,}.")
    else:
        sub_table = "Not available — `confirmed_label` column not present."
        sub_caption = ""

    out = ["## 2. Pseudo-label distribution and rule firing\n"]
    out.append("### 2.1 Label type counts\n")
    out.append(label_table)
    out.append("\n### 2.2 Suspicion score distribution\n")
    out.append(score_table)
    out.append("\n### 2.3 Per-rule firing counts (whole sample)\n")
    out.append(rule_table)
    out.append("\n### 2.4 Rule co-occurrence matrix (whole sample)\n")
    out.append(md_code(co_table, lang=""))
    out.append("\n*Diagonal entries = single-rule fire counts; "
               "off-diagonal entries = transactions where both rules fired.*\n")
    out.append("\n### 2.5 Rule firing — confirmed illicit vs licit subsets\n")
    out.append(sub_caption)
    out.append("")
    out.append(sub_table)
    return "\n".join(out)


def section_3_enrichment() -> str:
    if not os.path.exists(FILES["enriched_features"]):
        return ("## 3. Confirmed-label enrichment results\n\n"
                "Not available — `outputs/enriched_features.csv` is missing; "
                "run `python 02b_enrich_labels.py` first.\n")

    df = pd.read_csv(FILES["enriched_features"])
    n = len(df)
    matched_mask = df["confirmation_source"].notna()
    matched = df[matched_mask].copy()
    n_matched = len(matched)

    # Per-source matched (a tx can be in multiple sources).
    if n_matched:
        src_lists = matched["confirmation_source"].str.split(";")
        exploded = pd.DataFrame({
            "source": src_lists.explode().reset_index(drop=True),
            "label":  np.repeat(matched["confirmed_label"].to_numpy(),
                                src_lists.apply(len).to_numpy()),
        })
        per_source = (
            exploded
            .assign(illicit=lambda d: (d["label"] == 1).astype(int),
                    licit  =lambda d: (d["label"] == 0).astype(int))
            .groupby("source")[["illicit", "licit"]].sum()
            .reset_index()
        )
        per_source["total"] = per_source["illicit"] + per_source["licit"]
        per_source = per_source.sort_values("source").reset_index(drop=True)
        per_source_rows = [[r["source"], fmt(int(r["illicit"])),
                            fmt(int(r["licit"])), fmt(int(r["total"]))]
                           for _, r in per_source.iterrows()]
        per_source_table = md_table(per_source_rows,
                                    ["source", "illicit", "licit", "total"])
    else:
        per_source_table = "No confirmed matches present."

    # Multi-source matches.
    if n_matched:
        n_src = matched["confirmation_source"].str.count(";") + 1
        multi_rows = []
        for k in sorted(n_src.unique()):
            cnt = int((n_src == k).sum())
            multi_rows.append([int(k), fmt(cnt), f"{cnt/n_matched:.4%}"])
        multi_table = md_table(multi_rows,
                               ["# of sources matched", "transactions",
                                "share of matched"])
        n_multi = int((n_src > 1).sum())
    else:
        multi_table = "No matches."
        n_multi = 0

    # confirmation_strength distribution.
    if "confirmation_strength" in df.columns:
        strength = df["confirmation_strength"].fillna("(none)").value_counts()
        strength_rows = [[k, fmt(int(v)), f"{int(v)/n:.4%}"]
                         for k, v in strength.items()]
        strength_table = md_table(strength_rows,
                                  ["confirmation_strength", "count", "share"])
    else:
        strength_table = "Column not present."

    # Master lookup contributions — read directly from the JSON-ish summary
    # is not persisted, so we recompute here by re-loading the source files
    # only if available. We also report the values printed by 02b_enrich_labels.py
    # for reference. Skipped here to avoid a re-scan; flagged as best-effort.
    lookup_note = ("Master-lookup pre-dedup contribution counts are not "
                   "persisted to disk by 02b_enrich_labels.py — they are "
                   "printed to stdout only. The most recent run reported:")
    pre_dedup_table = md_table(
        [
            ["BABD-13",         "138,641", "402,671", "541,312"],
            ["BitcoinAbuse",    "47,099",  "0",       "47,099"],
            ["EllipticPlusPlus","14,266",  "251,088", "265,354"],
            ["OpenSanctions",   "11,782",  "0",       "11,782"],
        ],
        ["source", "illicit (addresses)", "licit (addresses)", "total"]
    )
    post_dedup_table = md_table(
        [
            ["Total unique addresses", "861,529"],
            ["Confirmed illicit",      "209,043"],
            ["Confirmed licit",        "652,486"],
        ],
        ["metric", "addresses"]
    )

    out = ["## 3. Confirmed-label enrichment results\n"]
    summary_rows = [
        ["Total transactions",                          fmt(n)],
        ["Transactions with any confirmed match",       fmt(n_matched)],
        ["Confirmed illicit",                           fmt(int((df['confirmed_label']==1).sum()))],
        ["Confirmed licit",                             fmt(int((df['confirmed_label']==0).sum()))],
        ["Transactions matched by multiple sources",    fmt(n_multi)],
    ]
    out.append(md_table(summary_rows, ["metric", "value"]))
    out.append("\n### 3.1 Per-source matched transactions (counts)\n")
    out.append(per_source_table)
    out.append("\n*A transaction can contribute to multiple rows when its "
               "addresses match more than one source — totals therefore "
               "exceed the unique matched count.*\n")
    out.append("\n### 3.2 Multi-source coverage\n")
    out.append(multi_table)
    out.append("\n### 3.3 confirmation_strength distribution\n")
    out.append(strength_table)
    out.append("\n### 3.4 Master lookup table — last reported run\n")
    out.append(lookup_note + "\n")
    out.append("**Per-source pre-dedup contributions:**\n")
    out.append(pre_dedup_table)
    out.append("\n**Post-dedup totals:**\n")
    out.append(post_dedup_table)
    return "\n".join(out)


_FULL_METRIC_KEYS = [
    "precision_macro", "recall_macro", "f1_macro",
    "f1_weighted",
    "precision_illicit", "recall_illicit",
    "precision_licit",   "recall_licit",
    "roc_auc", "pr_auc",
]


def _cv_table(cv_metrics: dict, folds: int) -> str:
    rows = []
    for k in _FULL_METRIC_KEYS:
        m = cv_metrics.get(k, {})
        if not m:
            continue
        rows.append([
            k,
            fmt(m.get("mean")),
            fmt(m.get("std")),
            ", ".join(fmt(x) for x in m.get("all", [])),
        ])
    return md_table(rows, ["metric", "mean", "std", f"per-fold ({folds})"])


def _cm_block(cm: list) -> str:
    if not cm or len(cm) != 2:
        return "Not available"
    return ("          pred_licit  pred_illicit\n"
            f"true_licit    {cm[0][0]:>9,}    {cm[0][1]:>9,}\n"
            f"true_illicit  {cm[1][0]:>9,}    {cm[1][1]:>9,}")


def _sweep_table(sweep: dict) -> str:
    if not sweep or sweep.get("skipped"):
        return None
    rows = []
    per_t = sweep.get("per_threshold", {})
    for t in range(7):
        m = per_t.get(str(t), {})
        if not m:
            continue
        rows.append([
            t,
            fmt(m.get("n_flagged")),
            fmt(m.get("n_tp")) if m.get("n_tp") is not None else "—",
            fmt(m.get("precision")),
            fmt(m.get("recall")),
            fmt(m.get("f1")),
        ])
    return md_table(rows,
                    ["threshold (≥)", "n_flagged", "n_tp",
                     "precision", "recall", "f1"])


def section_4_mechanism() -> str:
    j = safe_load_json(FILES["mechanism_eval"])
    if j is None:
        return ("## 4. Mechanism validation results\n\n"
                "Not available — `outputs/mechanism_eval.json` is missing; "
                "run `python 08_mechanism_eval.py` first.\n")

    schema = j.get("schema_version", 1)
    classifier = j.get("classifier", "?")
    n_conf = j.get("n_confirmed_transactions")
    n_ill  = j.get("n_confirmed_illicit")
    n_lit  = j.get("n_confirmed_licit")
    aug    = j.get("augmented_eval_set", {})

    # 4.1 Classifier metadata.
    bundle_info = "model.pkl not present"
    if os.path.exists(FILES["model"]):
        try:
            b = joblib.load(FILES["model"])
            clf = b["model"].named_steps["clf"]
            params = clf.get_params()
            keep = ["n_estimators", "learning_rate", "max_depth", "num_leaves",
                    "class_weight", "random_state", "n_jobs"]
            param_rows = [[k, fmt(params.get(k))] for k in keep if k in params]
            bundle_info = md_table(param_rows, ["hyperparameter", "value"])
        except Exception as exc:
            bundle_info = f"(error reading model bundle: {exc})"

    # 4.2 Eval set composition (new in v2).
    eval_rows = [
        ["Confirmed-only subset (rows)", fmt(n_conf)],
        ["  confirmed illicit",          fmt(n_ill)],
        ["  confirmed licit",            fmt(n_lit)],
        ["Augmented eval set (rows)",    fmt(aug.get("n_total"))],
        ["  confirmed illicit",          fmt(aug.get("n_confirmed_illicit"))],
        ["  confirmed licit",            fmt(aug.get("n_confirmed_licit"))],
        ["  presumed licit (sampled)",   fmt(aug.get("n_presumed_licit"))],
    ]
    eval_table = md_table(eval_rows, ["component", "value"])

    # 4.3 Augmented CV.
    aug_cv = j.get("supervised_cv_augmented", {}) or {}
    aug_cv_metrics = aug_cv.get("cv_metrics", {})
    aug_cv_table = _cv_table(aug_cv_metrics, aug_cv.get("n_cv_folds", 5))
    aug_cm_text  = _cm_block(aug_cv.get("confusion_matrix", []))

    # 4.4 Confirmed-only CV (legacy view).
    conf_cv = j.get("supervised_cv_confirmed", {}) or j.get("supervised_cv", {}) or {}
    if conf_cv.get("skipped"):
        conf_cv_table = f"Not available — {conf_cv.get('note')}"
        conf_cm_text  = "Not available"
    else:
        conf_cv_metrics = conf_cv.get("cv_metrics", {})
        conf_cv_table = _cv_table(conf_cv_metrics, conf_cv.get("n_cv_folds", 5))
        conf_cm_text  = _cm_block(conf_cv.get("confusion_matrix", []))

    # 4.5 Per-rule on augmented set.
    rv = j.get("rule_validation_augmented", j.get("rule_validation", {}))
    rule_rows = []
    for r in RULE_COLS + ["any_rule"]:
        m = rv.get(r, {})
        rule_rows.append([
            r,
            fmt(m.get("n_fired")),
            fmt(m.get("n_tp")) if m.get("n_tp") is not None else "—",
            fmt(m.get("precision")),
            fmt(m.get("recall")),
            fmt(m.get("f1")),
        ])
    rule_table = md_table(rule_rows,
                          ["rule", "n_fired", "n_tp",
                           "precision", "recall", "f1"])

    # 4.6 Per-source threshold sweeps.
    sweeps = j.get("threshold_sweeps", {})
    sweep_blocks = []
    sweep_order = ["augmented_full"] + sorted(
        k for k in sweeps if k != "augmented_full")
    for src in sweep_order:
        sweep = sweeps.get(src, {})
        if sweep.get("skipped"):
            sweep_blocks.append(
                f"**{src}** — skipped: {sweep.get('note')}\n")
            continue
        tbl = _sweep_table(sweep)
        if tbl is None:
            continue
        opt_t = sweep.get("optimal_threshold")
        opt_f1 = sweep.get("optimal_f1")
        cur_t = sweep.get("current_default")
        cur_f1 = sweep.get("current_default_f1")
        sweep_blocks.append(
            f"**{src}**\n\n{tbl}\n\n"
            f"- Optimal threshold: **{opt_t}** (F1={fmt(opt_f1)})\n"
            f"- Current default:   **{cur_t}** (F1={fmt(cur_f1)})\n")

    # 4.7 Per-source CV — primary single sources only.
    psb = j.get("per_source_breakdown", {})
    primary_sources = ["EllipticPlusPlus", "BABD-13",
                       "BitcoinAbuse", "OpenSanctions"]
    per_src_rows = []
    for src in primary_sources:
        s = psb.get(src, {})
        if not s:
            per_src_rows.append([src, "missing"] + ["—"] * 8 + ["not in JSON"])
            continue
        cv_s = s.get("supervised_cv", {})
        if cv_s.get("skipped"):
            per_src_rows.append([
                src,
                fmt(s.get("n_transactions")),
                fmt(s.get("n_illicit")),
                fmt(s.get("n_licit")),
                "—", "—", "—", "—", "—",
                cv_s.get("note", "skipped"),
            ])
            continue
        m = cv_s.get("cv_metrics", {})
        per_src_rows.append([
            src,
            fmt(s.get("n_transactions")),
            fmt(s.get("n_illicit")),
            fmt(s.get("n_licit")),
            fmt(m.get("f1_macro", {}).get("mean")),
            fmt(m.get("f1_weighted", {}).get("mean")),
            fmt(m.get("precision_illicit", {}).get("mean")),
            fmt(m.get("recall_illicit", {}).get("mean")),
            fmt(m.get("roc_auc", {}).get("mean")),
            "ran",
        ])
    per_src_table = md_table(per_src_rows,
                             ["source", "n_tx", "n_illicit", "n_licit",
                              "F1(macro)", "F1(weighted)",
                              "P(illicit)", "R(illicit)",
                              "ROC-AUC", "status"])

    out = ["## 4. Mechanism validation results\n"]
    out.append(f"- Classifier: **{classifier}** "
               f"(JSON schema v{schema})\n")
    if j.get("schema_changes"):
        out.append("\n*Schema changes:* " + " ".join(j["schema_changes"]) + "\n")

    out.append("\n### 4.1 Trained-model hyperparameters\n")
    out.append(bundle_info)

    out.append("\n### 4.2 Evaluation-set composition\n")
    out.append(eval_table)
    out.append("\n*The augmented evaluation set adds presumed-licit rows "
               "(sampled from rows where `label_type == \"pseudo\"` and "
               "`pseudo_label == 0`) to balance the confirmed-illicit pool. "
               "Class balance is therefore ~50/50 in the augmented set.*\n")

    out.append("\n### 4.3 5-fold stratified CV on the augmented eval set\n")
    out.append(aug_cv_table)
    out.append("\n**Confusion matrix (out-of-fold predictions)**\n")
    out.append(md_code(aug_cm_text))

    out.append("\n### 4.4 5-fold stratified CV on the confirmed-only subset (legacy view)\n")
    out.append(conf_cv_table)
    out.append("\n**Confusion matrix (out-of-fold predictions)**\n")
    out.append(md_code(conf_cm_text))

    out.append("\n### 4.5 Per-rule precision / recall on the augmented set\n")
    out.append(rule_table)
    out.append("\n*Augmenting the licit pool with rows whose suspicion_score "
               "was already 0 does not change rule-firing counts (the new "
               "presumed-licit rows have all `rule_*` flags equal to 0 by "
               "construction). The numbers therefore equal those on the "
               "confirmed-only subset.*\n")

    out.append("\n### 4.6 Per-source threshold sweeps\n")
    out.append("\n".join(sweep_blocks) if sweep_blocks
               else "Not available.\n")

    out.append("\n### 4.7 Per-source CV breakdown (primary single sources)\n")
    out.append(per_src_table)
    out.append("\n*BitcoinAbuse and OpenSanctions are single-class sources "
               "(illicit-only); supervised CV requires both classes, so they "
               "are reported with row counts only. Compound source rows "
               "(e.g. `BABD-13;BitcoinAbuse`) are captured in "
               "`per_source_breakdown` but excluded from this primary table.*\n")
    return "\n".join(out)


def section_5_elliptic() -> str:
    j = safe_load_json(FILES["elliptic_eval"])
    if j is None:
        return ("## 5. External benchmark validation (Elliptic++)\n\n"
                "Not available — `outputs/elliptic_eval.json` is missing; "
                "run `python 06_evaluate.py` first.\n")

    classifier = j.get("classifier", "?")
    out = ["## 5. External benchmark validation (Elliptic++)\n",
           f"- Dataset source: {j.get('dataset_source', '—')}\n",
           f"- Classifier: **{classifier}**\n"]

    for level in ("transaction_level", "actor_level"):
        block = j.get(level, {})
        if not block:
            out.append(f"\n### 5.x {level} — not present in JSON\n")
            continue
        cv = block.get("cv_metrics", {})
        n_samples = block.get("n_samples")
        labels = block.get("label_counts", {})
        rows = []
        for k in ("precision_macro", "recall_macro", "f1_macro",
                  "precision_weighted", "recall_weighted", "f1_weighted"):
            m = cv.get(k, {})
            rows.append([
                k,
                fmt(m.get("mean")),
                fmt(m.get("std")),
                ", ".join(fmt(x) for x in m.get("all", [])),
            ])
        cv_table = md_table(rows, ["metric", "mean", "std", "per-fold"])
        cm = block.get("confusion_matrix", [])
        cm_text = "\n".join(
            ["          pred_licit  pred_illicit"] +
            [f"true_licit    {cm[0][0]:>9,}    {cm[0][1]:>9,}",
             f"true_illicit  {cm[1][0]:>9,}    {cm[1][1]:>9,}"]
        ) if cm else "—"
        title = "transaction-level" if level == "transaction_level" else "actor-level"
        out.append(f"\n### 5.{1 if level=='transaction_level' else 2} {title} results\n")
        out.append(f"- n_samples: {fmt(n_samples)}\n"
                   f"- n_features: {fmt(block.get('n_features'))}\n"
                   f"- label counts: total={fmt(labels.get('total_labeled'))}, "
                   f"illicit={fmt(labels.get('illicit'))}, "
                   f"licit={fmt(labels.get('licit'))}\n")
        out.append(cv_table)
        out.append("\n**Confusion matrix (sum across folds)**\n")
        out.append(md_code(cm_text))

    out.append("\n### 5.3 Random Forest comparison\n")
    out.append("Not available — only the LightGBM run is captured in "
               "`elliptic_eval.json`; an RF baseline is not persisted by "
               "`06_evaluate.py`.\n")
    return "\n".join(out)


def section_6_temporal() -> str:
    j = safe_load_json(FILES["temporal_eval"])
    if j is None:
        return ("## 6. Temporal validation\n\n"
                "Not available — `outputs/temporal_eval.json` is missing; "
                "run `python 07_temporal_eval.py` first.\n")

    out = ["## 6. Temporal validation\n"]
    rows = [
        ["split_mode",        j.get("split_mode")],
        ["classifier",        j.get("classifier")],
        ["n_feature_cols",    j.get("n_feature_cols")],
        ["train_period",      j.get("train_period")],
        ["test_period",       j.get("test_period")],
    ]
    out.append(md_table([[k, fmt(v) if isinstance(v, (int, float)) else (v or "—")]
                         for k, v in rows], ["metric", "value"]))

    for tag, key in (("Train", "train_metrics"), ("Test", "test_metrics")):
        m = j.get(key, {}) or {}
        labels = m.get("label_counts", {}) or {}
        cm = m.get("confusion_matrix", [])
        cm_text = "\n".join(
            ["          pred_licit  pred_illicit"] +
            [f"true_licit    {cm[0][0]:>9,}    {cm[0][1]:>9,}",
             f"true_illicit  {cm[1][0]:>9,}    {cm[1][1]:>9,}"]
        ) if cm else "—"
        out.append(f"\n### 6.{1 if tag=='Train' else 2} {tag} metrics\n")
        out.append(md_table([
            ["n_samples", fmt(m.get("n_samples"))],
            ["illicit",   fmt(labels.get("illicit"))],
            ["licit",     fmt(labels.get("licit"))],
            ["precision", fmt(m.get("precision"))],
            ["recall",    fmt(m.get("recall"))],
            ["f1",        fmt(m.get("f1"))],
        ], ["metric", "value"]))
        out.append("\n**Confusion matrix**\n")
        out.append(md_code(cm_text))

    drift = j.get("performance_drift", {}) or {}
    out.append("\n### 6.3 Drift differential (test − train)\n")
    out.append(md_table([
        ["precision", fmt(drift.get("precision"))],
        ["recall",    fmt(drift.get("recall"))],
        ["f1",        fmt(drift.get("f1"))],
    ], ["metric", "delta"]))
    if drift.get("note"):
        out.append(f"\n*{drift['note']}*\n")

    out.append("\n### 6.4 Per-month test metrics\n")
    out.append("Not available — single-period split mode "
               f"(`{j.get('split_mode')}`) does not produce per-month "
               "breakdowns. A multi-period split would require ≥2 calendar "
               "months in the sample.\n")
    return "\n".join(out)


def section_7_feature_importance() -> str:
    if not os.path.exists(FILES["model"]):
        return ("## 7. Feature importance\n\n"
                "Not available — `outputs/model.pkl` is missing.\n")
    bundle = joblib.load(FILES["model"])
    feats  = bundle["features"]
    clf    = bundle["model"].named_steps["clf"]
    booster = clf.booster_
    split  = booster.feature_importance(importance_type="split")
    gain   = booster.feature_importance(importance_type="gain")

    df = pd.DataFrame({"feature": feats, "split": split, "gain": gain})

    top_split  = df.sort_values("split", ascending=False).head(20)
    top_gain   = df.sort_values("gain",  ascending=False).head(20)
    bottom     = df.sort_values("split", ascending=True).head(10)

    def make_table(d):
        return md_table(
            [[r["feature"], fmt(int(r["split"])), fmt(r["gain"])]
             for _, r in d.iterrows()],
            ["feature", "split", "gain"])

    # Group totals.
    feat_to_group = {}
    for grp, cols in FEATURE_GROUPS:
        for c in cols:
            feat_to_group[c] = grp
    df["group"] = df["feature"].map(feat_to_group).fillna("(ungrouped)")
    grp_totals = (
        df.groupby("group")[["split", "gain"]].sum()
        .sort_values("split", ascending=False).reset_index()
    )
    grp_table = md_table(
        [[r["group"], fmt(int(r["split"])), fmt(r["gain"])]
         for _, r in grp_totals.iterrows()],
        ["group", "Σ split", "Σ gain"])

    out = ["## 7. Feature importance\n",
           "Computed from the LightGBM booster inside `outputs/model.pkl`. "
           "`split` = number of trees that split on the feature; "
           "`gain` = total gain across those splits.\n"]
    out.append("\n### 7.1 Top 20 by split count\n")
    out.append(make_table(top_split))
    out.append("\n### 7.2 Top 20 by gain\n")
    out.append(make_table(top_gain))
    out.append("\n### 7.3 Bottom 10 by split count\n")
    out.append(make_table(bottom))
    out.append("\n### 7.4 Importance by feature group (sums)\n")
    out.append(grp_table)
    return "\n".join(out)


def section_8_clusters() -> str:
    if not os.path.exists(FILES["cluster_assignments"]):
        return ("## 8. Cluster analysis\n\n"
                "Not available — `outputs/cluster_assignments.csv` is missing; "
                "run `python 04_train.py` first.\n")
    if not os.path.exists(FILES["labeled_features"]):
        return ("## 8. Cluster analysis\n\n"
                "Not available — `outputs/labeled_features.csv` is missing.\n")

    ca = pd.read_csv(FILES["cluster_assignments"])
    feat = pd.read_csv(FILES["labeled_features"])
    df  = feat.merge(ca, on=["Block_No", "Trans_No"], how="inner")

    # K-Means clusters.
    km_n = ca["cluster_kmeans"].nunique() if "cluster_kmeans" in ca.columns else None

    profile_cols = ["coinjoin_strength", "num_outputs", "value_entropy",
                    "fresh_output_ratio", "peeling_flag", "equal_value_flag",
                    "total_input"]
    available = [c for c in profile_cols if c in df.columns]

    if "cluster_kmeans" in df.columns:
        km_summary = (
            df.groupby("cluster_kmeans")[available].mean().round(4).reset_index()
        )
        sizes = df["cluster_kmeans"].value_counts().sort_index()
        km_summary["size"] = km_summary["cluster_kmeans"].map(sizes)
        km_summary = km_summary[["cluster_kmeans", "size"] + available]

        # Pseudo-label distribution per cluster.
        pseudo_pivot = (
            df.assign(
                bucket=df["pseudo_label"].map(
                    {1: "illicit", 0: "licit", -1: "uncertain"}))
            .groupby(["cluster_kmeans", "bucket"]).size().unstack(fill_value=0)
            .reset_index()
        )
        # Confirmed-label distribution per cluster.
        df_conf = df[df["confirmation_source"].notna()].copy()
        if not df_conf.empty:
            conf_pivot = (
                df_conf.assign(
                    bucket=df_conf["confirmed_label"].map(
                        {1.0: "confirmed_illicit", 0.0: "confirmed_licit"}))
                .groupby(["cluster_kmeans", "bucket"]).size()
                .unstack(fill_value=0)
                .reset_index()
            )
        else:
            conf_pivot = pd.DataFrame()
    else:
        km_summary = pd.DataFrame()
        pseudo_pivot = pd.DataFrame()
        conf_pivot = pd.DataFrame()

    out = ["## 8. Cluster analysis\n"]
    out.append("### 8.1 K-Means\n")
    out.append(f"- Clusters: **{km_n}**\n")
    if not km_summary.empty:
        header = ["cluster", "size"] + available
        rows = []
        for _, r in km_summary.iterrows():
            row = [int(r["cluster_kmeans"]), fmt(int(r["size"]))]
            for c in available:
                row.append(fmt(r[c]))
            rows.append(row)
        out.append(md_table(rows, header))
    else:
        out.append("Cluster summary not available (cluster_kmeans missing).\n")

    out.append("\n### 8.2 Pseudo-label distribution per K-Means cluster\n")
    if not pseudo_pivot.empty:
        cols = [c for c in ["cluster_kmeans", "illicit", "licit", "uncertain"]
                if c in pseudo_pivot.columns]
        rows = [[int(r[cols[0]])] + [int(r[c]) for c in cols[1:]]
                for _, r in pseudo_pivot[cols].iterrows()]
        out.append(md_table(rows, cols))
    else:
        out.append("Not available.\n")

    out.append("\n### 8.3 Confirmed-label distribution per K-Means cluster\n")
    if not conf_pivot.empty:
        cols = [c for c in ["cluster_kmeans", "confirmed_illicit",
                            "confirmed_licit"] if c in conf_pivot.columns]
        rows = [[int(r[cols[0]])] + [int(r[c]) for c in cols[1:]]
                for _, r in conf_pivot[cols].iterrows()]
        out.append(md_table(rows, cols))
    else:
        out.append("No confirmed-label transactions intersect the cluster set.\n")

    out.append("\n### 8.4 DBSCAN (full-dataset)\n")
    if "cluster_dbscan" in ca.columns:
        ds       = ca["cluster_dbscan"].value_counts().sort_index()
        n_total  = len(ca)
        n_noise  = int(ds.get(-1, 0))
        n_unsamp = int(ds.get(-2, 0))   # legacy sentinel — should be 0 now
        cluster_ids = [c for c in ds.index if c not in (-1, -2)]
        n_clusters  = len(cluster_ids)
        rows = [
            ["DBSCAN clusters (excl. noise)", fmt(n_clusters)],
            ["Noise points (label = -1)",    fmt(n_noise)],
            ["Out-of-sample sentinel (-2)",  fmt(n_unsamp)],
            ["Total rows in cluster_assignments.csv", fmt(n_total)],
        ]
        out.append(md_table(rows, ["metric", "value"]))
        if n_unsamp > 0:
            out.append("\n*A non-zero `-2` count means the legacy sampling "
                       "path was used. Re-run `04_train.py` to assign every "
                       "row a real cluster.*\n")
        else:
            out.append("\n*04_train.py now runs DBSCAN on the full feature "
                       "matrix (no -2 sentinel). Cluster ids cover every row "
                       "in the dataset.*\n")
        if cluster_ids:
            top5 = (ca[ca["cluster_dbscan"].isin(cluster_ids)]["cluster_dbscan"]
                    .value_counts().head(5))
            out.append("\n**Largest 5 DBSCAN clusters**\n")
            out.append(md_table(
                [[int(k), fmt(int(v))] for k, v in top5.items()],
                ["cluster_id", "size"]))
    else:
        out.append("Not available — `cluster_dbscan` column missing.\n")
    return "\n".join(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def section_9_per_period() -> str:
    """Per-snapshot breakdown of label and rule firing rates."""
    if not (os.path.exists(FILES["labeled_features"])
            and os.path.exists(FILES["block_header"])):
        return ("## 9. Per-period breakdown\n\n"
                "Not available — `outputs/labeled_features.csv` or "
                "`data/Block_Header.csv` is missing.\n")

    # Build Block_No → period (YYYY-MM) map from Block_Header.
    bh = pd.read_csv(FILES["block_header"], usecols=["Block_No", "DT"], dtype=str)
    bh["Block_No"] = pd.to_numeric(bh["Block_No"], errors="coerce").astype("Int64")
    bh["block_dt"] = pd.to_datetime(bh["DT"], errors="coerce", format="mixed")
    bh = bh.dropna(subset=["Block_No", "block_dt"])
    period_map = dict(zip(bh["Block_No"].astype(int),
                          bh["block_dt"].dt.strftime("%Y-%m-%d")))

    df = pd.read_csv(FILES["labeled_features"], usecols=[
        "Block_No", "pseudo_label", "label_type", "confirmed_label",
        "suspicion_score",
    ] + RULE_COLS)
    df["period"] = pd.to_numeric(df["Block_No"],
                                 errors="coerce").map(period_map)
    df = df.dropna(subset=["period"])

    # Per-period label distribution.
    rows = []
    for p, sub in df.groupby("period"):
        n = len(sub)
        ci = int(((sub["label_type"] == "confirmed") & (sub["pseudo_label"] == 1)).sum())
        cl = int(((sub["label_type"] == "confirmed") & (sub["pseudo_label"] == 0)).sum())
        pi = int(((sub["label_type"] == "pseudo")    & (sub["pseudo_label"] == 1)).sum())
        pl = int(((sub["label_type"] == "pseudo")    & (sub["pseudo_label"] == 0)).sum())
        un = int((sub["pseudo_label"] == -1).sum())
        rows.append([p, fmt(n), fmt(ci), fmt(cl), fmt(pi), fmt(pl), fmt(un)])
    label_table = md_table(rows,
        ["period", "n_tx", "conf.illicit", "conf.licit",
         "pseudo.ill", "pseudo.lic", "uncertain"])

    # Per-period rule firing rates.
    rule_rows = []
    for p, sub in df.groupby("period"):
        rates = []
        for r in RULE_COLS:
            rate = float(sub[r].mean()) if r in sub.columns else float("nan")
            rates.append(f"{rate:.3f}")
        rule_rows.append([p] + rates)
    rule_header = ["period"] + [r.replace("rule_", "") for r in RULE_COLS]
    rule_table = md_table(rule_rows, rule_header)

    out = ["## 9. Per-period breakdown\n",
           "All rows bucketed by **block-confirmation date** "
           "(Block_No → Block_Header.DT), which is the unambiguous period "
           "key for the multi-period CUBiD dataset.\n"]
    out.append("\n### 9.1 Label distribution per snapshot\n")
    out.append(label_table)
    out.append("\n### 9.2 Rule firing rates per snapshot (mean of 0/1)\n")
    out.append(rule_table)
    return "\n".join(out)


def main():
    sections = [
        build_header(),
        section_1_dataset(),
        section_2_pseudo_labels(),
        section_3_enrichment(),
        section_4_mechanism(),
        section_5_elliptic(),
        section_6_temporal(),
        section_7_feature_importance(),
        section_8_clusters(),
        section_9_per_period(),
    ]
    body = "\n\n".join(s.rstrip() + "\n" for s in sections)

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(body)

    print(f"\n[OK] Section 3 inventory written to: "
          f"{os.path.abspath(OUT_MD)}")


if __name__ == "__main__":
    main()
