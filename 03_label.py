"""
03_label.py — rule-based pseudo-labelling with confirmed-label priority.

Operationalises FATF red-flag indicators (FATF, 2020) into a suspicion
score and assigns pseudo-labels for downstream supervised learning.

Label priority:
    1.  If a transaction's addresses were matched to an external confirmed
        source by 02b_enrich_labels.py, the confirmed label is used directly.
    2.  Otherwise, the FATF rule-based pseudo-label is applied as fallback.

Label scheme (pseudo_label column):
     1 = illicit  (confirmed illicit, or >= 2 red flags fired)
     0 = licit    (confirmed licit, or 0 red flags fired)
    -1 = uncertain (exactly 1 red flag fired, no confirmed match)

New column:
    label_type — "confirmed" if derived from an external source, "pseudo" otherwise.

Reads from: outputs/enriched_features.csv  (if present, produced by 02b_enrich_labels.py)
            outputs/transaction_features.csv  (fallback if enrichment was skipped)

Produces: outputs/labeled_features.csv

Usage:
    python 03_label.py
"""

import os
import sys
import numpy as np
import pandas as pd

OUTPUT_DIR        = "outputs"
ENRICHED_CSV      = os.path.join(OUTPUT_DIR, "enriched_features.csv")
FEATURES_CSV      = os.path.join(OUTPUT_DIR, "transaction_features.csv")
OUT_CSV           = os.path.join(OUTPUT_DIR, "labeled_features.csv")

# ── Rule thresholds ──────────────────────────────────────────────────────────
# Each threshold maps a FATF red-flag concept to a measurable boundary.
HIGH_FAN_OUT      = 10   # >= 10 output addresses → structuring / fund dispersion
HIGH_FRESH_RATIO  = 0.8  # >= 80 % fresh output addresses → address churn / layering
HIGH_COINJOIN     = 0.5  # strong CoinJoin signal → mixing service involvement
HIGH_ENTROPY      = 3.0  # bits — unusually diverse output-value distribution


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each transaction against FATF red-flag rules.
    Each rule that fires contributes 1 to the suspicion_score.

    Rules implemented:
        rule_equal_value   — equal-value splitting (CoinJoin / structuring)
        rule_peeling       — peeling-chain pattern (layering)
        rule_high_fan_out  — high output count (fund dispersion)
        rule_fresh_outputs — high proportion of new addresses (layering)
        rule_mixing        — strong CoinJoin signal (mixer involvement)
        rule_high_entropy  — anomalous value distribution
    """
    df = df.copy()

    df["rule_equal_value"]   = df["equal_value_flag"].astype(int)
    df["rule_peeling"]       = df["peeling_flag"].astype(int)
    df["rule_high_fan_out"]  = (df["num_outputs"]        >= HIGH_FAN_OUT).astype(int)
    df["rule_fresh_outputs"] = (df["fresh_output_ratio"] >= HIGH_FRESH_RATIO).astype(int)
    df["rule_mixing"]        = (df["coinjoin_strength"]  >= HIGH_COINJOIN).astype(int)
    df["rule_high_entropy"]  = (df["value_entropy"]      >= HIGH_ENTROPY).astype(int)

    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    df["suspicion_score"] = df[rule_cols].sum(axis=1)

    df["pseudo_label"] = np.where(
        df["suspicion_score"] >= 2,  1,
        np.where(df["suspicion_score"] == 0, 0, -1)
    )

    return df


def apply_confirmed_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overlay confirmed external labels onto the pseudo-labelled DataFrame.

    If the ``confirmed_label`` column is present (written by 02b_enrich_labels.py),
    it takes priority over the pseudo-label for rows where it is not NaN.

    Adds a ``label_type`` column: "confirmed" or "pseudo".
    """
    df = df.copy()

    if "confirmed_label" not in df.columns:
        df["label_type"] = "pseudo"
        return df

    confirmed_mask = df["confirmed_label"].notna()

    # Override pseudo_label with confirmed_label where available
    df.loc[confirmed_mask, "pseudo_label"] = (
        df.loc[confirmed_mask, "confirmed_label"].astype(int)
    )

    df["label_type"] = "pseudo"
    df.loc[confirmed_mask, "label_type"] = "confirmed"

    return df


if __name__ == "__main__":
    print("=" * 55)
    print("  Bitcoin Illicit Detection — Rule-Based Labelling")
    print("=" * 55 + "\n")

    # Prefer enriched feature matrix; fall back to raw features if enrichment
    # was skipped (e.g., run with --skip-enrichment in run_pipeline.py).
    if os.path.exists(ENRICHED_CSV):
        src_csv = ENRICHED_CSV
    elif os.path.exists(FEATURES_CSV):
        print(f"[INFO] Enriched features not found; falling back to {FEATURES_CSV}")
        print("       Run 02b_enrich_labels.py to enable confirmed-label priority.\n")
        src_csv = FEATURES_CSV
    else:
        print(f"[ERROR] Features file not found: {FEATURES_CSV}")
        print("  Run 02_feature.py first.")
        sys.exit(1)

    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df):,} transactions from {src_csv}\n")

    df = apply_rules(df)
    df = apply_confirmed_labels(df)

    total = len(df)

    # ── Label type breakdown ─────────────────────────────────────────────────
    if "label_type" in df.columns and df["label_type"].eq("confirmed").any():
        n_conf = int(df["label_type"].eq("confirmed").sum())
        conf_ill  = int(((df["label_type"]=="confirmed") & (df["pseudo_label"]==1)).sum())
        conf_lit  = int(((df["label_type"]=="confirmed") & (df["pseudo_label"]==0)).sum())
        pseudo_ill = int(((df["label_type"]=="pseudo") & (df["pseudo_label"]==1)).sum())
        pseudo_lit = int(((df["label_type"]=="pseudo") & (df["pseudo_label"]==0)).sum())
        uncertain  = int((df["pseudo_label"] == -1).sum())
        print("── Label distribution ───────────────────────────────")
        print(f"  Confirmed illicit   : {conf_ill:>8,}")
        print(f"  Confirmed licit     : {conf_lit:>8,}")
        print(f"  Pseudo illicit      : {pseudo_ill:>8,}")
        print(f"  Pseudo licit        : {pseudo_lit:>8,}")
        print(f"  Uncertain (pseudo)  : {uncertain:>8,}")
    else:
        print("── Pseudo-label distribution ────────────────────────")
        counts = df["pseudo_label"].value_counts().sort_index()
        for lbl, n in counts.items():
            name = {1: "Illicit", 0: "Licit", -1: "Uncertain"}[lbl]
            print(f"  {name:10s} ({lbl:2d}) : {n:>8,}  ({n / total:.1%})")

    # ── Rule firing rates ────────────────────────────────────────────────────
    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    print("\n── Rule firing rates ────────────────────────────────")
    for col in rule_cols:
        n = int(df[col].sum())
        print(f"  {col:25s}: {n:>8,}  ({n / total:.1%})")

    # ── Suspicion score distribution ─────────────────────────────────────────
    print("\n── Suspicion score distribution ─────────────────────")
    for score, n in df["suspicion_score"].value_counts().sort_index().items():
        print(f"  Score {score}: {n:>8,}  ({n / total:.1%})")

    df.to_csv(OUT_CSV, index=False)
    labeled = (df["pseudo_label"] != -1).sum()
    print(f"\n[OK] Labeled features saved → {OUT_CSV}")
    print(f"     Labeled rows (excl. uncertain): {labeled:,} / {total:,}")
