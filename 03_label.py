"""
03_label.py — rule-based pseudo-labelling of transactions.

Operationalises FATF red-flag indicators (FATF, 2020) into a suspicion
score and assigns pseudo-labels for downstream supervised learning.

Label scheme:
     1 = suspected illicit  (>= 2 red flags fired)
     0 = suspected licit    (0 red flags fired)
    -1 = uncertain          (exactly 1 red flag fired)

Produces: outputs/labeled_features.csv

Usage:
    python 03_label.py
"""

import os
import sys
import numpy as np
import pandas as pd

OUTPUT_DIR   = "outputs"
FEATURES_CSV = os.path.join(OUTPUT_DIR, "transaction_features.csv")
OUT_CSV      = os.path.join(OUTPUT_DIR, "labeled_features.csv")

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


if __name__ == "__main__":
    print("=" * 55)
    print("  Bitcoin Illicit Detection — Rule-Based Labelling")
    print("=" * 55 + "\n")

    if not os.path.exists(FEATURES_CSV):
        print(f"[ERROR] Features file not found: {FEATURES_CSV}")
        print("  Run 02_feature.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df):,} transactions from {FEATURES_CSV}\n")

    df = apply_rules(df)

    # ── Label distribution ───────────────────────────────────────────────────
    total  = len(df)
    counts = df["pseudo_label"].value_counts().sort_index()
    print("── Pseudo-label distribution ────────────────────────")
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
