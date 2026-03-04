"""
02_feature.py — engineer a transaction-level feature matrix (vectorised).

All features are computed with pandas groupby/agg/transform — no Python-level
loops over transactions. See inline comments for ddof and NaN choices.

Produces: outputs/transaction_features.csv

Usage:
    python 02_feature.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
OUT_CSV    = os.path.join(OUTPUT_DIR, "transaction_features.csv")
KEYS       = ["Block_No", "Trans_No"]


def load_data():
    td = pd.read_csv(os.path.join(DATA_DIR, "Trans_Detail.csv"))
    tx = pd.read_csv(os.path.join(DATA_DIR, "Transactions.csv"))
    ai = pd.read_csv(os.path.join(DATA_DIR, "Address_Info.csv"))
    wi = pd.read_csv(os.path.join(DATA_DIR, "Wallet_Info.csv"))
    return td, tx, ai, wi


def build_features(td: pd.DataFrame, tx: pd.DataFrame,
                   ai: pd.DataFrame, wi: pd.DataFrame) -> pd.DataFrame:

    # ── Output-side aggregations ──────────────────────────────────────────────
    print("  [1/7] Output-side aggregations...")
    ov_stats = (
        td.groupby(KEYS, sort=False)["Output_Value"]
        .agg(mean_output="mean", max_output="max", min_output="min")
        .reset_index()
    )
    # total_output: min_count=1 so all-NaN groups stay NaN (matches numpy sum)
    total_output = (
        td.groupby(KEYS, sort=False)["Output_Value"]
        .sum(min_count=1)
        .rename("total_output")
        .reset_index()
    )
    # std with ddof=0 to match original numpy behaviour; fillna(0) for 1-output groups
    std_output = (
        td.groupby(KEYS, sort=False)["Output_Value"]
        .std(ddof=0)
        .fillna(0)
        .rename("std_output")
        .reset_index()
    )

    # ── Input-side aggregations ───────────────────────────────────────────────
    print("  [2/7] Input-side aggregations...")
    total_input = (
        td.groupby(KEYS, sort=False)["Input_Value"]
        .sum(min_count=1)   # NaN for coinbase (no-input) transactions
        .rename("total_input")
        .reset_index()
    )

    # ── Address counts (nunique is the slowest step) ──────────────────────────
    print("  [3/7] Address nunique counts...")
    num_inputs = (
        td.groupby(KEYS, sort=False)["Input_Address"]
        .nunique()
        .rename("num_inputs")
        .reset_index()
    )
    num_outputs = (
        td.groupby(KEYS, sort=False)["Output_Address"]
        .nunique()
        .rename("num_outputs")
        .reset_index()
    )

    # ── Value entropy (fully vectorised via transform) ────────────────────────
    print("  [4/7] Value entropy...")
    td_ov = td[KEYS + ["Output_Value"]].copy()
    total_per_tx = td_ov.groupby(KEYS, sort=False)["Output_Value"].transform("sum")
    p = (td_ov["Output_Value"] / total_per_tx.clip(lower=1e-300)).clip(lower=1e-300)
    td_ov["entropy_term"] = -(p * np.log2(p))
    value_entropy = (
        td_ov.groupby(KEYS, sort=False)["entropy_term"]
        .sum()
        .rename("value_entropy")
        .reset_index()
    )

    # ── Fresh output ratio ────────────────────────────────────────────────────
    print("  [5/7] Fresh output ratio...")
    addr_col = next(
        (c for c in ("ADR", "Address", "address", "addr") if c in ai.columns),
        None
    )
    known = set(ai[addr_col].dropna().values) if addr_col else set(ai.iloc[:, 0].dropna().values)

    td_fr = td[KEYS + ["Output_Address"]].copy()
    valid = td_fr["Output_Address"].notna()
    td_fr["is_fresh"] = np.nan
    td_fr.loc[valid, "is_fresh"] = (
        ~td_fr.loc[valid, "Output_Address"].isin(known)
    ).astype(float)
    fresh_ratio = (
        td_fr.groupby(KEYS, sort=False)["is_fresh"]
        .mean()
        .rename("fresh_output_ratio")
        .reset_index()
    )

    # ── Merge all aggregations ────────────────────────────────────────────────
    print("  [6/7] Merging aggregations...")
    feat = (
        total_output
        .merge(ov_stats,     on=KEYS)
        .merge(std_output,   on=KEYS)
        .merge(total_input,  on=KEYS)
        .merge(num_inputs,   on=KEYS)
        .merge(num_outputs,  on=KEYS)
        .merge(value_entropy, on=KEYS)
        .merge(fresh_ratio,  on=KEYS)
    )

    # ── Derived features ──────────────────────────────────────────────────────
    feat["io_ratio"]          = feat["num_inputs"] / feat["num_outputs"].clip(lower=1)
    feat["value_ratio"]       = feat["total_input"] / feat["total_output"].clip(lower=1e-9)
    feat["equal_value_flag"]  = (
        feat["std_output"] / (feat["mean_output"].abs() + 1e-9) < 0.01
    ).astype(int)
    feat["peeling_flag"]      = (
        (feat["max_output"] / feat["total_output"].clip(lower=1e-9) >= 0.8)
        & (feat["num_outputs"] >= 2)
    ).astype(int)
    feat["coinjoin_strength"] = (
        0.6 * feat["equal_value_flag"]
        + 0.2 * (feat["num_inputs"]  / 10).clip(upper=1.0)
        + 0.2 * (feat["num_outputs"] / 10).clip(upper=1.0)
    ).round(4)

    # ── Wallet features with CoinJoin suppression ─────────────────────────────
    print("  [7/7] Wallet features...")
    wallet_map = (
        tx[["Block_No", "Trans_No", "Wallet"]]
        .merge(
            wi[["Wallet", "N_Transactions", "Total_Sent", "N_Addresses"]],
            on="Wallet", how="left"
        )
        .rename(columns={
            "N_Transactions": "wallet_total_tx",
            "Total_Sent":     "wallet_total_sent",
            "N_Addresses":    "wallet_size",
        })
    )
    feat = feat.merge(
        wallet_map[["Block_No", "Trans_No", "wallet_total_tx",
                    "wallet_total_sent", "wallet_size"]],
        on=KEYS, how="left"
    )
    cj_mask = feat["coinjoin_strength"] > 0.1
    feat.loc[cj_mask, ["wallet_total_tx", "wallet_total_sent", "wallet_size"]] = np.nan

    return feat


if __name__ == "__main__":
    print("=" * 55)
    print("  Bitcoin Illicit Detection — Feature Engineering")
    print("=" * 55 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading raw data...")
    td, tx, ai, wi = load_data()
    n_tx = td.groupby(KEYS).ngroups
    print(f"  {n_tx:,} transactions, {len(td):,} detail rows\n")

    t0   = time.time()
    feat = build_features(td, tx, ai, wi)
    elapsed = time.time() - t0

    # Canonical column order
    feat = feat[KEYS + [c for c in feat.columns if c not in KEYS]]
    feat.to_csv(OUT_CSV, index=False)

    print(f"\n[OK] Features saved → {OUT_CSV}")
    print(f"     Shape  : {feat.shape[0]:,} rows × {feat.shape[1]} cols")
    print(f"     Runtime: {elapsed:.1f} s")

    n_cj = (feat["coinjoin_strength"] > 0.1).sum()
    print(f"\n  CoinJoin suppression : {n_cj:,} transactions ({n_cj/len(feat):.1%})")
    print(f"  Peeling chains       : {int(feat['peeling_flag'].sum()):,}")
    print(f"  Equal-value outputs  : {int(feat['equal_value_flag'].sum()):,}")

    print("\nFeature summary:")
    print(feat.drop(columns=KEYS).describe().to_string())
