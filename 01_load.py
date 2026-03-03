"""
01_load.py — load all raw CSVs and perform basic validation.

Run this first to confirm your data loaded correctly. The output of this
script should be documented in your Methods chapter (dataset description).

Usage:
    python 01_load.py
"""

import pandas as pd
import sys
import os

# ── Path configuration ──────────────────────────────────────────────────────
DATA_DIR = "data"
FILES = {
    "transactions": "Transactions.csv",
    "tx_details":   "Trans_Detail.csv",
    "address_info": "Address_Info.csv",
    "wallet_info":  "Wallet_Info.csv",
}


def load_all(data_dir: str = DATA_DIR) -> dict:
    """Load all four CSVs. Returns a dict of {name: DataFrame}."""
    dfs = {}
    for key, fname in FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[ERROR] Missing file: {path}")
            sys.exit(1)
        dfs[key] = pd.read_csv(path)
        print(f"  Loaded {key:15s} — {dfs[key].shape[0]:>8,} rows, {dfs[key].shape[1]} cols")
    return dfs


def validate(dfs: dict) -> None:
    """Print a data quality summary."""

    print("\n── Column names ───────────────────────────────────────")
    for name, df in dfs.items():
        print(f"  {name}: {list(df.columns)}")

    print("\n── Missing values ─────────────────────────────────────")
    for name, df in dfs.items():
        missing = df.isnull().sum()
        if missing.any():
            print(f"\n  {name}:")
            print(missing[missing > 0].to_string())
        else:
            print(f"  {name}: none")

    print("\n── Duplicate rows ─────────────────────────────────────")
    for name, df in dfs.items():
        dupes = df.duplicated().sum()
        print(f"  {name}: {dupes:,} duplicate rows")

    print("\n── Key dataset statistics ─────────────────────────────")
    tx  = dfs["transactions"]
    txd = dfs["tx_details"]
    print(f"  Unique wallets      : {tx['Wallet'].nunique():>10,}")
    print(f"  Unique blocks       : {txd['Block_No'].nunique():>10,}")
    n_tx = txd.groupby(["Block_No", "Trans_No"]).ngroups
    print(f"  Unique transactions : {n_tx:>10,}")

    print("\n── Output value distribution ──────────────────────────")
    print(txd["Output_Value"].describe().to_string())


if __name__ == "__main__":
    print("=" * 55)
    print("  Bitcoin Illicit Transaction Detection — Data Load")
    print("=" * 55 + "\n")
    dfs = load_all()
    validate(dfs)
    print("\n[OK] All files loaded and validated.\n")