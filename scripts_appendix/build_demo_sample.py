"""
build_demo_sample.py — produce small versions of the labeled feature matrix,
cluster assignments, and raw CUBiD CSVs, for fast Streamlit dashboard
screenshots.

Sample size: ≈ 5,000 transactions, stratified to include some confirmed
illicit and licit examples (so the per-tx and per-wallet pages have
non-empty results) while preserving the rule-firing distribution that
the dashboard's plots will display.

Writes:
    outputs/labeled_features_demo.csv
    outputs/cluster_assignments_demo.csv
    data_demo/Trans_Detail.csv
    data_demo/Transactions.csv
    data_demo/Address_Info.csv
    data_demo/Wallet_Info.csv
    data_demo/Block_Header.csv
"""

import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(os.path.join(__file__, "..")))
LBL  = "outputs/labeled_features.csv"
CLU  = "outputs/cluster_assignments.csv"
TD   = "data/Trans_Detail.csv"
TX   = "data/Transactions.csv"
AI   = "data/Address_Info.csv"
WI   = "data/Wallet_Info.csv"
BH   = "data/Block_Header.csv"

OUT_LBL = "outputs/labeled_features_demo.csv"
OUT_CLU = "outputs/cluster_assignments_demo.csv"

DEMO_DATA_DIR = "data_demo"

# Target sample sizes per stratum (sums to ~5,000 total)
N_CONF_ILL = 200
N_CONF_LIT = 100
N_PSEUDO_ILL = 1500
N_PSEUDO_LIT = 1500
N_UNCERTAIN  = 1700

RND = np.random.default_rng(42)


def stratified_sample() -> pd.DataFrame:
    print(f"[1/6] Reading {LBL}…")
    t0 = time.time()
    df = pd.read_csv(LBL)
    print(f"      loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    def take(mask, n, name):
        avail = df[mask]
        n_take = min(n, len(avail))
        idx = RND.choice(len(avail), size=n_take, replace=False)
        sub = avail.iloc[idx]
        print(f"      {name}: target={n}, took={len(sub):,} "
              f"(of {len(avail):,} available)")
        return sub

    pieces = [
        take((df["label_type"] == "confirmed") & (df["pseudo_label"] == 1),
             N_CONF_ILL,   "confirmed illicit"),
        take((df["label_type"] == "confirmed") & (df["pseudo_label"] == 0),
             N_CONF_LIT,   "confirmed licit"),
        take((df["label_type"] == "pseudo") & (df["pseudo_label"] == 1),
             N_PSEUDO_ILL, "pseudo illicit"),
        take((df["label_type"] == "pseudo") & (df["pseudo_label"] == 0),
             N_PSEUDO_LIT, "pseudo licit"),
        take(df["pseudo_label"] == -1,
             N_UNCERTAIN,  "uncertain"),
    ]
    sample = pd.concat(pieces, ignore_index=True)
    print(f"      → demo sample: {len(sample):,} rows")
    return sample


def write_cluster_subset(sample: pd.DataFrame) -> None:
    print(f"[2/6] Filtering {CLU} to sample keys…")
    cl = pd.read_csv(CLU)
    keys = set(zip(sample["Block_No"].astype(int),
                   sample["Trans_No"].astype(str)))
    cl["Block_No"] = cl["Block_No"].astype(int)
    cl["Trans_No"] = cl["Trans_No"].astype(str)
    sub_keys = pd.Series(list(zip(cl["Block_No"], cl["Trans_No"])),
                         index=cl.index)
    cl_sub = cl[sub_keys.isin(keys)]
    cl_sub.to_csv(OUT_CLU, index=False)
    print(f"      kept {len(cl_sub):,} rows → {OUT_CLU}")


def stream_filter_trans_detail(keys: set, addr_set: set, wallet_set: set,
                                out_path: str) -> tuple[set, set]:
    """
    Stream-filter Trans_Detail.csv: keep rows whose (Block_No, Trans_No) is
    in *keys*. Side-effect: collect input/output addresses and wallets seen.
    """
    print(f"[3/6] Streaming {TD} (5.3 GB)…")
    t0 = time.time()
    chunks = pd.read_csv(TD, chunksize=1_000_000, dtype=str, low_memory=False)
    parts = []
    rows_seen = 0
    for ch in chunks:
        rows_seen += len(ch)
        ch["_bn"] = pd.to_numeric(ch["Block_No"], errors="coerce").astype("Int64")
        ch["_tn"] = ch["Trans_No"].astype(str)
        mask = pd.Series(
            [(b, t) in keys for b, t in zip(ch["_bn"].astype("Int64"),
                                            ch["_tn"])],
            index=ch.index)
        sub = ch[mask].drop(columns=["_bn", "_tn"]).copy()
        if not sub.empty:
            parts.append(sub)
            for col in ("Input_Address", "Output_Address"):
                addr_set.update(sub[col].dropna().tolist())
            for col in ("Input_Wallet", "Output_Wallet"):
                wallet_set.update(sub[col].dropna().tolist())
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(out_path, index=False)
    print(f"      streamed {rows_seen:,} rows in {time.time()-t0:.1f}s; "
          f"kept {len(out):,} → {out_path}")
    return addr_set, wallet_set


def filter_transactions(keys: set, out_path: str) -> None:
    print(f"[4/6] Filtering {TX}…")
    tx = pd.read_csv(TX, dtype=str, low_memory=False)
    bn = pd.to_numeric(tx["Block_No"], errors="coerce").astype("Int64")
    tn = tx["Trans_No"].astype(str)
    mask = pd.Series([(b, t) in keys for b, t in zip(bn, tn)],
                     index=tx.index)
    sub = tx[mask]
    sub.to_csv(out_path, index=False)
    print(f"      kept {len(sub):,} of {len(tx):,} → {out_path}")


def filter_address_info(addr_set: set, out_path: str) -> None:
    print(f"[5/6] Streaming {AI} (2.5 GB) for {len(addr_set):,} addresses…")
    t0 = time.time()
    parts = []
    rows_seen = 0
    for ch in pd.read_csv(AI, chunksize=500_000, dtype=str, low_memory=False):
        rows_seen += len(ch)
        addr_col = next(
            (c for c in ("ADR", "Address", "address") if c in ch.columns),
            None)
        if addr_col is None:
            continue
        sub = ch[ch[addr_col].isin(addr_set)]
        if not sub.empty:
            parts.append(sub)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(out_path, index=False)
    print(f"      streamed {rows_seen:,} rows in {time.time()-t0:.1f}s; "
          f"kept {len(out):,} → {out_path}")


def filter_wallet_info(wallet_set: set, out_path: str) -> None:
    print(f"[6/6] Filtering {WI}…")
    wi = pd.read_csv(WI, dtype=str, low_memory=False)
    sub = wi[wi["Wallet"].isin(wallet_set)]
    sub.to_csv(out_path, index=False)
    print(f"      kept {len(sub):,} of {len(wi):,} → {out_path}")


def main():
    sample = stratified_sample()
    sample.to_csv(OUT_LBL, index=False)
    print(f"      wrote {OUT_LBL}")

    write_cluster_subset(sample)

    os.makedirs(DEMO_DATA_DIR, exist_ok=True)
    keys = set(zip(sample["Block_No"].astype(int),
                   sample["Trans_No"].astype(str)))
    addr_set: set = set()
    wallet_set: set = set()

    stream_filter_trans_detail(
        keys, addr_set, wallet_set,
        os.path.join(DEMO_DATA_DIR, "Trans_Detail.csv"))
    filter_transactions(
        keys, os.path.join(DEMO_DATA_DIR, "Transactions.csv"))
    filter_address_info(
        addr_set, os.path.join(DEMO_DATA_DIR, "Address_Info.csv"))
    filter_wallet_info(
        wallet_set, os.path.join(DEMO_DATA_DIR, "Wallet_Info.csv"))

    # Block_Header is small — copy as-is
    print(f"      copying Block_Header.csv as-is…")
    pd.read_csv(BH).to_csv(
        os.path.join(DEMO_DATA_DIR, "Block_Header.csv"), index=False)

    print(f"\n[OK] Demo sample ready under outputs/ and {DEMO_DATA_DIR}/")


if __name__ == "__main__":
    main()
