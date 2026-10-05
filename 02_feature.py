"""
02_feature.py — engineer a transaction-level feature matrix.

All features are computed with pandas groupby/agg/transform — no Python-level
loops over transactions.

Feature groups (38 total):
  A. Output statistics     (6)  — value distribution, entropy
  B. Input statistics      (1)  — total input
  C. Address counts        (4)  — num_inputs, num_outputs, io_ratio, value_ratio
  D. Graph topology        (2)  — distinct input/output wallets per tx
  E. FATF rule signals     (3)  — equal_value_flag, peeling_flag, coinjoin_strength
  F. Fresh address ratio   (1)  — fresh_output_ratio
  G. Wallet context        (7)  — size, tx count, sent, age, sent_ratio, balance, balance_ratio
  H. Wallet temporal       (3)  — balance velocity, range, std (Wallet_Balance_Block)
  I. Tx metadata           (6)  — fee, fee_normalised, lock_time, hour, day_of_week, is_night
  J. Output-address stats  (6)  — mean tx-count, mean balance, zero-balance frac,
                                   mean age, mean sent-ratio, mean addr balance velocity

Produces: outputs/transaction_features.csv

Usage:
    python 02_feature.py
"""

import os
import sys
import time
from typing import Optional
import numpy as np
import pandas as pd

DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
OUT_CSV    = os.path.join(OUTPUT_DIR, "transaction_features.csv")
KEYS       = ["Block_No", "Trans_No"]

# All wallet-level columns — suppressed to NaN when CoinJoin strength > 0.1
WALLET_FEATURE_COLS = [
    "wallet_total_tx", "wallet_total_sent", "wallet_size",
    "wallet_age", "wallet_sent_ratio", "wallet_final_balance",
    "wallet_balance_ratio", "wallet_balance_velocity",
    "wallet_balance_range", "wallet_balance_std",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    td = pd.read_csv(os.path.join(DATA_DIR, "Trans_Detail.csv"))
    tx = pd.read_csv(os.path.join(DATA_DIR, "Transactions.csv"))
    ai = pd.read_csv(os.path.join(DATA_DIR, "Address_Info.csv"))
    wi = pd.read_csv(os.path.join(DATA_DIR, "Wallet_Info.csv"))
    return td, tx, ai, wi


def load_block_header():
    """Block-level fee averages for fee normalisation."""
    path = os.path.join(DATA_DIR, "Block_Header.csv")
    if not os.path.exists(path):
        return None
    bh = pd.read_csv(path)
    bh.columns = bh.columns.str.lstrip("\ufeff")
    bh["block_avg_fee"] = (
        pd.to_numeric(bh["Trans_Fees"], errors="coerce") /
        pd.to_numeric(bh["Number_Transactions"], errors="coerce").clip(lower=1)
    )
    return bh[["Block_No", "block_avg_fee"]]


def _chunked_balance_agg(path: str, id_col: str, id_dtype,
                          chunk_size: int = 1_500_000) -> Optional[pd.DataFrame]:
    """
    Stream-aggregate a large balance time-series CSV without loading it all into RAM.

    Tracks per-entity: earliest balance, latest balance, max balance,
    running Welford variance (for std), and count.
    Peak RAM = O(unique entities) + one chunk at a time — typically < 300 MB.

    Returns a DataFrame with columns [id_col, 'velocity', 'range', 'std'].
    """
    size_gb = os.path.getsize(path) / 1e9
    print(f"  Streaming {os.path.basename(path)} ({size_gb:.1f} GB) in "
          f"{chunk_size//1_000_000}M-row chunks…")
    t0 = time.time()

    # stats[entity_id] = [min_blk, min_bal, max_blk, max_bal, n, mean, M2]
    # Welford online algorithm for variance: M2 = sum of squared deviations
    stats: dict = {}

    try:
        reader = pd.read_csv(
            path,
            usecols=[id_col, "Block_No", "Balance"],
            dtype={"Block_No": np.int32, "Balance": np.float32},
            chunksize=chunk_size,
        )
        chunks_read = 0
        for chunk in reader:
            chunks_read += 1
            if chunks_read % 10 == 0:
                elapsed = time.time() - t0
                print(f"    …chunk {chunks_read}  ({elapsed:.0f}s elapsed)")

            for eid, grp in chunk.groupby(id_col, sort=False):
                blk = grp["Block_No"].values
                bal = grp["Balance"].values.astype(np.float64)

                min_i = blk.argmin()
                max_i = blk.argmax()

                if eid not in stats:
                    # [min_blk, min_bal, max_blk, max_bal, n, mean, M2]
                    n    = len(bal)
                    mean = bal.mean()
                    M2   = ((bal - mean) ** 2).sum()
                    stats[eid] = [
                        int(blk[min_i]), float(bal[min_i]),
                        int(blk[max_i]), float(bal[max_i]),
                        n, mean, M2,
                    ]
                else:
                    s = stats[eid]
                    if blk[min_i] < s[0]:
                        s[0], s[1] = int(blk[min_i]), float(bal[min_i])
                    if blk[max_i] > s[2]:
                        s[2], s[3] = int(blk[max_i]), float(bal[max_i])
                    # Parallel Welford merge
                    n_b   = len(bal)
                    mean_b = bal.mean()
                    M2_b   = ((bal - mean_b) ** 2).sum()
                    n_a, mean_a, M2_a = s[4], s[5], s[6]
                    n_new  = n_a + n_b
                    delta  = mean_b - mean_a
                    mean_new = mean_a + delta * n_b / n_new
                    M2_new   = M2_a + M2_b + delta ** 2 * n_a * n_b / n_new
                    s[4], s[5], s[6] = n_new, mean_new, M2_new

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.0f}s  ({len(stats):,} entities, {chunks_read} chunks)")

        rows = []
        for eid, s in stats.items():
            n = s[4]
            std = float(np.sqrt(s[6] / n)) if n > 1 else 0.0
            rows.append({
                id_col:      eid,
                "velocity":  (s[3] - s[1]) / max(n, 1),   # (last_bal - first_bal) / n_obs
                "range":     max(0.0, s[3] - s[1]),        # max_bal - first_bal
                "std":       std,
            })
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"    [WARN] Skipping temporal features from {os.path.basename(path)}: {e}")
        return None


def load_wallet_temporal():
    """
    Per-wallet balance velocity / range / std from Wallet_Balance_Block.csv.
    Uses chunked streaming — peak RAM ~200-300 MB regardless of file size.
    """
    path = os.path.join(DATA_DIR, "Wallet_Balance_Block.csv")
    if not os.path.exists(path):
        return None
    agg = _chunked_balance_agg(path, "Wallet", np.int32)
    if agg is None:
        return None
    agg.rename(columns={
        "velocity": "wallet_balance_velocity",
        "range":    "wallet_balance_range",
        "std":      "wallet_balance_std",
    }, inplace=True)
    return agg[["Wallet", "wallet_balance_velocity", "wallet_balance_range", "wallet_balance_std"]]


def load_address_temporal():
    """
    Per-address balance velocity / std from Address_Balance_Block.csv.
    Uses chunked streaming — peak RAM ~200-300 MB regardless of file size.
    """
    path = os.path.join(DATA_DIR, "Address_Balance_Block.csv")
    if not os.path.exists(path):
        return None
    agg = _chunked_balance_agg(path, "ADR", str)
    if agg is None:
        return None
    agg.rename(columns={
        "velocity": "addr_balance_velocity",
        "std":      "addr_balance_std",
    }, inplace=True)
    return agg[["ADR", "addr_balance_velocity", "addr_balance_std"]]


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(td, tx, ai, wi, bh=None, wallet_temporal=None, addr_temporal=None):

    # ── [1] Output-side stats ──────────────────────────────────────────────────
    print("  [1/10] Output-side aggregations...")
    ov_stats = (
        td.groupby(KEYS, sort=False)["Output_Value"]
        .agg(mean_output="mean", max_output="max", min_output="min")
        .reset_index()
    )
    total_output = (
        td.groupby(KEYS, sort=False)["Output_Value"]
        .sum(min_count=1).rename("total_output").reset_index()
    )
    std_output = (
        td.groupby(KEYS, sort=False)["Output_Value"]
        .std(ddof=0).fillna(0).rename("std_output").reset_index()
    )

    # ── [2] Input-side stats ───────────────────────────────────────────────────
    print("  [2/10] Input-side aggregations...")
    total_input = (
        td.groupby(KEYS, sort=False)["Input_Value"]
        .sum(min_count=1).rename("total_input").reset_index()
    )

    # ── [3] Address + wallet counts ───────────────────────────────────────────
    print("  [3/10] Address/wallet nunique counts...")
    num_inputs = (
        td.groupby(KEYS, sort=False)["Input_Address"]
        .nunique().rename("num_inputs").reset_index()
    )
    num_outputs = (
        td.groupby(KEYS, sort=False)["Output_Address"]
        .nunique().rename("num_outputs").reset_index()
    )
    n_in_wallets = (
        td.groupby(KEYS, sort=False)["Input_Wallet"]
        .nunique().rename("n_distinct_input_wallets").reset_index()
    )
    n_out_wallets = (
        td.groupby(KEYS, sort=False)["Output_Wallet"]
        .nunique().rename("n_distinct_output_wallets").reset_index()
    )

    # ── [4] Value entropy ─────────────────────────────────────────────────────
    print("  [4/10] Value entropy...")
    td_ov = td[KEYS + ["Output_Value"]].copy()
    total_per_tx = td_ov.groupby(KEYS, sort=False)["Output_Value"].transform("sum")
    p = (td_ov["Output_Value"] / total_per_tx.clip(lower=1e-300)).clip(lower=1e-300)
    td_ov["entropy_term"] = -(p * np.log2(p))
    value_entropy = (
        td_ov.groupby(KEYS, sort=False)["entropy_term"]
        .sum().rename("value_entropy").reset_index()
    )

    # ── [5] Fresh output ratio ────────────────────────────────────────────────
    print("  [5/10] Fresh output ratio...")
    adr_col = next(
        (c for c in ("ADR", "Address", "address", "addr") if c in ai.columns), None
    )
    known = set(ai[adr_col].dropna().values) if adr_col else set(ai.iloc[:, 0].dropna().values)
    td_fr = td[KEYS + ["Output_Address"]].copy()
    valid = td_fr["Output_Address"].notna()
    td_fr["is_fresh"] = np.nan
    td_fr.loc[valid, "is_fresh"] = (~td_fr.loc[valid, "Output_Address"].isin(known)).astype(float)
    fresh_ratio = (
        td_fr.groupby(KEYS, sort=False)["is_fresh"]
        .mean().rename("fresh_output_ratio").reset_index()
    )

    # ── [6] Output-address static features (Address_Info) ────────────────────
    print("  [6/10] Output-address static features (Address_Info)...")
    ai_sub = ai.copy()
    ai_sub.columns = ai_sub.columns.str.strip()
    adr_key = next((c for c in ("ADR", "Address", "address") if c in ai_sub.columns), ai_sub.columns[0])

    if "First_Active_Block" in ai_sub.columns and "Last_Active_Block" in ai_sub.columns:
        ai_sub["_addr_age"] = (ai_sub["Last_Active_Block"] - ai_sub["First_Active_Block"]).clip(lower=0)
    else:
        ai_sub["_addr_age"] = np.nan

    if "Total_Sent" in ai_sub.columns and "Total_Received" in ai_sub.columns:
        ai_sub["_addr_sent_ratio"] = (
            ai_sub["Total_Sent"] / ai_sub["Total_Received"].clip(lower=1)
        ).clip(upper=1.0)
    else:
        ai_sub["_addr_sent_ratio"] = np.nan

    if "Final_Balance" in ai_sub.columns:
        ai_sub["_addr_zero_bal"] = (ai_sub["Final_Balance"] == 0).astype(float)
    else:
        ai_sub["_addr_zero_bal"] = np.nan

    if addr_temporal is not None:
        ai_sub = ai_sub.merge(addr_temporal, left_on=adr_key, right_on="ADR", how="left")

    td_adr = td[KEYS + ["Output_Address"]].copy()
    td_adr = td_adr.merge(
        ai_sub.rename(columns={adr_key: "Output_Address"}),
        on="Output_Address", how="left",
    )

    adr_agg = {}
    for new_col, src_col in [
        ("mean_out_addr_tx_count",    "N_Transactions"),
        ("mean_out_addr_balance",     "Final_Balance"),
        ("frac_out_addr_zero_balance","_addr_zero_bal"),
        ("mean_out_addr_age",         "_addr_age"),
        ("mean_out_addr_sent_ratio",  "_addr_sent_ratio"),
        ("mean_out_addr_bal_velocity","addr_balance_velocity"),
    ]:
        if src_col in td_adr.columns:
            adr_agg[new_col] = (src_col, "mean")

    addr_feats = td_adr.groupby(KEYS, sort=False).agg(**adr_agg).reset_index() if adr_agg else None

    # ── [7] Transaction metadata ──────────────────────────────────────────────
    print("  [7/10] Transaction metadata (fee, timing, lock_time)...")
    tx_meta = tx[KEYS + [c for c in ["DT", "TM", "Lock_Time", "Fees"] if c in tx.columns]].copy()

    tx_meta["tx_fee"] = pd.to_numeric(tx_meta.get("Fees", pd.Series(dtype=float)), errors="coerce")
    tx_meta["lock_time_flag"] = (
        pd.to_numeric(tx_meta.get("Lock_Time", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0
    ).astype(int)

    if "DT" in tx_meta.columns and "TM" in tx_meta.columns:
        try:
            # Date format auto-detected: the reference dataset uses
            # DD/MM/YYYY, the multi-period CUBiD dataset uses YYYY/MM/DD.
            # Both are unambiguous (the year is always 4 digits), so we let
            # pandas infer the format rather than forcing dayfirst=True
            # (which silently mis-parses the YYYY/MM/DD form).
            dt_parsed = pd.to_datetime(
                tx_meta["DT"].astype(str) + " " + tx_meta["TM"].astype(str),
                errors="coerce", format="mixed",
            )
            tx_meta["tx_datetime"]    = dt_parsed
            tx_meta["tx_hour"]        = dt_parsed.dt.hour.astype("Int16")
            tx_meta["tx_day_of_week"] = dt_parsed.dt.dayofweek.astype("Int16")
            tx_meta["is_night"]       = ((dt_parsed.dt.hour >= 0) & (dt_parsed.dt.hour < 6)).astype(int)
        except Exception:
            tx_meta["tx_hour"] = tx_meta["tx_day_of_week"] = tx_meta["is_night"] = np.nan

    if bh is not None:
        tx_meta = tx_meta.merge(bh, on="Block_No", how="left")
        tx_meta["tx_fee_normalized"] = tx_meta["tx_fee"] / tx_meta["block_avg_fee"].clip(lower=1e-9)
    else:
        tx_meta["tx_fee_normalized"] = np.nan

    tx_keep = KEYS + [c for c in [
        "tx_datetime",
        "tx_fee", "tx_fee_normalized", "lock_time_flag",
        "tx_hour", "tx_day_of_week", "is_night",
    ] if c in tx_meta.columns]
    tx_meta = tx_meta[tx_keep]

    # ── [8] Merge all aggregations ────────────────────────────────────────────
    print("  [8/10] Merging aggregations...")
    feat = (
        total_output
        .merge(ov_stats,      on=KEYS)
        .merge(std_output,    on=KEYS)
        .merge(total_input,   on=KEYS)
        .merge(num_inputs,    on=KEYS)
        .merge(num_outputs,   on=KEYS)
        .merge(n_in_wallets,  on=KEYS)
        .merge(n_out_wallets, on=KEYS)
        .merge(value_entropy, on=KEYS)
        .merge(fresh_ratio,   on=KEYS)
        .merge(tx_meta,       on=KEYS, how="left")
    )
    if addr_feats is not None:
        feat = feat.merge(addr_feats, on=KEYS, how="left")

    # ── [9] Derived features ──────────────────────────────────────────────────
    print("  [9/10] Derived FATF rule signals...")
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

    # ── [10] Wallet features (extended) ───────────────────────────────────────
    print("  [10/10] Wallet features (extended)...")
    wi_rename = {
        "N_Transactions": "wallet_total_tx",
        "Total_Sent":     "wallet_total_sent",
        "N_Addresses":    "wallet_size",
    }
    wi_extra = {c: f"_wi_{c}" for c in [
        "First_Active_Block", "Last_Active_Block",
        "N_Sent_Transactions", "Final_Balance", "Total_Received",
    ] if c in wi.columns}

    wi_sub = wi[["Wallet"] + list(wi_rename) + list(wi_extra)].copy()
    wi_sub.rename(columns={**wi_rename, **wi_extra}, inplace=True)

    if "_wi_First_Active_Block" in wi_sub and "_wi_Last_Active_Block" in wi_sub:
        wi_sub["wallet_age"] = (
            wi_sub["_wi_Last_Active_Block"] - wi_sub["_wi_First_Active_Block"]
        ).clip(lower=0)

    if "wallet_total_tx" in wi_sub and "_wi_N_Sent_Transactions" in wi_sub:
        wi_sub["wallet_sent_ratio"] = (
            wi_sub["_wi_N_Sent_Transactions"] / wi_sub["wallet_total_tx"].clip(lower=1)
        ).clip(upper=1.0)

    if "_wi_Final_Balance" in wi_sub:
        wi_sub["wallet_final_balance"] = wi_sub["_wi_Final_Balance"]

    if "_wi_Final_Balance" in wi_sub and "_wi_Total_Received" in wi_sub:
        wi_sub["wallet_balance_ratio"] = (
            wi_sub["_wi_Final_Balance"] / wi_sub["_wi_Total_Received"].clip(lower=1)
        ).clip(upper=1.0, lower=0.0)

    wi_sub.drop(columns=[c for c in wi_sub.columns if c.startswith("_wi_")], inplace=True)

    if wallet_temporal is not None:
        wi_sub = wi_sub.merge(wallet_temporal, on="Wallet", how="left")

    wallet_map = tx[KEYS + ["Wallet"]].merge(wi_sub, on="Wallet", how="left")
    wallet_join_cols = [c for c in wi_sub.columns if c != "Wallet"]
    feat = feat.merge(wallet_map[KEYS + wallet_join_cols], on=KEYS, how="left")

    # CoinJoin suppression: zero out all wallet features when mixing is likely
    cj_mask     = feat["coinjoin_strength"] > 0.1
    suppress    = [c for c in WALLET_FEATURE_COLS if c in feat.columns]
    feat.loc[cj_mask, suppress] = np.nan

    return feat


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Bitcoin Illicit Detection — Feature Engineering")
    print("=" * 55 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading core data...")
    td, tx, ai, wi = load_data()
    n_tx = td.groupby(KEYS).ngroups
    print(f"  {n_tx:,} transactions, {len(td):,} detail rows\n")

    print("Loading auxiliary data...")
    bh              = load_block_header()
    wallet_temporal = load_wallet_temporal()
    addr_temporal   = load_address_temporal()

    t0   = time.time()
    feat = build_features(td, tx, ai, wi, bh, wallet_temporal, addr_temporal)
    elapsed = time.time() - t0

    feat = feat[KEYS + [c for c in feat.columns if c not in KEYS]]
    feat.to_csv(OUT_CSV, index=False)

    print(f"\n[OK] Features saved → {OUT_CSV}")
    print(f"     Shape  : {feat.shape[0]:,} rows × {feat.shape[1]} cols")
    print(f"     Runtime: {elapsed:.1f} s")

    cj_n = (feat["coinjoin_strength"] > 0.1).sum()
    print(f"\n  CoinJoin suppression : {cj_n:,} tx ({cj_n/len(feat):.1%})")
    print(f"  Peeling chains       : {int(feat['peeling_flag'].sum()):,}")
    print(f"  Equal-value outputs  : {int(feat['equal_value_flag'].sum()):,}")
    print("\nFeature summary:")
    print(feat.select_dtypes(include=[np.number]).drop(columns=KEYS, errors="ignore")
          .describe().to_string())
