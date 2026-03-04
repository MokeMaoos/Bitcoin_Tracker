"""
ui_utils.py — shared data loading, caching, and helper functions for the Streamlit UI.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Ensure project root is on path so utils.py can be imported
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUTPUT_DIR   = os.path.join(_ROOT, "outputs")
DATA_DIR     = os.path.join(_ROOT, "data")
MODEL_PATH   = os.path.join(OUTPUT_DIR, "model.pkl")
LABELED_CSV  = os.path.join(OUTPUT_DIR, "labeled_features.csv")
CLUSTER_CSV  = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")

# ── Rule metadata ──────────────────────────────────────────────────────────────
RULE_META = {
    "rule_equal_value": {
        "label":     "Equal-Value Outputs",
        "fatf":      "CoinJoin / Structuring",
        "threshold": "σ/μ < 1%",
        "icon":      "⚖️",
    },
    "rule_peeling": {
        "label":     "Peeling Chain",
        "fatf":      "Layering",
        "threshold": "Largest output ≥ 80% of total",
        "icon":      "🔗",
    },
    "rule_high_fan_out": {
        "label":     "High Fan-Out",
        "fatf":      "Fund Dispersion",
        "threshold": "≥ 10 unique output addresses",
        "icon":      "🌐",
    },
    "rule_fresh_outputs": {
        "label":     "Fresh Output Addresses",
        "fatf":      "Address Churn / Layering",
        "threshold": "≥ 80% unseen output addresses",
        "icon":      "🆕",
    },
    "rule_mixing": {
        "label":     "Mixing Service Signal",
        "fatf":      "Mixer Involvement",
        "threshold": "CoinJoin strength ≥ 0.5",
        "icon":      "🔀",
    },
    "rule_high_entropy": {
        "label":     "High Value Entropy",
        "fatf":      "Anomalous Value Distribution",
        "threshold": "Shannon entropy ≥ 3.0 bits",
        "icon":      "📊",
    },
}

LABEL_MAP    = {1: "Illicit", 0: "Licit", -1: "Uncertain"}
LABEL_COLORS = {1: "#ef4444", 0: "#22c55e", -1: "#f59e0b"}

# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading labeled features…")
def load_labeled():
    if not os.path.exists(LABELED_CSV):
        return None
    return pd.read_csv(LABELED_CSV)


@st.cache_data(show_spinner="Loading cluster assignments…")
def load_clusters():
    if not os.path.exists(CLUSTER_CSV):
        return None
    return pd.read_csv(CLUSTER_CSV)


@st.cache_data(show_spinner="Loading model…")
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading raw transaction data (this may take a moment)…")
def load_raw():
    td = pd.read_csv(os.path.join(DATA_DIR, "Trans_Detail.csv"))
    tx = pd.read_csv(os.path.join(DATA_DIR, "Transactions.csv"))
    ai = pd.read_csv(os.path.join(DATA_DIR, "Address_Info.csv"))
    wi = pd.read_csv(os.path.join(DATA_DIR, "Wallet_Info.csv"))
    return td, tx, ai, wi


# ── Rule derivation ────────────────────────────────────────────────────────────

def derive_rules(feat_row: dict) -> dict:
    """
    Re-compute the 6 rule flags from a feature dict.
    Works for both pre-computed rows and live-computed feature dicts.
    """
    return {
        "rule_equal_value":   int(feat_row.get("equal_value_flag", 0)),
        "rule_peeling":       int(feat_row.get("peeling_flag", 0)),
        "rule_high_fan_out":  int(feat_row.get("num_outputs", 0) >= 10),
        "rule_fresh_outputs": int(feat_row.get("fresh_output_ratio", 0) >= 0.8),
        "rule_mixing":        int(feat_row.get("coinjoin_strength", 0) >= 0.5),
        "rule_high_entropy":  int(feat_row.get("value_entropy", 0) >= 3.0),
    }


# ── Rule evidence ──────────────────────────────────────────────────────────────

def rule_evidence(feat_row: dict, grp, ai) -> dict:
    """
    Build per-rule evidence dicts containing the measured value string and
    underlying data that justifies or refutes each flag.

    grp : DataFrame — rows from Trans_Detail for this transaction (may be None/empty)
    ai  : DataFrame — Address_Info (needed for fresh-address check)
    """
    ov = np.array([])
    oa = np.array([])
    if grp is not None and not grp.empty:
        ov = grp["Output_Value"].dropna().values
        oa = grp["Output_Address"].dropna().values

    # Resolve known-address column
    addr_col = next(
        (c for c in ("ADR", "Address", "address", "addr", "Addr") if c in ai.columns),
        None,
    )
    known = set(ai[addr_col].dropna().values) if addr_col else set(ai.iloc[:, 0].dropna().values)

    ev = {}

    # rule_equal_value
    mean_v = abs(float(feat_row.get("mean_output", 1e-9))) + 1e-9
    std_v  = float(feat_row.get("std_output", 0))
    cv     = std_v / mean_v
    ev["rule_equal_value"] = {
        "value_str": f"σ/μ = {cv:.2%}",
        "sections": {
            "Output values (BTC)": ov.tolist(),
        },
    }

    # rule_peeling
    total = float(feat_row.get("total_output", 1e-9)) or 1e-9
    max_o = float(feat_row.get("max_output", 0))
    pct   = max_o / total
    pairs = sorted(zip(oa.tolist(), ov.tolist()), key=lambda x: -x[1]) if len(ov) else []
    ev["rule_peeling"] = {
        "value_str": f"Largest output = {pct:.1%} of total ({max_o:.6f} / {total:.6f} BTC)",
        "sections": {
            "Outputs by value (Address → BTC)": pairs,
        },
    }

    # rule_high_fan_out
    n_out = int(feat_row.get("num_outputs", 0))
    unique_oa = list(dict.fromkeys(oa.tolist())) if len(oa) else []
    ev["rule_high_fan_out"] = {
        "value_str": f"{n_out} unique output address{'es' if n_out != 1 else ''}",
        "sections": {
            "Output addresses": unique_oa,
        },
    }

    # rule_fresh_outputs
    fresh_pct   = float(feat_row.get("fresh_output_ratio", 0))
    fresh_addrs = [a for a in oa if a not in known]
    known_addrs = [a for a in oa if a in known]
    ev["rule_fresh_outputs"] = {
        "value_str": f"{fresh_pct:.1%} of output addresses are unseen "
                     f"({len(fresh_addrs)}/{len(oa)} addresses)",
        "sections": {
            "Fresh (unseen) addresses": fresh_addrs,
            "Known addresses":          known_addrs,
        },
    }

    # rule_mixing
    cj      = float(feat_row.get("coinjoin_strength", 0))
    eq_flag = int(feat_row.get("equal_value_flag", 0))
    n_in    = int(feat_row.get("num_inputs", 0))
    fan_in  = round(min(n_in / 10.0, 1.0), 3)
    fan_out = round(min(n_out / 10.0, 1.0), 3)
    ev["rule_mixing"] = {
        "value_str": f"CoinJoin strength = {cj:.4f}",
        "sections": {
            "Formula breakdown": [
                f"0.6 × equal_value_flag ({eq_flag})  =  {0.6 * eq_flag:.3f}",
                f"0.2 × min(num_inputs/10, 1)  ({n_in} inputs → {fan_in})  =  {0.2 * fan_in:.3f}",
                f"0.2 × min(num_outputs/10, 1)  ({n_out} outputs → {fan_out})  =  {0.2 * fan_out:.3f}",
                f"Total = {cj:.4f}  (threshold: ≥ 0.5)",
            ],
        },
    }

    # rule_high_entropy
    ent = float(feat_row.get("value_entropy", 0))
    ev["rule_high_entropy"] = {
        "value_str": f"Value entropy = {ent:.4f} bits",
        "sections": {
            "Output values (BTC)": ov.tolist(),
        },
    }

    return ev


# ── Live prediction ────────────────────────────────────────────────────────────

def run_live_prediction(block_no, trans_no, td, tx, ai, wi, bundle):
    """
    Compute transaction features on-the-fly and return (feat_dict, prob).
    Returns (None, None) if the transaction is not found in td.
    """
    from utils import (
        is_equal_value, compute_value_entropy, is_peeling,
        compute_fresh_output_ratio, compute_coinjoin_strength,
    )

    grp = td[(td["Block_No"] == block_no) & (td["Trans_No"] == trans_no)]
    if grp.empty:
        return None, None

    ov          = grp["Output_Value"].dropna().values
    oa          = grp["Output_Address"].dropna().values
    num_inputs  = grp["Input_Address"].nunique()
    num_outputs = grp["Output_Address"].nunique()
    cj_str      = compute_coinjoin_strength(ov, num_inputs, num_outputs)

    tx_row = tx[(tx["Block_No"] == block_no) & (tx["Trans_No"] == trans_no)]
    wtx = wsent = wsize = np.nan
    if not tx_row.empty and cj_str <= 0.1:
        wid = tx_row["Wallet"].values[0]
        w   = wi[wi["Wallet"] == wid]
        if not w.empty:
            wtx   = w["N_Transactions"].values[0]
            wsent = w["Total_Sent"].values[0]
            wsize = w["N_Addresses"].values[0]

    feat = {
        "num_inputs":          num_inputs,
        "num_outputs":         num_outputs,
        "total_input":         float(grp["Input_Value"].sum()),
        "total_output":        float(ov.sum()),
        "mean_output":         float(ov.mean()) if len(ov) else 0.0,
        "std_output":          float(ov.std())  if len(ov) > 1 else 0.0,
        "max_output":          float(ov.max())  if len(ov) else 0.0,
        "min_output":          float(ov.min())  if len(ov) else 0.0,
        "io_ratio":            num_inputs / max(num_outputs, 1),
        "value_ratio":         float(grp["Input_Value"].sum()) / max(float(ov.sum()), 1e-9),
        "equal_value_flag":    is_equal_value(ov),
        "value_entropy":       compute_value_entropy(ov),
        "peeling_flag":        is_peeling(ov),
        "fresh_output_ratio":  compute_fresh_output_ratio(oa, ai),
        "coinjoin_strength":   cj_str,
        "wallet_total_tx":     wtx,
        "wallet_total_sent":   wsent,
        "wallet_size":         wsize,
    }

    model  = bundle["model"]
    fcols  = bundle["features"]
    feat_df = pd.DataFrame([feat])
    X      = feat_df[fcols].fillna(feat_df[fcols].median())
    prob   = float(model.predict_proba(X)[0][1])
    return feat, prob
