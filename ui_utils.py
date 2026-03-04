"""
ui_utils.py — shared data loading, caching, and helper functions for the Streamlit UI.

External integrations:
  - Mempool.space API  (no key needed)   → enrich tx with fee/size/timing
  - Bitcoin Abuse DB   (free key)        → flag addresses with abuse reports
  - Claude API         (Anthropic key)   → generate natural-language risk analysis
  - Elliptic dataset   (Kaggle download) → ground-truth benchmark evaluation
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
    Compute all 38 transaction features on-the-fly and return (feat_dict, prob).
    Features that require large pre-aggregated files (balance velocity) are
    left as NaN and imputed by the model's median imputer.
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

    # ── Wallet features ────────────────────────────────────────────────────────
    tx_row = tx[(tx["Block_No"] == block_no) & (tx["Trans_No"] == trans_no)]
    wtx = wsent = wsize = wage = wsent_ratio = wfbal = wbal_ratio = np.nan
    if not tx_row.empty and cj_str <= 0.1:
        wid = tx_row["Wallet"].values[0]
        w   = wi[wi["Wallet"] == wid]
        if not w.empty:
            wtx   = w["N_Transactions"].values[0]  if "N_Transactions"  in w.columns else np.nan
            wsent = w["Total_Sent"].values[0]       if "Total_Sent"      in w.columns else np.nan
            wsize = w["N_Addresses"].values[0]      if "N_Addresses"     in w.columns else np.nan
            if "First_Active_Block" in w.columns and "Last_Active_Block" in w.columns:
                wage = max(0, int(w["Last_Active_Block"].values[0]) - int(w["First_Active_Block"].values[0]))
            if "N_Sent_Transactions" in w.columns and not np.isnan(wtx) and float(wtx) > 0:
                wsent_ratio = float(w["N_Sent_Transactions"].values[0]) / float(wtx)
            if "Final_Balance" in w.columns:
                wfbal = float(w["Final_Balance"].values[0])
            if "Final_Balance" in w.columns and "Total_Received" in w.columns:
                tr = float(w["Total_Received"].values[0]) or 1.0
                wbal_ratio = min(1.0, max(0.0, float(w["Final_Balance"].values[0]) / tr))

    # ── Transaction metadata ───────────────────────────────────────────────────
    tx_fee = lock_time_flag = tx_hour = tx_day = is_night = np.nan
    if not tx_row.empty:
        if "Fees" in tx_row.columns:
            tx_fee = float(pd.to_numeric(tx_row["Fees"].values[0], errors="coerce") or np.nan)
        lock_raw       = tx_row["Lock_Time"].values[0] if "Lock_Time" in tx_row.columns else 0
        lock_time_flag = int(float(lock_raw or 0) > 0)
        if "DT" in tx_row.columns and "TM" in tx_row.columns:
            try:
                dt_str = str(tx_row["DT"].values[0]) + " " + str(tx_row["TM"].values[0])
                dt_obj = pd.to_datetime(dt_str, dayfirst=True, errors="coerce")
                if not pd.isna(dt_obj):
                    tx_hour  = int(dt_obj.hour)
                    tx_day   = int(dt_obj.dayofweek)
                    is_night = int(0 <= dt_obj.hour < 6)
            except Exception:
                pass

    # ── Output-address static features ────────────────────────────────────────
    adr_col = next((c for c in ("ADR", "Address", "address") if c in ai.columns), ai.columns[0])
    try:
        ai_idx = ai.set_index(adr_col)
    except Exception:
        ai_idx = ai.set_index(ai.columns[0])

    tx_cnts, balances, ages, sent_ratios, zero_flags = [], [], [], [], []
    for a in oa:
        if a in ai_idx.index:
            r = ai_idx.loc[a]
            tx_cnts.append(float(r.get("N_Transactions", np.nan)))
            balances.append(float(r.get("Final_Balance",  np.nan)))
            fa = r.get("First_Active_Block", np.nan)
            la = r.get("Last_Active_Block",  np.nan)
            ages.append(float(la - fa) if not (pd.isna(fa) or pd.isna(la)) else np.nan)
            ts = r.get("Total_Sent",     np.nan)
            tr = r.get("Total_Received", np.nan)
            sent_ratios.append(float(ts) / max(float(tr), 1.0)
                               if not (pd.isna(ts) or pd.isna(tr)) else np.nan)
            fb = float(r.get("Final_Balance", np.nan))
            zero_flags.append(0.0 if pd.isna(fb) else float(fb == 0))

    def _safe_mean(lst):
        vals = [v for v in lst if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else np.nan

    # ── Graph topology ─────────────────────────────────────────────────────────
    n_in_wallets  = grp["Input_Wallet"].nunique()  if "Input_Wallet"  in grp.columns else np.nan
    n_out_wallets = grp["Output_Wallet"].nunique() if "Output_Wallet" in grp.columns else np.nan

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
        "n_distinct_input_wallets":  n_in_wallets,
        "n_distinct_output_wallets": n_out_wallets,
        "wallet_total_tx":       wtx,
        "wallet_total_sent":     wsent,
        "wallet_size":           wsize,
        "wallet_age":            wage,
        "wallet_sent_ratio":     wsent_ratio,
        "wallet_final_balance":  wfbal,
        "wallet_balance_ratio":  wbal_ratio,
        "wallet_balance_velocity": np.nan,  # requires Wallet_Balance_Block; model imputes
        "wallet_balance_range":    np.nan,
        "wallet_balance_std":      np.nan,
        "tx_fee":            tx_fee,
        "tx_fee_normalized": np.nan,        # requires Block_Header lookup; model imputes
        "lock_time_flag":    lock_time_flag,
        "tx_hour":           tx_hour,
        "tx_day_of_week":    tx_day,
        "is_night":          is_night,
        "mean_out_addr_tx_count":     _safe_mean(tx_cnts),
        "mean_out_addr_balance":      _safe_mean(balances),
        "frac_out_addr_zero_balance": _safe_mean(zero_flags),
        "mean_out_addr_age":          _safe_mean(ages),
        "mean_out_addr_sent_ratio":   _safe_mean(sent_ratios),
        "mean_out_addr_bal_velocity": np.nan,  # requires Address_Balance_Block; model imputes
    }

    model   = bundle["model"]
    fcols   = bundle["features"]
    feat_df = pd.DataFrame([feat])
    X       = feat_df.reindex(columns=fcols).fillna(feat_df.reindex(columns=fcols).median())
    prob    = float(model.predict_proba(X)[0][1])
    return feat, prob


# ── External API helpers ───────────────────────────────────────────────────────

def _get_secret(key: str) -> str:
    """Read a key from Streamlit secrets or environment variables."""
    try:
        import streamlit as _st
        val = _st.secrets.get(key, "")
    except Exception:
        val = ""
    return val or os.environ.get(key, "")


# ── Mempool.space (no key required) ───────────────────────────────────────────

def fetch_mempool_tx(trans_no: str) -> dict | None:
    """
    Fetch live transaction data from Mempool.space.
    Returns a dict with fee, size, vsize, weight, fee_rate, status fields.
    Returns None on failure (network error / tx not found).
    """
    import requests
    try:
        r = requests.get(
            f"https://mempool.space/api/tx/{trans_no}",
            timeout=6,
        )
        if r.status_code == 200:
            data = r.json()
            fee  = data.get("fee", 0)
            vsz  = data.get("vsize") or data.get("size", 1)
            return {
                "fee_sats":        fee,
                "fee_btc":         fee / 1e8,
                "fee_rate_sat_vb": round(fee / max(vsz, 1), 2),
                "size_bytes":      data.get("size"),
                "vsize_vbytes":    data.get("vsize"),
                "weight":          data.get("weight"),
                "version":         data.get("version"),
                "locktime":        data.get("locktime"),
                "vin_count":       len(data.get("vin", [])),
                "vout_count":      len(data.get("vout", [])),
                "confirmed":       data.get("status", {}).get("confirmed", False),
                "block_height":    data.get("status", {}).get("block_height"),
                "block_time":      data.get("status", {}).get("block_time"),
            }
    except Exception:
        pass
    return None


def fetch_mempool_address(address: str) -> dict | None:
    """Fetch address summary from Mempool.space (tx count, balance)."""
    import requests
    try:
        r = requests.get(
            f"https://mempool.space/api/address/{address}",
            timeout=6,
        )
        if r.status_code == 200:
            data = r.json()
            chain = data.get("chain_stats", {})
            return {
                "tx_count":       chain.get("tx_count", 0),
                "funded_sum":     chain.get("funded_txo_sum", 0) / 1e8,
                "spent_sum":      chain.get("spent_txo_sum", 0) / 1e8,
                "funded_count":   chain.get("funded_txo_count", 0),
                "spent_count":    chain.get("spent_txo_count", 0),
            }
    except Exception:
        pass
    return None


# ── Bitcoin Abuse DB ───────────────────────────────────────────────────────────

def check_bitcoin_abuse(address: str, api_key: str = "") -> dict | None:
    """
    Check a Bitcoin address against the Bitcoin Abuse database.
    Returns dict with 'count' and 'recent' report list, or None on failure.
    Register for a free API key at: https://www.bitcoinabuse.com/api-docs
    """
    import requests
    key = api_key or _get_secret("BITCOIN_ABUSE_KEY")
    if not key:
        return None
    try:
        r = requests.get(
            "https://www.bitcoinabuse.com/api/reports/check",
            params={"address": address, "api_token": key},
            timeout=6,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def check_addresses_abuse(addresses: list[str], api_key: str = "") -> dict:
    """
    Check multiple addresses. Returns {address: abuse_result} dict.
    Skips blanks and NaN values. Results cached in session_state.
    """
    results = {}
    key = api_key or _get_secret("BITCOIN_ABUSE_KEY")
    if not key:
        return results
    for addr in addresses:
        if not addr or str(addr) == "nan":
            continue
        result = check_bitcoin_abuse(str(addr), key)
        if result is not None:
            results[str(addr)] = result
    return results


# ── Claude AI Analysis ─────────────────────────────────────────────────────────

def generate_claude_analysis(
    feat_row: dict,
    rules: dict,
    prob: float,
    block_no: int,
    trans_no: str,
    api_key: str = "",
) -> str:
    """
    Call Claude API to generate a natural-language forensic risk analysis.
    Returns the analysis text, or an error message string.

    Set ANTHROPIC_API_KEY in .streamlit/secrets.toml or as an environment variable.
    """
    import anthropic

    key = api_key or _get_secret("ANTHROPIC_API_KEY")
    if not key:
        return "❌ No Anthropic API key configured. Add ANTHROPIC_API_KEY to .streamlit/secrets.toml"

    fired     = [RULE_META[k]["label"] for k, v in rules.items() if v and k in RULE_META]
    not_fired = [RULE_META[k]["label"] for k, v in rules.items() if not v and k in RULE_META]
    score     = sum(1 for v in rules.values() if v)
    pred      = "ILLICIT" if prob >= 0.5 else "LICIT"

    prompt = f"""You are a blockchain forensics expert specialising in Bitcoin AML (Anti-Money Laundering) compliance, with deep knowledge of FATF guidance and common illicit transaction patterns (CoinJoin mixing, peeling chains, darknet market payments, ransomware, structuring, layering).

Analyse the following Bitcoin transaction and provide a professional risk assessment:

TRANSACTION
  Block:  {block_no}
  Hash:   {trans_no}
  Model verdict: {pred}  ({prob:.1%} illicit probability)
  Suspicion score: {score}/6 rules fired

FATF RED-FLAG RULES
  Fired ({len(fired)}):     {', '.join(fired) if fired else 'None'}
  Not fired ({len(not_fired)}): {', '.join(not_fired) if not_fired else 'None'}

FEATURE VALUES
  Inputs:             {int(feat_row.get('num_inputs', 0))} addresses  |  Total in:  {float(feat_row.get('total_input', 0) or 0):.6f} BTC
  Outputs:            {int(feat_row.get('num_outputs', 0))} addresses  |  Total out: {float(feat_row.get('total_output', 0) or 0):.6f} BTC
  I/O address ratio:  {float(feat_row.get('io_ratio', 0) or 0):.3f}
  Value entropy:      {float(feat_row.get('value_entropy', 0) or 0):.3f} bits  (≥3.0 = anomalous)
  CoinJoin strength:  {float(feat_row.get('coinjoin_strength', 0) or 0):.4f}  (≥0.5 = mixing signal)
  Equal-value flag:   {'Yes' if feat_row.get('equal_value_flag') else 'No'}
  Peeling chain flag: {'Yes' if feat_row.get('peeling_flag') else 'No'}
  Std of outputs:     {float(feat_row.get('std_output', 0) or 0):.6f} BTC
  Max output:         {float(feat_row.get('max_output', 0) or 0):.6f} BTC  ({float(feat_row.get('max_output', 0) or 0) / max(float(feat_row.get('total_output', 1e-9) or 1e-9), 1e-9):.1%} of total)

Provide your analysis in exactly these four sections (use these headings):

**PATTERN**
What transaction pattern does this most resemble? Name the specific pattern type (e.g. CoinJoin mix, peel chain, exchange consolidation, darknet settlement, ransomware payout, structuring). Cite the specific features that lead you to this conclusion.

**RISK SYNTHESIS**
Synthesise all signals. Which features are most diagnostic and why? Does the combination of signals strengthen or weaken the case for illicit activity? Are there innocent explanations for these patterns?

**RECOMMENDED ACTION**
What should a compliance analyst do next? Be specific (e.g. trace input addresses, check against OFAC SDN list, file SAR, escalate to senior analyst).

**CAVEATS**
What additional information would materially change this assessment? What are the limitations of rule-based detection for this type of pattern?
"""

    try:
        client  = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except anthropic.AuthenticationError:
        return "❌ Invalid Anthropic API key. Check ANTHROPIC_API_KEY in .streamlit/secrets.toml"
    except anthropic.RateLimitError:
        return "❌ Anthropic rate limit reached. Please wait a moment and try again."
    except Exception as e:
        return f"❌ Claude API error: {e}"
