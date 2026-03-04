"""
pages/3_Wallet.py — Wallet-level risk analysis.

Shows aggregate risk verdict, transaction timeline, and per-transaction
risk scores for every transaction belonging to a wallet.
"""

import os
import sys

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_utils import (
    load_labeled, load_raw, load_model_bundle,
    RULE_META, LABEL_MAP, LABEL_COLORS,
)

st.set_page_config(
    page_title="Wallet · Bitcoin Tracker",
    page_icon="👛",
    layout="wide",
)

st.title("👛  Wallet Analysis")

# ── Wallet input ───────────────────────────────────────────────────────────────
default_wallet = st.session_state.get("sel_wallet", "")
wallet_input = st.text_input(
    "Wallet ID  (numeric)",
    value=str(default_wallet) if default_wallet else "",
    placeholder="e.g. 678952961",
    help="Wallet IDs are numeric integers (e.g. 678952961). Find one via the Explorer page.",
)

if not wallet_input:
    st.info("Enter a numeric wallet ID to analyse all its transactions.")
    st.stop()

# Wallet IDs are integers in both Transactions (float64) and Wallet_Info (int64)
try:
    wallet_id = int(float(wallet_input.strip()))
except ValueError:
    st.error("Wallet ID must be a number (e.g. 97273369).")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
df     = load_labeled()
bundle = load_model_bundle()

with st.spinner("Loading raw data…"):
    td, tx, ai, wi = load_raw()

# Transactions.Wallet is float64, Wallet_Info.Wallet is int64 — compare as float
wallet_txs = tx[tx["Wallet"] == float(wallet_id)][["Block_No", "Trans_No"]]
if wallet_txs.empty:
    st.error(f"Wallet `{wallet_id}` not found in the dataset.")
    st.stop()

# ── Wallet metadata ────────────────────────────────────────────────────────────
wi_row = wi[wi["Wallet"] == wallet_id]
if not wi_row.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wallet ID",           wallet_id)
    c2.metric("Total Transactions",  f"{int(wi_row['N_Transactions'].values[0]):,}")
    c3.metric("Total Sent (BTC)",    f"{float(wi_row['Total_Sent'].values[0]):.4f}")
    c4.metric("Unique Addresses",    f"{int(wi_row['N_Addresses'].values[0]):,}")
else:
    st.metric("Wallet ID", wallet_id)

st.divider()

# ── Pre-computed predictions ───────────────────────────────────────────────────
rule_cols = [c for c in (df.columns if df is not None else []) if c.startswith("rule_")]

prob_rows = []
if df is not None and bundle is not None:
    model    = bundle["model"]
    fcols    = bundle["features"]

    joined = wallet_txs.merge(df, on=["Block_No", "Trans_No"], how="inner")
    for _, row in joined.iterrows():
        feat_df = pd.DataFrame([{c: row.get(c, np.nan) for c in fcols}])
        X       = feat_df.fillna(feat_df.median())
        p       = float(model.predict_proba(X)[0][1])
        entry   = {
            "Block_No":    int(row["Block_No"]),
            "Trans_No":    str(row["Trans_No"]),
            "illicit_prob": p,
            "verdict":     "⛔ ILLICIT" if p >= 0.5 else "✅ LICIT",
            "suspicion_score": int(row.get("suspicion_score", 0)),
        }
        for rc in rule_cols:
            entry[rc] = int(row.get(rc, 0))
        prob_rows.append(entry)

prob_df = pd.DataFrame(prob_rows)

# ── Aggregate verdict ──────────────────────────────────────────────────────────
if not prob_df.empty:
    avg_prob = float(prob_df["illicit_prob"].mean())
    flagged  = int((prob_df["illicit_prob"] >= 0.5).sum())
    n_tx     = len(prob_df)

    if avg_prob >= 0.5:
        verdict_label = "⛔  HIGH RISK"
        verdict_color = LABEL_COLORS[1]
    elif avg_prob >= 0.25:
        verdict_label = "⚠️  MEDIUM RISK"
        verdict_color = LABEL_COLORS[-1]
    else:
        verdict_label = "✅  LOW RISK"
        verdict_color = LABEL_COLORS[0]

    vc1, vc2, vc3, vc4 = st.columns(4)
    vc1.metric("Overall Verdict",          verdict_label)
    vc2.metric("Avg Illicit Probability",  f"{avg_prob:.1%}")
    vc3.metric("Flagged Transactions",     f"{flagged} / {n_tx}")
    vc4.metric("Transactions Analysed",    n_tx)

    # ── Risk timeline ──────────────────────────────────────────────────────────
    st.subheader("Transaction Risk Timeline")
    st.caption("Each point is one transaction. Hover for details.")

    fig = px.scatter(
        prob_df,
        x="Block_No", y="illicit_prob",
        color="illicit_prob",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 1],
        hover_data={"Trans_No": True, "verdict": True, "suspicion_score": True,
                    "illicit_prob": ":.2%"},
        labels={"Block_No": "Block Number", "illicit_prob": "Illicit Probability"},
    )
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="red",
        annotation_text="Decision boundary (0.5)",
        annotation_position="top right",
    )
    fig.update_layout(
        height=380, margin=dict(t=20, b=0),
        coloraxis_colorbar=dict(title="Prob"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Transaction table ──────────────────────────────────────────────────────
    st.subheader("All Wallet Transactions")

    display_cols = ["Block_No", "Trans_No", "verdict", "illicit_prob", "suspicion_score"] + rule_cols
    display_cols = [c for c in display_cols if c in prob_df.columns]
    display_prob_df = prob_df[display_cols].copy()

    sel_event = st.dataframe(
        display_prob_df.reset_index(drop=True),
        use_container_width=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "illicit_prob": st.column_config.NumberColumn("Illicit prob", format="%.2%"),
            "Trans_No":     st.column_config.TextColumn("Trans No", width="medium"),
        },
    )

    selected_rows = sel_event.selection.rows if hasattr(sel_event, "selection") else []
    if selected_rows:
        sel_row = display_prob_df.reset_index(drop=True).iloc[selected_rows[0]]
        sel_block = int(prob_df.reset_index(drop=True).iloc[selected_rows[0]]["Block_No"])
        sel_trans = str(prob_df.reset_index(drop=True).iloc[selected_rows[0]]["Trans_No"])
        st.info(
            f"Selected: Block `{sel_block}` / `{sel_trans[:20]}…` — "
            f"{sel_row['verdict']}"
        )
        if st.button("View Transaction Details →", type="primary"):
            st.session_state["sel_block"] = sel_block
            st.session_state["sel_trans"] = sel_trans
            st.switch_page("pages/2_Transaction.py")

else:
    st.warning(
        "No pre-computed predictions available for this wallet. "
        "Ensure the pipeline has been run and labeled_features.csv exists."
    )
    st.dataframe(wallet_txs, use_container_width=True)
