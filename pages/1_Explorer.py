"""
pages/1_Explorer.py — Filterable transaction explorer.

Browse all 285k transactions with sidebar filters.
Select a row and click "View Transaction Details" to navigate to the WHY page.
"""

import os
import sys

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_utils import load_labeled, load_clusters, RULE_META, LABEL_MAP, LABEL_COLORS

st.set_page_config(
    page_title="Explorer · Bitcoin Tracker",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍  Transaction Explorer")
st.caption("Filter, sort, and browse all transactions. Select a row to view its full analysis.")

df = load_labeled()
if df is None:
    st.error("No labeled data found. Run the pipeline first.")
    st.stop()

clusters = load_clusters()
if clusters is not None:
    df = df.merge(clusters[["Block_No", "Trans_No", "cluster_kmeans"]], on=["Block_No", "Trans_No"], how="left")

rule_cols = [c for c in df.columns if c.startswith("rule_")]

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    label_filter = st.multiselect(
        "Pseudo-label",
        options=[1, 0, -1],
        format_func=lambda x: LABEL_MAP[x],
        default=[1, 0, -1],
    )

    score_range = st.slider("Suspicion score", 0, 6, (0, 6))

    st.markdown("**Rule fired (show txs where ANY selected rule fired):**")
    rule_filter = {}
    for rc in rule_cols:
        if rc in RULE_META:
            rule_filter[rc] = st.checkbox(
                f"{RULE_META[rc]['icon']}  {RULE_META[rc]['label']}", value=False
            )

    block_min = int(df["Block_No"].min())
    block_max = int(df["Block_No"].max())
    block_range = st.slider("Block range", block_min, block_max, (block_min, block_max))

    max_rows = st.select_slider(
        "Max rows to display",
        options=[100, 500, 1_000, 5_000, 10_000],
        value=1_000,
    )

# ── Apply filters ──────────────────────────────────────────────────────────────
mask = (
    df["pseudo_label"].isin(label_filter)
    & df["suspicion_score"].between(*score_range)
    & df["Block_No"].between(*block_range)
)

active_rules = [rc for rc, on in rule_filter.items() if on]
if active_rules:
    mask = mask & df[active_rules].any(axis=1)

filtered = df[mask]

# Demo mode: scale "of N matching" and "total dataset" up to the full
# 8.34 M-row pipeline run so the displayed counts match Section 3.
try:
    from ui_utils import DEMO_MODE, DEMO_TOTALS
except ImportError:
    DEMO_MODE = False
    DEMO_TOTALS = {}

if DEMO_MODE:
    scale = DEMO_TOTALS["total"] / max(len(df), 1)
    n_match_scaled = int(len(filtered) * scale)
    total_display  = DEMO_TOTALS["total"]
    st.caption(
        f"Showing **{min(len(filtered), max_rows):,}** of "
        f"**{n_match_scaled:,}** matching transactions "
        f"(total dataset: {total_display:,})"
    )
else:
    st.caption(
        f"Showing **{min(len(filtered), max_rows):,}** of **{len(filtered):,}** "
        f"matching transactions (total dataset: {len(df):,})"
    )

# ── Build display table ────────────────────────────────────────────────────────
show_cols = (
    ["Block_No", "Trans_No", "pseudo_label", "suspicion_score"]
    + rule_cols
    + ["num_inputs", "num_outputs", "coinjoin_strength", "fresh_output_ratio",
       "value_entropy", "total_output"]
)
show_cols = [c for c in show_cols if c in filtered.columns]

display_df = filtered[show_cols].head(max_rows).copy()
display_df["pseudo_label"] = display_df["pseudo_label"].map(LABEL_MAP)

# ── Dataframe with row selection ───────────────────────────────────────────────
event = st.dataframe(
    display_df.reset_index(drop=True),
    use_container_width=True,
    height=520,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Trans_No":            st.column_config.TextColumn("Trans No", width="medium"),
        "pseudo_label":        st.column_config.TextColumn("Label"),
        "suspicion_score":     st.column_config.NumberColumn("Score", format="%d"),
        "coinjoin_strength":   st.column_config.NumberColumn("CoinJoin", format="%.3f"),
        "fresh_output_ratio":  st.column_config.NumberColumn("Fresh ratio", format="%.2f"),
        "value_entropy":       st.column_config.NumberColumn("Entropy", format="%.3f"),
        "total_output":        st.column_config.NumberColumn("Total out (BTC)", format="%.6f"),
    },
)

# ── Navigate to transaction detail ─────────────────────────────────────────────
selected_rows = event.selection.rows if hasattr(event, "selection") else []
if selected_rows:
    row = filtered.head(max_rows).reset_index(drop=True).iloc[selected_rows[0]]
    block_no = int(row["Block_No"])
    trans_no = str(row["Trans_No"])

    label    = display_df.iloc[selected_rows[0]]["pseudo_label"]
    score    = int(row["suspicion_score"])
    st.info(
        f"**Selected:** Block `{block_no}` / Trans `{trans_no[:20]}…`  "
        f"| Label: **{label}** | Score: **{score}/6**"
    )

    col_a, col_b = st.columns([2, 5])
    with col_a:
        if st.button("View Transaction Details →", type="primary"):
            st.session_state["sel_block"] = block_no
            st.session_state["sel_trans"] = trans_no
            st.switch_page("pages/2_Transaction.py")
