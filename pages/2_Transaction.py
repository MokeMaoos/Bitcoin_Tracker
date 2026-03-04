"""
pages/2_Transaction.py — Transaction detail and explainability (WHY page).

Shows:
  - Model verdict + confidence
  - Blockchain explorer links (mempool.space, blockchain.com)
  - Per-rule evidence with raw data (expandable)
  - Feature comparison vs population
  - Raw inputs/outputs table
  - Cluster assignment
  - Wallet summary + link to wallet page
  - External: Mempool.space live data (fee, size, timing)
  - External: Bitcoin Abuse DB address reports
  - External: Claude AI natural-language forensic analysis
"""

import os
import sys

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_utils import (
    load_labeled, load_clusters, load_model_bundle, load_raw,
    run_live_prediction, rule_evidence, derive_rules,
    fetch_mempool_tx, check_addresses_abuse, generate_claude_analysis,
    RULE_META, LABEL_MAP, LABEL_COLORS, _get_secret,
)

st.set_page_config(
    page_title="Transaction Detail · Bitcoin Tracker",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎  Transaction Detail")

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("tx_form"):
    c1, c2, c3 = st.columns([1, 4, 1])
    default_block = st.session_state.get("sel_block", "")
    default_trans = st.session_state.get("sel_trans", "")
    block_input = c1.text_input("Block No", value=str(default_block) if default_block else "",
                                placeholder="e.g. 705592")
    trans_input = c2.text_input("Transaction Hash", value=str(default_trans) if default_trans else "",
                                placeholder="e.g. d8bbd0d759ab17f3…")
    submitted = c3.form_submit_button("Analyse", type="primary", use_container_width=True)

if not (block_input and trans_input):
    st.info("Enter a Block No and Transaction hash above, or select a transaction from the Explorer.")
    st.stop()

try:
    block_no = int(block_input)
    trans_no = str(trans_input).strip()
except ValueError:
    st.error("Block No must be an integer.")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
df       = load_labeled()
clusters = load_clusters()
bundle   = load_model_bundle()

with st.spinner("Loading raw transaction data…"):
    td, tx, ai, wi = load_raw()

# Pre-computed row from labeled_features.csv
precomputed = None
if df is not None:
    rows = df[(df["Block_No"] == block_no) & (df["Trans_No"] == trans_no)]
    if not rows.empty:
        precomputed = rows.iloc[0].to_dict()

# Raw detail rows
grp = td[(td["Block_No"] == block_no) & (td["Trans_No"] == trans_no)]

if grp.empty and precomputed is None:
    st.error(f"Transaction Block `{block_no}` / `{trans_no}` not found in the dataset.")
    st.stop()

# ── Get prediction ─────────────────────────────────────────────────────────────
prob        = None
feat_row    = precomputed or {}
data_source = "pre-computed"

if precomputed is not None and bundle is not None:
    fcols   = bundle["features"]
    feat_df = pd.DataFrame([{c: precomputed.get(c, np.nan) for c in fcols}])
    X       = feat_df.fillna(feat_df.median())
    prob    = float(bundle["model"].predict_proba(X)[0][1])

# ── Live re-predict button ─────────────────────────────────────────────────────
if bundle is not None and not grp.empty:
    _, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("🔄 Re-predict Live", help="Recompute features from raw data and re-run the model"):
            live_feat, live_prob = run_live_prediction(block_no, trans_no, td, tx, ai, wi, bundle)
            if live_feat is not None:
                feat_row    = live_feat
                prob        = live_prob
                data_source = "live"
                st.success("Live prediction complete.")

# ── Derive rule flags ──────────────────────────────────────────────────────────
rules = {}
if precomputed is not None:
    # Use stored rule columns if available
    for rc in RULE_META:
        rules[rc] = int(precomputed.get(rc, 0))
else:
    rules = derive_rules(feat_row)

# ── Verdict banner ─────────────────────────────────────────────────────────────
st.divider()

pred_label = "ILLICIT" if (prob or 0) >= 0.5 else "LICIT"
score      = int(precomputed.get("suspicion_score", sum(rules.values()))) if precomputed else sum(rules.values())
pseudo_lbl = LABEL_MAP.get(int((precomputed or {}).get("pseudo_label", -1)), "Unknown")

cv1, cv2, cv3, cv4 = st.columns(4)
cv1.metric("Model Verdict",  ("⛔ ILLICIT" if pred_label == "ILLICIT" else "✅ LICIT"))
if prob is not None:
    cv2.metric("Confidence",     f"{prob:.1%} illicit probability")
cv3.metric("Suspicion Score", f"{score}/6 rules fired")
cv4.metric("Rule-Based Label", pseudo_lbl)

# Blockchain explorer links
st.markdown(
    f"**View on blockchain explorers:** &nbsp;"
    f"[🔗 mempool.space](https://mempool.space/tx/{trans_no}) &nbsp;·&nbsp; "
    f"[🔗 blockchain.com](https://www.blockchain.com/btc/tx/{trans_no})"
)

st.divider()

# ── Rule evidence section ──────────────────────────────────────────────────────
st.subheader("Red-Flag Rule Analysis")
st.caption(
    "Each card below corresponds to one FATF red-flag indicator. "
    "Fired rules (red) are expanded by default. Click any card to see the underlying data."
)

ev = rule_evidence(feat_row, grp if not grp.empty else None, ai)

col_left, col_right = st.columns(2)
rule_items = list(RULE_META.items())
for i, (rule_key, meta) in enumerate(rule_items):
    fired    = bool(rules.get(rule_key, 0))
    badge    = "⛔ FIRED" if fired else "✅ Not fired"
    col      = col_left if i % 2 == 0 else col_right

    with col:
        with st.expander(
            f"{meta['icon']}  **{meta['label']}** — {badge}",
            expanded=fired,
        ):
            st.markdown(f"**FATF concern:** {meta['fatf']}")
            st.markdown(f"**Decision threshold:** {meta['threshold']}")

            if rule_key in ev:
                e = ev[rule_key]
                st.markdown(f"**Measured value:** `{e['value_str']}`")

                for section_title, section_data in e["sections"].items():
                    if not section_data:
                        st.caption(f"_{section_title}: (none)_")
                        continue

                    st.markdown(f"**{section_title}:**")

                    if isinstance(section_data, list):
                        if section_data and isinstance(section_data[0], tuple):
                            # (address, value) pairs
                            addr_df = pd.DataFrame(section_data, columns=["Address", "Value (BTC)"])
                            addr_df["Value (BTC)"] = addr_df["Value (BTC)"].round(8)
                            st.dataframe(addr_df, use_container_width=True, height=min(200, 40 + 35 * len(addr_df)))
                        elif section_data and isinstance(section_data[0], str) and len(section_data[0]) > 30:
                            # address list — show as scrollable table
                            addr_df = pd.DataFrame({"Address": section_data})
                            st.dataframe(addr_df, use_container_width=True, height=min(200, 40 + 35 * len(addr_df)))
                        elif section_data and isinstance(section_data[0], float):
                            # value array
                            val_df = pd.DataFrame({"Value (BTC)": [round(v, 8) for v in section_data]})
                            st.dataframe(val_df, use_container_width=True, height=min(200, 40 + 35 * len(val_df)))
                        else:
                            # formula breakdown strings
                            for line in section_data:
                                st.code(line)

# ── Feature comparison chart ───────────────────────────────────────────────────
if df is not None and feat_row:
    st.divider()
    st.subheader("Feature Values vs Population Medians")
    st.caption(
        "Blue = this transaction · Grey = median across all transactions · "
        "Red = median across illicit transactions"
    )

    compare_feats = [
        "num_inputs", "num_outputs", "io_ratio", "total_output",
        "value_entropy", "coinjoin_strength", "fresh_output_ratio",
        "peeling_flag", "equal_value_flag",
    ]
    compare_feats = [c for c in compare_feats if c in df.columns]

    med_all     = df[compare_feats].median()
    med_illicit = df[df["pseudo_label"] == 1][compare_feats].median()
    this_vals   = {c: float(feat_row.get(c) or 0) for c in compare_feats}

    fig = go.Figure()
    for series, color, name in [
        ([this_vals[c]        for c in compare_feats], "#3b82f6", "This TX"),
        ([float(med_all[c])   for c in compare_feats], "#9ca3af", "Median (all)"),
        ([float(med_illicit[c]) for c in compare_feats], "#ef4444", "Median (illicit)"),
    ]:
        fig.add_trace(go.Bar(name=name, x=compare_feats, y=series, marker_color=color))

    fig.update_layout(
        barmode="group",
        height=380,
        margin=dict(t=20, b=0),
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Raw inputs / outputs ───────────────────────────────────────────────────────
if not grp.empty:
    st.divider()
    st.subheader("Raw Transaction Data")

    c_in, c_out = st.columns(2)
    with c_in:
        st.markdown("**Inputs**")
        inputs = grp[["Input_Address", "Input_Value"]].dropna(subset=["Input_Address"]).reset_index(drop=True)
        st.dataframe(inputs, use_container_width=True, height=min(300, 40 + 35 * len(inputs)))

    with c_out:
        st.markdown("**Outputs**")
        outputs = grp[["Output_Address", "Output_Value"]].dropna(subset=["Output_Address"]).reset_index(drop=True)
        st.dataframe(outputs, use_container_width=True, height=min(300, 40 + 35 * len(outputs)))

# ── Cluster assignment ─────────────────────────────────────────────────────────
if clusters is not None:
    cl_row = clusters[(clusters["Block_No"] == block_no) & (clusters["Trans_No"] == trans_no)]
    if not cl_row.empty:
        st.divider()
        st.subheader("Cluster Assignments")
        cc1, cc2 = st.columns(2)
        km = int(cl_row["cluster_kmeans"].values[0])
        db = int(cl_row["cluster_dbscan"].values[0])
        cc1.metric("K-Means Cluster", f"Cluster {km}")
        cc2.metric(
            "DBSCAN Cluster",
            "Outlier (noise)" if db == -1 else ("Not in sample" if db == -2 else f"Cluster {db}"),
        )

# ── Wallet summary ─────────────────────────────────────────────────────────────
tx_row    = tx[(tx["Block_No"] == block_no) & (tx["Trans_No"] == trans_no)]
wallet_id = None
if not tx_row.empty:
    raw_wid = tx_row["Wallet"].values[0]
    try:
        wallet_id = int(float(raw_wid))   # Wallet column is float64 in Transactions.csv
    except (ValueError, TypeError):
        wallet_id = None

if wallet_id is not None:
    wi_row = wi[wi["Wallet"] == wallet_id]
    st.divider()
    st.subheader("Wallet Information")

    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("Wallet ID", wallet_id)
    if not wi_row.empty:
        wc2.metric("Total Transactions", f"{int(wi_row['N_Transactions'].values[0]):,}")
        wc3.metric("Total Sent (BTC)",   f"{float(wi_row['Total_Sent'].values[0]):.4f}")
        wc4.metric("Unique Addresses",   f"{int(wi_row['N_Addresses'].values[0]):,}")

    if st.button("View Full Wallet Analysis →"):
        st.session_state["sel_wallet"] = str(wallet_id)
        st.switch_page("pages/3_Wallet.py")

# ══════════════════════════════════════════════════════════════════════════════
# EXTERNAL DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🌐  External Data Sources")

ext_tab1, ext_tab2, ext_tab3 = st.tabs([
    "📡  Mempool.space",
    "🚨  Bitcoin Abuse DB",
    "🤖  Claude AI Analysis",
])

# ── Tab 1: Mempool.space ──────────────────────────────────────────────────────
with ext_tab1:
    st.caption(
        "Live transaction data from [Mempool.space](https://mempool.space) — "
        "no API key required."
    )

    mempool_key = f"mempool_{trans_no}"
    if mempool_key not in st.session_state:
        st.session_state[mempool_key] = None

    col_fetch, _ = st.columns([1, 4])
    if col_fetch.button("Fetch from Mempool.space", key="fetch_mempool"):
        with st.spinner("Querying Mempool.space…"):
            st.session_state[mempool_key] = fetch_mempool_tx(trans_no)

    mpdata = st.session_state.get(mempool_key)
    if mpdata is not None:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Fee",           f"{mpdata['fee_sats']:,} sats  ({mpdata['fee_btc']:.8f} BTC)")
        mc2.metric("Fee Rate",      f"{mpdata['fee_rate_sat_vb']} sat/vB")
        mc3.metric("Size",          f"{mpdata['size_bytes']:,} B  /  {mpdata['vsize_vbytes']} vB")
        mc4.metric("Confirmed",     "✅ Yes" if mpdata["confirmed"] else "⏳ Pending")

        mc5, mc6, mc7, mc8 = st.columns(4)
        mc5.metric("Version",       mpdata.get("version", "—"))
        mc6.metric("Locktime",      mpdata.get("locktime", "—"))
        mc7.metric("Inputs (live)", mpdata.get("vin_count", "—"))
        mc8.metric("Outputs (live)",mpdata.get("vout_count", "—"))

        if mpdata.get("block_height"):
            import datetime
            bt = mpdata.get("block_time")
            dt = datetime.datetime.fromtimestamp(bt).strftime("%Y-%m-%d %H:%M UTC") if bt else "—"
            st.caption(f"Confirmed in block **{mpdata['block_height']:,}** at {dt}")

        st.markdown(
            f"[🔗 View full details on Mempool.space](https://mempool.space/tx/{trans_no})"
        )
    elif st.session_state.get(mempool_key) is None and "fetch_mempool" not in st.session_state:
        st.info("Click the button above to fetch live transaction data.")
    else:
        st.warning(
            "Transaction not found on Mempool.space. "
            "It may be too old, or the hash may be from a test dataset."
        )

# ── Tab 2: Bitcoin Abuse DB ───────────────────────────────────────────────────
with ext_tab2:
    st.caption(
        "Check output addresses against the "
        "[Bitcoin Abuse Database](https://www.bitcoinabuse.com/api-docs). "
        "Free API key required — register at bitcoinabuse.com."
    )

    abuse_key = _get_secret("BITCOIN_ABUSE_KEY")
    if not abuse_key:
        st.warning(
            "No Bitcoin Abuse API key found.  \n"
            "Add `BITCOIN_ABUSE_KEY = \"your_key\"` to `.streamlit/secrets.toml` and restart."
        )
    else:
        abuse_cache_key = f"abuse_{trans_no}"
        if st.button("Check Output Addresses", key="check_abuse"):
            oa_list = []
            if not grp.empty:
                oa_list = grp["Output_Address"].dropna().unique().tolist()[:20]  # cap at 20
            with st.spinner(f"Checking {len(oa_list)} addresses…"):
                st.session_state[abuse_cache_key] = check_addresses_abuse(oa_list)

        abuse_results = st.session_state.get(abuse_cache_key, {})
        if abuse_results:
            any_flagged = any(r.get("count", 0) > 0 for r in abuse_results.values())
            if any_flagged:
                st.error("⚠️  One or more output addresses have abuse reports!")
            else:
                st.success("✅  No abuse reports found for checked addresses.")

            rows = []
            for addr, result in abuse_results.items():
                count = result.get("count", 0)
                rows.append({
                    "Address": addr,
                    "Abuse Reports": count,
                    "Status": "⚠️ FLAGGED" if count > 0 else "✅ Clean",
                })
            ab_df = pd.DataFrame(rows)
            st.dataframe(ab_df, use_container_width=True)
        else:
            st.info("Click the button above to check output addresses.")

# ── Tab 3: Claude AI Analysis ─────────────────────────────────────────────────
with ext_tab3:
    st.caption(
        "Ask Claude (Anthropic's AI) to synthesise all signals and generate a "
        "forensic risk narrative. Requires an Anthropic API key."
    )

    claude_key = _get_secret("ANTHROPIC_API_KEY")
    if not claude_key:
        st.warning(
            "No Anthropic API key found.  \n"
            "Add `ANTHROPIC_API_KEY = \"sk-ant-...\"` to `.streamlit/secrets.toml` and restart."
        )
        st.markdown(
            "Get a key at [console.anthropic.com](https://console.anthropic.com/)"
        )
    else:
        claude_cache_key = f"claude_{trans_no}"

        if st.button("🤖  Generate Claude Analysis", type="primary", key="gen_claude"):
            with st.spinner("Claude is analysing the transaction…"):
                analysis = generate_claude_analysis(
                    feat_row=feat_row,
                    rules=rules,
                    prob=prob or 0.0,
                    block_no=block_no,
                    trans_no=trans_no,
                )
                st.session_state[claude_cache_key] = analysis

        analysis = st.session_state.get(claude_cache_key)
        if analysis:
            if analysis.startswith("❌"):
                st.error(analysis)
            else:
                st.markdown(analysis)
                st.caption(
                    "⚠️ AI-generated analysis. Claude has no real-time blockchain access — "
                    "it reasons from the feature values and rule flags provided. "
                    "Always verify against primary sources before taking compliance action."
                )
