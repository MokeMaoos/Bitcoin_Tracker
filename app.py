"""
app.py — Bitcoin Illicit Transaction Detection — Overview Dashboard.

Run with:
    cd /mnt/data/Bitcoin_Tracker
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui_utils import (
    load_labeled, load_clusters, load_model_bundle,
    RULE_META, LABEL_MAP, LABEL_COLORS, OUTPUT_DIR,
    DEMO_MODE, DEMO_TOTALS,
)

st.set_page_config(
    page_title="Bitcoin Tracker",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("₿  Bitcoin Illicit Transaction Detection")
st.caption("Weakly-supervised detection pipeline · FATF red-flag methodology (BMAN31260)")

# ── Load data ──────────────────────────────────────────────────────────────────
df       = load_labeled()
clusters = load_clusters()
bundle   = load_model_bundle()

if df is None:
    st.error(
        "No labeled features found. Run the pipeline first:  \n"
        "```\npython run_pipeline.py\n```"
    )
    st.stop()

# ── Metric cards ───────────────────────────────────────────────────────────────
# In demo mode the loaded df is a 5,000-row stratified sample for fast
# rendering; the displayed counts come from DEMO_TOTALS so they continue
# to reflect the full 8.34 M-row pipeline run.
if DEMO_MODE:
    total     = DEMO_TOTALS["total"]
    illicit   = DEMO_TOTALS["illicit"]
    licit     = DEMO_TOTALS["licit"]
    uncertain = DEMO_TOTALS["uncertain"]
else:
    total     = len(df)
    illicit   = int((df["pseudo_label"] == 1).sum())
    licit     = int((df["pseudo_label"] == 0).sum())
    uncertain = int((df["pseudo_label"] == -1).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("⛔ Illicit",          f"{illicit:,}",   f"{illicit/total:.1%}")
c3.metric("✅ Licit",            f"{licit:,}",     f"{licit/total:.1%}")
c4.metric("⚠️ Uncertain",        f"{uncertain:,}", f"{uncertain/total:.1%}")

# ── Model info banner ──────────────────────────────────────────────────────────
if bundle is not None:
    clf_name  = bundle.get("classifier", "random_forest").replace("_", " ").title()
    n_feats   = len(bundle.get("features", []))
    st.info(
        f"**Model:** {clf_name} · "
        f"**Features used:** {n_feats} · "
        f"**Labeled training set:** {illicit + licit:,} transactions "
        f"({illicit:,} illicit + {licit:,} licit, uncertain excluded)"
    )

st.divider()

# ── Row 1: Label pie + Suspicion score histogram ───────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Label Distribution")
    fig = px.pie(
        {"Label": ["Illicit", "Licit", "Uncertain"],
         "Count": [illicit, licit, uncertain]},
        values="Count", names="Label", hole=0.4,
        color="Label",
        color_discrete_map={
            "Illicit":   LABEL_COLORS[1],
            "Licit":     LABEL_COLORS[0],
            "Uncertain": LABEL_COLORS[-1],
        },
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Suspicion Score Distribution")
    if DEMO_MODE:
        score_counts = pd.Series(DEMO_TOTALS["suspicion_score"]).sort_index()
    else:
        score_counts = df["suspicion_score"].value_counts().sort_index()
    fig2 = px.bar(
        x=score_counts.index, y=score_counts.values,
        labels={"x": "Suspicion Score (rules fired)", "y": "Transactions"},
        color=score_counts.index,
        color_continuous_scale="Reds",
        text=score_counts.values,
    )
    fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig2.update_layout(
        margin=dict(t=0, b=0), height=300,
        coloraxis_showscale=False,
        xaxis=dict(tickmode="linear"),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 2: Rule firing rates + Feature importances ─────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Rule Firing Rates")
    rule_cols   = [c for c in df.columns if c.startswith("rule_") and c in RULE_META]
    rule_labels = [RULE_META[c]["label"] for c in rule_cols]
    if DEMO_MODE:
        rule_counts = [DEMO_TOTALS["rule_firing"][c] for c in rule_cols]
    else:
        rule_counts = [int(df[c].sum()) for c in rule_cols]
    rule_pcts   = [n / total for n in rule_counts]

    fig3 = px.bar(
        x=rule_counts, y=rule_labels,
        orientation="h",
        text=[f"{p:.1%}" for p in rule_pcts],
        labels={"x": "Transactions fired", "y": ""},
        color=rule_counts,
        color_continuous_scale="Oranges",
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(
        margin=dict(t=0, b=0), height=300,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Top Feature Importances")
    if bundle is not None:
        clf        = bundle["model"].named_steps["clf"]
        feat_names = bundle["features"]
        if hasattr(clf, "feature_importances_"):
            importances = sorted(
                zip(feat_names, clf.feature_importances_),
                key=lambda x: x[1], reverse=True,
            )[:15]
            fi_df = pd.DataFrame(importances, columns=["Feature", "Importance"])
            fig4  = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                text=fi_df["Importance"].round(4),
            )
            fig4.update_traces(textposition="outside")
            fig4.update_layout(
                margin=dict(t=0, b=0), height=350,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Feature importances not available for this model type.")
    else:
        st.info("Model not found. Run the pipeline first.")

st.divider()

# ── Row 3: New feature insights ────────────────────────────────────────────────
st.subheader("New Feature Insights")
st.caption(
    "Distributions of the 20 new features added from previously unused data sources, "
    "split by pseudo-label (Illicit vs Licit). Uncertain transactions excluded."
)

labeled_only = df[df["pseudo_label"] != -1].copy()
labeled_only["Label"] = labeled_only["pseudo_label"].map({1: "Illicit", 0: "Licit"})

new_feat_tabs = st.tabs([
    "👛 Wallet",
    "🕐 Timing",
    "🌐 Graph",
    "📍 Output Addresses",
])

# ── Wallet features ────────────────────────────────────────────────────────────
with new_feat_tabs[0]:
    wf_col1, wf_col2 = st.columns(2)
    wallet_feats = [
        ("wallet_age",           "Wallet Age (blocks)",           "Purples"),
        ("wallet_sent_ratio",    "Wallet Sent Ratio",             "Oranges"),
        ("wallet_final_balance", "Wallet Final Balance (satoshi)","Blues"),
        ("wallet_balance_velocity","Wallet Balance Velocity",     "Reds"),
    ]
    for i, (col, label, scale) in enumerate(wallet_feats):
        if col not in labeled_only.columns:
            continue
        target = wf_col1 if i % 2 == 0 else wf_col2
        with target:
            sub = labeled_only[[col, "Label"]].dropna()
            if sub.empty:
                st.caption(f"_{label}: no data_")
                continue
            # Cap extreme outliers at 99th percentile for readability
            p99 = sub[col].quantile(0.99)
            sub = sub[sub[col] <= p99]
            fig = px.histogram(
                sub, x=col, color="Label", barmode="overlay",
                nbins=60, opacity=0.7,
                color_discrete_map={"Illicit": LABEL_COLORS[1], "Licit": LABEL_COLORS[0]},
                labels={col: label},
                title=label,
            )
            fig.update_layout(height=280, margin=dict(t=30, b=0), showlegend=(i == 0))
            st.plotly_chart(fig, use_container_width=True)

# ── Timing features ────────────────────────────────────────────────────────────
with new_feat_tabs[1]:
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        if "tx_hour" in labeled_only.columns:
            hr = labeled_only[["tx_hour", "Label"]].dropna()
            hr["tx_hour"] = hr["tx_hour"].astype(int)
            fig = px.histogram(
                hr, x="tx_hour", color="Label", barmode="overlay",
                nbins=24, opacity=0.7, range_x=[0, 23],
                color_discrete_map={"Illicit": LABEL_COLORS[1], "Licit": LABEL_COLORS[0]},
                labels={"tx_hour": "Hour of Day (UTC)"},
                title="Transaction Hour Distribution",
            )
            fig.update_layout(height=280, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        if "tx_day_of_week" in labeled_only.columns:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dw = labeled_only[["tx_day_of_week", "Label", "pseudo_label"]].dropna()
            dw["tx_day_of_week"] = dw["tx_day_of_week"].astype(int)
            dw_counts = (
                dw.groupby(["tx_day_of_week", "Label"])
                .size().reset_index(name="count")
            )
            dw_counts["Day"] = dw_counts["tx_day_of_week"].map(
                dict(enumerate(day_names))
            )
            fig = px.bar(
                dw_counts, x="Day", y="count", color="Label",
                barmode="group",
                color_discrete_map={"Illicit": LABEL_COLORS[1], "Licit": LABEL_COLORS[0]},
                title="Transaction Day of Week",
                category_orders={"Day": day_names},
            )
            fig.update_layout(height=280, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with t_col2:
        if "is_night" in labeled_only.columns:
            night = labeled_only[["is_night", "Label"]].dropna()
            night["is_night"] = night["is_night"].astype(int)
            night_rate = night.groupby("Label")["is_night"].mean().reset_index()
            night_rate.columns = ["Label", "Night Rate"]
            fig = px.bar(
                night_rate, x="Label", y="Night Rate", color="Label",
                color_discrete_map={"Illicit": LABEL_COLORS[1], "Licit": LABEL_COLORS[0]},
                text=night_rate["Night Rate"].map("{:.1%}".format),
                title="Night-Time Transactions (00:00–05:59 UTC)",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=280, margin=dict(t=30, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if "lock_time_flag" in labeled_only.columns:
            lt = labeled_only[["lock_time_flag", "Label"]].dropna()
            lt_rate = lt.groupby("Label")["lock_time_flag"].mean().reset_index()
            lt_rate.columns = ["Label", "Lock-Time Rate"]
            fig = px.bar(
                lt_rate, x="Label", y="Lock-Time Rate", color="Label",
                color_discrete_map={"Illicit": LABEL_COLORS[1], "Licit": LABEL_COLORS[0]},
                text=lt_rate["Lock-Time Rate"].map("{:.1%}".format),
                title="Transactions with Non-Zero Lock Time",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=280, margin=dict(t=30, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ── Graph topology features ────────────────────────────────────────────────────
with new_feat_tabs[2]:
    g_col1, g_col2 = st.columns(2)
    graph_feats = [
        ("n_distinct_input_wallets",  "Distinct Input Wallets per Tx"),
        ("n_distinct_output_wallets", "Distinct Output Wallets per Tx"),
    ]
    for i, (col, label) in enumerate(graph_feats):
        target = g_col1 if i == 0 else g_col2
        if col not in labeled_only.columns:
            continue
        with target:
            sub = labeled_only[[col, "Label"]].dropna()
            p99 = sub[col].quantile(0.99)
            sub = sub[sub[col] <= p99]
            fig = px.histogram(
                sub, x=col, color="Label", barmode="overlay",
                nbins=40, opacity=0.7,
                color_discrete_map={"Illicit": LABEL_COLORS[1], "Licit": LABEL_COLORS[0]},
                labels={col: label}, title=label,
            )
            fig.update_layout(height=300, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

    if "tx_fee" in labeled_only.columns:
        fee = labeled_only[["tx_fee", "Label"]].dropna()
        p99 = fee["tx_fee"].quantile(0.99)
        fee = fee[fee["tx_fee"] <= p99]
        fig = px.box(
            fee, x="Label", y="tx_fee", color="Label",
            color_discrete_map={"Illicit": LABEL_COLORS[1], "Licit": LABEL_COLORS[0]},
            labels={"tx_fee": "Transaction Fee (satoshi)"},
            title="Transaction Fee by Label",
        )
        fig.update_layout(height=300, margin=dict(t=30, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ── Output-address features ────────────────────────────────────────────────────
with new_feat_tabs[3]:
    oa_col1, oa_col2 = st.columns(2)
    oa_feats = [
        ("mean_out_addr_age",          "Mean Output Address Age (blocks)", "Purples"),
        ("frac_out_addr_zero_balance", "Fraction of Output Addresses with Zero Balance", "Reds"),
        ("mean_out_addr_tx_count",     "Mean Output Address Tx Count", "Blues"),
        ("mean_out_addr_sent_ratio",   "Mean Output Address Sent Ratio", "Oranges"),
    ]
    for i, (col, label, scale) in enumerate(oa_feats):
        if col not in labeled_only.columns:
            continue
        target = oa_col1 if i % 2 == 0 else oa_col2
        with target:
            sub = labeled_only[[col, "Label"]].dropna()
            if sub.empty:
                st.caption(f"_{label}: no data_")
                continue
            p99 = sub[col].quantile(0.99)
            sub = sub[sub[col] <= p99]
            fig = px.histogram(
                sub, x=col, color="Label", barmode="overlay",
                nbins=50, opacity=0.7,
                color_discrete_map={"Illicit": LABEL_COLORS[1], "Licit": LABEL_COLORS[0]},
                labels={col: label}, title=label,
            )
            fig.update_layout(height=280, margin=dict(t=30, b=0), showlegend=(i == 0))
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Cluster summary ────────────────────────────────────────────────────────────
if clusters is not None:
    st.subheader("K-Means Cluster Sizes")
    km_counts = clusters["cluster_kmeans"].value_counts().sort_index()
    fig5 = px.bar(
        x=[f"Cluster {i}" for i in km_counts.index],
        y=km_counts.values,
        labels={"x": "", "y": "Transactions"},
        text=km_counts.values,
        color=km_counts.values,
        color_continuous_scale="Purples",
    )
    fig5.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig5.update_layout(
        margin=dict(t=0, b=30), height=280,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.divider()

# ── Pipeline status ────────────────────────────────────────────────────────────
with st.expander("Pipeline status"):
    files = {
        "transaction_features.csv": "Feature engineering (02)",
        "labeled_features.csv":     "Rule-based labelling (03)",
        "cluster_assignments.csv":  "Cluster assignments (04)",
        "model.pkl":                "Trained model (04)",
        "elliptic_eval.json":       "Elliptic evaluation (06)",
    }
    for fname, label in files.items():
        path   = os.path.join(OUTPUT_DIR, fname)
        exists = os.path.exists(path)
        icon   = "✅" if exists else "❌"
        if exists:
            size_mb = os.path.getsize(path) / 1e6
            st.markdown(f"{icon} **{label}** — `{fname}` ({size_mb:.1f} MB)")
        else:
            st.markdown(f"{icon} **{label}** — `{fname}` not found")

    if bundle is not None:
        st.markdown(
            f"**Classifier:** {bundle.get('classifier', 'unknown').replace('_',' ').title()}  \n"
            f"**Features:**   {len(bundle.get('features', []))}"
        )
