"""
app.py — Bitcoin Illicit Transaction Detection — Overview Dashboard.

Run with:
    cd /home/cp/Bitcoin_Tracker
    streamlit run app.py
"""

import os
import sys

import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui_utils import (
    load_labeled, load_clusters, load_model_bundle,
    RULE_META, LABEL_MAP, LABEL_COLORS,
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
total     = len(df)
illicit   = int((df["pseudo_label"] == 1).sum())
licit     = int((df["pseudo_label"] == 0).sum())
uncertain = int((df["pseudo_label"] == -1).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("⛔ Illicit",          f"{illicit:,}",   f"{illicit/total:.1%}")
c3.metric("✅ Licit",            f"{licit:,}",     f"{licit/total:.1%}")
c4.metric("⚠️ Uncertain",        f"{uncertain:,}", f"{uncertain/total:.1%}")

st.divider()

# ── Row 1: Label pie + Suspicion score histogram ───────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Label Distribution")
    pie_df = {
        "Label": ["Illicit", "Licit", "Uncertain"],
        "Count": [illicit, licit, uncertain],
    }
    fig = px.pie(
        pie_df, values="Count", names="Label", hole=0.4,
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

# ── Row 2: Rule firing rates + Feature importances ────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Rule Firing Rates")
    rule_cols   = [c for c in df.columns if c.startswith("rule_")]
    rule_labels = [RULE_META[c]["label"] for c in rule_cols if c in RULE_META]
    rule_counts = [int(df[c].sum()) for c in rule_cols if c in RULE_META]
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
        clf         = bundle["model"].named_steps["clf"]
        feat_names  = bundle["features"]
        importances = sorted(
            zip(feat_names, clf.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )[:10]
        import pandas as pd
        fi_df = pd.DataFrame(importances, columns=["Feature", "Importance"])
        fig4  = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            text=fi_df["Importance"].round(4),
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(
            margin=dict(t=0, b=0), height=300,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Model not found. Run the pipeline first.")

# ── Cluster summary ────────────────────────────────────────────────────────────
if clusters is not None:
    st.divider()
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

# ── Pipeline status ────────────────────────────────────────────────────────────
st.divider()
with st.expander("Pipeline status"):
    from ui_utils import OUTPUT_DIR
    files = {
        "transaction_features.csv": "Feature engineering (02)",
        "labeled_features.csv":     "Rule-based labelling (03)",
        "cluster_assignments.csv":  "Cluster assignments (04)",
        "model.pkl":                "Trained model (04)",
    }
    for fname, label in files.items():
        path    = os.path.join(OUTPUT_DIR, fname)
        exists  = os.path.exists(path)
        icon    = "✅" if exists else "❌"
        size_mb = os.path.getsize(path) / 1e6 if exists else 0
        st.markdown(f"{icon} **{label}** — `{fname}` ({size_mb:.1f} MB)" if exists
                    else f"{icon} **{label}** — `{fname}` not found")
