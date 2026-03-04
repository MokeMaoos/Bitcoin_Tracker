"""
pages/4_Features.py — Feature distribution analysis, correlation heatmap,
and Random Forest feature importances.
"""

import os
import sys

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_utils import load_labeled, load_model_bundle, LABEL_COLORS

st.set_page_config(
    page_title="Features · Bitcoin Tracker",
    page_icon="📈",
    layout="wide",
)

st.title("📈  Feature Analysis")

df     = load_labeled()
bundle = load_model_bundle()

if df is None:
    st.error("No labeled data found. Run the pipeline first.")
    st.stop()

NUMERIC_FEATURES = [
    "num_inputs", "num_outputs", "total_input", "total_output",
    "mean_output", "std_output", "max_output", "min_output",
    "io_ratio", "value_ratio", "value_entropy",
    "coinjoin_strength", "fresh_output_ratio",
    "wallet_total_tx", "wallet_total_sent", "wallet_size",
]
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in df.columns]

tab1, tab2, tab3 = st.tabs(["📊  Distributions", "🔲  Correlation", "🏆  Importances"])

# ── Tab 1: Distributions ───────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        feat_sel = st.selectbox("Select feature", NUMERIC_FEATURES, index=0)
    with col_b:
        log_y = st.checkbox("Log scale (y-axis)", value=True)

    df_plot = df[df[feat_sel].notna()].copy()
    df_plot["Label"] = df_plot["pseudo_label"].map({1: "Illicit", 0: "Licit", -1: "Uncertain"})

    fig = px.histogram(
        df_plot, x=feat_sel, color="Label",
        color_discrete_map={
            "Illicit":   LABEL_COLORS[1],
            "Licit":     LABEL_COLORS[0],
            "Uncertain": LABEL_COLORS[-1],
        },
        barmode="overlay",
        opacity=0.72,
        nbins=80,
        log_y=log_y,
        labels={feat_sel: feat_sel, "count": "Transactions"},
    )
    fig.update_layout(height=420, margin=dict(t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics by label
    st.subheader("Summary Statistics by Label")
    stats = (
        df_plot.groupby("Label")[feat_sel]
        .agg(Count="count", Mean="mean", Median="median", Std="std",
             Min="min", Max="max")
        .round(4)
    )
    st.dataframe(stats, use_container_width=True)

    # Scatter: any two features
    st.divider()
    st.subheader("Scatter Plot")
    col_x, col_y, col_sample = st.columns([2, 2, 1])
    feat_x  = col_x.selectbox("X axis", NUMERIC_FEATURES, index=0, key="scatter_x")
    feat_y  = col_y.selectbox("Y axis", NUMERIC_FEATURES,
                               index=min(4, len(NUMERIC_FEATURES) - 1), key="scatter_y")
    n_samp  = col_sample.select_slider("Sample", [1_000, 5_000, 10_000, 20_000], value=5_000)

    sdf = (
        df[[feat_x, feat_y, "pseudo_label"]]
        .dropna()
        .sample(min(n_samp, len(df)), random_state=42)
        .copy()
    )
    sdf["Label"] = sdf["pseudo_label"].map({1: "Illicit", 0: "Licit", -1: "Uncertain"})

    fig_sc = px.scatter(
        sdf, x=feat_x, y=feat_y, color="Label",
        color_discrete_map={
            "Illicit":   LABEL_COLORS[1],
            "Licit":     LABEL_COLORS[0],
            "Uncertain": LABEL_COLORS[-1],
        },
        opacity=0.5,
    )
    fig_sc.update_traces(marker_size=3)
    fig_sc.update_layout(height=400, margin=dict(t=20, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 2: Correlation heatmap ─────────────────────────────────────────────────
with tab2:
    st.subheader("Pearson Correlation Matrix")
    st.caption("Sampled from 10,000 transactions. Includes `pseudo_label` as the target variable.")

    @st.cache_data(show_spinner="Computing correlation matrix…")
    def compute_corr():
        cols  = NUMERIC_FEATURES + ["pseudo_label"]
        cols  = [c for c in cols if c in df.columns]
        sample = df[cols].dropna().sample(min(10_000, len(df)), random_state=42)
        return sample.corr()

    corr = compute_corr()
    fig2 = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig2.update_layout(height=620, margin=dict(t=20, b=0))
    fig2.update_traces(textfont_size=9)
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Feature importances ─────────────────────────────────────────────────
with tab3:
    if bundle is None:
        st.info("Model not found. Run the pipeline first.")
    else:
        st.subheader("Random Forest Feature Importances")
        st.caption("Mean decrease in impurity (MDI) across all 200 trees.")

        clf        = bundle["model"].named_steps["clf"]
        feat_names = bundle["features"]
        fi         = sorted(
            zip(feat_names, clf.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        fi_df = pd.DataFrame(fi, columns=["Feature", "Importance"])

        fig3 = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            text=fi_df["Importance"].round(4),
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(
            height=520,
            margin=dict(t=20, b=0),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Numeric table
        st.dataframe(
            fi_df.assign(Importance=fi_df["Importance"].round(5)),
            use_container_width=True,
        )
