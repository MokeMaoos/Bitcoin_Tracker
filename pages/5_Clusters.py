"""
pages/5_Clusters.py — 2-D PCA cluster visualisation.

Projects the full feature space down to 2 principal components and
renders an interactive scatter, coloured by pseudo-label, K-Means cluster,
or suspicion score.
"""

import os
import sys

import streamlit as st
import plotly.express as px
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_utils import load_labeled, load_clusters, LABEL_COLORS

st.set_page_config(
    page_title="Clusters · Bitcoin Tracker",
    page_icon="🔵",
    layout="wide",
)

st.title("🔵  Cluster Visualisation (PCA)")
st.caption(
    "2-D PCA projection of the 18-feature transaction space. "
    "Sampled for performance — hover any point for details."
)

df       = load_labeled()
clusters = load_clusters()

if df is None:
    st.error("No labeled data found. Run the pipeline first.")
    st.stop()

FEATURES = [
    "num_inputs", "num_outputs", "total_input", "total_output",
    "mean_output", "std_output", "max_output", "min_output",
    "io_ratio", "value_ratio", "value_entropy",
    "coinjoin_strength", "fresh_output_ratio",
]
FEATURES = [c for c in FEATURES if c in df.columns]

# ── Controls ───────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
sample_size = ctrl1.select_slider(
    "Sample size", options=[1_000, 2_000, 5_000, 10_000, 20_000], value=5_000
)
color_by = ctrl2.radio(
    "Colour by",
    ["Pseudo-label", "K-Means cluster", "Suspicion score"],
)
show_hulls = ctrl3.checkbox("Show cluster convex hulls (K-Means only)", value=False)

# ── Cached PCA ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing PCA projection…")
def compute_pca(n: int):
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA

    df_local = load_labeled()
    if df_local is None:
        return None, None

    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(df_local), size=min(n, len(df_local)), replace=False)
    sub  = df_local.iloc[idx].copy().reset_index(drop=True)

    X = sub[FEATURES].values
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]
    return sub, pca.explained_variance_ratio_


with st.spinner("Computing PCA…"):
    sub, evr = compute_pca(sample_size)

if sub is None:
    st.error("Failed to compute PCA.")
    st.stop()

# Merge cluster assignments
if clusters is not None:
    sub = sub.merge(
        clusters[["Block_No", "Trans_No", "cluster_kmeans"]],
        on=["Block_No", "Trans_No"], how="left",
    )

st.caption(
    f"**PC1** explains **{evr[0]:.1%}** of variance · "
    f"**PC2** explains **{evr[1]:.1%}** of variance · "
    f"Showing **{len(sub):,}** transactions"
)

# ── Build plot ─────────────────────────────────────────────────────────────────
hover_cols = {"Block_No": True, "Trans_No": True, "suspicion_score": True}

if color_by == "Pseudo-label":
    sub["Label"] = sub["pseudo_label"].map({1: "Illicit", 0: "Licit", -1: "Uncertain"})
    fig = px.scatter(
        sub, x="PC1", y="PC2", color="Label",
        color_discrete_map={
            "Illicit":   LABEL_COLORS[1],
            "Licit":     LABEL_COLORS[0],
            "Uncertain": LABEL_COLORS[-1],
        },
        opacity=0.55,
        hover_data=hover_cols,
    )

elif color_by == "K-Means cluster" and "cluster_kmeans" in sub.columns:
    sub["Cluster"] = sub["cluster_kmeans"].fillna(-1).astype(int).astype(str)
    fig = px.scatter(
        sub, x="PC1", y="PC2", color="Cluster",
        opacity=0.55,
        hover_data=hover_cols,
    )

else:  # Suspicion score
    fig = px.scatter(
        sub, x="PC1", y="PC2", color="suspicion_score",
        color_continuous_scale="Reds",
        range_color=[0, 6],
        opacity=0.55,
        hover_data=hover_cols,
        labels={"suspicion_score": "Score"},
    )

fig.update_traces(marker_size=3)
fig.update_layout(height=620, margin=dict(t=20, b=0))
st.plotly_chart(fig, use_container_width=True)

# ── PCA component loadings table ───────────────────────────────────────────────
with st.expander("Show PCA loadings (which features drive each component)"):
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    import pandas as pd

    @st.cache_data(show_spinner="Computing loadings…")
    def get_loadings():
        df_l = load_labeled()
        X    = df_l[FEATURES].dropna().values
        X    = SimpleImputer(strategy="median").fit_transform(X)
        X    = StandardScaler().fit_transform(X)
        pca  = PCA(n_components=2, random_state=42)
        pca.fit(X)
        return pd.DataFrame(pca.components_.T, index=FEATURES,
                            columns=["PC1 loading", "PC2 loading"]).round(4)

    loadings = get_loadings()
    st.dataframe(loadings.sort_values("PC1 loading", ascending=False), use_container_width=True)
