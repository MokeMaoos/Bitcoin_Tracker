"""
pages/6_Evaluate.py — Elliptic ground-truth benchmark evaluation results.

Displays the output of 06_evaluate.py: precision/recall/F1 of the
Random Forest methodology validated against the Elliptic expert-labeled
dataset (Weber et al., 2019).
"""

import os
import sys
import json

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Evaluate · Bitcoin Tracker",
    page_icon="📋",
    layout="wide",
)

st.title("📋  Elliptic Ground-Truth Evaluation")
st.caption(
    "Validates the detection methodology against the "
    "[Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) "
    "(Weber et al., 2019) — 203,769 expert-labeled Bitcoin transactions."
)

RESULTS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "elliptic_eval.json",
)
ELLIPTIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "elliptic",
)

# ── Check for data / results ───────────────────────────────────────────────────
has_results = os.path.exists(RESULTS_JSON)
has_data    = os.path.exists(os.path.join(ELLIPTIC_DIR, "elliptic_txs_features.csv"))

if not has_data:
    st.warning("### Elliptic dataset not downloaded yet")
    st.markdown("""
**To run the evaluation:**

The dataset downloads automatically via `kagglehub`. You just need a free Kaggle account set up once:

1. Go to [kaggle.com](https://www.kaggle.com/) → **Account** → **API** → **Create New Token**
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. Run from the terminal:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   python 06_evaluate.py
   ```
4. Refresh this page to see the results.

The dataset (~50 MB) is cached after the first download — subsequent runs are instant.

**Why this matters:**
The Elliptic dataset provides expert-annotated ground truth for 203k real Bitcoin transactions
(~2% illicit: ransomware, darknet markets, scams; ~21% licit: exchanges, miners).
By training and evaluating our Random Forest on Elliptic's data, we can measure what
the methodology achieves against *real* labels rather than pseudo-labels.
    """)
    st.stop()

if not has_results:
    st.info(
        "Elliptic data found but evaluation not run yet.  \n"
        "Run `python 06_evaluate.py` from the terminal, then refresh."
    )
    st.stop()

# ── Load results ───────────────────────────────────────────────────────────────
with open(RESULTS_JSON) as f:
    res = json.load(f)

st.success(f"Results loaded from `outputs/elliptic_eval.json`  —  {res.get('dataset', '')}")
st.divider()

# ── Feature set selector ───────────────────────────────────────────────────────
feat_options = ["Local features only (93)", "All features (166)"]
if res.get("rf_local_features"):
    feat_options.append("RF baseline — local features (comparison)")

feat_set = st.radio("Feature set", feat_options, horizontal=True)

if "RF baseline" in feat_set:
    key = "rf_local_features"
elif "Local" in feat_set:
    key = "local_features"
else:
    key = "all_features"
data = res.get(key, {})

if not data:
    st.error(f"No results found for key '{key}' in the JSON.")
    st.stop()

cv   = data.get("cv_metrics", {})
counts = data.get("label_counts", {})

# ── Overview metrics ───────────────────────────────────────────────────────────
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Labeled Transactions",
           f"{counts.get('total_labeled', 0):,}")
mc2.metric("Illicit",
           f"{counts.get('illicit', 0):,}",
           f"{counts.get('illicit', 0) / max(counts.get('total_labeled', 1), 1):.1%}")
mc3.metric("F1 Score (macro)",
           f"{cv.get('f1_macro', {}).get('mean', 0):.3f}",
           f"± {cv.get('f1_macro', {}).get('std', 0):.3f}")
mc4.metric("Precision (macro)",
           f"{cv.get('precision_macro', {}).get('mean', 0):.3f}",
           f"± {cv.get('precision_macro', {}).get('std', 0):.3f}")
mc5.metric("Recall (macro)",
           f"{cv.get('recall_macro', {}).get('mean', 0):.3f}",
           f"± {cv.get('recall_macro', {}).get('std', 0):.3f}")

st.divider()

col_l, col_r = st.columns(2)

# ── CV metrics bar chart ───────────────────────────────────────────────────────
with col_l:
    st.subheader("5-Fold Cross-Validation Metrics")

    metrics_data = []
    for m_key, avg_label, weighted_label in [
        ("macro",    "Macro",    "Macro avg"),
        ("weighted", "Weighted", "Weighted avg"),
    ]:
        for metric in ["precision", "recall", "f1"]:
            full_key = f"{metric}_{m_key}"
            v = cv.get(full_key, {})
            metrics_data.append({
                "Metric":  f"{metric.capitalize()} ({avg_label})",
                "Mean":    v.get("mean", 0),
                "Std":     v.get("std", 0),
            })

    mdf = pd.DataFrame(metrics_data)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mdf["Metric"], y=mdf["Mean"],
        error_y=dict(type="data", array=mdf["Std"].tolist(), visible=True),
        marker_color="#3b82f6",
        text=mdf["Mean"].round(3),
        textposition="outside",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="Random baseline")
    fig.update_layout(
        height=380, margin=dict(t=20, b=0),
        yaxis=dict(range=[0, 1.05], title="Score"),
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Confusion matrix ───────────────────────────────────────────────────────────
with col_r:
    st.subheader("Confusion Matrix (Full Training Set)")
    cm = data.get("confusion_matrix", [[0, 0], [0, 0]])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Licit", "Actual Illicit"],
        columns=["Predicted Licit", "Predicted Illicit"],
    )
    fig2 = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig2.update_layout(height=380, margin=dict(t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Feature importances ────────────────────────────────────────────────────────
st.divider()
st.subheader("Top-20 Feature Importances (Elliptic model)")
st.caption(
    "Elliptic's features are anonymised (f1–f166). "
    "The local features (f2–f94) correspond conceptually to our hand-crafted "
    "transaction-level features; f95–f166 add neighbourhood graph aggregates."
)

fi = data.get("feature_importances", [])
if fi:
    n_feats = data.get("n_features", len(fi))
    offset  = 2 if "local" in key else 2
    feat_names = [f"f{i}" for i in range(offset, offset + len(fi))]
    fi_pairs = sorted(zip(feat_names, fi), key=lambda x: -x[1])[:20]
    fi_df = pd.DataFrame(fi_pairs, columns=["Feature", "Importance"])

    fig3 = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Oranges",
        text=fi_df["Importance"].round(4),
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(
        height=500, margin=dict(t=20, b=0),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Comparison note ────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖  How to interpret these results for your dissertation"):
    st.markdown("""
### Methodology Validation

This evaluation answers: *"When our Random Forest approach is applied to a dataset
with **known ground-truth labels**, how well does it perform?"*

**Key points to document:**

1. **Ground truth**: Elliptic's labels were assigned by blockchain analytics experts
   who traced actual illicit activity (ransomware payments, darknet sales, fraud).
   This is the gold standard for Bitcoin AML research.

2. **Feature set difference**: Our production model uses 18 hand-crafted features
   derived from FATF red-flag indicators. Elliptic provides 166 anonymised features
   (local + neighbourhood graph). The local-only subset (f2–f94) is most comparable
   to our approach.

3. **Interpretation of results**: An F1 score above ~0.80 on the Elliptic benchmark
   demonstrates that tree-based classifiers on transaction-level features are
   capable of detecting illicit activity at a high rate. This validates the
   methodology independently of the pseudo-labelling step.

4. **Limitations**: Elliptic transactions are from 2011–2019; our sample may be
   from a different time window. Class imbalance (~2% illicit) makes recall
   particularly important for operational use.

5. **Cite as**: Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin:
   Experimenting with Graph Convolutional Networks for Financial Forensics.*
   FATML Workshop, KDD 2019.
    """)
