"""
pages/6_Evaluate.py — Elliptic++ ground-truth benchmark evaluation results.

Displays the output of 06_evaluate.py: precision/recall/F1 of the
LightGBM / Random Forest methodology validated against the Elliptic++
dataset (Elmougy and Liu, 2023) at both transaction and actor levels.
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

st.title("📋  Elliptic++ Ground-Truth Evaluation")
st.caption(
    "Validates the detection methodology against the "
    "[Elliptic++ dataset](https://github.com/git-disl/EllipticPlusPlus) "
    "(Elmougy and Liu, 2023) — transaction-level and actor-level labelled data."
)

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON = os.path.join(_ROOT, "outputs", "elliptic_eval.json")

# Locate the Elliptic++ data — sibling directory takes priority
_SIBLING_DIR = os.path.normpath(os.path.join(_ROOT, "..", "EllipticPlusPlus-main"))
_LEGACY_DIR  = os.path.join(_ROOT, "data", "elliptic_pp")
_TX_CHECK    = os.path.join("Transactions Dataset", "txs_features.csv")

def _is_lfs(path):
    try:
        with open(path, "rb") as f:
            return f.read(27) == b"version https://git-lfs.git"
    except OSError:
        return False

_sibling_tx = os.path.join(_SIBLING_DIR, _TX_CHECK)
_legacy_tx  = os.path.join(_LEGACY_DIR,  _TX_CHECK)

if os.path.exists(_sibling_tx) and not _is_lfs(_sibling_tx):
    has_data = True
elif os.path.exists(_legacy_tx) and not _is_lfs(_legacy_tx):
    has_data = True
else:
    has_data = False

# Detect the specific case: files present but are LFS pointers
_lfs_present = (
    (os.path.exists(_sibling_tx) and _is_lfs(_sibling_tx)) or
    (os.path.exists(_legacy_tx)  and _is_lfs(_legacy_tx))
)

has_results = os.path.exists(RESULTS_JSON)

# ── Data not yet downloaded / LFS pointers ────────────────────────────────────
if not has_data:
    if _lfs_present:
        st.error("### Git LFS pointer files found — real data not yet downloaded")
        st.markdown(f"""
The `EllipticPlusPlus-main/` folder you downloaded contains **Git LFS pointer
files**, not actual CSV data.  GitHub ZIP downloads never include LFS objects.

**To fetch the real data (Option A — Git LFS):**
```bash
brew install git-lfs        # macOS; see https://git-lfs.com for other OS
git lfs install
cd "{_SIBLING_DIR}"
git init
git remote add origin https://github.com/git-disl/EllipticPlusPlus.git
git lfs pull origin
```

**Alternatively (Option B — Google Drive):**
1. Go to the [Google Drive folder](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l)
2. Download **Transactions Dataset** and **Actors Dataset**
3. Copy the CSVs into `{_SIBLING_DIR}/`
   (keeping the same sub-folder names)

Then run:
```bash
python 06_evaluate.py
```
and refresh this page.
        """)
    else:
        st.warning("### Elliptic++ dataset not found")
        st.markdown(f"""
**To run the evaluation (one-time setup):**

**Option A — Git LFS clone (recommended):**
```bash
brew install git-lfs
git lfs install
git clone https://github.com/git-disl/EllipticPlusPlus.git \\
    "{_SIBLING_DIR}"
```

**Option B — Google Drive download:**
1. Go to the [Google Drive folder](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l)
2. Download the **Transactions Dataset** and **Actors Dataset** folders
3. Place them as:
   ```
   {_SIBLING_DIR}/Transactions Dataset/txs_features.csv
   {_SIBLING_DIR}/Transactions Dataset/txs_classes.csv
   {_SIBLING_DIR}/Actors Dataset/wallets_features.csv
   {_SIBLING_DIR}/Actors Dataset/wallets_classes.csv
   ```

Then run `python 06_evaluate.py` and refresh this page.
        """)
    st.stop()

if not has_results:
    st.info(
        "Elliptic++ data found but evaluation not yet run.  \n"
        "Run `python 06_evaluate.py` from the terminal, then refresh."
    )
    st.stop()

# ── Load results ───────────────────────────────────────────────────────────────
with open(RESULTS_JSON) as f:
    res = json.load(f)

source = res.get("dataset_source", "Elliptic++")
clf    = res.get("classifier", "unknown")
st.success(f"Results loaded — **{source}**  ·  classifier: `{clf}`")
st.divider()

# ── Level selector ─────────────────────────────────────────────────────────────
level_options = []
if res.get("transaction_level"):
    level_options.append("Transaction level (203k txs, 183 features)")
if res.get("actor_level"):
    level_options.append("Actor / wallet level (823k addresses, 56 features)")

if not level_options:
    st.error("No results found in the JSON for either level.  "
             "Re-run `python 06_evaluate.py`.")
    st.stop()

chosen = st.radio("Evaluation level", level_options, horizontal=True)
key    = "transaction_level" if "Transaction" in chosen else "actor_level"
data   = res[key]


def _render_level(data: dict):
    cv     = data.get("cv_metrics", {})
    counts = data.get("label_counts", {})

    # ── Overview metrics ───────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Labelled Transactions",
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

    # ── CV metrics bar chart ───────────────────────────────────────────────────
    with col_l:
        st.subheader("5-Fold Cross-Validation Metrics")
        metrics_data = []
        for m_key, avg_label in [("macro", "Macro"), ("weighted", "Weighted")]:
            for metric in ["precision", "recall", "f1"]:
                full_key = f"{metric}_{m_key}"
                v = cv.get(full_key, {})
                metrics_data.append({
                    "Metric": f"{metric.capitalize()} ({avg_label})",
                    "Mean":   v.get("mean", 0),
                    "Std":    v.get("std", 0),
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

    # ── Confusion matrix ───────────────────────────────────────────────────────
    with col_r:
        st.subheader("Confusion Matrix (Full Training Set)")
        cm = data.get("confusion_matrix", [[0, 0], [0, 0]])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Licit", "Actual Illicit"],
            columns=["Predicted Licit", "Predicted Illicit"],
        )
        fig2 = px.imshow(
            cm_df, text_auto=True,
            color_continuous_scale="Blues", aspect="auto",
        )
        fig2.update_layout(height=380, margin=dict(t=20, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Feature importances ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Top-20 Feature Importances")
    fi = data.get("feature_importances", [])
    if fi:
        feat_names = [f"f{i}" for i in range(1, len(fi) + 1)]
        fi_pairs   = sorted(zip(feat_names, fi), key=lambda x: -x[1])[:20]
        fi_df      = pd.DataFrame(fi_pairs, columns=["Feature", "Importance"])
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


_render_level(data)

# ── Interpretation ─────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖  How to interpret these results for your dissertation"):
    st.markdown("""
### Methodology Validation

This evaluation answers: *"When our classifier is applied to a dataset with
**known ground-truth labels**, how well does it perform?"*

**Key points to document:**

1. **Ground truth**: Elliptic++ labels were assigned by blockchain analytics
   experts who traced actual illicit activity (ransomware, darknet markets,
   fraud).  This is the gold standard for Bitcoin AML research.

2. **Two evaluation levels**:
   - *Transaction level*: 183 features per transaction, ~46k labelled rows
     (4.5k illicit, 42k licit).  Most directly comparable to our FATF-derived
     feature set.
   - *Actor level*: 56 features per wallet address.  Tests whether the same
     classifier architecture transfers to the address-level perspective.

3. **Interpretation of results**: An F1 score above ~0.80 at either level
   demonstrates that tree-based classifiers on graph-derived and local features
   are capable of detecting illicit activity at a high rate, validating the
   methodological approach.

4. **Limitations**: Elliptic++ transactions are from 2011–2019; temporal drift
   relative to newer CUBiD data may affect direct performance comparisons.
   Class imbalance (~9% illicit at transaction level) makes recall particularly
   important for operational use.

5. **Cite as**: Elmougy, Y. and Liu, L. (2023). *Demystifying Fraudulent
   Transactions and Illicit Nodes in the Bitcoin Network for Financial
   Forensics.* Proceedings of the 29th ACM SIGKDD Conference on Knowledge
   Discovery and Data Mining (KDD 2023). doi:10.1145/3580305.3599803
    """)
