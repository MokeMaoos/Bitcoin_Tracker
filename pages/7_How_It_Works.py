"""
pages/7_How_It_Works.py — Interactive pipeline explainer for non-technical readers.

Walks through every stage of the detection pipeline with plain-English
explanations, key code snippets, and interactive demos.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_utils import RULE_META, load_labeled, load_model_bundle

st.set_page_config(
    page_title="How It Works · Bitcoin Tracker",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠  How It Works — Pipeline Explained")
st.caption(
    "A plain-English, interactive walk-through of the Bitcoin illicit transaction "
    "detection pipeline (BMAN31260). No prior knowledge of machine learning or "
    "blockchain required."
)

# ── Top-level pipeline diagram ─────────────────────────────────────────────────
st.subheader("The Pipeline at a Glance")
st.graphviz_chart("""
digraph pipeline {
    rankdir=LR
    graph [bgcolor=transparent splines=ortho]
    node  [fontname="Helvetica" fontsize=13 style="filled,rounded" shape=box]
    edge  [fontname="Helvetica" fontsize=10 color="#6b7280"]

    Raw  [label="📁  Raw Data\n4 CSV files" fillcolor="#fef9c3" color="#ca8a04"]
    S01  [label="① Load &\nValidate"         fillcolor="#dbeafe" color="#2563eb"]
    S02  [label="② Feature\nEngineering"     fillcolor="#dbeafe" color="#2563eb"]
    S03  [label="③ Rule-Based\nLabelling"    fillcolor="#fce7f3" color="#db2777"]
    S04a [label="④a  Clustering\n(K-Means / DBSCAN)" fillcolor="#dcfce7" color="#16a34a"]
    S04b [label="④b  Classification\n(Random Forest)" fillcolor="#dcfce7" color="#16a34a"]
    S05  [label="⑤  Live\nPrediction"        fillcolor="#f3e8ff" color="#7c3aed"]

    F1   [label="transaction_features.csv" shape=note style=filled fillcolor="#fefce8" color="#a16207" fontsize=11]
    F2   [label="labeled_features.csv"     shape=note style=filled fillcolor="#fefce8" color="#a16207" fontsize=11]
    F3   [label="model.pkl\ncluster_assignments.csv" shape=note style=filled fillcolor="#fefce8" color="#a16207" fontsize=11]

    Raw  -> S01 -> S02 -> F1 -> S03 -> F2 -> S04a
                                            F2 -> S04b
    S04a -> F3
    S04b -> F3
    F3   -> S05
}
""", use_container_width=True)

st.info(
    "**How to use this page:** each tab below covers one stage of the pipeline. "
    "Click through them in order — each stage feeds its output into the next."
)

st.divider()

# ── Per-stage tabs ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "① Data Loading",
    "② Feature Engineering",
    "③ Rule-Based Labelling",
    "④ Model Training",
    "⑤ Live Prediction",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("① Data Load & Validation")
    st.markdown("""
The first step simply reads the four raw CSV files and checks them for
missing values, duplicates, and obvious data quality issues.
No data is transformed here — this step just confirms the pipeline has
everything it needs before any real work begins.
""")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("What data does the pipeline use?")
        st.markdown("""
| File | Contents |
|------|----------|
| `Transactions.csv` | One row per transaction — block number, transaction hash, wallet ID |
| `Trans_Detail.csv` | One row per input/output of each transaction — addresses and BTC values |
| `Address_Info.csv` | Known Bitcoin addresses in the dataset |
| `Wallet_Info.csv`  | Wallet-level aggregates — total sent, number of addresses, tx count |
""")

    with col_b:
        st.subheader("Why four separate files?")
        st.markdown("""
Bitcoin transactions are naturally **graph-shaped**: a single transaction can
have many input addresses (who is paying) and many output addresses (who is
receiving). Storing this in a flat table would create huge amounts of
duplication, so the dataset is split into a header table (`Transactions.csv`)
and a detail table (`Trans_Detail.csv`) — a standard database normalisation
pattern.

`Address_Info.csv` lets the pipeline identify *fresh* addresses — ones that
have never appeared before — which is a key illicit indicator (see Step ③).
""")

    st.subheader("Key code — file registry")
    st.code("""
# 01_load.py
FILES = {
    "transactions": "Transactions.csv",   # 1 row per tx
    "tx_details":   "Trans_Detail.csv",   # 1 row per input/output
    "address_info": "Address_Info.csv",   # known addresses
    "wallet_info":  "Wallet_Info.csv",    # wallet aggregates
}

for key, fname in FILES.items():
    dfs[key] = pd.read_csv(os.path.join(DATA_DIR, fname))
""", language="python")

    st.success("**Output:** four validated DataFrames in memory — no CSV files written at this stage.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("② Feature Engineering")
    st.markdown("""
Raw Bitcoin transactions are just lists of addresses and BTC amounts.
Machine-learning models need *numbers*. This step converts each transaction
into **18 numeric features** that capture its structural and economic
properties — the things that distinguish normal commerce from money-laundering
techniques like CoinJoin mixing, peeling chains, and fund layering.
""")

    st.subheader("The 18 features")

    feat_tab1, feat_tab2, feat_tab3 = st.tabs([
        "📏 Basic statistics",
        "🔍 Structural indicators",
        "👛 Wallet context",
    ])

    with feat_tab1:
        st.markdown("""
| Feature | What it measures | Why it matters |
|---------|-----------------|----------------|
| `num_inputs` | Unique input addresses | Many inputs = many wallets pooling funds → mixing signal |
| `num_outputs` | Unique output addresses | Many outputs = funds being dispersed to many recipients |
| `total_input` | Sum of all input values (BTC) | Raw transaction size |
| `total_output` | Sum of all output values (BTC) | Should be ≤ input (difference = miner fee) |
| `mean_output` | Average output value | Low mean + equal values → structuring |
| `std_output` | Std dev of output values | Near-zero std → all outputs equal → CoinJoin |
| `max_output` | Largest single output | High fraction of total → peeling chain |
| `min_output` | Smallest single output | |
| `io_ratio` | `num_inputs / num_outputs` | High ratio = many payers, few recipients |
| `value_ratio` | `total_input / total_output` | Should be ~1.0; large deviations are suspicious |
""")

    with feat_tab2:
        st.markdown("### Equal-Value Flag — CoinJoin / Structuring Detection")
        st.markdown("""
CoinJoin is a technique where multiple parties combine their transactions so
that all outputs have the *exact same value*, making it impossible to trace
which input funded which output. The pipeline detects this by computing the
**coefficient of variation** (σ/μ) of output values. If outputs are nearly
identical, σ/μ is tiny.
""")
        st.code("""
# 02_feature.py
feat["equal_value_flag"] = (
    feat["std_output"] / (feat["mean_output"].abs() + 1e-9) < 0.01
).astype(int)
# σ/μ < 1% → all outputs are within 1% of each other → suspicious
""", language="python")

        st.markdown("---")
        st.markdown("### Peeling Chain Flag — Layering Detection")
        st.markdown("""
A peeling chain is a money-laundering layering technique. The attacker sends
most of a transaction's value to a *new* address (the "peel") and keeps a
small "change" output — then repeats. Each hop makes the trail harder to
follow. The pipeline detects this by looking for transactions where one output
dominates (≥ 80% of total value) and at least one other output exists.
""")
        st.code("""
# 02_feature.py
feat["peeling_flag"] = (
    (feat["max_output"] / feat["total_output"].clip(lower=1e-9) >= 0.8)
    & (feat["num_outputs"] >= 2)           # needs at least one "change" output
).astype(int)
""", language="python")

        st.markdown("---")
        st.markdown("### CoinJoin Strength — Composite Mixing Score")
        st.markdown("""
Rather than a binary flag, this gives a **continuous score from 0 to 1**
by combining three sub-signals. A score ≥ 0.5 is treated as a strong mixing
indicator by the rule engine.
""")
        st.code("""
# 02_feature.py
feat["coinjoin_strength"] = (
    0.6 * feat["equal_value_flag"]                          # dominant signal
    + 0.2 * (feat["num_inputs"]  / 10).clip(upper=1.0)     # many payers
    + 0.2 * (feat["num_outputs"] / 10).clip(upper=1.0)     # many recipients
).round(4)
# Weight breakdown: equal-value outputs are the strongest CoinJoin signal (60%),
# then large fan-in and fan-out each add 20%.
""", language="python")

        st.markdown("---")
        st.markdown("### Value Entropy — Anomalous Distribution Detection")
        st.markdown("""
Shannon entropy measures *how evenly spread* the output values are.
Legitimate transactions tend to have simple, predictable output patterns (e.g.
one payment + one change). Illicit transactions often show *unusual*
distributions — either all equal (CoinJoin) or wildly irregular (layering).
The pipeline flags entropy ≥ 3.0 bits as anomalous.
""")
        st.code("""
# 02_feature.py  — fully vectorised, no Python loops
total_per_tx = td_ov.groupby(KEYS)["Output_Value"].transform("sum")
p = (td_ov["Output_Value"] / total_per_tx.clip(lower=1e-300)).clip(lower=1e-300)
td_ov["entropy_term"] = -(p * np.log2(p))   # Shannon entropy: -Σ p·log₂(p)
value_entropy = td_ov.groupby(KEYS)["entropy_term"].sum()
""", language="python")

        st.markdown("---")
        st.markdown("### Fresh Output Ratio — Address Churn / Layering Detection")
        st.markdown("""
Illicit actors constantly generate *new* Bitcoin addresses to avoid address
reuse, which would make their activity trivial to track.
This feature measures what fraction of a transaction's output addresses
have never been seen in `Address_Info.csv` before. A high ratio means the
outputs are going to addresses the dataset has never encountered — suspicious.
""")
        st.code("""
# 02_feature.py
known = set(ai["ADR"].dropna().values)   # all known addresses
td_fr["is_fresh"] = ~td_fr["Output_Address"].isin(known)
fresh_ratio = td_fr.groupby(KEYS)["is_fresh"].mean()
# 1.0 = ALL outputs go to brand-new addresses
# 0.0 = ALL outputs go to previously seen addresses
""", language="python")

    with feat_tab3:
        st.markdown("""
Three wallet-level aggregates are joined from `Wallet_Info.csv`:

| Feature | Meaning | Note |
|---------|---------|------|
| `wallet_total_tx` | Total transactions the wallet has made | High-volume wallets may be exchanges (legitimate) |
| `wallet_total_sent` | Total BTC sent by the wallet | Very high values can indicate exchange or mixing service |
| `wallet_size` | Number of unique addresses the wallet controls | Large wallets are often exchanges |

**Important — CoinJoin suppression:**
Wallet features are deliberately set to `NaN` for transactions with
`coinjoin_strength > 0.1`. This is because CoinJoin transactions pool funds
from *multiple* wallets, so the wallet ID in `Transactions.csv` may not
correspond to the true sender — using those features would corrupt the model.
""")
        st.code("""
# 02_feature.py
cj_mask = feat["coinjoin_strength"] > 0.1
feat.loc[cj_mask, ["wallet_total_tx", "wallet_total_sent", "wallet_size"]] = np.nan
# The model's imputer will fill these with the dataset median at training time.
""", language="python")

    st.success("**Output:** `outputs/transaction_features.csv` — one row per transaction, 18 numeric columns.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RULE-BASED LABELLING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("③ Rule-Based Pseudo-Labelling")
    st.markdown("""
The dataset has **no ground-truth labels** — we don't know in advance which
transactions are truly illicit. This is a classic problem in financial crime
detection: real labels are rare, expensive, and often confidential.

The solution is **weak supervision**: instead of labelling by hand, the
pipeline applies six rules derived directly from the
[FATF (Financial Action Task Force)](https://www.fatf-gafi.org/) red-flag
methodology for virtual asset crime. Each rule that "fires" adds 1 to a
**suspicion score**, and the score determines the pseudo-label:
""")

    sc1, sc2, sc3 = st.columns(3)
    sc1.error("**Score ≥ 2 → Illicit (1)**\nTwo or more red flags fired")
    sc2.warning("**Score = 1 → Uncertain (−1)**\nOne red flag — ambiguous signal")
    sc3.success("**Score = 0 → Licit (0)**\nNo red flags fired")

    st.markdown("---")
    st.subheader("The Six FATF Red-Flag Rules")

    for rule_key, meta in RULE_META.items():
        with st.expander(f"{meta['icon']}  **{meta['label']}** — threshold: `{meta['threshold']}`"):
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                if rule_key == "rule_equal_value":
                    st.markdown("""
**What it detects:** CoinJoin mixing and structured payments.

When all outputs of a transaction have nearly equal value, it strongly suggests
a *CoinJoin* — a technique where multiple parties contribute inputs and
receive equal outputs so their individual flows are indistinguishable.
Structuring (deliberately splitting large sums into small equal payments to
avoid reporting thresholds) produces the same pattern.
""")
                elif rule_key == "rule_peeling":
                    st.markdown("""
**What it detects:** Peeling chains (layering).

In a peeling chain, the attacker sends a large fraction to a new address and
keeps a small "change" amount, then repeats. Each hop creates a new link in
the chain, making the transaction trail long and difficult to trace — a classic
*layering* technique in money laundering.
""")
                elif rule_key == "rule_high_fan_out":
                    st.markdown("""
**What it detects:** Fund dispersion (placement → layering).

Sending funds to 10 or more distinct output addresses in a single transaction
is unusual for ordinary commerce. It is a hallmark of *payment processors for
illicit services* (e.g. darknet markets) or automated fund distribution across
many mule accounts.
""")
                elif rule_key == "rule_fresh_outputs":
                    st.markdown("""
**What it detects:** Address churn / layering.

A high proportion of *never-before-seen* output addresses means funds are
moving to freshly generated wallets. Legitimate users typically reuse addresses
or send to known counterparties. Illicit actors generate new addresses for each
hop to disrupt chain analysis.
""")
                elif rule_key == "rule_mixing":
                    st.markdown("""
**What it detects:** Mixer / tumbler involvement.

The composite CoinJoin strength score combines equal-value outputs with
high fan-in and fan-out to give a continuous signal of how "mixer-like"
a transaction is. A score above 0.5 means the transaction exhibits multiple
structural properties of a mixing service simultaneously.
""")
                elif rule_key == "rule_high_entropy":
                    st.markdown("""
**What it detects:** Anomalous value distribution.

Shannon entropy measures the spread of output values. Ordinary transactions
(a payment + change) have *low* entropy (predictable values). Very high
entropy means outputs are spread across many wildly different values —
consistent with automated laundering tools that deliberately randomise
output amounts to defeat pattern matching.
""")

            with rc2:
                st.markdown(f"**FATF concern**")
                st.info(meta["fatf"])
                st.markdown(f"**Decision threshold**")
                st.code(meta["threshold"])

    st.markdown("---")
    st.subheader("Why pseudo-labels instead of real labels?")
    st.markdown("""
This approach is called **weakly supervised learning**. It has trade-offs:

- ✅ No manual labelling required — scalable to millions of transactions
- ✅ Rules are directly grounded in expert FATF guidance — defensible in a compliance context
- ✅ Uncertain cases (score = 1) are *excluded* from training, keeping the training set clean
- ⚠️ Labels are noisy — some licit transactions will resemble illicit ones and vice versa
- ⚠️ Novel laundering techniques not covered by the six rules will be missed
- ⚠️ Legitimate high-volume services (exchanges, mixers-as-a-service) may be over-flagged
""")

    st.subheader("Key code — labelling logic")
    st.code("""
# 03_label.py
df["rule_equal_value"]   = df["equal_value_flag"].astype(int)
df["rule_peeling"]       = df["peeling_flag"].astype(int)
df["rule_high_fan_out"]  = (df["num_outputs"]        >= 10).astype(int)
df["rule_fresh_outputs"] = (df["fresh_output_ratio"] >= 0.8).astype(int)
df["rule_mixing"]        = (df["coinjoin_strength"]  >= 0.5).astype(int)
df["rule_high_entropy"]  = (df["value_entropy"]      >= 3.0).astype(int)

df["suspicion_score"] = df[[rule columns]].sum(axis=1)   # 0 – 6

df["pseudo_label"] = np.where(
    df["suspicion_score"] >= 2,   1,   # illicit
    np.where(df["suspicion_score"] == 0, 0, -1)  # licit / uncertain
)
""", language="python")

    st.success("**Output:** `outputs/labeled_features.csv` — the feature matrix plus `suspicion_score` and `pseudo_label` columns.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("④ Model Training")
    st.markdown("""
Step 4 trains **two complementary models** — one unsupervised and one
supervised — and saves both to disk for use by the prediction interface.
""")

    model_tab1, model_tab2 = st.tabs(["🔵 Unsupervised Clustering", "🌲 Supervised Classification"])

    with model_tab1:
        st.subheader("K-Means Clustering (k = 3)")
        st.markdown("""
K-Means partitions *all* transactions (including uncertain ones) into 3 groups
based purely on their 18 features — no labels used. The choice of k = 3
mirrors the expected structure: licit, ambiguous, and illicit clusters.

**Why cluster at all, if we have a classifier?**
Clustering is *unsupervised* — it makes no assumptions about labels, so it can
surface natural groupings that the rule-based labels may have missed. It also
lets analysts visually inspect the transaction space (see the Clusters page)
and spot outliers without relying on the rule engine.
""")
        st.code("""
# 04_train.py
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = km.fit_predict(X_scaled)   # all transactions, scaled features
""", language="python")

        st.subheader("DBSCAN (density-based outlier detection)")
        st.markdown("""
DBSCAN groups transactions that are *dense* in feature space and marks
isolated transactions as **outliers (label = −1)**. Unlike K-Means, it
does not require specifying k and can find clusters of arbitrary shape.

Because DBSCAN's memory usage is O(n²), it runs on a **random 10,000-
transaction sample** and remaining transactions receive label −2 (not sampled).
Transactions marked as DBSCAN outliers are structurally anomalous — they don't
fit neatly into any cluster, which is itself a suspicious signal.
""")
        st.code("""
# 04_train.py  — sampled to avoid O(n²) memory usage
db = DBSCAN(eps=1.5, min_samples=5)
db_sample_lbls = db.fit_predict(X_sample)
# -1 = outlier (noise point)
# -2 = not included in the 10k sample
""", language="python")

    with model_tab2:
        st.subheader("Random Forest Classifier")
        st.markdown("""
The Random Forest is trained on the **pseudo-labeled** subset (transactions
with `pseudo_label` ∈ {0, 1} — uncertain rows are excluded). It learns which
combinations of features best separate licit from illicit transactions.
""")

        why1, why2, why3 = st.columns(3)
        with why1:
            st.info("**Why Random Forest?**\n\nHandles mixed numeric features without scaling worries, "
                    "natively supports `class_weight='balanced'` for the skewed label distribution, "
                    "and produces calibrated `predict_proba` outputs usable as a risk score.")
        with why2:
            st.info("**Why `class_weight='balanced'`?**\n\nIllicit transactions are a small minority "
                    "of the dataset. Without balancing, the model would optimise for predicting "
                    "everything as licit (the majority class) and achieve high accuracy while "
                    "catching almost nothing.")
        with why3:
            st.info("**Why 5-fold cross-validation?**\n\nA single train/test split would give a noisy "
                    "estimate of performance. Stratified 5-fold CV ensures every fifth of the "
                    "data serves as the test set once, giving a much more robust evaluation.")

        st.subheader("The sklearn Pipeline")
        st.code("""
# 04_train.py
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # fill NaN wallet features
    ("scaler",  StandardScaler()),                   # zero mean, unit variance
    ("clf",     RandomForestClassifier(
        n_estimators=200,      # 200 trees — enough for stable feature importances
        max_depth=10,          # prevents overfitting on noisy pseudo-labels
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,             # use all CPU cores
    )),
])

# 5-fold stratified cross-validation before final fit
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(pipeline, X, y, cv=cv,
                        scoring=["precision_macro", "recall_macro", "f1_macro"])

pipeline.fit(X, y)   # final model trained on all labeled data
""", language="python")

        st.subheader("What the model actually learns")
        st.markdown("""
The Random Forest splits the feature space repeatedly, asking questions like:
- *"Is `coinjoin_strength` > 0.5 AND `num_outputs` > 12?"*
- *"Is `peeling_flag` = 1 AND `fresh_output_ratio` > 0.7?"*

Each of the 200 trees votes, and the fraction of trees voting "illicit"
becomes the **illicit probability** shown in the Transaction Detail page.
The Overview dashboard's **Top Feature Importances** chart shows which
features the forest found most discriminating.
""")

        # Live feature importances if model is available
        bundle = load_model_bundle()
        if bundle is not None:
            import plotly.express as px
            import pandas as pd
            clf        = bundle["model"].named_steps["clf"]
            feat_names = bundle["features"]
            fi         = sorted(zip(feat_names, clf.feature_importances_),
                                key=lambda x: x[1], reverse=True)
            fi_df = pd.DataFrame(fi, columns=["Feature", "Importance"])
            fig   = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Greens",
                text=fi_df["Importance"].round(3),
                title="Feature importances from your trained model",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=420, margin=dict(t=40, b=0),
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline to see your model's actual feature importances here.")

    st.success(
        "**Outputs:** `outputs/model.pkl` (the trained Pipeline + metadata) and "
        "`outputs/cluster_assignments.csv` (K-Means + DBSCAN labels for every transaction)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("⑤ Live Prediction")
    st.markdown("""
Once the model is trained, the app can score *any* transaction in the dataset
on demand — even if it was in the uncertain (−1) set that was excluded from
training. It can also **re-compute features live** from the raw detail rows,
so the prediction reflects the actual on-chain data rather than the batch-
computed features.
""")

    lp_col1, lp_col2 = st.columns(2)
    with lp_col1:
        st.subheader("How a prediction works")
        st.markdown("""
1. The user enters a **Block No** and **Transaction Hash** on the Transaction
   Detail page.
2. The app looks up the raw input/output rows from `Trans_Detail.csv`.
3. `run_live_prediction()` in `ui_utils.py` recomputes all 18 features from
   scratch for that single transaction.
4. The 18 features are fed into the trained `Pipeline`, which imputes missing
   values, scales, and runs them through the 200 Random Forest trees.
5. `predict_proba()` returns the fraction of trees that voted "illicit" —
   this is the **illicit probability** (0 – 100%).
6. Simultaneously, the six FATF rules are re-evaluated on the live features
   and displayed as fired/not-fired cards with the underlying raw data.
""")

    with lp_col2:
        st.subheader("Key code — live feature computation")
        st.code("""
# ui_utils.py — run_live_prediction()
grp = td[(td["Block_No"] == block_no) & (td["Trans_No"] == trans_no)]

ov           = grp["Output_Value"].dropna().values
num_inputs   = grp["Input_Address"].nunique()
num_outputs  = grp["Output_Address"].nunique()
cj_str       = compute_coinjoin_strength(ov, num_inputs, num_outputs)

feat = {
    "num_inputs":         num_inputs,
    "num_outputs":        num_outputs,
    "total_output":       float(ov.sum()),
    "equal_value_flag":   is_equal_value(ov),   # σ/μ < 1%
    "peeling_flag":       is_peeling(ov),        # max ≥ 80% of total
    "value_entropy":      compute_value_entropy(ov),
    "fresh_output_ratio": compute_fresh_output_ratio(oa, ai),
    "coinjoin_strength":  cj_str,
    # ... (all 18 features)
}

prob = float(bundle["model"].predict_proba(feat_df)[0][1])
# [0][1] → first (only) row, class-1 (illicit) probability
""", language="python")

    st.markdown("---")
    st.subheader("Interpreting the output")

    out_col1, out_col2, out_col3 = st.columns(3)
    with out_col1:
        st.error("**≥ 50% illicit probability**\n\nModel verdict: **ILLICIT**\n\n"
                 "The majority of the 200 decision trees voted that this transaction's "
                 "feature profile matches the illicit training examples.")
    with out_col2:
        st.warning("**30 – 50%**\n\nBorderline — treat as **uncertain**\n\n"
                   "The forest is split. A high suspicion score alongside a borderline "
                   "probability strengthens the case for manual review.")
    with out_col3:
        st.success("**< 30% illicit probability**\n\nModel verdict: **LICIT**\n\n"
                   "The feature profile is consistent with ordinary transaction patterns. "
                   "Still review if any rules fired — a single rule flag is non-zero evidence.")

    st.markdown("---")
    st.subheader("External enrichment (Transaction Detail page)")
    st.markdown("""
The Transaction Detail page adds three optional data sources that the model
does not use (they are not available at training time), but which a human
analyst can use to verify the model's verdict:

| Source | What it adds | Why useful |
|--------|-------------|------------|
| **Mempool.space** | Live fee, size, confirmation status | Fee rate anomalies; confirms the transaction is real |
| **Bitcoin Abuse DB** | Crowd-sourced abuse reports for output addresses | Directly links addresses to known scam/ransomware campaigns |
| **Claude AI** | Natural-language forensic narrative | Synthesises all signals into plain English for a compliance report |
""")

    st.info(
        "**Tip:** if the model says ILLICIT but no rules fired, it has detected a pattern "
        "in the *combination* of features that doesn't individually cross any single "
        "threshold — this is the key advantage of a supervised model over pure rule engines."
    )

    st.markdown("---")
    st.subheader("End-to-end summary")
    st.code("""
# Run the full pipeline from scratch:
python run_pipeline.py

# Then launch the dashboard:
streamlit run app.py

# Or predict a single transaction from the command line:
python 05_predict.py --tx 705592 d8bbd0d759ab17f3...
python 05_predict.py --wallet 12345
""", language="bash")
