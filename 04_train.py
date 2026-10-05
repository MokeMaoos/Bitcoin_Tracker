"""
04_train.py — train the illicit-transaction detection model.

Staged pipeline (methodology §6.4):
    1. Unsupervised clustering (K-Means + DBSCAN) on all transactions
    2. Supervised classification on labelled transactions
       — LightGBM (primary, if installed) or Random Forest (fallback)
    3. 5-fold stratified cross-validation (precision, recall, F1)
    4. Save model bundle → outputs/model_<mode>.pkl
       Save cluster assignments → outputs/cluster_assignments.csv

Label modes (--label-mode):
    hybrid         — use confirmed labels where available, pseudo-labels elsewhere
                     (default when confirmed labels are present in the data)
    pseudo_only    — use only pseudo-labels (original behaviour; default when
                     no confirmed labels are present)
    confirmed_only — train exclusively on rows where label_type == "confirmed"
                     (warns if fewer than 100 illicit confirmed examples)

Usage:
    python 04_train.py
    python 04_train.py --label-mode confirmed_only
    python 04_train.py --label-mode pseudo_only
    python 04_train.py --label-mode hybrid
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.cluster         import KMeans, DBSCAN
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import classification_report
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline

try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    USE_LGBM = False

OUTPUT_DIR   = "outputs"
LABELED_CSV  = os.path.join(OUTPUT_DIR, "labeled_features.csv")
CLUSTER_CSV  = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")

_VALID_MODES = ("hybrid", "pseudo_only", "confirmed_only")
_MIN_ILLICIT_CONFIRMED = 100  # warn if confirmed_only has fewer illicit examples

# All 38 features produced by 02_feature.py
# Original 18 + 20 new features from unused CSVs
FEATURE_COLS = [
    # ── Output statistics ──────────────────────────────────────────────────────
    "num_inputs", "num_outputs", "total_input", "total_output",
    "mean_output", "std_output", "max_output", "min_output",
    "io_ratio", "value_ratio",
    # ── FATF rule signals ──────────────────────────────────────────────────────
    "equal_value_flag", "value_entropy", "peeling_flag",
    "fresh_output_ratio", "coinjoin_strength",
    # ── Graph topology ─────────────────────────────────────────────────────────
    "n_distinct_input_wallets", "n_distinct_output_wallets",
    # ── Wallet context (suppressed for CoinJoin tx) ────────────────────────────
    "wallet_total_tx", "wallet_total_sent", "wallet_size",
    "wallet_age", "wallet_sent_ratio", "wallet_final_balance", "wallet_balance_ratio",
    # ── Wallet temporal (from Wallet_Balance_Block) ────────────────────────────
    "wallet_balance_velocity", "wallet_balance_range", "wallet_balance_std",
    # ── Transaction metadata ───────────────────────────────────────────────────
    "tx_fee", "tx_fee_normalized", "lock_time_flag",
    "tx_hour", "tx_day_of_week", "is_night",
    # ── Output-address features (from Address_Info + Address_Balance_Block) ────
    "mean_out_addr_tx_count", "mean_out_addr_balance",
    "frac_out_addr_zero_balance", "mean_out_addr_age",
    "mean_out_addr_sent_ratio", "mean_out_addr_bal_velocity",
]

RANDOM_STATE = 42
# DBSCAN now runs on the full transaction set — the previous 10,000-row
# sampling restriction was dropped (it produced cluster sizes of ≤14 and
# left ~96 % of rows tagged with a -2 sentinel).  Memory and time are not
# constrained for this evaluation.
DBSCAN_EPS         = 1.5
DBSCAN_MIN_SAMPLES = 5
DBSCAN_ALGORITHM   = "ball_tree"  # O(n log n) for moderate dimensionality


def load_data(path: str, label_mode: str):
    """
    Load and filter the labeled feature matrix according to *label_mode*.

    Returns (all_tx, training_subset) where training_subset is the set of rows
    used for supervised learning.
    """
    df = pd.read_csv(path)

    has_confirmed = (
        "label_type" in df.columns
        and df["label_type"].eq("confirmed").any()
    )

    print(f"  Total transactions           : {len(df):,}")
    print(f"  Has confirmed labels         : {has_confirmed}")

    if label_mode == "hybrid":
        # All rows with a definitive label (confirmed or pseudo, but not uncertain)
        labeled = df[df["pseudo_label"] != -1].copy()

    elif label_mode == "pseudo_only":
        # Exclude confirmed rows entirely — use only FATF pseudo-labels
        if has_confirmed:
            labeled = df[
                (df["pseudo_label"] != -1) &
                (df.get("label_type", pd.Series("pseudo", index=df.index)) != "confirmed")
            ].copy()
        else:
            labeled = df[df["pseudo_label"] != -1].copy()

    elif label_mode == "confirmed_only":
        if not has_confirmed:
            print("[ERROR] --label-mode confirmed_only requested but no confirmed "
                  "labels are present in the data.")
            print("  Run 02b_enrich_labels.py and 03_label.py first.")
            sys.exit(1)
        labeled = df[df["label_type"] == "confirmed"].copy()
        n_ill = int((labeled["pseudo_label"] == 1).sum())
        if n_ill < _MIN_ILLICIT_CONFIRMED:
            print(f"[WARN] Only {n_ill} confirmed illicit examples available "
                  f"(threshold: {_MIN_ILLICIT_CONFIRMED}).  "
                  "Cross-validation metrics may be unreliable with so few positives.")

    else:
        print(f"[ERROR] Unknown label mode: {label_mode!r}.  "
              f"Choose from: {_VALID_MODES}")
        sys.exit(1)

    print(f"  Label mode                   : {label_mode}")
    print(f"  Training rows                : {len(labeled):,}")
    print(f"    Illicit (pseudo_label = 1) : {(labeled['pseudo_label'] == 1).sum():,}")
    print(f"    Licit   (pseudo_label = 0) : {(labeled['pseudo_label'] == 0).sum():,}")
    return df, labeled


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return only the FEATURE_COLS that actually exist in df."""
    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] {len(missing)} feature cols not found in data "
              f"(run 02_feature.py to regenerate): {missing[:5]}{'...' if len(missing)>5 else ''}")
    return present


def build_classifier():
    """Return a fitted-ready sklearn Pipeline using LightGBM or Random Forest."""
    if USE_LGBM:
        print("  Using LightGBM classifier")
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        # LightGBM handles NaN natively — imputer still needed for sklearn CV API
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     clf),
        ])
    else:
        print("  LightGBM not available — using Random Forest")
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])


def run_clustering(all_tx: pd.DataFrame, feat_cols: list,
                   imp: SimpleImputer, scaler=None):
    """
    Apply K-Means and DBSCAN to the full transaction set.

    DBSCAN was previously run on a 10,000-row sample due to a memory
    restriction; it is now run on the full feature matrix (eps=1.5,
    min_samples=5, ball_tree).  Every row in the output therefore has a
    real DBSCAN cluster id (or the noise label -1) — no -2 sentinel is
    written.
    """
    X_imp = imp.transform(all_tx[feat_cols])
    X_scaled = scaler.transform(X_imp) if scaler is not None else X_imp

    print("\n  Running K-Means (k=3)...")
    km        = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    print(f"  Running DBSCAN on full set "
          f"({len(X_scaled):,} rows, eps={DBSCAN_EPS}, "
          f"min_samples={DBSCAN_MIN_SAMPLES}, algorithm={DBSCAN_ALGORITHM!r})...")
    t0 = time.time()
    db        = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                       algorithm=DBSCAN_ALGORITHM, n_jobs=-1)
    db_labels = db.fit_predict(X_scaled)
    elapsed = time.time() - t0

    cluster_ids = sorted(set(db_labels) - {-1})
    n_clusters  = len(cluster_ids)
    n_noise     = int((db_labels == -1).sum())
    print(f"    DBSCAN clusters : {n_clusters}")
    print(f"    DBSCAN noise    : {n_noise:,}  ({n_noise/len(db_labels):.1%})")
    if n_clusters:
        sizes = pd.Series(db_labels[db_labels != -1]).value_counts().head(5)
        print("    Largest 5 clusters (id, size):")
        for cid, sz in sizes.items():
            print(f"      cluster {int(cid):>4}: {int(sz):>10,}")
    print(f"    DBSCAN runtime  : {elapsed:.1f} s")

    result = all_tx[["Block_No", "Trans_No"]].copy()
    result["cluster_kmeans"] = km_labels
    result["cluster_dbscan"] = db_labels
    return result, km


def train_supervised(labeled: pd.DataFrame, feat_cols: list):
    """
    Fit a preprocessing + classifier pipeline.
    Returns the fitted pipeline and the fitted imputer/scaler for clustering reuse.
    """
    X_raw = labeled[feat_cols]
    y     = labeled["pseudo_label"].values

    pipeline = build_classifier()

    print("\n  Cross-validating (5-fold stratified)...")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline, X_raw, y, cv=cv,
        scoring=["precision_macro", "recall_macro", "f1_macro"],
        return_train_score=False,
    )
    print(f"    Precision : {scores['test_precision_macro'].mean():.3f}"
          f"  (+/- {scores['test_precision_macro'].std():.3f})")
    print(f"    Recall    : {scores['test_recall_macro'].mean():.3f}"
          f"  (+/- {scores['test_recall_macro'].std():.3f})")
    print(f"    F1        : {scores['test_f1_macro'].mean():.3f}"
          f"  (+/- {scores['test_f1_macro'].std():.3f})")

    print("\n  Training final model on full labeled set...")
    pipeline.fit(X_raw, y)

    # Feature importances
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = sorted(
            zip(feat_cols, clf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print("\n── Top-15 feature importances ──────────────────────")
        for feat, imp_val in importances[:15]:
            print(f"  {feat:35s}: {imp_val:.4f}")

    imp_fitted    = pipeline.named_steps["imputer"]
    scaler_fitted = pipeline.named_steps.get("scaler", None)
    return pipeline, imp_fitted, scaler_fitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the illicit-transaction detection model."
    )
    parser.add_argument(
        "--label-mode",
        choices=list(_VALID_MODES),
        default=None,
        help=(
            "Which labels to train on.  "
            "'hybrid' uses confirmed labels where available and pseudo-labels "
            "elsewhere (default when confirmed labels are present).  "
            "'pseudo_only' uses only FATF pseudo-labels (original behaviour).  "
            "'confirmed_only' trains exclusively on externally confirmed rows."
        ),
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Bitcoin Illicit Detection — Model Training")
    print(f"  Classifier: {'LightGBM' if USE_LGBM else 'Random Forest'}")
    print("=" * 55 + "\n")

    if not os.path.exists(LABELED_CSV):
        print(f"[ERROR] Labeled features not found: {LABELED_CSV}")
        print("  Run 03_label.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine default label mode from the data
    _probe = pd.read_csv(LABELED_CSV, usecols=lambda c: c == "label_type", nrows=0)
    _has_label_type = "label_type" in _probe.columns
    if not _has_label_type:
        # label_type column absent → enrichment was skipped
        _has_confirmed = False
    else:
        _peek = pd.read_csv(LABELED_CSV, usecols=["label_type"])
        _has_confirmed = _peek["label_type"].eq("confirmed").any()

    if args.label_mode is None:
        label_mode = "hybrid" if _has_confirmed else "pseudo_only"
        print(f"[INFO] --label-mode not specified; auto-selected: {label_mode!r}")
    else:
        label_mode = args.label_mode

    model_path = os.path.join(OUTPUT_DIR, f"model_{label_mode}.pkl")

    print("Loading data...")
    all_tx, labeled = load_data(LABELED_CSV, label_mode)

    if labeled["pseudo_label"].nunique() < 2:
        print("[ERROR] Labeled set contains only one class — cannot train.")
        sys.exit(1)

    feat_cols = get_feature_cols(labeled)
    print(f"\n  Using {len(feat_cols)} feature columns")

    print("\n── Supervised Classification ────────────────────────")
    pipeline, imp_fitted, scaler_fitted = train_supervised(labeled, feat_cols)

    print("\n── Unsupervised Clustering ──────────────────────────")
    cluster_df, km_model = run_clustering(all_tx, feat_cols, imp_fitted, scaler_fitted)
    cluster_df.to_csv(CLUSTER_CSV, index=False)
    print(f"  Cluster assignments saved → {CLUSTER_CSV}")

    bundle = {
        "model":      pipeline,
        "features":   feat_cols,
        "km_model":   km_model,
        "classifier": "lightgbm" if USE_LGBM else "random_forest",
        "label_mode": label_mode,
    }
    joblib.dump(bundle, model_path)
    # Also write the canonical model.pkl so that 05_predict.py (which uses a
    # hard-coded path) always has an up-to-date model to load.
    canonical_path = os.path.join(OUTPUT_DIR, "model.pkl")
    joblib.dump(bundle, canonical_path)
    print(f"\n[OK] Model bundle saved → {model_path}")
    print(f"     Canonical copy  → {canonical_path}")
    print(f"     Label mode   : {label_mode}")
    print(f"     Features used: {len(feat_cols)}")
    print(f"     Classifier   : {'LightGBM' if USE_LGBM else 'Random Forest'}")
    print(f"\n  Predict with:")
    print(f"    python 05_predict.py --wallet <wallet_id>")
    print(f"    python 05_predict.py --tx <block_no> <trans_no>")
