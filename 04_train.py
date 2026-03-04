"""
04_train.py — train the illicit-transaction detection model.

Staged pipeline (methodology §6.4):
    1. Unsupervised clustering (K-Means + DBSCAN) on all transactions
    2. Supervised classification on pseudo-labeled transactions
       — LightGBM (primary, if installed) or Random Forest (fallback)
    3. 5-fold stratified cross-validation (precision, recall, F1)
    4. Save model bundle → outputs/model.pkl
       Save cluster assignments → outputs/cluster_assignments.csv

Usage:
    python 04_train.py
"""

import os
import sys
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
MODEL_PATH   = os.path.join(OUTPUT_DIR, "model.pkl")
CLUSTER_CSV  = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")

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

RANDOM_STATE  = 42
DBSCAN_SAMPLE = 10_000


def load_data(path: str):
    df      = pd.read_csv(path)
    labeled = df[df["pseudo_label"] != -1].copy()
    print(f"  Total transactions           : {len(df):,}")
    print(f"  Labeled (excl. uncertain)    : {len(labeled):,}")
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
    """Apply K-Means and DBSCAN to the full transaction set."""
    X_imp = imp.transform(all_tx[feat_cols])
    X_scaled = scaler.transform(X_imp) if scaler is not None else X_imp

    print("\n  Running K-Means (k=3)...")
    km        = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    n_sample = min(DBSCAN_SAMPLE, len(X_scaled))
    rng      = np.random.default_rng(RANDOM_STATE)
    idx      = rng.choice(len(X_scaled), size=n_sample, replace=False)

    print(f"  Running DBSCAN on {n_sample:,}-tx sample (eps=1.5, min_samples=5)...")
    db             = DBSCAN(eps=1.5, min_samples=5)
    db_sample_lbls = db.fit_predict(X_scaled[idx])

    n_clusters = len(set(db_sample_lbls)) - (1 if -1 in db_sample_lbls else 0)
    n_outliers = int((db_sample_lbls == -1).sum())
    print(f"    DBSCAN clusters : {n_clusters}")
    print(f"    DBSCAN outliers : {n_outliers:,}  ({n_outliers/n_sample:.1%})")

    db_labels      = np.full(len(X_scaled), -2, dtype=int)
    db_labels[idx] = db_sample_lbls

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
    print("=" * 55)
    print("  Bitcoin Illicit Detection — Model Training")
    print(f"  Classifier: {'LightGBM' if USE_LGBM else 'Random Forest'}")
    print("=" * 55 + "\n")

    if not os.path.exists(LABELED_CSV):
        print(f"[ERROR] Labeled features not found: {LABELED_CSV}")
        print("  Run 03_label.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    all_tx, labeled = load_data(LABELED_CSV)

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
        "model":    pipeline,
        "features": feat_cols,
        "km_model": km_model,
        "classifier": "lightgbm" if USE_LGBM else "random_forest",
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n[OK] Model bundle saved → {MODEL_PATH}")
    print(f"     Features used: {len(feat_cols)}")
    print(f"     Classifier   : {'LightGBM' if USE_LGBM else 'Random Forest'}")
