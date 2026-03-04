"""
04_train.py — train the illicit-transaction detection model.

Staged pipeline (methodology §6.4):
    1. Unsupervised clustering (K-Means + DBSCAN) on all transactions
    2. Supervised Random Forest on pseudo-labeled transactions
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

OUTPUT_DIR   = "outputs"
LABELED_CSV  = os.path.join(OUTPUT_DIR, "labeled_features.csv")
MODEL_PATH   = os.path.join(OUTPUT_DIR, "model.pkl")
CLUSTER_CSV  = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")

FEATURE_COLS = [
    "num_inputs", "num_outputs", "total_input", "total_output",
    "mean_output", "std_output", "max_output", "min_output",
    "io_ratio", "value_ratio", "equal_value_flag", "value_entropy",
    "peeling_flag", "fresh_output_ratio", "coinjoin_strength",
    "wallet_total_tx", "wallet_total_sent", "wallet_size",
]

RANDOM_STATE  = 42
DBSCAN_SAMPLE = 10_000   # DBSCAN distance matrix is O(n²) — sample to avoid OOM


def load_data(path: str):
    df      = pd.read_csv(path)
    labeled = df[df["pseudo_label"] != -1].copy()
    print(f"  Total transactions           : {len(df):,}")
    print(f"  Labeled (excl. uncertain)    : {len(labeled):,}")
    print(f"    Illicit (pseudo_label = 1) : {(labeled['pseudo_label'] == 1).sum():,}")
    print(f"    Licit   (pseudo_label = 0) : {(labeled['pseudo_label'] == 0).sum():,}")
    return df, labeled


def run_clustering(all_tx: pd.DataFrame, imp: SimpleImputer,
                   scaler: StandardScaler) -> pd.DataFrame:
    """Apply K-Means and DBSCAN to the full transaction set."""
    X_imp    = imp.transform(all_tx[FEATURE_COLS])
    X_scaled = scaler.transform(X_imp)

    # K-Means (k=3: licit / ambiguous / illicit)
    print("\n  Running K-Means (k=3)...")
    km       = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    # DBSCAN — run on a random sample to avoid O(n²) memory usage
    n_sample = min(DBSCAN_SAMPLE, len(X_scaled))
    rng      = np.random.default_rng(RANDOM_STATE)
    idx      = rng.choice(len(X_scaled), size=n_sample, replace=False)
    X_sample = X_scaled[idx]

    print(f"  Running DBSCAN on {n_sample:,}-transaction sample (eps=1.5, min_samples=5)...")
    db             = DBSCAN(eps=1.5, min_samples=5)
    db_sample_lbls = db.fit_predict(X_sample)

    n_clusters = len(set(db_sample_lbls)) - (1 if -1 in db_sample_lbls else 0)
    n_outliers = int((db_sample_lbls == -1).sum())
    print(f"    DBSCAN clusters : {n_clusters}")
    print(f"    DBSCAN outliers : {n_outliers:,}  ({n_outliers / n_sample:.1%})")

    # Assign -2 (not sampled) for transactions outside the DBSCAN sample
    db_labels          = np.full(len(X_scaled), -2, dtype=int)
    db_labels[idx]     = db_sample_lbls

    result = all_tx[["Block_No", "Trans_No"]].copy()
    result["cluster_kmeans"] = km_labels
    result["cluster_dbscan"] = db_labels   # -2 = not included in DBSCAN sample
    return result, km


def train_supervised(labeled: pd.DataFrame) -> tuple:
    """
    Fit a preprocessing pipeline (imputer → scaler → RandomForest).
    Returns the fitted pipeline and the fitted imputer/scaler for reuse.
    """
    X_raw = labeled[FEATURE_COLS]
    y     = labeled["pseudo_label"].values

    pipeline = Pipeline([
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
    clf          = pipeline.named_steps["clf"]
    importances  = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                          key=lambda x: x[1], reverse=True)
    print("\n── Top-10 feature importances ───────────────────────")
    for feat, imp_val in importances[:10]:
        print(f"  {feat:25s}: {imp_val:.4f}")

    # Reuse fitted preprocessing for clustering step
    imp_fitted    = pipeline.named_steps["imputer"]
    scaler_fitted = pipeline.named_steps["scaler"]
    return pipeline, imp_fitted, scaler_fitted


if __name__ == "__main__":
    print("=" * 55)
    print("  Bitcoin Illicit Detection — Model Training")
    print("=" * 55 + "\n")

    if not os.path.exists(LABELED_CSV):
        print(f"[ERROR] Labeled features not found: {LABELED_CSV}")
        print("  Run 03_label.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    all_tx, labeled = load_data(LABELED_CSV)

    if labeled["pseudo_label"].nunique() < 2:
        print("[ERROR] Labeled set contains only one class — cannot train.")
        sys.exit(1)

    # ── Supervised training ───────────────────────────────────────────────────
    print("\n── Supervised Classification ────────────────────────")
    pipeline, imp_fitted, scaler_fitted = train_supervised(labeled)

    # ── Unsupervised clustering ───────────────────────────────────────────────
    print("\n── Unsupervised Clustering ──────────────────────────")
    cluster_df, km_model = run_clustering(all_tx, imp_fitted, scaler_fitted)
    cluster_df.to_csv(CLUSTER_CSV, index=False)
    print(f"  Cluster assignments saved → {CLUSTER_CSV}")

    # ── Save model bundle ─────────────────────────────────────────────────────
    bundle = {
        "model":    pipeline,       # sklearn Pipeline — predict_proba works directly
        "features": FEATURE_COLS,
        "km_model": km_model,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n[OK] Model bundle saved → {MODEL_PATH}")
