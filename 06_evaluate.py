"""
06_evaluate.py — validate the detection methodology against the Elliptic dataset.

Downloads the Elliptic Bitcoin dataset (Weber et al., 2019) via kagglehub
and evaluates our Random Forest / LightGBM approach against its expert labels.

This provides the ground-truth benchmark required by §6.4 of the methodology:
    "Data of both known legal and illegal transactions will be blindly inputted
     in the model, and the results will be compared to the real answers."

Elliptic dataset:
    203,769 transactions, ~2% illicit (ransomware, darknet, fraud)
    166 anonymised features: f1 (time step) + f2-f94 (local) + f95-f166 (graph)
    Labels: 1 = illicit, 2 = licit, "unknown" = unlabelled

Two evaluation runs:
    local_features : f2-f94  (93 features — most comparable to our FATF feature set)
    all_features   : f2-f166 (165 features — local + neighbourhood graph aggregates)

Produces: outputs/elliptic_eval.json  (consumed by pages/6_Evaluate.py)

Usage:
    python 06_evaluate.py

Requirements:
    pip install kagglehub
    kaggle API credentials at ~/.kaggle/kaggle.json
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import confusion_matrix, classification_report
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    USE_LGBM = False

OUTPUT_DIR   = "outputs"
ELLIPTIC_DIR = os.path.join("data", "elliptic")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "elliptic_eval.json")
RANDOM_STATE = 42


# ── Download / locate dataset ──────────────────────────────────────────────────

def get_elliptic_path() -> str:
    """
    Return path to the Elliptic dataset directory.
    Downloads via kagglehub if not already present.
    """
    local_check = os.path.join(ELLIPTIC_DIR, "elliptic_txs_features.csv")
    if os.path.exists(local_check):
        print(f"  Elliptic data found at {ELLIPTIC_DIR}")
        return ELLIPTIC_DIR

    print("  Downloading Elliptic dataset via kagglehub…")
    try:
        import kagglehub
        path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
        print(f"  Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[ERROR] Could not download dataset: {e}")
        print(
            "\nManual setup:\n"
            "  1. Go to kaggle.com → Account → API → Create New Token\n"
            "  2. Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json\n"
            "  4. Re-run this script.\n"
        )
        sys.exit(1)


def load_elliptic(data_path: str):
    """
    Load and merge Elliptic features + class labels.
    Returns (df_labeled, feature_cols_local, feature_cols_all)
    """
    feats_file   = None
    classes_file = None

    # Search recursively for the files (kagglehub may nest them)
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if "features" in f.lower() and f.endswith(".csv"):
                feats_file = os.path.join(root, f)
            if "classes" in f.lower() and f.endswith(".csv"):
                classes_file = os.path.join(root, f)

    if feats_file is None or classes_file is None:
        print(f"[ERROR] Could not find Elliptic CSV files under {data_path}")
        sys.exit(1)

    print(f"  Features : {feats_file}")
    print(f"  Classes  : {classes_file}")

    # Features: txId + f1...f166 (no header — assign names)
    feats = pd.read_csv(feats_file, header=None)
    n_cols = feats.shape[1]
    col_names = ["txId"] + [f"f{i}" for i in range(1, n_cols)]
    feats.columns = col_names

    # Classes: txId, class
    classes = pd.read_csv(classes_file)
    classes.columns = classes.columns.str.strip()

    # Merge
    df = feats.merge(classes, on="txId", how="inner")

    # Keep only labeled rows (class 1 = illicit, class 2 = licit)
    df = df[df["class"].isin(["1", "2", 1, 2])].copy()
    df["label"] = (df["class"].astype(str) == "1").astype(int)  # 1=illicit, 0=licit

    # Feature subsets
    # f1 = time step (excluded — temporal leakage risk in CV)
    local_cols = [f"f{i}" for i in range(2, 95)   if f"f{i}" in df.columns]   # 93 local
    all_cols   = [f"f{i}" for i in range(2, n_cols) if f"f{i}" in df.columns]  # all non-time

    print(f"\n  Labeled transactions : {len(df):,}")
    print(f"  Illicit              : {df['label'].sum():,}  ({df['label'].mean():.1%})")
    print(f"  Licit                : {(df['label']==0).sum():,}")
    print(f"  Local features       : {len(local_cols)}")
    print(f"  All features         : {len(all_cols)}")

    return df, local_cols, all_cols


# ── Model builders ─────────────────────────────────────────────────────────────

def build_rf_pipeline():
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


def build_lgbm_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_pipeline(pipeline, X, y, label: str) -> dict:
    """Run 5-fold CV + full-dataset confusion matrix. Returns a result dict."""
    print(f"\n  [{label}] 5-fold stratified CV…")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=[
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
        ],
        return_train_score=False,
    )
    elapsed = time.time() - t0

    cv_metrics = {}
    for metric in ["precision_macro", "recall_macro", "f1_macro",
                   "precision_weighted", "recall_weighted", "f1_weighted"]:
        key = metric.replace("test_", "")
        cv_metrics[key] = {
            "mean": float(scores[f"test_{metric}"].mean()),
            "std":  float(scores[f"test_{metric}"].std()),
            "all":  scores[f"test_{metric}"].tolist(),
        }

    print(f"    F1 (macro)    : {cv_metrics['f1_macro']['mean']:.3f} "
          f"± {cv_metrics['f1_macro']['std']:.3f}")
    print(f"    Precision     : {cv_metrics['precision_macro']['mean']:.3f}")
    print(f"    Recall        : {cv_metrics['recall_macro']['mean']:.3f}")
    print(f"    Time          : {elapsed:.1f}s")

    # Fit on full set for confusion matrix + feature importances
    print(f"  [{label}] Fitting on full labeled set…")
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    cm     = confusion_matrix(y, y_pred).tolist()

    clf = pipeline.named_steps["clf"]
    fi  = clf.feature_importances_.tolist() if hasattr(clf, "feature_importances_") else []

    return {
        "cv_metrics":          cv_metrics,
        "confusion_matrix":    cm,
        "feature_importances": fi,
        "n_features":          X.shape[1],
        "n_samples":           int(len(y)),
        "label_counts": {
            "total_labeled": int(len(y)),
            "illicit":       int(y.sum()),
            "licit":         int((y == 0).sum()),
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Elliptic Ground-Truth Evaluation")
    print(f"  Classifier: {'LightGBM' if USE_LGBM else 'Random Forest'}")
    print("=" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Step 1 — Locating Elliptic dataset…")
    data_path = get_elliptic_path()

    print("\nStep 2 — Loading and merging Elliptic data…")
    df, local_cols, all_cols = load_elliptic(data_path)

    results = {
        "dataset": f"Elliptic (Weber et al., 2019) — {len(df):,} labeled transactions",
        "classifier": "lightgbm" if USE_LGBM else "random_forest",
    }

    build_pipe = build_lgbm_pipeline if USE_LGBM else build_rf_pipeline

    # ── Run 1: local features only (most comparable to our FATF feature set)
    print("\nStep 3 — Evaluation A: local features only (f2–f94)…")
    X_local = df[local_cols].values
    y       = df["label"].values
    results["local_features"] = evaluate_pipeline(
        build_pipe(), X_local, y, "Local features"
    )

    # ── Run 2: all features (local + neighbourhood graph aggregates)
    print("\nStep 4 — Evaluation B: all features (f2–f166)…")
    X_all = df[all_cols].values
    results["all_features"] = evaluate_pipeline(
        build_pipe(), X_all, y, "All features"
    )

    # ── Also run Random Forest for comparison if LightGBM was primary
    if USE_LGBM:
        print("\nStep 5 — Evaluation C: Random Forest on local features (baseline comparison)…")
        results["rf_local_features"] = evaluate_pipeline(
            build_rf_pipeline(), X_local, y, "RF local (baseline)"
        )

    # ── Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[OK] Results saved → {RESULTS_PATH}")
    print(f"\nSummary:")
    for run_key, run_label in [
        ("local_features",    "LightGBM / local features (f2-f94)  "),
        ("all_features",      "LightGBM / all features  (f2-f166)  "),
        ("rf_local_features", "Random Forest / local features       "),
    ]:
        if run_key in results:
            f1  = results[run_key]["cv_metrics"]["f1_macro"]["mean"]
            pre = results[run_key]["cv_metrics"]["precision_macro"]["mean"]
            rec = results[run_key]["cv_metrics"]["recall_macro"]["mean"]
            print(f"  {run_label}: F1={f1:.3f}  P={pre:.3f}  R={rec:.3f}")
    print(f"\nOpen the Streamlit app → Evaluate page to view charts.")
