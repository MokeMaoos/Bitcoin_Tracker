"""
06_evaluate.py — validate the detection methodology against Elliptic++.

Evaluates our LightGBM / Random Forest approach against the Elliptic++
dataset (Elmougy and Liu, 2023) at two levels of granularity:

    transaction_level — 203,769 transactions, 183 features
                        Labels: 1 = illicit, 2 = licit, 3 = unknown
    actor_level       — 822,942 wallet addresses, 56 features
                        Labels: same encoding

Both evaluations use 5-fold stratified cross-validation and exclude rows
with unknown labels (class = 3).

This provides the ground-truth benchmark required by §6.4 of the methodology:
    "Data of both known legal and illegal transactions will be blindly inputted
     in the model, and the results will be compared to the real answers."

Produces: outputs/elliptic_eval.json  (consumed by pages/6_Evaluate.py)

Usage:
    python 06_evaluate.py

Data acquisition — IMPORTANT (Git LFS):
    The GitHub repository uses Git LFS for the large CSV files.  Downloading
    the ZIP archive from GitHub gives only 3-line pointer files, not real data.
    You must obtain the actual files via one of two methods:

    Option A — Git LFS clone (recommended):
        brew install git-lfs          # macOS; or see https://git-lfs.com
        git lfs install
        git clone https://github.com/git-disl/EllipticPlusPlus.git \
            ../EllipticPlusPlus-main
        # The script will find the data automatically in that sibling directory.

    Option B — Google Drive download:
        1. Go to: https://drive.google.com/drive/folders/
           1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l
        2. Download the 'Transactions Dataset' and 'Actors Dataset' folders.
        3. Place them as:
             data/elliptic_pp/Transactions Dataset/txs_features.csv
             data/elliptic_pp/Transactions Dataset/txs_classes.csv
             data/elliptic_pp/Actors Dataset/wallets_features.csv
             data/elliptic_pp/Actors Dataset/wallets_classes.csv

    The script is idempotent: it searches for the data in both locations and
    proceeds without re-downloading if the files are already present.

Reference:
    Elmougy, Y. and Liu, L. (2023). Demystifying Fraudulent Transactions and
    Illicit Nodes in the Bitcoin Network for Financial Forensics.
    KDD 2023. doi:10.1145/3580305.3599803
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
from sklearn.metrics         import confusion_matrix
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    USE_LGBM = False

OUTPUT_DIR   = "outputs"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "elliptic_eval.json")
RANDOM_STATE = 42

# Non-feature columns to exclude when building X matrices
_TX_SKIP_COLS     = {"txId", "Time step", "class"}
_WALLET_SKIP_COLS = {"address", "Time step", "class"}

# Label encoding: integer class values used in the Elliptic++ files
_ILLICIT_CLASS = 1
_LICIT_CLASS   = 2
_UNKNOWN_CLASS = 3

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── Data file location helpers ─────────────────────────────────────────────────

def _is_lfs_pointer(path: str) -> bool:
    """Return True if the file is a Git LFS pointer rather than real data."""
    try:
        with open(path, "rb") as fh:
            return fh.read(27) == b"version https://git-lfs.git"
    except OSError:
        return False


def _resolve_data_dir():
    """
    Locate the Elliptic++ data directory.  Searches in priority order:

    1. Sibling EllipticPlusPlus-main/ directory (beside this repo) — the
       natural location when cloned with Git LFS.
    2. data/elliptic_pp/ inside this repo — the Google Drive download path.

    Returns the first directory where txs_features.csv exists as a real file
    (not an LFS pointer), or None if neither location has real data.
    """
    candidates = [
        os.path.join(_HERE, "..", "EllipticPlusPlus-main"),
        os.path.join(_HERE, "data", "elliptic_pp"),
        os.path.join("data", "elliptic_pp"),
    ]
    for candidate in candidates:
        tx_path = os.path.join(candidate, "Transactions Dataset", "txs_features.csv")
        if os.path.exists(tx_path) and not _is_lfs_pointer(tx_path):
            return os.path.normpath(candidate)
    return None


def _check_and_exit_if_missing():
    """
    Verify that real (non-LFS-pointer) Elliptic++ CSV files are present.
    Returns the resolved data directory path, or exits with a diagnostic message.
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.normpath(os.path.join(_here, "..", "EllipticPlusPlus-main"))
    tx_sibling = os.path.join(sibling, "Transactions Dataset", "txs_features.csv")

    # Detect LFS pointer in the sibling directory specifically
    if os.path.exists(tx_sibling) and _is_lfs_pointer(tx_sibling):
        print("[ERROR] The EllipticPlusPlus-main/ directory contains Git LFS "
              "pointer files, not actual data.")
        print(
            "\nThe ZIP download from GitHub does not include LFS objects.\n"
            "To fetch the real data files, run:\n\n"
            "    brew install git-lfs          # install Git LFS (macOS)\n"
            "    git lfs install\n"
            f"    cd \"{sibling}\"\n"
            "    git init\n"
            "    git remote add origin https://github.com/git-disl/EllipticPlusPlus.git\n"
            "    git lfs pull origin\n\n"
            "Alternatively, download the files directly from Google Drive:\n"
            "    https://drive.google.com/drive/folders/"
            "1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l\n"
            "  and place them under:\n"
            f"    {sibling}/Transactions Dataset/txs_features.csv\n"
            f"    {sibling}/Transactions Dataset/txs_classes.csv\n"
            f"    {sibling}/Actors Dataset/wallets_features.csv\n"
            f"    {sibling}/Actors Dataset/wallets_classes.csv\n"
        )
        sys.exit(1)

    data_dir = _resolve_data_dir()
    if data_dir is None:
        print("[ERROR] Elliptic++ data files not found.")
        print(
            "\nOption A — Git LFS clone (recommended):\n"
            "    brew install git-lfs\n"
            "    git lfs install\n"
            "    git clone https://github.com/git-disl/EllipticPlusPlus.git "
            f"\"{sibling}\"\n\n"
            "Option B — Google Drive download:\n"
            "    https://drive.google.com/drive/folders/"
            "1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l\n"
            "  Place files under data/elliptic_pp/ (see docstring for paths).\n"
        )
        sys.exit(1)

    return data_dir


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_transactions(data_dir: str) -> tuple:
    """
    Load the Elliptic++ transaction-level dataset.

    txs_features.csv contains txId, Time step, class, and 183 feature columns.
    The class label (1/2/3) is embedded in the features file; txs_classes.csv
    is loaded as a fallback to fill missing labels.

    Returns (df_labeled, feature_cols).
    """
    feat_path = os.path.join(data_dir, "Transactions Dataset", "txs_features.csv")
    cls_path  = os.path.join(data_dir, "Transactions Dataset", "txs_classes.csv")

    print(f"  Features : {feat_path}")
    df = pd.read_csv(feat_path)
    df.columns = df.columns.str.strip()

    # If 'class' is not in the features file, merge from txs_classes.csv
    if "class" not in df.columns:
        classes = pd.read_csv(cls_path)
        classes.columns = classes.columns.str.strip()
        # Normalise label encoding to integers (handles "1","2","3" strings)
        lbl_col = [c for c in classes.columns if c != "txId"][0]
        classes[lbl_col] = pd.to_numeric(
            classes[lbl_col].astype(str).str.strip(), errors="coerce"
        ).fillna(_UNKNOWN_CLASS).astype(int)
        df = df.merge(
            classes[["txId", lbl_col]].rename(columns={lbl_col: "class"}),
            on="txId", how="left",
        )
        df["class"] = df["class"].fillna(_UNKNOWN_CLASS).astype(int)

    # Ensure class is integer
    df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(_UNKNOWN_CLASS).astype(int)

    # Exclude unknown rows
    df_labeled = df[df["class"].isin([_ILLICIT_CLASS, _LICIT_CLASS])].copy()
    df_labeled["label"] = (df_labeled["class"] == _ILLICIT_CLASS).astype(int)

    # Feature columns = all columns except the non-feature identifiers
    feat_cols = [c for c in df.columns if c not in _TX_SKIP_COLS]

    # Drop all-NaN columns
    all_nan = [c for c in feat_cols if df_labeled[c].isna().all()]
    if all_nan:
        print(f"  [WARN] Dropping {len(all_nan)} all-NaN feature columns: "
              f"{all_nan[:5]}{'…' if len(all_nan) > 5 else ''}")
        feat_cols = [c for c in feat_cols if c not in all_nan]

    print(f"\n  Transaction-level dataset:")
    print(f"    Total rows       : {len(df):,}")
    print(f"    Labeled rows     : {len(df_labeled):,}")
    print(f"    Illicit          : {df_labeled['label'].sum():,}  "
          f"({df_labeled['label'].mean():.1%})")
    print(f"    Licit            : {(df_labeled['label'] == 0).sum():,}")
    print(f"    Feature columns  : {len(feat_cols)}")

    return df_labeled, feat_cols


def _load_actors(data_dir: str) -> tuple:
    """
    Load the Elliptic++ actor-level dataset.

    wallets_features.csv contains the address ID and 56 feature columns.
    wallets_classes.csv contains the address ID and class label.
    Both are merged on the address column.

    Returns (df_labeled, feature_cols).
    """
    feat_path = os.path.join(data_dir, "Actors Dataset", "wallets_features.csv")
    cls_path  = os.path.join(data_dir, "Actors Dataset", "wallets_classes.csv")

    print(f"  Features : {feat_path}")
    print(f"  Classes  : {cls_path}")

    feats = pd.read_csv(feat_path)
    feats.columns = feats.columns.str.strip()

    classes = pd.read_csv(cls_path)
    classes.columns = classes.columns.str.strip()

    # Identify ID and label columns in classes file
    addr_col  = classes.columns[0]   # first col = address identifier
    label_col = [c for c in classes.columns if c != addr_col][0]

    classes[label_col] = pd.to_numeric(
        classes[label_col].astype(str).str.strip(), errors="coerce"
    ).fillna(_UNKNOWN_CLASS).astype(int)

    # Merge features with labels
    feat_addr_col = feats.columns[0]  # address is first column in features file
    df = feats.merge(
        classes[[addr_col, label_col]].rename(
            columns={addr_col: feat_addr_col, label_col: "class"}
        ),
        on=feat_addr_col,
        how="inner",
    )
    df.columns = df.columns.str.strip()

    # Ensure class is integer
    df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(_UNKNOWN_CLASS).astype(int)

    # Exclude unknown rows
    df_labeled = df[df["class"].isin([_ILLICIT_CLASS, _LICIT_CLASS])].copy()
    df_labeled["label"] = (df_labeled["class"] == _ILLICIT_CLASS).astype(int)

    # Feature columns = all except address, class, Time step
    feat_cols = [
        c for c in df.columns
        if c not in _WALLET_SKIP_COLS and c != feat_addr_col
    ]

    # Drop all-NaN columns
    all_nan = [c for c in feat_cols if df_labeled[c].isna().all()]
    if all_nan:
        print(f"  [WARN] Dropping {len(all_nan)} all-NaN feature columns: "
              f"{all_nan[:5]}{'…' if len(all_nan) > 5 else ''}")
        feat_cols = [c for c in feat_cols if c not in all_nan]

    print(f"\n  Actor-level dataset:")
    print(f"    Total rows       : {len(df):,}")
    print(f"    Labeled rows     : {len(df_labeled):,}")
    print(f"    Illicit          : {df_labeled['label'].sum():,}  "
          f"({df_labeled['label'].mean():.1%})")
    print(f"    Licit            : {(df_labeled['label'] == 0).sum():,}")
    print(f"    Feature columns  : {len(feat_cols)}")

    return df_labeled, feat_cols


# ── Model builders ─────────────────────────────────────────────────────────────

def _build_rf_pipeline() -> Pipeline:
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


def _build_lgbm_pipeline() -> Pipeline:
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

def evaluate_pipeline(pipeline: Pipeline, X: np.ndarray, y: np.ndarray,
                      label: str) -> dict:
    """
    Run 5-fold stratified CV followed by a full-dataset fit for the confusion
    matrix.  Returns a result dict compatible with the structure consumed by
    pages/6_Evaluate.py.
    """
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
    for metric in [
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    ]:
        cv_metrics[metric] = {
            "mean": float(scores[f"test_{metric}"].mean()),
            "std":  float(scores[f"test_{metric}"].std()),
            "all":  scores[f"test_{metric}"].tolist(),
        }

    print(f"    F1 (macro)    : {cv_metrics['f1_macro']['mean']:.3f} "
          f"± {cv_metrics['f1_macro']['std']:.3f}")
    print(f"    Precision     : {cv_metrics['precision_macro']['mean']:.3f}")
    print(f"    Recall        : {cv_metrics['recall_macro']['mean']:.3f}")
    print(f"    Time          : {elapsed:.1f}s")

    # Fit on full labeled set for confusion matrix + feature importances
    print(f"  [{label}] Fitting on full labeled set for confusion matrix…")
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
    print("  Elliptic++ Ground-Truth Evaluation")
    print(f"  Classifier  : {'LightGBM' if USE_LGBM else 'Random Forest'}")
    print(f"  Dataset     : Elliptic++ (Elmougy and Liu, 2023)")
    print("=" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Verify data presence ───────────────────────────────────────────
    print("Step 1 — Verifying Elliptic++ data files…")
    data_dir = _check_and_exit_if_missing()
    print(f"  Data found at {data_dir}")

    build_pipe = _build_lgbm_pipeline if USE_LGBM else _build_rf_pipeline
    clf_name   = "lightgbm" if USE_LGBM else "random_forest"
    print(f"  Classifier  : {clf_name}")

    results = {
        "dataset_source": "Elliptic++ (Elmougy and Liu, 2023)",
        "classifier":     clf_name,
    }

    # ── Step 2: Transaction-level evaluation ───────────────────────────────────
    print("\nStep 2 — Loading transaction-level data…")
    tx_df, tx_feat_cols = _load_transactions(data_dir)

    print("\nStep 3 — Transaction-level 5-fold CV…")
    X_tx = tx_df[tx_feat_cols].values
    y_tx = tx_df["label"].values
    results["transaction_level"] = evaluate_pipeline(
        build_pipe(), X_tx, y_tx, "Transaction level"
    )
    results["transaction_level"]["feature_cols_used"] = len(tx_feat_cols)

    # ── Step 3: Actor-level evaluation ────────────────────────────────────────
    print("\nStep 4 — Loading actor-level data…")
    actor_df, actor_feat_cols = _load_actors(data_dir)

    print("\nStep 5 — Actor-level 5-fold CV…")
    # Note: 822k rows — this step may take several minutes with LightGBM.
    X_actor = actor_df[actor_feat_cols].values
    y_actor = actor_df["label"].values
    results["actor_level"] = evaluate_pipeline(
        build_pipe(), X_actor, y_actor, "Actor level"
    )
    results["actor_level"]["feature_cols_used"] = len(actor_feat_cols)

    # ── Save results ───────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[OK] Results saved → {RESULTS_PATH}")
    print(f"\nSummary ({clf_name}):")

    for run_key, run_label in [
        ("transaction_level", "Transaction level"),
        ("actor_level",       "Actor level      "),
    ]:
        r   = results[run_key]
        f1  = r["cv_metrics"]["f1_macro"]["mean"]
        pre = r["cv_metrics"]["precision_macro"]["mean"]
        rec = r["cv_metrics"]["recall_macro"]["mean"]
        print(f"  {run_label}: F1={f1:.3f}  P={pre:.3f}  R={rec:.3f}")

    print("\nOpen the Streamlit app → Evaluate page to view charts.")
