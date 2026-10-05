"""
07_temporal_eval.py — temporal validation via chronological train/test split.

Evaluates classifier generalisation across time by splitting the CUBiD
transaction data chronologically into a training window (earlier months) and
a test window (later months).  Performance drift — the absolute difference
between in-sample and out-of-sample metrics — quantifies how well the
methodology transfers to unseen time periods.

Expected data layout (full mode):
    outputs/labeled_features.csv must contain a ``tx_datetime`` column
    produced by 02_feature.py, and must span all 12 monthly samples of the
    CUBiD dataset:
        Train  (8 samples): March–December 2022, March–December 2023
        Test   (4 samples): March–December 2024

Fallback mode (partial or single-period data):
    If fewer than 12 expected monthly samples are present (e.g., when running
    on the 200 k development sample that spans only one time period), the
    script logs a warning, performs a 67 / 33 chronological split on whatever
    data is available, and records ``split_mode: "fallback"`` in the output.
    When no ``tx_datetime`` column is present the split is performed on row
    order, which approximates chronological order assuming the CSV was written
    by 02_feature.py (which processes blocks in ascending Block_No order).

Classifier:
    LightGBM (primary) or Random Forest (fallback) — identical configuration
    to 04_train.py.  FEATURE_COLS and ``build_classifier`` are imported
    directly from that module via importlib to avoid duplication.

Produces: outputs/temporal_eval.json

Usage:
    python 07_temporal_eval.py
"""

import os
import sys
import json
import time
import warnings
import importlib.util
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)

# ── Import shared constants from 04_train.py via importlib ────────────────────
# (filename starts with a digit, so normal import syntax is unavailable)

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_train04", os.path.join(_HERE, "04_train.py")
)
_train04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train04)

FEATURE_COLS     = _train04.FEATURE_COLS
build_classifier = _train04.build_classifier
get_feature_cols = _train04.get_feature_cols
RANDOM_STATE     = _train04.RANDOM_STATE

OUTPUT_DIR   = "outputs"
LABELED_CSV  = os.path.join(OUTPUT_DIR, "labeled_features.csv")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "temporal_eval.json")

# Expected (year, month) pairs for the CUBiD multi-period dataset.
# The full multi-period sample is 24 quarterly snapshots spanning
# March 2019 – December 2024, taken on the 8th of March, June, September,
# and December each year.
EXPECTED_MONTHS = [
    (y, m) for y in range(2019, 2025) for m in (3, 6, 9, 12)
]
# Training window covers 2019 – 2022 (16 quarterly samples); the held-out
# test window is 2023 – 2024 (8 quarterly samples). Cutoff is inclusive
# of December 2022.
TRAIN_CUTOFF = pd.Period("2022-12", "M")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_labeled(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} transactions from {path}")

    # Prefer block-confirmation date over the broadcast `tx_datetime`:
    # `tx_datetime` is the broadcast/origination time (which can lag the
    # block by days due to mempool latency), and on the multi-period
    # dataset it was also affected by an earlier dayfirst-parsing bug
    # in 02_feature.py. Re-derive period membership from Block_No
    # joined to data/Block_Header.csv, which carries the unambiguous
    # confirmation date.
    bh_path = os.path.join("data", "Block_Header.csv")
    if "Block_No" in df.columns and os.path.exists(bh_path):
        bh = pd.read_csv(bh_path, usecols=["Block_No", "DT"], dtype=str)
        bh["Block_No"] = pd.to_numeric(bh["Block_No"], errors="coerce").astype("Int64")
        bh["block_dt"] = pd.to_datetime(bh["DT"], errors="coerce", format="mixed")
        bh = bh.dropna(subset=["Block_No", "block_dt"])
        bh_map = dict(zip(bh["Block_No"].astype(int), bh["block_dt"]))
        df["tx_datetime"] = pd.to_numeric(df["Block_No"], errors="coerce").map(bh_map)
        df["tx_datetime"] = pd.to_datetime(df["tx_datetime"], errors="coerce")
        n_parsed = df["tx_datetime"].notna().sum()
        print(f"  tx_datetime re-derived from Block_Header: "
              f"{n_parsed:,} / {len(df):,} rows")
    elif "tx_datetime" in df.columns:
        df["tx_datetime"] = pd.to_datetime(df["tx_datetime"], errors="coerce")
        n_parsed = df["tx_datetime"].notna().sum()
        print(f"  tx_datetime parsed from file: {n_parsed:,} / {len(df):,} rows")
    return df


# ── Split determination ────────────────────────────────────────────────────────

def determine_split(df: pd.DataFrame):
    """
    Return (train_idx, test_idx, split_mode, train_period_str, test_period_str).

    Full mode: all 12 expected CUBiD monthly samples present → split at the
    December 2023 / March 2024 boundary.

    Fallback mode: chronological 67 / 33 split on whatever months are present.
    This is triggered when the dataset spans fewer than 12 expected months
    (e.g., the 200 k development sample).
    """
    if "tx_datetime" not in df.columns or df["tx_datetime"].isna().all():
        print("[WARN] tx_datetime column absent or entirely NaN; "
              "splitting by row order (proxy for Block_No order).")
        n      = len(df)
        cutoff = int(n * 0.67)
        return (
            df.index[:cutoff],
            df.index[cutoff:],
            "fallback_no_datetime",
            f"Rows 0–{cutoff - 1} ({cutoff:,} transactions)",
            f"Rows {cutoff}–{n - 1} ({n - cutoff:,} transactions)",
        )

    # Sort by datetime (ascending = chronological)
    df = df.sort_values("tx_datetime").reset_index(drop=True)

    periods = df["tx_datetime"].dt.to_period("M").dropna()
    present = set(periods.unique())
    expected = {pd.Period(f"{y}-{m:02d}", "M") for y, m in EXPECTED_MONTHS}

    n_expected_present = len(expected & present)
    print(f"  Expected monthly samples present: {n_expected_present} / {len(EXPECTED_MONTHS)}")

    if expected.issubset(present):
        # Full mode — deterministic cutoff at December 2023
        train_mask = periods <= TRAIN_CUTOFF
        train_idx  = df.index[train_mask]
        test_idx   = df.index[~train_mask]
        return (
            train_idx,
            test_idx,
            "full",
            "March 2019 – December 2022 (16 quarterly samples)",
            "March 2023 – December 2024 (8 quarterly samples)",
        )

    # Fallback — 67 / 33 chronological split
    print("[WARN] Not all 12 expected monthly samples found. "
          "Using 67 / 33 chronological fallback split.")

    n      = len(df)
    cutoff = int(n * 0.67)
    sorted_idx = df.index  # already sorted above

    present_sorted = sorted(present)
    mid_idx        = max(0, int(len(present_sorted) * 0.67) - 1)
    approx_cutoff  = str(present_sorted[mid_idx]) if present_sorted else "unknown"
    approx_end     = str(present_sorted[-1]) if present_sorted else "unknown"
    approx_start   = str(present_sorted[0]) if present_sorted else "unknown"

    return (
        sorted_idx[:cutoff],
        sorted_idx[cutoff:],
        "fallback_partial",
        f"{approx_start} – {approx_cutoff} (earliest 67 % of sorted transactions)",
        f"after {approx_cutoff} – {approx_end} (latest 33 % of sorted transactions)",
    )


# ── Metrics computation ────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return precision, recall, F1 (macro), and confusion matrix."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "precision": round(float(prec), 4),
        "recall":    round(float(rec), 4),
        "f1":        round(float(f1), 4),
        "confusion_matrix": cm,
        "n_samples": int(len(y_true)),
        "label_counts": {
            "illicit": int((y_true == 1).sum()),
            "licit":   int((y_true == 0).sum()),
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Bitcoin Illicit Detection — Temporal Validation")
    print(f"  Classifier: {'LightGBM' if _train04.USE_LGBM else 'Random Forest'}")
    print("=" * 60 + "\n")

    if not os.path.exists(LABELED_CSV):
        print(f"[ERROR] Labeled features not found: {LABELED_CSV}")
        print("  Run 02_feature.py → 03_label.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    print("Step 1 — Loading labeled features…")
    df = load_labeled(LABELED_CSV)

    # ── Step 2: Determine train / test split ───────────────────────────────────
    print("\nStep 2 — Determining chronological split…")
    train_idx, test_idx, split_mode, train_period, test_period = determine_split(df)

    # Re-sort (if fallback, row order is preserved; full mode was sorted above)
    if "tx_datetime" in df.columns and df["tx_datetime"].notna().any():
        df = df.sort_values("tx_datetime").reset_index(drop=True)

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    print(f"\n  Split mode   : {split_mode}")
    print(f"  Train period : {train_period}")
    print(f"  Test period  : {test_period}")
    print(f"  Train rows   : {len(df_train):,}")
    print(f"  Test rows    : {len(df_test):,}")

    # ── Step 3: Select feature columns ────────────────────────────────────────
    print("\nStep 3 — Selecting feature columns…")
    feat_cols = get_feature_cols(df_train)
    print(f"  Using {len(feat_cols)} feature columns")

    # ── Step 4: Filter to labeled transactions (exclude pseudo_label = -1) ────
    train_lbl = df_train[df_train["pseudo_label"] != -1].copy()
    test_lbl  = df_test[df_test["pseudo_label"] != -1].copy()

    print(f"\n  Labeled train : {len(train_lbl):,}  "
          f"(illicit={int((train_lbl['pseudo_label']==1).sum()):,}, "
          f"licit={int((train_lbl['pseudo_label']==0).sum()):,})")
    print(f"  Labeled test  : {len(test_lbl):,}  "
          f"(illicit={int((test_lbl['pseudo_label']==1).sum()):,}, "
          f"licit={int((test_lbl['pseudo_label']==0).sum()):,})")

    if train_lbl["pseudo_label"].nunique() < 2:
        print("[ERROR] Training set contains only one class — cannot train.")
        sys.exit(1)

    if test_lbl.empty or test_lbl["pseudo_label"].nunique() < 2:
        print("[WARN] Test set is empty or single-class — test metrics may be unreliable.")

    # ── Step 5: Train classifier on training set ───────────────────────────────
    print("\nStep 4 — Training classifier on training window…")
    t0       = time.time()
    pipeline = build_classifier()
    X_train  = train_lbl[feat_cols]
    y_train  = train_lbl["pseudo_label"].values
    pipeline.fit(X_train, y_train)
    print(f"  Fitted in {time.time() - t0:.1f} s")

    # ── Step 6: Evaluate ───────────────────────────────────────────────────────
    print("\nStep 5 — Evaluating…")

    # In-sample (training set)
    y_train_pred = pipeline.predict(X_train)
    train_metrics = compute_metrics(y_train, y_train_pred)
    print(f"  Train (in-sample) — Precision: {train_metrics['precision']:.3f}  "
          f"Recall: {train_metrics['recall']:.3f}  F1: {train_metrics['f1']:.3f}")

    # Out-of-sample (test set)
    if not test_lbl.empty and test_lbl["pseudo_label"].nunique() >= 2:
        X_test       = test_lbl[feat_cols]
        y_test       = test_lbl["pseudo_label"].values
        y_test_pred  = pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred)
    else:
        # Edge case: test set too small or single-class
        test_metrics = {
            "precision": None, "recall": None, "f1": None,
            "confusion_matrix": [], "n_samples": len(test_lbl),
            "label_counts": {"illicit": 0, "licit": 0},
            "note": "Test set empty or single-class — metrics unavailable.",
        }

    if test_metrics.get("f1") is not None:
        print(f"  Test (held-out) — Precision: {test_metrics['precision']:.3f}  "
              f"Recall: {test_metrics['recall']:.3f}  F1: {test_metrics['f1']:.3f}")

    # Performance drift: signed difference (test – train); negative = degradation
    def _drift(key):
        t = test_metrics.get(key)
        r = train_metrics.get(key)
        return round(float(t) - float(r), 4) if (t is not None and r is not None) else None

    performance_drift = {
        "precision": _drift("precision"),
        "recall":    _drift("recall"),
        "f1":        _drift("f1"),
        "note": ("Signed difference test – train; negative values indicate "
                 "performance degradation on unseen time periods."),
    }

    if performance_drift["f1"] is not None:
        print(f"\n  Performance drift (test – train):")
        print(f"    F1 drift       : {performance_drift['f1']:+.3f}")
        print(f"    Precision drift: {performance_drift['precision']:+.3f}")
        print(f"    Recall drift   : {performance_drift['recall']:+.3f}")

    # ── Step 7: Save results ───────────────────────────────────────────────────
    results = {
        "split_mode":          split_mode,
        "train_period":        train_period,
        "test_period":         test_period,
        "classifier":          "lightgbm" if _train04.USE_LGBM else "random_forest",
        "n_feature_cols":      len(feat_cols),
        "train_metrics":       train_metrics,
        "test_metrics":        test_metrics,
        "performance_drift":   performance_drift,
    }

    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[OK] Results saved → {RESULTS_PATH}")
    print(f"\n  Split mode : {split_mode}")
    print(f"  Train F1   : {train_metrics['f1']:.3f}")
    if test_metrics.get("f1") is not None:
        print(f"  Test F1    : {test_metrics['f1']:.3f}")
        print(f"  F1 drift   : {performance_drift['f1']:+.3f}")
