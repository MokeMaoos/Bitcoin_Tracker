"""
08_mechanism_eval.py — mechanism validation against confirmed external labels.

Evaluates the detection mechanism (FATF rules + supervised classifier) using
transactions whose addresses match an external confirmed-label source from
02b_enrich_labels.py.  This is the primary, non-circular ground-truth
benchmark for the dissertation.

What changed in this revision (driven by methodological review)
--------------------------------------------------------------
1. Held-out confusion matrix.
   The previous version computed the confusion matrix by fitting the
   classifier on the full confirmed-label set and predicting on the same set,
   producing a perfect-classification matrix that disagreed with the per-fold
   F1.  The matrix is now derived from out-of-fold predictions
   (StratifiedKFold) so it agrees with the reported macro F1.

2. Augmented presumed-licit pool.
   The raw confirmed-label set is ~85 % illicit because three of four label
   sources (BitcoinAbuse, OpenSanctions, partial BABD-13) only contribute
   illicit examples.  The script now augments the licit pool with a random
   sample of unmatched CUBiD transactions (presumed-licit on the assumption
   that the base rate of illicit Bitcoin activity is well below 1 %), so
   the augmented evaluation set is approximately 50/50.  The provenance of
   each row is recorded in the new ``eval_label_source`` column with values
   ``confirmed_illicit`` / ``confirmed_licit`` / ``presumed_licit``.

3. Per-source threshold sweeps.
   The combined-set threshold sweep produced base-rate-artefact results.
   The sweep is now run separately for each source that has both classes
   (Elliptic++, BABD-13) and on the augmented full set.  Single-class
   sources (BitcoinAbuse, OpenSanctions) are explicitly excluded with a
   note rather than reported as "insufficient illicit".

4. Expanded metrics.
   Each CV evaluation now reports macro and weighted F1, per-class
   precision and recall, ROC-AUC and PR-AUC (both from out-of-fold
   probabilities), in addition to macro precision / recall / F1.

5. Single-class skip message.
   Sources with only one class now print
       "Single-class source — supervised CV requires both illicit and licit
        examples (X illicit, 0 licit)."
   instead of the misleading "insufficient confirmed illicit examples".

Reads from: outputs/labeled_features.csv
Produces:   outputs/mechanism_eval.json

Usage:
    python 08_mechanism_eval.py
"""

import importlib.util
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

# ── Import shared constants from 04_train.py ──────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_train04", os.path.join(_HERE, "04_train.py")
)
_train04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train04)

build_classifier = _train04.build_classifier
get_feature_cols = _train04.get_feature_cols
RANDOM_STATE     = _train04.RANDOM_STATE
USE_LGBM         = _train04.USE_LGBM

OUTPUT_DIR        = "outputs"
LABELED_CSV       = os.path.join(OUTPUT_DIR, "labeled_features.csv")
RESULTS_PATH      = os.path.join(OUTPUT_DIR, "mechanism_eval.json")
EVAL_SET_CSV      = os.path.join(OUTPUT_DIR, "mechanism_eval_set.csv")

MIN_ILLICIT_CONFIRMED = 10   # absolute floor for any evaluation
MIN_ILLICIT_CV        = 100  # floor for "reliable" cross-validation
DEFAULT_RULE_THRESHOLD = 2   # current production rule-aggregation cutoff
TARGET_BALANCE_RATIO  = 1.0  # target |licit|/|illicit| in augmented set

RULE_COLS = [
    "rule_equal_value",
    "rule_peeling",
    "rule_high_fan_out",
    "rule_fresh_outputs",
    "rule_mixing",
    "rule_high_entropy",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labeled(path: str) -> pd.DataFrame:
    """Read the full labeled feature matrix; verify required columns exist."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} transactions from {path}")

    required = {"label_type", "confirmed_label", "pseudo_label"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] labeled_features.csv is missing required columns: {missing}")
        print("  Re-run 02b_enrich_labels.py and 03_label.py.")
        sys.exit(1)
    return df


def build_augmented_eval_set(df: pd.DataFrame,
                             rng_seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Build the primary evaluation subset.

    Components:
        confirmed_illicit  — every row with label_type == "confirmed" and
                             pseudo_label == 1.
        confirmed_licit    — every row with label_type == "confirmed" and
                             pseudo_label == 0.
        presumed_licit     — random sample of label_type == "pseudo" rows
                             (preferring pseudo_label == 0); count chosen
                             so the licit pool roughly equals the illicit
                             pool, for a ~50/50 evaluation balance.

    Adds a new column ``eval_label_source`` recording the provenance.
    """
    rng = np.random.default_rng(rng_seed)

    confirmed   = df[df["label_type"] == "confirmed"].copy()
    conf_illicit = confirmed[confirmed["pseudo_label"] == 1].copy()
    conf_licit   = confirmed[confirmed["pseudo_label"] == 0].copy()
    n_ill = len(conf_illicit)
    n_lit = len(conf_licit)
    target_licit = max(int(round(n_ill * TARGET_BALANCE_RATIO)) - n_lit, 0)

    print(f"  Confirmed illicit : {n_ill:,}")
    print(f"  Confirmed licit   : {n_lit:,}")
    print(f"  Augmenting licit pool with {target_licit:,} presumed-licit "
          f"rows from the unmatched CUBiD pool.")

    # Pseudo pool — exclude uncertain (-1) and exclude any row that is
    # already in the confirmed set.
    pseudo_pool = df[
        (df["label_type"] == "pseudo") &
        (df["pseudo_label"] != -1)
    ].copy()
    pseudo_licit_pool = pseudo_pool[pseudo_pool["pseudo_label"] == 0]
    pseudo_other_pool = pseudo_pool[pseudo_pool["pseudo_label"] != 0]

    pieces = []
    take_from_licit = min(target_licit, len(pseudo_licit_pool))
    if take_from_licit > 0:
        idx = rng.choice(len(pseudo_licit_pool), size=take_from_licit,
                         replace=False)
        pieces.append(pseudo_licit_pool.iloc[idx])
    remaining = target_licit - take_from_licit
    if remaining > 0 and len(pseudo_other_pool) > 0:
        idx = rng.choice(len(pseudo_other_pool),
                         size=min(remaining, len(pseudo_other_pool)),
                         replace=False)
        pieces.append(pseudo_other_pool.iloc[idx])
    presumed_licit = (
        pd.concat(pieces, ignore_index=False) if pieces else
        pseudo_pool.iloc[0:0]
    ).copy()
    # Force pseudo_label to 0 for the supervised target — these are presumed-licit.
    presumed_licit["pseudo_label"] = 0

    conf_illicit["eval_label_source"] = "confirmed_illicit"
    conf_licit  ["eval_label_source"] = "confirmed_licit"
    presumed_licit["eval_label_source"] = "presumed_licit"

    augmented = pd.concat(
        [conf_illicit, conf_licit, presumed_licit],
        ignore_index=False
    ).reset_index(drop=True)

    n_p = len(presumed_licit)
    n_total = len(augmented)
    n_illicit_aug = int((augmented["pseudo_label"] == 1).sum())
    n_licit_aug   = int((augmented["pseudo_label"] == 0).sum())
    print(f"  Augmented eval set: {n_total:,} rows  "
          f"(illicit={n_illicit_aug:,}, licit={n_licit_aug:,}, "
          f"of which presumed_licit={n_p:,})")
    return augmented


# ── Generic CV metric bundle ──────────────────────────────────────────────────

def cv_metric_bundle(X: pd.DataFrame, y: np.ndarray,
                     n_folds: int = 5, label: str = "") -> dict:
    """
    Run stratified k-fold CV with the standard pipeline and return a bundle of
    per-fold metrics, their mean/std, and a held-out (out-of-fold) confusion
    matrix that is internally consistent with the per-fold scores.
    """
    if y.size == 0 or set(np.unique(y)) != {0, 1}:
        return {
            "skipped": True,
            "note": ("Single-class source — supervised CV requires both "
                     f"illicit and licit examples ({int((y==1).sum())} illicit, "
                     f"{int((y==0).sum())} licit)."),
            "n_samples": int(y.size),
            "label_counts": {
                "illicit": int((y == 1).sum()),
                "licit":   int((y == 0).sum()),
            },
        }

    n_per_class = int(min((y == 0).sum(), (y == 1).sum()))
    folds = min(n_folds, n_per_class)
    if folds < 2:
        return {
            "skipped": True,
            "note": f"Too few examples per class ({n_per_class}) to run CV.",
            "n_samples": int(y.size),
            "label_counts": {
                "illicit": int((y == 1).sum()),
                "licit":   int((y == 0).sum()),
            },
        }

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    metric_keys = [
        "precision_macro", "recall_macro", "f1_macro",
        "f1_weighted",
        "precision_illicit", "recall_illicit",
        "precision_licit",   "recall_licit",
        "roc_auc", "pr_auc",
    ]
    per_fold = {k: [] for k in metric_keys}
    y_oof_pred  = np.full(len(y), -1, dtype=int)
    y_oof_proba = np.full(len(y), np.nan, dtype=float)

    print(f"\n  Running {folds}-fold stratified CV on {len(y):,} rows  "
          f"({label or 'eval set'})…")
    t0 = time.time()
    for fi, (tr, te) in enumerate(cv.split(X, y), 1):
        pipe = build_classifier()
        pipe.fit(X.iloc[tr], y[tr])
        y_pred  = pipe.predict(X.iloc[te])
        y_proba = pipe.predict_proba(X.iloc[te])[:, 1]
        y_oof_pred[te]  = y_pred
        y_oof_proba[te] = y_proba

        per_fold["precision_macro"].append(
            precision_score(y[te], y_pred, average="macro", zero_division=0))
        per_fold["recall_macro"].append(
            recall_score   (y[te], y_pred, average="macro", zero_division=0))
        per_fold["f1_macro"].append(
            f1_score       (y[te], y_pred, average="macro", zero_division=0))
        per_fold["f1_weighted"].append(
            f1_score       (y[te], y_pred, average="weighted", zero_division=0))
        per_fold["precision_illicit"].append(
            precision_score(y[te], y_pred, pos_label=1, zero_division=0))
        per_fold["recall_illicit"].append(
            recall_score   (y[te], y_pred, pos_label=1, zero_division=0))
        per_fold["precision_licit"].append(
            precision_score(y[te], y_pred, pos_label=0, zero_division=0))
        per_fold["recall_licit"].append(
            recall_score   (y[te], y_pred, pos_label=0, zero_division=0))
        per_fold["roc_auc"].append(
            roc_auc_score  (y[te], y_proba))
        per_fold["pr_auc"].append(
            average_precision_score(y[te], y_proba))

    elapsed = time.time() - t0

    cv_metrics = {}
    for k in metric_keys:
        vals = per_fold[k]
        cv_metrics[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)), 4),
            "all":  [round(float(v), 4) for v in vals],
        }

    cm = confusion_matrix(y, y_oof_pred, labels=[0, 1]).tolist()

    print(f"    F1 (macro)    : "
          f"{cv_metrics['f1_macro']['mean']:.4f} ± "
          f"{cv_metrics['f1_macro']['std']:.4f}")
    print(f"    F1 (weighted) : "
          f"{cv_metrics['f1_weighted']['mean']:.4f}")
    print(f"    P / R illicit : "
          f"{cv_metrics['precision_illicit']['mean']:.4f} / "
          f"{cv_metrics['recall_illicit']['mean']:.4f}")
    print(f"    P / R licit   : "
          f"{cv_metrics['precision_licit']['mean']:.4f} / "
          f"{cv_metrics['recall_licit']['mean']:.4f}")
    print(f"    ROC-AUC       : "
          f"{cv_metrics['roc_auc']['mean']:.4f}")
    print(f"    PR-AUC        : "
          f"{cv_metrics['pr_auc']['mean']:.4f}")
    print(f"    Time          : {elapsed:.1f} s")

    return {
        "skipped": False,
        "cv_metrics": cv_metrics,
        "confusion_matrix": cm,
        "confusion_matrix_note": (
            "From out-of-fold predictions — internally consistent with "
            "per-fold CV metrics."),
        "n_features": X.shape[1],
        "n_samples":  int(y.size),
        "label_counts": {
            "illicit": int((y == 1).sum()),
            "licit":   int((y == 0).sum()),
        },
        "n_cv_folds": folds,
        "unreliable_warning": int((y == 1).sum()) < MIN_ILLICIT_CV,
    }


# ── Rule validation against confirmed labels ──────────────────────────────────

def evaluate_rules(df: pd.DataFrame) -> dict:
    """Per-rule precision / recall against the binary label in df."""
    y_true = (df["pseudo_label"] == 1).astype(int).values
    n_illicit = int(y_true.sum())
    results = {}

    for rule in RULE_COLS:
        if rule not in df.columns:
            results[rule] = {"note": "Rule column absent."}
            continue
        fired = (df[rule] == 1).astype(int).values
        n_fired = int(fired.sum())
        if n_fired == 0:
            results[rule] = {
                "n_fired": 0, "precision": None, "recall": None,
                "note": "Rule did not fire on any evaluation row.",
            }
            continue
        tp = int((fired & y_true).sum())
        prec = tp / n_fired
        rec  = tp / n_illicit if n_illicit > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        results[rule] = {
            "n_fired":   n_fired,
            "n_tp":      tp,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        }

    if "suspicion_score" in df.columns:
        any_fired = (df["suspicion_score"] >= 1).astype(int).values
        n_any = int(any_fired.sum())
        tp_any = int((any_fired & y_true).sum())
        prec_any = tp_any / n_any if n_any > 0 else 0.0
        rec_any  = tp_any / n_illicit if n_illicit > 0 else 0.0
        f1_any = ((2 * prec_any * rec_any / (prec_any + rec_any))
                  if (prec_any + rec_any) > 0 else 0.0)
        results["any_rule"] = {
            "n_fired":   n_any,
            "n_tp":      tp_any,
            "precision": round(prec_any, 4),
            "recall":    round(rec_any, 4),
            "f1":        round(f1_any, 4),
        }
    return results


# ── Suspicion-score threshold sweep on a given subset ─────────────────────────

def sweep_thresholds(df: pd.DataFrame) -> dict:
    """
    Sweep integer thresholds 0–6 of suspicion_score against the binary label
    in df and identify the empirically optimal threshold.  Caller is
    responsible for ensuring the subset has both classes — single-class
    subsets are excluded upstream.
    """
    if "suspicion_score" not in df.columns:
        return {"note": "suspicion_score column absent."}

    y_true = (df["pseudo_label"] == 1).astype(int).values
    n_illicit = int(y_true.sum())
    n_licit   = int((y_true == 0).sum())
    if n_illicit == 0 or n_licit == 0:
        return {
            "skipped": True,
            "note": ("Single-class subset — threshold sweep would produce "
                     f"base-rate artefacts ({n_illicit} illicit, {n_licit} licit)."),
        }

    per_threshold = {}
    best_f1, best_t = -1.0, None
    for t in range(7):
        flagged = (df["suspicion_score"] >= t).astype(int).values
        n_f = int(flagged.sum())
        if n_f == 0:
            per_threshold[str(t)] = {
                "n_flagged": 0, "precision": None, "recall": None, "f1": None,
                "note": "No transactions flagged at this threshold.",
            }
            continue
        tp = int((flagged & y_true).sum())
        prec = tp / n_f
        rec  = tp / n_illicit
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_threshold[str(t)] = {
            "n_flagged": n_f,
            "n_tp":      tp,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        }
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    return {
        "per_threshold":      per_threshold,
        "optimal_threshold":  best_t,
        "optimal_f1":         round(best_f1, 4) if best_f1 >= 0 else None,
        "current_default":    DEFAULT_RULE_THRESHOLD,
        "current_default_f1": per_threshold.get(str(DEFAULT_RULE_THRESHOLD), {}).get("f1"),
    }


# ── Per-source breakdown ──────────────────────────────────────────────────────

def split_sources(value: str) -> list:
    """Split the ``confirmation_source`` value into individual source names."""
    if not value or pd.isna(value):
        return []
    # Handle both delimiters defensively (older runs used "," ; current uses ";").
    tokens = []
    for chunk in str(value).split(";"):
        for sub in chunk.split(","):
            s = sub.strip()
            if s:
                tokens.append(s)
    return tokens


def evaluate_per_source(df: pd.DataFrame) -> dict:
    """
    For each individual source name appearing in confirmation_source, build
    the subset of confirmed-label rows attributed to that source and report:
      - rule firing rates,
      - supervised CV (skipped or single-class flagged),
      - threshold sweep (only when the subset contains both classes).
    """
    if "confirmation_source" not in df.columns:
        return {"note": "confirmation_source column absent."}

    sources = set()
    for v in df["confirmation_source"].dropna().unique():
        sources.update(split_sources(v))

    feat_cols = get_feature_cols(df)
    results = {}
    for src in sorted(sources):
        mask = df["confirmation_source"].fillna("").apply(
            lambda v: src in split_sources(v)
        )
        sub = df[mask].copy()
        n_ill = int((sub["pseudo_label"] == 1).sum())
        n_lit = int((sub["pseudo_label"] == 0).sum())

        src_res = {
            "n_transactions": int(len(sub)),
            "n_illicit":      n_ill,
            "n_licit":        n_lit,
            "rule_firing_rates": {
                r: round(float(sub[r].mean()), 4)
                for r in RULE_COLS if r in sub.columns
            },
        }

        if n_lit == 0 or n_ill == 0:
            src_res["supervised_cv"] = {
                "skipped": True,
                "note": ("Single-class source — supervised CV requires both "
                         f"illicit and licit examples ({n_ill} illicit, "
                         f"{n_lit} licit)."),
            }
            src_res["threshold_sweep"] = {
                "skipped": True,
                "note": ("Single-class source — threshold sweep would produce "
                         f"base-rate artefacts ({n_ill} illicit, {n_lit} licit)."),
            }
        elif n_ill < MIN_ILLICIT_CONFIRMED:
            src_res["supervised_cv"] = {
                "skipped": True,
                "note": (f"Insufficient illicit examples for reliable CV "
                         f"({n_ill} illicit, minimum {MIN_ILLICIT_CONFIRMED})."),
            }
            src_res["threshold_sweep"] = sweep_thresholds(sub)
        else:
            X = sub[feat_cols]
            y = (sub["pseudo_label"] == 1).astype(int).values
            src_res["supervised_cv"] = cv_metric_bundle(X, y, label=src)
            src_res["threshold_sweep"] = sweep_thresholds(sub)

        results[src] = src_res

    return results


# ── Pretty printing helpers ───────────────────────────────────────────────────

def _print_cv_summary(name: str, bundle: dict) -> None:
    if bundle.get("skipped"):
        print(f"  {name}: skipped — {bundle.get('note')}")
        return
    cm = bundle["cv_metrics"]
    print(f"  {name}:")
    print(f"    F1 (macro / weighted) : "
          f"{cm['f1_macro']['mean']:.4f} / {cm['f1_weighted']['mean']:.4f}")
    print(f"    P / R illicit         : "
          f"{cm['precision_illicit']['mean']:.4f} / "
          f"{cm['recall_illicit']['mean']:.4f}")
    print(f"    P / R licit           : "
          f"{cm['precision_licit']['mean']:.4f} / "
          f"{cm['recall_licit']['mean']:.4f}")
    print(f"    ROC-AUC / PR-AUC      : "
          f"{cm['roc_auc']['mean']:.4f} / {cm['pr_auc']['mean']:.4f}")


def _print_threshold_sweep(name: str, sweep: dict) -> None:
    if sweep.get("skipped"):
        print(f"  Threshold sweep ({name}): skipped — {sweep.get('note')}")
        return
    print(f"  Threshold sweep ({name}):")
    for t, m in sweep.get("per_threshold", {}).items():
        if m.get("f1") is None:
            print(f"    >= {t}: (none flagged)")
        else:
            print(f"    >= {t}: P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"F1={m['f1']:.4f}  (flagged={m['n_flagged']:,})")
    print(f"    optimal threshold = {sweep['optimal_threshold']}  "
          f"(F1={sweep['optimal_f1']:.4f})")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Bitcoin Illicit Detection — Mechanism Validation")
    print(f"  Classifier  : {'LightGBM' if USE_LGBM else 'Random Forest'}")
    print(f"  Evaluation  : confirmed labels + presumed-licit augmentation")
    print("=" * 60 + "\n")

    if not os.path.exists(LABELED_CSV):
        print(f"[ERROR] Labeled features not found: {LABELED_CSV}")
        print("  Run the pipeline through 03_label.py first.")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load and build augmented eval set ─────────────────────────────
    print("Step 1 — Loading labeled features and building augmented eval set…")
    df_full = load_labeled(LABELED_CSV)
    df_eval = build_augmented_eval_set(df_full)
    df_eval.to_csv(EVAL_SET_CSV, index=False)
    print(f"  Augmented eval set written to {EVAL_SET_CSV}")

    # Confirmed-only (legacy) view, kept for back-compat reporting.
    df_confirmed = df_full[df_full["label_type"] == "confirmed"].copy()
    n_conf = len(df_confirmed)
    n_conf_ill = int((df_confirmed["pseudo_label"] == 1).sum())
    n_conf_lit = int((df_confirmed["pseudo_label"] == 0).sum())

    # ── Step 2: Rule validation on the augmented set ──────────────────────────
    print("\nStep 2 — Rule validation against augmented eval labels…")
    rules_aug = evaluate_rules(df_eval)
    for r in RULE_COLS:
        m = rules_aug.get(r, {})
        if m.get("precision") is None:
            print(f"  {r:25s}: {m.get('note', 'n/a')}")
        else:
            print(f"  {r:25s}: P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"F1={m['f1']:.4f}  (fired={m['n_fired']:,})")
    if "any_rule" in rules_aug:
        ar = rules_aug["any_rule"]
        print(f"\n  any_rule                 : "
              f"P={ar['precision']:.4f}  R={ar['recall']:.4f}  "
              f"F1={ar['f1']:.4f}  (fired={ar['n_fired']:,})")

    # ── Step 3: Per-source threshold sweeps ───────────────────────────────────
    print("\nStep 3 — Per-source threshold sweeps "
          "(only sources with both classes)…")
    per_source_sweeps = {}
    # 3a — augmented full-set sweep.
    aug_sweep = sweep_thresholds(df_eval)
    per_source_sweeps["augmented_full"] = aug_sweep
    _print_threshold_sweep("augmented_full", aug_sweep)
    # 3b — sweep per single source where confirmed-licit exists.
    src_sweeps_meta = evaluate_per_source(df_confirmed)
    for src, res in sorted(src_sweeps_meta.items()):
        sweep = res.get("threshold_sweep", {})
        per_source_sweeps[src] = sweep
        _print_threshold_sweep(src, sweep)

    # ── Step 4: Supervised CV (augmented) ─────────────────────────────────────
    print("\nStep 4 — Supervised CV on augmented eval set…")
    feat_cols = get_feature_cols(df_eval)
    X_aug = df_eval[feat_cols]
    y_aug = (df_eval["pseudo_label"] == 1).astype(int).values
    aug_cv = cv_metric_bundle(X_aug, y_aug, label="augmented_full")

    # ── Step 5: Supervised CV (legacy: confirmed-only) ────────────────────────
    print("\nStep 5 — Supervised CV on confirmed-only subset (legacy view)…")
    if n_conf_ill < MIN_ILLICIT_CONFIRMED or n_conf_lit == 0:
        confirmed_cv = {
            "skipped": True,
            "note": ("Single-class or insufficient — confirmed-only subset has "
                     f"{n_conf_ill} illicit, {n_conf_lit} licit."),
        }
        print(f"  Skipped: {confirmed_cv['note']}")
    else:
        X_c = df_confirmed[feat_cols]
        y_c = (df_confirmed["pseudo_label"] == 1).astype(int).values
        confirmed_cv = cv_metric_bundle(X_c, y_c, label="confirmed_only")

    # ── Step 6: Per-source breakdown (CV + sweep) ─────────────────────────────
    print("\nStep 6 — Per-source breakdown…")
    source_results = src_sweeps_meta  # already populated in step 3b
    for src in sorted(source_results.keys()):
        res = source_results[src]
        print(f"\n  Source: {src}")
        print(f"    Rows : {res['n_transactions']:,}  "
              f"(illicit={res['n_illicit']}, licit={res['n_licit']})")
        _print_cv_summary("Supervised CV", res["supervised_cv"])

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "status":     "complete",
        "classifier": "lightgbm" if USE_LGBM else "random_forest",
        "schema_version": 2,
        "schema_changes": [
            "v2: confusion matrix is now from out-of-fold predictions; "
            "augmented eval set added; per-source threshold sweep; "
            "expanded metrics (ROC-AUC, PR-AUC, per-class P/R, weighted F1).",
        ],
        "n_confirmed_transactions": n_conf,
        "n_confirmed_illicit":      n_conf_ill,
        "n_confirmed_licit":        n_conf_lit,
        "augmented_eval_set": {
            "n_total":           int(len(df_eval)),
            "n_confirmed_illicit": int((df_eval["eval_label_source"]
                                         == "confirmed_illicit").sum()),
            "n_confirmed_licit":   int((df_eval["eval_label_source"]
                                         == "confirmed_licit").sum()),
            "n_presumed_licit":    int((df_eval["eval_label_source"]
                                         == "presumed_licit").sum()),
        },
        "rule_validation_augmented": rules_aug,
        "supervised_cv_augmented":   aug_cv,
        "supervised_cv_confirmed":   confirmed_cv,
        "threshold_sweeps":          per_source_sweeps,
        "per_source_breakdown":      source_results,
    }

    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[OK] Results saved → {RESULTS_PATH}")
    print(f"     Augmented eval set saved → {EVAL_SET_CSV}")
    print(f"\n  Confirmed-only subset      : {n_conf:,} "
          f"(illicit={n_conf_ill:,}, licit={n_conf_lit:,})")
    print(f"  Augmented eval subset      : {len(df_eval):,} "
          f"(illicit={int((df_eval['pseudo_label']==1).sum()):,}, "
          f"licit={int((df_eval['pseudo_label']==0).sum()):,})")
    if not aug_cv.get("skipped"):
        cm = aug_cv["cv_metrics"]
        print(f"  CV macro F1 (augmented)    : "
              f"{cm['f1_macro']['mean']:.4f}  "
              f"(weighted {cm['f1_weighted']['mean']:.4f})")
        print(f"  ROC-AUC / PR-AUC (augmented): "
              f"{cm['roc_auc']['mean']:.4f} / {cm['pr_auc']['mean']:.4f}")
