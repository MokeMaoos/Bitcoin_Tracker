"""
run_pipeline.py — run the full detection pipeline end-to-end.

Pipeline stages (in order):
    01_load.py          — data loading and validation
    02_feature.py       — feature engineering (38 features per transaction)
    02b_enrich_labels.py — external label enrichment (OFAC, Elliptic++, etc.)
                           Skipped if --skip-enrichment is passed.
    03_label.py         — FATF rule-based pseudo-labelling with confirmed-label
                           priority where 02b_enrich_labels.py has run
    04_train.py         — unsupervised clustering + supervised classification

External label retrieval (stage 02b) occurs only during training and validation.
The saved model artefact is self-contained: once trained and saved, 05_predict.py
operates entirely offline on any Bitcoin transaction data without runtime
network calls or external lookups.

Usage:
    python run_pipeline.py                    # full pipeline with enrichment
    python run_pipeline.py --skip-enrichment  # skip external label retrieval
    python run_pipeline.py --refresh          # re-download external label data
    python run_pipeline.py --label-mode confirmed_only
"""

import argparse
import subprocess
import sys
import time
import os

STEPS_ALWAYS = [
    ("01_load.py",    "Data load & validation"),
    ("02_feature.py", "Feature engineering"),
]
STEP_ENRICH = ("02b_enrich_labels.py", "External label enrichment")
STEPS_AFTER = [
    ("03_label.py",   "Rule-based pseudo-labelling"),
    ("04_train.py",   "Model training"),
]


def run_step(script: str, label: str, extra_args: list = None) -> bool:
    print(f"\n{'=' * 55}")
    print(f"  STEP: {label}")
    print(f"{'=' * 55}")
    cmd     = [sys.executable, script] + (extra_args or [])
    t0      = time.time()
    result  = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode}")
        return False
    print(f"\n  Completed in {elapsed:.1f} s")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bitcoin illicit-transaction detection pipeline."
    )
    parser.add_argument(
        "--skip-enrichment", action="store_true",
        help="Skip 02b_enrich_labels.py (external data retrieval). "
             "Downstream stages will use pseudo-labels only.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Pass --refresh to 02b_enrich_labels.py to re-download "
             "all external label data even if cached copies exist.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["hybrid", "pseudo_only", "confirmed_only"],
        default=None,
        help="Label mode forwarded to 04_train.py.",
    )
    args = parser.parse_args()

    total_start = time.time()
    print("\nBitcoin Illicit Transaction Detection — Full Pipeline")

    # ── Always-on steps ────────────────────────────────────────────────────────
    for script, label in STEPS_ALWAYS:
        if not run_step(script, label):
            print("\nPipeline aborted.")
            sys.exit(1)

    # ── Optional enrichment step ───────────────────────────────────────────────
    if args.skip_enrichment:
        print(f"\n{'=' * 55}")
        print(f"  STEP: {STEP_ENRICH[1]} — SKIPPED (--skip-enrichment)")
        print(f"{'=' * 55}")
        print("  Downstream stages will fall back to pseudo-labels only.")
    else:
        enrich_args = ["--refresh"] if args.refresh else []
        if not run_step(STEP_ENRICH[0], STEP_ENRICH[1], extra_args=enrich_args):
            print("\nPipeline aborted.")
            sys.exit(1)

    # ── Labelling and training ─────────────────────────────────────────────────
    for script, label in STEPS_AFTER:
        extra = []
        if script == "04_train.py" and args.label_mode:
            extra = ["--label-mode", args.label_mode]
        if not run_step(script, label, extra_args=extra):
            print("\nPipeline aborted.")
            sys.exit(1)

    total = time.time() - total_start
    label_mode = args.label_mode or ("hybrid" if not args.skip_enrichment else "pseudo_only")
    print(f"\n{'=' * 55}")
    print(f"  Pipeline complete in {total:.1f} s")
    print(f"  Model saved to: outputs/model_{label_mode}.pkl")
    print(f"                  outputs/model.pkl  (canonical alias)")
    print(f"  Predict with:")
    print(f"    python 05_predict.py --wallet <wallet_id>")
    print(f"    python 05_predict.py --tx <block_no> <trans_no>")
    print(f"{'=' * 55}\n")
