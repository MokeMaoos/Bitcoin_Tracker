"""
run_pipeline.py — run the full detection pipeline end-to-end.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import time
import os

STEPS = [
    ("01_load.py",    "Data load & validation"),
    ("02_feature.py", "Feature engineering"),
    ("03_label.py",   "Rule-based pseudo-labelling"),
    ("04_train.py",   "Model training"),
]


def run_step(script: str, label: str) -> bool:
    print(f"\n{'=' * 55}")
    print(f"  STEP: {label}")
    print(f"{'=' * 55}")
    t0     = time.time()
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode}")
        return False
    print(f"\n  Completed in {elapsed:.1f} s")
    return True


if __name__ == "__main__":
    total_start = time.time()
    print("\nBitcoin Illicit Transaction Detection — Full Pipeline")

    for script, label in STEPS:
        if not run_step(script, label):
            print("\nPipeline aborted.")
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{'=' * 55}")
    print(f"  Pipeline complete in {total:.1f} s")
    print(f"  Model saved to: outputs/model.pkl")
    print(f"  Predict with:")
    print(f"    python 05_predict.py --wallet <wallet_id>")
    print(f"    python 05_predict.py --tx <block_no> <trans_no>")
    print(f"{'=' * 55}\n")
