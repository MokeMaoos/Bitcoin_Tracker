"""Build outputs/pipeline_architecture.png — Bitcoin AML pipeline diagram."""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY    = "#2F5597"
ORANGE  = "#ED7D31"
GREY    = "#7F7F7F"
LIGHT   = "#F2F2F2"
WHITE   = "#FFFFFF"


def box(ax, x, y, w, h, text, fill=NAVY, fc=WHITE, fontsize=10, lw=0.7):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.04,rounding_size=0.12",
                       linewidth=lw, edgecolor=NAVY, facecolor=fill)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=fc, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color=NAVY, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, color=color, lw=1.0,
                        mutation_scale=12,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis("off")

# Title
ax.text(9, 11.5, "Bitcoin Illicit-Transaction Detection Pipeline",
        ha="center", fontsize=15, color=NAVY, fontweight="bold")
ax.text(9, 11.05, "BMAN31260 — operationalising FATF red-flag indicators on the CUBiD database",
        ha="center", fontsize=10, color=GREY, style="italic")

# Layer labels (left)
for y, lab in [(9.5, "INPUT"), (7.0, "PROCESSING"),
               (4.0, "MODELLING"), (1.4, "EVALUATION")]:
    ax.text(0.15, y, lab, fontsize=9, color=GREY, fontweight="bold",
            rotation=90, va="center")

# ── Input layer ────────────────────────────────────────────────────────────
input_tables = [
    "Transactions",
    "Trans_Detail",
    "Address_Info",
    "Wallet_Info",
    "Block_Header",
    "Address_Activity",
    "Wallet_Activity",
]
for i, name in enumerate(input_tables):
    box(ax, 1.0 + i * 2.4, 9.0, 2.1, 0.85, name, fill=NAVY, fontsize=9)

# Brace to processing
arrow(ax, 9.0, 8.95, 9.0, 8.10, color=GREY)

# ── Processing layer ──────────────────────────────────────────────────────
proc_y = 6.6
proc_h = 1.1
proc_specs = [
    (1.0, "Feature Engineering\n(35 deployed features)\n02_feature.py"),
    (5.5, "Label Enrichment\n(4 external sources)\n02b_enrich_labels.py"),
    (10.0, "Pseudo-labelling\n(6 FATF rules)\n03_label.py"),
    (14.5, "Hybrid Label Set\nconfirmed > pseudo > NA"),
]
for x, t in proc_specs:
    box(ax, x, proc_y, 3.5, proc_h, t, fill=ORANGE, fontsize=8.5)
for i in range(len(proc_specs) - 1):
    x1 = proc_specs[i][0] + 3.5
    x2 = proc_specs[i + 1][0]
    arrow(ax, x1, proc_y + proc_h / 2, x2, proc_y + proc_h / 2,
          color=ORANGE)

# Bridge to modelling layer
arrow(ax, 16.25, proc_y, 16.25, 4.6, color=ORANGE)
arrow(ax, 16.25, 4.6, 4.0, 4.6, color=ORANGE)
arrow(ax, 16.25, 4.6, 11.0, 4.6, color=ORANGE)

# ── Modelling layer ───────────────────────────────────────────────────────
mod_y = 3.4
mod_h = 1.1
box(ax, 1.5, mod_y, 5.5, mod_h,
    "Unsupervised Clustering\nK-Means (k=3)\nDBSCAN (eps=1.5, min_samples=5)\n04_train.py",
    fill=NAVY, fontsize=9)
box(ax, 8.5, mod_y, 5.5, mod_h,
    "Supervised Classification\nLightGBM (primary)\nRandom Forest (fallback)\n04_train.py",
    fill=NAVY, fontsize=9)

# Side branch: Streamlit dashboard
box(ax, 15.2, mod_y, 2.5, mod_h,
    "Streamlit\nDashboard\napp.py + 7 pages",
    fill=GREY, fontsize=9)
arrow(ax, 14.0, mod_y + mod_h / 2, 15.2, mod_y + mod_h / 2, color=GREY)

# Bridge from modelling to evaluation
arrow(ax, 4.25, mod_y, 4.25, 2.1, color=NAVY)
arrow(ax, 11.25, mod_y, 11.25, 2.1, color=NAVY)

# ── Evaluation layer ──────────────────────────────────────────────────────
eval_y = 1.0
eval_h = 1.0
eval_specs = [
    (1.0, "Mechanism Validation\n(confirmed-label CV)\n08_mechanism_eval.py"),
    (6.5, "Temporal Validation\n(2019-22 vs 2023-24)\n07_temporal_eval.py"),
    (12.0, "External Benchmark\n(Elliptic++)\n06_evaluate.py"),
]
for x, t in eval_specs:
    box(ax, x, eval_y, 5.0, eval_h, t, fill=NAVY, fontsize=8.5)

# Legend
ax.text(0.4, 0.15, "Input data", fontsize=8, color=NAVY, fontweight="bold")
ax.text(3.4, 0.15, "Processing", fontsize=8, color=ORANGE, fontweight="bold")
ax.text(6.4, 0.15, "Modelling / evaluation", fontsize=8, color=NAVY,
        fontweight="bold")
ax.text(11.4, 0.15, "Auxiliary (dashboard)", fontsize=8, color=GREY,
        fontweight="bold")

os.makedirs("outputs", exist_ok=True)
out_path = os.path.abspath("outputs/pipeline_architecture.png")
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
print(f"[OK] Saved {out_path}")
