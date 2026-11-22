#!/usr/bin/env python3
"""
make_fig4_v3_2panel.py - Figure 4 (two-panel variant): ablation, and the phage-held-out sweep decomposed.

Panel A  ablation (phage-held-out bar corrected to the --dedup value, 0.764)
Panel B  ROC-AUC across BLASTp identity thresholds, pooled and split
Panel C  PR-AUC relative to each subset's own base rate

All sweep points measured in one environment with --dedup, grouping by BLASTp
percent identity with a 50% query-coverage floor. The 100% point is bit-identical
to the published normalised-Levenshtein run, which anchors the curve.

The split uses a FIXED set of 39 phages at every threshold - a BLASTp-connected
component that the grouping never mixes with any outside phage. At 100% those 39
sit in 33 separate folds; by 70% they are a single fold. Nothing about the
predictions changes across that range except the fold boundaries.

The arrays below are transcribed, not computed here: the per-subset values from
results/a1_family_split.csv (written by analysis/05_metric_audit/fold_breakdown.py)
and the pooled values from the results/blast_sweep_id*.log run logs.

    python manuscript/make_fig4_v3_2panel.py --outdir .
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Panel A --------------------------------------------------------------
ABLATION = [
    ("RBP + serotype\n(PHL-RBP+S)", 0.817, 0.764),
    ("RBP only\n(no host)",         0.701, 0.673),
    ("Serotype only\n(no phage)",   0.589, None),
]

# ---- Panels B and C -------------------------------------------------------
THRESH = ["100", "95", "90", "80", "70", "60", "50"]

POOLED_ROC = [0.764, 0.686, 0.661, 0.631, 0.588, 0.579, 0.575]
POOLED_PR  = [0.262, 0.101, 0.097, 0.088, 0.058, 0.056, 0.056]
POOLED_BASE = 0.0333

CLUSTER_ROC = [0.709, 0.629, 0.629, 0.496, 0.439, 0.439, 0.439]
CLUSTER_PR  = [0.306, 0.027, 0.027, 0.025, 0.014, 0.014, 0.014]
CLUSTER_BASE = 0.0168          # 70 positives / 4,166 pairs

OTHER_ROC = [0.772, 0.694, 0.670, 0.665, 0.639, 0.624, 0.623]
OTHER_PR  = [0.252, 0.130, 0.127, 0.128, 0.099, 0.095, 0.099]
OTHER_BASE = 0.0450            # 263 positives / 5,840 pairs

# folds the 39-phage cluster is spread across, per threshold
CLUSTER_FOLDS = [33, 6, 6, 2, 1, 1, 1]

C_BLUE = "#1f77b4"      # the other 66 phages
C_RED = "#d62728"       # the 39-phage cluster
C_GREY = "#555555"      # pooled: a derived aggregate, deliberately neutral
C_RULE = "#7f7f7f"


def panel_a(ax):
    x = np.arange(len(ABLATION))
    w = 0.4
    logo = [v for _, v, _ in ABLATION]
    phage = [p for _, _, p in ABLATION]

    ax.bar(x - w / 2, logo, w, color=C_BLUE, label="Bacterial LOGO (100%)")
    for i, p in enumerate(phage):
        if p is not None:
            ax.bar(x[i] + w / 2, p, w, color=C_RED,
                   label="Phage-held-out" if i == 0 else None)

    for i, v in enumerate(logo):
        ax.text(x[i] - w / 2, v + 0.004, f"{v:.3f}", ha="center",
                va="bottom", fontsize=7.5)
    for i, p in enumerate(phage):
        if p is not None:
            ax.text(x[i] + w / 2, p + 0.004, f"{p:.3f}", ha="center",
                    va="bottom", fontsize=7.5)

    ax.axhline(0.5, ls=":", c=C_RULE, lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in ABLATION], fontsize=7.5)
    ax.set_ylim(0.45, 0.90)
    ax.set_yticks(np.arange(0.45, 0.91, 0.05))
    ax.set_ylabel("ROC-AUC", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.60, 1.03))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def sweep_axes(ax):
    x = np.arange(len(THRESH))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}%" for t in THRESH], fontsize=8)
    ax.set_xlim(-0.45, len(THRESH) - 0.55)
    ax.set_xlabel("BLASTp identity permitted between folds", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#ececec", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    return x


def panel_b(ax):
    x = sweep_axes(ax)
    ax.axhline(0.5, ls=":", c=C_RULE, lw=1.1)
    ax.text(len(THRESH) - 0.6, 0.508, "chance", fontsize=7.5, color=C_RULE,
            ha="right", va="bottom")

    ax.plot(x, OTHER_ROC, "-o", color=C_BLUE, lw=1.9, ms=4.5,
            label="Other 66 phages")
    ax.plot(x, POOLED_ROC, "--", color=C_GREY, lw=1.4, label="All phages")
    ax.plot(x, CLUSTER_ROC, "-s", color=C_RED, lw=1.9, ms=4.2,
            label="39-phage cluster")

    ax.set_ylim(0.40, 0.82)
    ax.set_ylabel("ROC-AUC", fontsize=9)


def panel_c(ax):
    x = sweep_axes(ax)
    ax.axhline(1.0, ls=":", c=C_RULE, lw=1.1)
    ax.text(len(THRESH) - 0.6, 1.05, "base rate", fontsize=7.5, color=C_RULE,
            ha="right", va="bottom")

    other = [p / OTHER_BASE for p in OTHER_PR]
    pooled = [p / POOLED_BASE for p in POOLED_PR]
    cluster = [p / CLUSTER_BASE for p in CLUSTER_PR]

    ax.plot(x, other, "-o", color=C_BLUE, lw=1.9, ms=4.5,
            label="Other 66 phages")
    ax.plot(x, pooled, "--", color=C_GREY, lw=1.4, label="All phages")
    ax.plot(x, cluster, "-s", color=C_RED, lw=1.9, ms=4.2,
            label="39-phage cluster")

    ax.set_yscale("log")
    ax.set_ylim(0.6, 30)
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×"])
    ax.minorticks_off()
    ax.set_ylabel("PR-AUC relative to subset base rate", fontsize=9)

    # the mechanism: the cluster goes from 33 folds to 6 in one step
    ax.annotate(f"{CLUSTER_FOLDS[0]} folds", (0, cluster[0]),
                textcoords="offset points", xytext=(4, 6),
                fontsize=7.5, color=C_RED, ha="left")
    ax.annotate("1 fold", (4, cluster[4]), textcoords="offset points",
                xytext=(0, -14), fontsize=7.5, color=C_RED, ha="center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})

    fig, (ax1, ax3) = plt.subplots(
        1, 2, figsize=(9.2, 3.6),
        gridspec_kw={"width_ratios": [1.05, 1.25]})

    panel_a(ax1)
    panel_c(ax3)

    handles, labels = ax3.get_legend_handles_labels()
    order = [0, 2, 1]        # other 66, cluster, pooled
    ax3.legend([handles[i] for i in order], [labels[i] for i in order],
               fontsize=7.5, frameon=False, loc="lower left",
               bbox_to_anchor=(0.0, 0.02))

    for ax, lab in ((ax1, "A"), (ax3, "B")):
        ax.set_title(lab, loc="left", fontsize=11, fontweight="bold", x=-0.02)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"FigNew1_ablation_2panel.{ext}"),
                    dpi=300)
    print("wrote FigNew1_ablation_2panel.pdf / .png")

    print(f"\n{'thr':>5}{'pooled':>9}{'cluster':>9}{'other':>8}"
          f"{'  |':>4}{'p-lift':>8}{'c-lift':>8}{'o-lift':>8}")
    for i, t in enumerate(THRESH):
        print(f"{t:>5}{POOLED_ROC[i]:>9.3f}{CLUSTER_ROC[i]:>9.3f}"
              f"{OTHER_ROC[i]:>8.3f}{'  |':>4}"
              f"{POOLED_PR[i]/POOLED_BASE:>7.2f}x"
              f"{CLUSTER_PR[i]/CLUSTER_BASE:>7.2f}x"
              f"{OTHER_PR[i]/OTHER_BASE:>7.2f}x")


if __name__ == "__main__":
    main()
