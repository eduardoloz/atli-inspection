#!/usr/bin/env python3
"""Figure for results/dataset_augmentation_summary.md: per-class training-set
composition before vs after the x3 Defective_Damper image-level oversampling.

Counts computed from the label files on the server (2026-07-09):
~/atli/ATLI_target_tightNI_noCPLID/train/labels (original) and
~/atli/ATLI_noCPLID_OS3/train/labels (after oversampling).

Output: results/figures/dataset_summary/fig_train_composition.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "dataset_summary"
OUT.mkdir(exist_ok=True)

# same validated palette as generate_clean_vs_prior.py
BLUE, AQUA = "#2a78d6", "#1baf7a"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

classes = ["Birdnest", "Broken\nInsulator", "Defective\nDamper",
           "Flashover\nInsulator", "Normal\nDamper", "Normal\nInsulators",
           "Self-Exploded\nInsulator"]
inst_orig = [162, 140, 82, 302, 833, 604, 221]
inst_aug = [172, 142, 246, 304, 1013, 710, 223]
imgs_orig = [153, 91, 36, 93, 187, 252, 148]
imgs_aug = [163, 93, 108, 95, 243, 302, 150]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
x = np.arange(len(classes))
w = 0.36
for ax, orig, aug, title in [
    (axes[0], imgs_orig, imgs_aug,
     "Training images containing the class"),
    (axes[1], inst_orig, inst_aug,
     "Training instances (annotated boxes)"),
]:
    ax.bar(x - w / 2 - 0.01, orig, w, color=BLUE, label="Original train (561 imgs)")
    ax.bar(x + w / 2 + 0.01, aug, w, color=AQUA,
           label="After ×3 oversampling (633 imgs)")
    for xi, v in zip(x - w / 2 - 0.01, orig):
        ax.text(xi, v + max(orig + aug) * 0.012, f"{v:,}", ha="center",
                va="bottom", fontsize=8, color=INK2)
    for xi, v in zip(x + w / 2 + 0.01, aug):
        ax.text(xi, v + max(orig + aug) * 0.012, f"{v:,}", ha="center",
                va="bottom", fontsize=8,
                color=INK if v > orig[list(x).index(round(xi))] else INK2)
    ax.set_xticks(x, classes, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title)
axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
fig.suptitle("Train split before vs after augmentation — only Defective_Damper "
             "images are duplicated (×3); other classes grow only by co-occurrence; "
             "val/test untouched", fontsize=10, color=INK2, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "fig_train_composition.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "fig_train_composition.png")
