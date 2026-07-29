#!/usr/bin/env python3
"""Generate print-quality vector figures for the paper from the logged benchmark numbers.

Sources of every number:
  - results/cv_proper_results.md   (proper 5-fold CV)
  - results/sota_cv_results.md     (MMDetection SOTA CV, same folds)
  - results/cplid_before_after.md  (clean no-CPLID benchmark, 3 seeds)
  - CLAUDE.md experiment log       (768 native retrain 4-seed, deploy ladder, TRT parity)
Outputs PDF (vector) into paper/figures/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#2166ac"
GREEN = "#1a9850"
GRAY = "#666666"
ORANGE = "#e08214"

# ---------------------------------------------------------------- Fig 1: accuracy vs params
fig, ax = plt.subplots(figsize=(3.5, 2.5))
pts = [
    ("YOLOv11n + our recipe", 2.6, 0.785, GREEN, "o", 60),
    ("RTMDet-Tiny", 4.8, 0.747, BLUE, "o", 45),
    ("Dynamic R-CNN", 41.0, 0.718, BLUE, "o", 45),
    ("DINO-4scale", 47.0, 0.785, BLUE, "o", 45),
]
for name, p, m, c, mk, s in pts:
    ax.scatter(p, m, c=c, marker=mk, s=s, zorder=3, edgecolors="white", linewidths=0.5)
ax.annotate("YOLOv11n + our recipe\n2.6M / 0.785", (2.6, 0.785), textcoords="offset points",
            xytext=(6, 6), fontsize=7.5, color=GREEN, fontweight="bold")
ax.annotate("RTMDet-Tiny\n4.8M / 0.747", (4.8, 0.747), textcoords="offset points",
            xytext=(6, -14), fontsize=7.5, color=BLUE)
ax.annotate("Dynamic R-CNN\n41M / 0.718", (41.0, 0.718), textcoords="offset points",
            xytext=(-30, -16), fontsize=7.5, color=BLUE)
ax.annotate("DINO-4scale\n47M / 0.785", (47.0, 0.785), textcoords="offset points",
            xytext=(-40, 7), fontsize=7.5, color=BLUE)
ax.set_xscale("log")
ax.set_xlim(1.8, 90)
ax.set_ylim(0.70, 0.80)
ax.set_xticks([2, 5, 10, 20, 50])
ax.set_xticklabels(["2", "5", "10", "20", "50"])
ax.set_xlabel("Parameters (millions, log scale)")
ax.set_ylabel("mAP@0.5 (5-fold CV)")
ax.grid(True, which="both", axis="both", alpha=0.25, linewidth=0.4)
fig.savefig(OUT / "fig_frontier.pdf")
fig.savefig(OUT / "fig_frontier.png", dpi=300)
plt.close(fig)

# ------------------------------------------------- Fig 2: per-class clean benchmark (3 seeds)
classes = ["Birdnest", "Broken\nInsulator", "Defective\nDamper", "Flashover\nInsulator",
           "Normal\nDamper", "Normal\nInsulators", "Self-Expl.\nInsulator", "overall\n(mAP)"]
base = [0.959, 0.667, 0.614, 0.590, 0.743, 0.826, 0.755, 0.736]
base_e = [0.010, 0.023, 0.082, 0.011, 0.023, 0.015, 0.016, 0.015]
champ = [0.984, 0.699, 0.622, 0.660, 0.813, 0.838, 0.871, 0.784]
champ_e = [0.004, 0.029, 0.073, 0.024, 0.019, 0.013, 0.027, 0.011]

x = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(7.0, 2.3))
ax.errorbar(x - 0.12, base, yerr=base_e, fmt="o", ms=5, capsize=2.5, lw=0, elinewidth=0.9,
            color=BLUE, label="base (640 px)")
ax.errorbar(x + 0.12, champ, yerr=champ_e, fmt="D", ms=4.5, capsize=2.5, lw=0, elinewidth=0.9,
            color=GREEN, label="champ (1280 px, 3$\\times$ oversampling, scale 0.9)")
for xi_, b, c in zip(x, base, champ):
    ax.plot([xi_ - 0.12, xi_ + 0.12], [b, c], color=GRAY, lw=0.5, alpha=0.5, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0.5, 1.03)
ax.set_ylabel("AP@0.5")
ax.axvline(6.5, color=GRAY, lw=0.6, ls=":")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
fig.savefig(OUT / "fig_perclass_clean.pdf")
fig.savefig(OUT / "fig_perclass_clean.png", dpi=300)
plt.close(fig)

# ------------------------------------------------- Fig: confusion matrices (clean re-render)
# Values transcribed from the archived Ultralytics exports
# (results/figures/confusion_matrices/CURRENT_{baseline,champion}_normalized.png);
# every column sums to 1.00 +- 0.01 (rounding), verified.
CLS = ["Birdnest", "Broken Ins.", "Defective\nDamper", "Flashover Ins.",
       "Normal Damper", "Normal Ins.", "Self-Expl. Ins.", "Background"]
cm_base = np.array([
    [0.75, 0,    0,    0,    0,    0.01, 0,    0.07],
    [0,    0.46, 0,    0,    0,    0,    0,    0.04],
    [0,    0,    0.67, 0,    0.01, 0,    0,    0.02],
    [0,    0,    0,    0.60, 0,    0,    0,    0.07],
    [0,    0,    0.21, 0,    0.76, 0,    0,    0.43],
    [0,    0,    0,    0,    0,    0.79, 0,    0.33],
    [0,    0,    0,    0,    0,    0,    0.74, 0.03],
    [0.25, 0.54, 0.13, 0.40, 0.23, 0.20, 0.26, 0],
])
cm_champ = np.array([
    [0.72, 0,    0,    0,    0,    0,    0,    0.04],
    [0,    0.43, 0,    0,    0,    0,    0,    0.01],
    [0,    0,    0.67, 0,    0.01, 0,    0,    0.03],
    [0,    0,    0,    0.63, 0,    0,    0,    0.09],
    [0,    0,    0.15, 0,    0.80, 0,    0,    0.35],
    [0,    0,    0,    0,    0,    0.83, 0,    0.44],
    [0,    0,    0,    0,    0,    0,    0.85, 0.04],
    [0.28, 0.57, 0.18, 0.37, 0.19, 0.16, 0.15, 0],
])
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
for ax, cm, title in zip(axes, [cm_base, cm_champ], ["(a) base", "(b) champ"]):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=0.85)
    for i in range(8):
        for j in range(8):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if v > 0.45 else "#1a3a5c")
    ax.set_xticks(range(8))
    ax.set_xticklabels(CLS, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(8))
    ax.set_yticklabels([c.replace("\n", " ") for c in CLS], fontsize=6)
    ax.set_xlabel("True", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(length=0)
axes[0].set_ylabel("Predicted", fontsize=7)
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "fig_cm.pdf")
fig.savefig(OUT / "fig_cm.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------- Fig: deployment ladder
cfgs = ["640\nnative", "768 infer\n(1280-tr.)", "768\nretrain", "1024 infer\n(1280-tr.)",
        "1280\nnative"]
map_ = [0.736, 0.754, 0.766, 0.783, 0.784]
dd = [0.614, 0.593, 0.664, 0.634, 0.622]
xi = np.arange(len(cfgs))
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(xi, map_, "-o", color=BLUE, label="mAP@0.5", ms=4.5)
ax.plot(xi, dd, "-o", color=GREEN, label="Defective Damper AP", ms=4.5)
ax.axhline(0.745, color=BLUE, ls="--", lw=0.7, alpha=0.6)
ax.axhline(0.59, color=GREEN, ls="--", lw=0.7, alpha=0.6)
ax.annotate("mAP floor 0.745", (3.25, 0.7315), fontsize=6.5, color=BLUE, alpha=0.8)
ax.annotate("DD floor 0.59 (provisional)", (2.85, 0.578), fontsize=6.5, color=GREEN, alpha=0.8)
ax.scatter([2], [0.766], s=140, facecolors="none", edgecolors="black", linewidths=1.0, zorder=4)
ax.annotate("deployed TRT FP16 engine:\n0.768 / DD 0.732", xy=(2, 0.766), xycoords="data",
            xytext=(0.02, 0.985), textcoords="axes fraction", fontsize=6.5, va="top",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY, shrinkB=9))
for i, (m, d) in enumerate(zip(map_, dd)):
    ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points",
                xytext=(0, 9) if i == 2 else (0, 5), fontsize=6.5, ha="center", color=BLUE)
    ax.annotate(f"{d:.3f}", (i, d), textcoords="offset points", xytext=(0, -11), fontsize=6.5,
                ha="center", color=GREEN)
ax.set_xticks(xi)
ax.set_xticklabels(cfgs, fontsize=6.5)
ax.set_ylim(0.54, 0.88)
ax.set_ylabel("score on clean test split")
ax.legend(loc="upper right", framealpha=0.9, fontsize=6.5)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
fig.savefig(OUT / "fig_deploy.pdf")
fig.savefig(OUT / "fig_deploy.png", dpi=300)
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("fig_*.pdf")))
