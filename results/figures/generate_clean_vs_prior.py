#!/usr/bin/env python3
"""Figures comparing pre-purge (CPLID-contaminated) benchmarks with the clean
no-CPLID benchmarks, plus the edge-deployment frontier.

Data sources: results/cv_proper_results.md (old proper CV), CLAUDE.md experiment
log + results/config_benchmark.csv (old single-split), the 2026-07-07/08 clean
benchmark + OBB logs on the server (seed-aggregated), and the optimization
campaign's resolution ladder / native-768 results (opt/orchestrator README).

Output: results/figures/clean_vs_prior/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "clean_vs_prior"
OUT.mkdir(exist_ok=True)

# palette (validated): baseline=blue, champion=aqua; text/surface tokens
BLUE, AQUA = "#2a78d6", "#1baf7a"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def bar_pair(ax, labels, base, champ, base_err=None, champ_err=None,
             label_fmt="{:.3f}", label_all=True):
    x = np.arange(len(labels))
    w = 0.36
    err_kw = dict(ecolor=INK2, elinewidth=1, capsize=2, capthick=1)
    ax.bar(x - w / 2 - 0.01, base, w, color=BLUE, label="Baseline",
           yerr=base_err, error_kw=err_kw)
    ax.bar(x + w / 2 + 0.01, champ, w, color=AQUA, label="Champion",
           yerr=champ_err, error_kw=err_kw)
    if label_all:
        for xi, v, e in zip(x - w / 2 - 0.01, base, base_err or [None] * len(x)):
            ax.text(xi, (v + (e or 0)) + 0.012, label_fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color=INK2)
        for xi, v, e in zip(x + w / 2 + 0.01, champ, champ_err or [None] * len(x)):
            ax.text(xi, (v + (e or 0)) + 0.012, label_fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ── Fig A: before/after decontamination, baseline vs champion (YOLOv11n) ────
eras = ["Pre-purge\nsingle split\n(199-img test, 11 seeds)",
        "Pre-purge\nproper 5-fold CV\n(contaminated pool)",
        "Clean no-CPLID\nsingle split\n(120-img test, 3 seeds)"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
bar_pair(axes[0], eras,
         base=[0.691, 0.722, 0.736], champ=[0.746, 0.785, 0.784],
         base_err=[0, 0.019, 0.015], champ_err=[0.009, 0.023, 0.011])
axes[0].set_title("mAP@0.5 — before vs after CPLID removal")
axes[0].set_ylim(0.5, 0.9)
axes[0].legend(frameon=False, loc="upper left")
bar_pair(axes[1], eras,
         base=[0.641, 0.743, 0.614], champ=[0.699, 0.812, 0.622],
         base_err=[0.036, 0.074, 0.082], champ_err=[0.028, 0.080, 0.073])
axes[1].set_title("Defective_Damper AP — before vs after")
axes[1].set_ylim(0.4, 0.95)
fig.suptitle("Champion's margin over baseline survives decontamination "
             "(test sets differ per era — compare within era)",
             fontsize=10, color=INK2, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_a_era_comparison.png", dpi=150, bbox_inches="tight")

# ── Fig B: clean-data detection benchmark, per class ─────────────────────────
classes = ["Overall\n(mAP@0.5)", "Birdnest", "Broken\nInsulator", "Defective\nDamper",
           "Flashover\nInsulator", "Normal\nDamper", "Normal\nInsulators",
           "Self-Exploded\nInsulator"]
det_base = [0.736, 0.959, 0.667, 0.614, 0.590, 0.743, 0.826, 0.755]
det_base_e = [0.015, 0.010, 0.023, 0.082, 0.011, 0.023, 0.015, 0.016]
det_champ = [0.784, 0.984, 0.699, 0.622, 0.660, 0.813, 0.838, 0.871]
det_champ_e = [0.011, 0.004, 0.029, 0.073, 0.024, 0.019, 0.013, 0.027]
fig, ax = plt.subplots(figsize=(11, 4.2))
bar_pair(ax, classes, det_base, det_champ, det_base_e, det_champ_e, "{:.2f}")
ax.set_title("Clean no-CPLID detection benchmark — per-class AP@0.5 (3 seeds, 120-img test)")
ax.set_ylim(0.4, 1.05)
ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig_b_perclass_det.png", dpi=150, bbox_inches="tight")

# ── Fig C: clean-data OBB benchmark, per class ───────────────────────────────
obb_base = [0.714, 0.924, 0.691, 0.653, 0.629, 0.756, 0.669, 0.678]
obb_base_e = [0.018, 0.020, 0.082, 0.021, 0.016, 0.020, 0.033, 0.036]
obb_champ = [0.765, 0.950, 0.846, 0.768, 0.681, 0.767, 0.724, 0.621]
obb_champ_e = [0.015, 0.003, 0.049, 0.033, 0.025, 0.020, 0.027, 0.068]
fig, ax = plt.subplots(figsize=(11, 4.2))
bar_pair(ax, classes, obb_base, obb_champ, obb_base_e, obb_champ_e, "{:.2f}")
ax.set_title("Clean no-CPLID OBB benchmark (oriented boxes) — per-class AP@0.5 "
             "(3 seeds, 107-img merged-split test)")
ax.set_ylim(0.4, 1.05)
ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig_c_perclass_obb.png", dpi=150, bbox_inches="tight")

# ── Fig D: deployment frontier (accuracy vs resolution, Nano fps annotated) ──
configs = ["640\nnative baseline", "768 infer\n(1280-trained)", "768\nnative retrain",
           "1024 infer\n(1280-trained)", "1280\nnative champion"]
fps = ["~25 fps", "~18 fps", "~18 fps", "~10 fps", "~6 fps"]
mAP = [0.736, 0.754, 0.769, 0.783, 0.784]
dd = [0.614, 0.593, 0.670, 0.634, 0.622]
x = np.arange(len(configs))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.axhline(0.745, color=BLUE, linewidth=1, linestyle=(0, (4, 4)), alpha=0.55)
ax.text(4.42, 0.747, "mAP floor 0.745", fontsize=8, color=BLUE, va="bottom", ha="right")
ax.axhline(0.590, color=AQUA, linewidth=1, linestyle=(0, (4, 4)), alpha=0.55)
ax.text(-0.42, 0.585, "DD floor 0.59", fontsize=8, color=AQUA, va="top", ha="left")
ax.plot(x, mAP, color=BLUE, linewidth=2, marker="o", markersize=7, label="mAP@0.5")
ax.plot(x, dd, color=AQUA, linewidth=2, marker="o", markersize=7, label="Defective_Damper AP")
for xi, v in zip(x, mAP):
    if xi == 2:
        ax.text(xi + 0.14, v - 0.004, f"{v:.3f}", ha="left", va="top", fontsize=8, color=INK2)
    else:
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
for xi, v in zip(x, dd):
    ax.text(xi, v - 0.014, f"{v:.3f}", ha="center", va="top", fontsize=8, color=INK2)
ax.scatter([2], [0.769], s=230, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=5)
ax.annotate("deployment candidate\n(TRT FP16 engine: 0.768 / DD 0.732)",
            xy=(2, 0.769), xytext=(2, 0.86), ha="center", fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.set_xticks(x, [f"{c}\n{f}" for c, f in zip(configs, fps)])
ax.set_ylim(0.52, 0.9)
ax.set_ylabel("AP@0.5 (clean test split)")
ax.set_title("Deployment frontier — accuracy vs input size, projected Jetson Nano fps under each config")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig_d_deploy_frontier.png", dpi=150, bbox_inches="tight")

# ── Fig E: class composition of the 249 CPLID images removed from ATLI ───────
cls_short = ["Birdnest", "Broken\nInsulator", "Defective\nDamper", "Flashover\nInsulator",
             "Normal\nDamper", "Normal\nInsulators", "Self-Exploded\nInsulator"]
removed_inst = [6, 0, 3, 0, 277, 30, 249]
removed_imgs = [6, 0, 3, 0, 72, 12, 249]
x = np.arange(len(cls_short))
fig, ax = plt.subplots(figsize=(9.5, 4.2))
bars = ax.bar(x, removed_inst, 0.55, color=BLUE)
for xi, v, n in zip(x, removed_inst, removed_imgs):
    ax.text(xi, v + 4, f"{v}", ha="center", va="bottom", fontsize=9, color=INK)
    if n:
        ax.text(xi, v + 22, f"({n} imgs)", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x, cls_short)
ax.set_ylabel("annotation instances on the 249 removed images")
ax.set_title("What the removed CPLID images contained (249 imgs; every one carries a "
             "Self-Exploded_Insulator)")
ax.set_ylim(0, 330)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_e_removed_composition.png", dpi=150, bbox_inches="tight")

# ── Fig F: before vs after CPLID removal, per class, baseline & champion ─────
# before = pre-purge proper 5-fold CV; after = clean no-CPLID single split (3 seeds)
pre_base = [0.824, 0.592, 0.743, 0.593, 0.716, 0.786, 0.800]
pre_base_e = [0.063, 0.073, 0.082, 0.024, 0.037, 0.031, 0.031]
post_base = [0.959, 0.667, 0.614, 0.590, 0.743, 0.826, 0.755]
post_base_e = [0.010, 0.023, 0.082, 0.011, 0.023, 0.015, 0.016]
pre_champ = [0.884, 0.662, 0.812, 0.656, 0.790, 0.808, 0.883]
pre_champ_e = [0.046, 0.099, 0.089, 0.044, 0.043, 0.029, 0.042]
post_champ = [0.984, 0.699, 0.622, 0.660, 0.813, 0.838, 0.871]
post_champ_e = [0.004, 0.029, 0.073, 0.024, 0.019, 0.013, 0.027]
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
for ax, pre, pre_e, post, post_e, title in (
        (axes[0], pre_base, pre_base_e, post_base, post_base_e, "Baseline (YOLOv11n, 640)"),
        (axes[1], pre_champ, pre_champ_e, post_champ, post_champ_e,
         "Champion (YOLOv11n, 1280 + OS3 + scale 0.9)")):
    x = np.arange(len(cls_short))
    w = 0.36
    err_kw = dict(ecolor=INK2, elinewidth=1, capsize=2, capthick=1)
    ax.bar(x - w / 2 - 0.01, pre, w, color=BLUE, label="Before (pre-purge 5-fold CV)",
           yerr=pre_e, error_kw=err_kw)
    ax.bar(x + w / 2 + 0.01, post, w, color=AQUA, label="After (clean no-CPLID, 3 seeds)",
           yerr=post_e, error_kw=err_kw)
    for xi, v, e in zip(x - w / 2 - 0.01, pre, pre_e):
        ax.text(xi, v + e + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    for xi, v, e in zip(x + w / 2 + 0.01, post, post_e):
        ax.text(xi, v + e + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, cls_short)
    ax.set_ylim(0.4, 1.1)
    ax.set_title(title)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, loc="lower right")
fig.suptitle("Per-class AP@0.5 before vs after CPLID removal (eval protocols differ; "
             "before = CV over the contaminated pool, after = clean 120-img split)",
             fontsize=10, color=INK2)
fig.tight_layout()
fig.savefig(OUT / "fig_f_before_after_perclass.png", dpi=150, bbox_inches="tight")

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))], sep="\n  ")
