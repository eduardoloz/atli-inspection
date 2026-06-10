#!/usr/bin/env python3
"""Generate charts and tables summarizing Eduardo's dataset contributions
and ablation experiment results for PI presentation."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "pi_charts")
os.makedirs(OUT, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. DATASET COMPOSITION
# ──────────────────────────────────────────────────────────────

classes = [
    "Birdnest",
    "Broken\nInsulator",
    "Defective\nDamper",
    "Flashover\nInsulator",
    "Normal\nDamper",
    "Normal\nInsulators",
    "Self-Exploded\nInsulator",
]
classes_short = [
    "Birdnest", "Broken_Ins", "Defect_Dam",
    "Flash_Ins", "Norm_Dam", "Norm_Ins", "Self-Exp_Ins",
]

# Full dataset instance counts (from CLAUDE.md / paper)
target_full = np.array([241, 194, 113, 434, 1460, 873, 554])

# Eduardo's contributions (from build scripts + memory)
# Eduardo v1 (Condition B): Defective_Damper, Normal_Damper, Normal_Insulators only
eduardo_b = np.array([0, 0, 151, 0, 819, 992, 0])
# Eduardo full (Condition C): adds 152 Broken_Insulator (remapped from Defective_Insulators)
eduardo_c = np.array([0, 152, 151, 0, 819, 992, 0])

# Train split (70%) approximate counts
target_train = np.array([169, 135, 79, 304, 1022, 611, 388])

# ── Chart 1: Target dataset class imbalance ──
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4CAF50" if c > 400 else "#FF9800" if c > 200 else "#F44336" for c in target_full]
bars = ax.bar(classes, target_full, color=colors, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Instance Count", fontsize=12)
ax.set_title("ATLI Target Dataset — Class Distribution (1,046 images)", fontsize=14, fontweight="bold")
for bar, val in zip(bars, target_full):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(target_full) * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Add legend for color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#F44336", label="Rare (< 200)"),
    Patch(facecolor="#FF9800", label="Moderate (200–400)"),
    Patch(facecolor="#4CAF50", label="Abundant (> 400)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "1_target_class_imbalance.png"), dpi=150)
plt.close()

# ── Chart 2: Before vs After (Train split) — Condition A vs C ──
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(classes))
w = 0.25
b1 = ax.bar(x - w, target_train, w, label="A: Target Only (train)", color="#2196F3", edgecolor="white")
b2 = ax.bar(x, target_train + eduardo_b, w, label="B: + Eduardo (no DI)", color="#FF9800", edgecolor="white")
b3 = ax.bar(x + w, target_train + eduardo_c, w, label="C: + Eduardo Full (DI→Broken)", color="#4CAF50", edgecolor="white")
ax.set_ylabel("Instance Count (train split)", fontsize=12)
ax.set_title("Training Set Composition — Before & After Eduardo's Annotations", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=9)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Annotate the key changes
for i, (a, c) in enumerate(zip(target_train, target_train + eduardo_c)):
    if c > a:
        pct = (c - a) / a * 100
        ax.annotate(f"+{pct:.0f}%", xy=(x[i] + w, c), fontsize=8,
                    ha="center", va="bottom", fontweight="bold", color="#2E7D32")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "2_before_after_train_composition.png"), dpi=150)
plt.close()

# ── Chart 3: Eduardo's contribution breakdown ──
edu_classes = ["Broken\nInsulator", "Defective\nDamper", "Normal\nDamper", "Normal\nInsulators"]
edu_counts = [152, 151, 819, 992]
edu_colors = ["#F44336", "#FF9800", "#4CAF50", "#4CAF50"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
bars = ax1.bar(edu_classes, edu_counts, color=edu_colors, edgecolor="white")
for bar, val in zip(bars, edu_counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
             str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylabel("Instances Added", fontsize=12)
ax1.set_title("Eduardo's Annotations\n(300 images, 2,114 instances)", fontsize=12, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Pie chart of contribution
ax2.pie(edu_counts, labels=edu_classes, autopct="%1.1f%%", colors=edu_colors,
        startangle=90, textprops={"fontsize": 10})
ax2.set_title("Distribution of Added Instances", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "3_eduardo_contribution.png"), dpi=150)
plt.close()

# ──────────────────────────────────────────────────────────────
# 2. ABLATION RESULTS (YOLOv5n SGD 150+100)
# ──────────────────────────────────────────────────────────────

# Per-class mAP@0.5 on shared Eduardo-free test set (156 imgs, 563 instances)
results = {
    "A: Target Only":        [0.775, 0.511, 0.404, 0.748, 0.825, 0.709, 0.876],
    "B: + Eduardo (no DI)":  [0.730, 0.570, 0.394, 0.694, 0.828, 0.749, 0.857],
    "C: + Eduardo Full":     [0.775, 0.543, 0.406, 0.732, 0.802, 0.743, 0.876],
}
overall = {
    "A: Target Only":        {"mAP50": 0.692, "mAP50_95": 0.420, "P": 0.821, "R": 0.643},
    "B: + Eduardo (no DI)":  {"mAP50": 0.689, "mAP50_95": 0.407, "P": 0.806, "R": 0.618},
    "C: + Eduardo Full":     {"mAP50": 0.697, "mAP50_95": 0.415, "P": 0.847, "R": 0.608},
}

classes_eval = [
    "Birdnest", "Broken\nInsulator", "Defective\nDamper", "Flashover\nInsulator",
    "Normal\nDamper", "Normal\nInsulators", "Self-Exploded\nInsulator",
]

# ── Chart 4: Per-class mAP@0.5 comparison ──
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(classes_eval))
w = 0.25
cond_colors = {"A: Target Only": "#2196F3", "B: + Eduardo (no DI)": "#FF9800", "C: + Eduardo Full": "#4CAF50"}
for i, (cond, vals) in enumerate(results.items()):
    bars = ax.bar(x + (i - 1) * w, vals, w, label=cond, color=cond_colors[cond], edgecolor="white")
ax.set_ylabel("mAP@0.5", fontsize=12)
ax.set_title("Per-Class Detection Performance — A vs B vs C\n(YOLOv5n SGD, 150+100 epochs, shared test set: 156 imgs)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(classes_eval, fontsize=9)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
ax.legend(fontsize=10, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "4_perclass_map_comparison.png"), dpi=150)
plt.close()

# ── Chart 5: Broken_Insulator deep dive (the target class) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# mAP improvement
conds = ["A: Target\nOnly", "B: + Eduardo\n(no DI)", "C: + Eduardo\nFull"]
broken_maps = [0.511, 0.570, 0.543]
broken_colors = ["#2196F3", "#FF9800", "#4CAF50"]
bars = ax1.bar(conds, broken_maps, color=broken_colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, broken_maps):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax1.set_ylabel("mAP@0.5", fontsize=12)
ax1.set_title("Broken Insulator — mAP@0.5", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 0.75)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
# Add improvement annotations
ax1.annotate(f"+11.5%", xy=(1, 0.570), xytext=(1.3, 0.62),
             fontsize=11, fontweight="bold", color="#E65100",
             arrowprops=dict(arrowstyle="->", color="#E65100"))
ax1.annotate(f"+6.3%", xy=(2, 0.543), xytext=(2.3, 0.59),
             fontsize=11, fontweight="bold", color="#2E7D32",
             arrowprops=dict(arrowstyle="->", color="#2E7D32"))

# Training instances
broken_train = [135, 135, 287]
bars2 = ax2.bar(conds, broken_train, color=broken_colors, edgecolor="white", width=0.5)
for bar, val in zip(bars2, broken_train):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
ax2.set_ylabel("Training Instances", fontsize=12)
ax2.set_title("Broken Insulator — Train Set Size", fontsize=12, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.suptitle("Broken Insulator: Weakest Class Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "5_broken_insulator_deep_dive.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 6: Overall mAP summary ──
fig, ax = plt.subplots(figsize=(8, 5))
conds_short = ["A: Target Only", "B: + Eduardo", "C: + Eduardo Full"]
map50 = [0.692, 0.689, 0.697]
map50_95 = [0.420, 0.407, 0.415]
x = np.arange(len(conds_short))
w = 0.3
b1 = ax.bar(x - w/2, map50, w, label="mAP@0.5", color="#2196F3", edgecolor="white")
b2 = ax.bar(x + w/2, map50_95, w, label="mAP@0.5:0.95", color="#90CAF9", edgecolor="white")
for bar, val in zip(b1, map50):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar, val in zip(b2, map50_95):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("mAP", fontsize=12)
ax.set_title("Overall Detection Performance\n(YOLOv5n SGD, 150+100 epochs)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(conds_short, fontsize=10)
ax.set_ylim(0, 0.85)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "6_overall_map_summary.png"), dpi=150)
plt.close()

# ── Chart 7: Delta from baseline (A) ──
fig, ax = plt.subplots(figsize=(12, 5))
baseline = np.array(results["A: Target Only"])
x = np.arange(len(classes_eval))
for i, (cond, vals) in enumerate(list(results.items())[1:]):  # skip A
    delta = np.array(vals) - baseline
    color = list(cond_colors.values())[i + 1]
    ax.bar(x + i * 0.35, delta, 0.35, label=cond, color=color, edgecolor="white")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("Δ mAP@0.5 (vs. Target Only)", fontsize=12)
ax.set_title("Impact of Eduardo's Annotations — Change from Baseline\n(Positive = improvement)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x + 0.175)
ax.set_xticklabels(classes_eval, fontsize=9)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "7_delta_from_baseline.png"), dpi=150)
plt.close()

print(f"All charts saved to {OUT}/")
print("Files generated:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
