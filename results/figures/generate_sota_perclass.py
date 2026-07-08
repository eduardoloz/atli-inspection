"""Per-class AP@0.5 bar graph: top-5 models on ATLI (for slide deck)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/Users/eddie/Research/Vegas/results/figures/writeup")

classes = ["Self-Exploded\nInsulator", "Birdnest", "Normal\nDamper",
           "Normal\nInsulators", "Defective\nDamper", "Flashover\nInsulator",
           "Broken\nInsulator"]

models = [
    ("YOLOv11n champion (ours)†", "#2a78d6",
     [0.855, 0.928, 0.789, 0.766, 0.745, 0.879, 0.650]),
    ("YOLOv11n baseline (ours)†", "#1baf7a",
     [0.855, 0.801, 0.733, 0.758, 0.727, 0.614, 0.680]),
    ("DINO-4scale (47M)", "#eda100",
     [0.881, 0.835, 0.826, 0.810, 0.743, 0.597, 0.582]),
    ("RTM-DET Tiny (4.8M)", "#008300",
     [0.833, 0.788, 0.742, 0.759, 0.631, 0.542, 0.543]),
    ("Dynamic-RCNN (41M)", "#4a3aa7",
     [0.737, 0.745, 0.707, 0.758, 0.660, 0.508, 0.655]),
]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

fig, ax = plt.subplots(figsize=(11, 7.5), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

n = len(models)
bar_h, gap = 0.13, 0.015
group_span = n * bar_h + (n - 1) * gap
y = np.arange(len(classes))

vals_by_class = list(zip(*[m[2] for m in models]))
for mi, (label, color, vals) in enumerate(models):
    offs = -group_span / 2 + bar_h / 2 + mi * (bar_h + gap)
    bars = ax.barh(y + offs, vals, height=bar_h, color=color, label=label,
                   edgecolor="#fcfcfb", linewidth=1, zorder=3)
    for ci, v in enumerate(vals):
        if v == max(vals_by_class[ci]):  # label only the class winner
            ax.annotate(f"{v:.3f}", xy=(v + 0.008, ci + offs), va="center",
                        fontsize=9, fontweight="bold", color=INK2, zorder=4)

ax.set_yticks(y)
ax.set_yticklabels(classes, fontsize=10.5, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.set_xlabel("AP@0.5", fontsize=11, color=INK2)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=MUTED)

ax.set_title("Per-class AP@0.5 — top 5 models on ATLI",
             fontsize=14, fontweight="bold", color=INK, loc="left", pad=14)
ax.legend(loc="lower right", fontsize=9.5, frameon=False, labelcolor=INK2)

fig.text(0.01, 0.01,
         "† YOLO values are 5-fold CV means; SOTA models scored on the single "
         "199-image test split — protocols differ, gaps involving YOLO are indicative.",
         fontsize=8.5, color=MUTED)

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT / "fig7_sota_perclass.png", facecolor="#fcfcfb",
            bbox_inches="tight")
print("saved", OUT / "fig7_sota_perclass.png")
