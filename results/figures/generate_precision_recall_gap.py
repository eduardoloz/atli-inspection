"""Compact callout figure: precision-recall gap per class for the 1280px
champion (mix15_1280) — shows recall, not precision, is the bottleneck on
the hard classes. Sized to drop into a small poster callout box."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Helvetica Neue"

OUT = Path(__file__).parent
DATA = json.load(open(Path(__file__).parent.parent / "eval_eduardo_results.json"))

CLASSES = ["Birdnest", "Normal_Insulators", "Normal_Damper", "Self-Exploded_Insulator",
           "Flashover_Insulator", "Defective_Damper", "Broken_Insulator"]
LABELS = ["Birdnest", "Normal Insulators", "Normal Damper", "Self-Exploded Insulator",
          "Flashover Insulator", "Defective Damper", "Broken Insulator"]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_P = "#70AD47"   # green — precision
C_R = "#ED7D31"   # orange — recall
C_LINE = "#c3c2b7"

folds = DATA["mix15_1280"]["folds"]
rows = []
for c, lbl in zip(CLASSES, LABELS):
    p = np.mean([f[f"{c}_P"] for f in folds])
    r = np.mean([f[f"{c}_R"] for f in folds])
    rows.append((lbl, p, r))
rows.sort(key=lambda r: r[1] - r[2])  # smallest (most negative/most recall-lagging) gap last isn't what we want
rows.sort(key=lambda r: r[1] - r[2], reverse=True)  # largest P-R gap first (worst recall lag on top)

fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

y = np.arange(len(rows))
for yi, (lbl, p, r) in zip(y, rows):
    ax.plot([r, p], [yi, yi], color=C_LINE, linewidth=2.2, zorder=2)
ax.scatter([r[2] for r in rows], y, color=C_R, s=110, zorder=3, label="Recall",
           edgecolor="#fcfcfb", linewidth=1.2)
ax.scatter([r[1] for r in rows], y, color=C_P, s=110, zorder=3, label="Precision",
           edgecolor="#fcfcfb", linewidth=1.2)

for yi, (lbl, p, r) in zip(y, rows):
    gap = p - r
    if gap > 0.03:
        mid = (p + r) / 2
        ax.annotate(f"−{gap:.2f}", xy=(mid, yi + 0.22), ha="center", fontsize=8.5,
                    color=INK2, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK)
ax.set_xlim(0.55, 1.0)
ax.set_xlabel("Score", fontsize=10.5, color=INK2)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=MUTED)

title = "Recall is the bottleneck, not precision"
ax.set_title(title, fontsize=16, fontweight="bold", color=INK, loc="left", pad=12)
ax.legend(loc="lower left", fontsize=9.5, frameon=False, labelcolor=INK2)

subtitle = ("1280px champion (mix15_1280), 5-fold CV means. Precision stays "
            "high (0.82-0.90) across all classes; recall is what drags Broken/"
            "Defective/Flashover down — the model misses defects more than it "
            "false-alarms on them.")
wrapped = textwrap.fill(subtitle, width=52)
fig.text(0.02, 0.02, wrapped, fontsize=8, color=MUTED, linespacing=1.4)

plt.tight_layout(rect=[0, 0.18, 1, 1])
fig.savefig(OUT / "fig_precision_recall_gap.png", facecolor="#fcfcfb", bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "fig_precision_recall_gap.png")
