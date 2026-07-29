"""Per-class Precision/Recall/AP@0.5 table for the two YOLOv11n deployment
champions: the 1280px CV champion (OBB + deg15 rotation + all-defect
oversample + mixup) and the 640px deployment champion (same 640px base +
blur-aug on top). Same underlying numbers as generate_v11_perclass_prmap.py /
generate_v11_perclass_prmap_poster.py (those render the data as two separate
bar charts); this renders it as one table, poster_fig_cv_ladder_table-styled
(see paper/poster/figs_src/gen_poster_figures.py Fig 12), for easier
side-by-side reading. Column labels follow the project's lever-progression
naming (see gen_poster_figures.py WB_1280/WB_640/LEFT_COL). Source:
eval_eduardo_results.json, 5-fold eduardo-CV means."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Helvetica Neue"

OUT = Path(__file__).parent
DATA = json.load(open(Path(__file__).parent.parent / "eval_eduardo_results.json"))

CLASSES = ["Birdnest", "Self-Exploded_Insulator", "Normal_Insulators", "Normal_Damper",
           "Broken_Insulator", "Flashover_Insulator", "Defective_Damper"]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

CONDS = ["mix15_1280", "blurmix_640"]
COL_RECIPE = ["OBB + defect OS x3 +\n15 deg + mixup=0.15",
              "OBB + defect OS x3 + 15 deg +\nmixup=0.15 + blur-aug"]
COL_TAG = ["(1280px CV Champion)", "(640px Deployment)"]


def summarize(key):
    folds = DATA[key]["folds"]
    out = {"P": [], "R": [], "mAP50": []}
    for c in CLASSES:
        out[c] = {"AP": [], "P": [], "R": []}
    for f in folds:
        out["P"].append(f["P"])
        out["R"].append(f["R"])
        out["mAP50"].append(f["mAP50"])
        for c in CLASSES:
            out[c]["AP"].append(f[f"{c}_AP"])
            out[c]["P"].append(f[f"{c}_P"])
            out[c]["R"].append(f[f"{c}_R"])
    overall = {k: np.mean(out[k]) for k in ["P", "R", "mAP50"]}
    per_class = {c: {m: np.mean(out[c][m]) for m in ["AP", "P", "R"]} for c in CLASSES}
    return overall, per_class


SUMMARY = {c: summarize(c) for c in CONDS}

fig, ax = plt.subplots(figsize=(8.4, 4.55), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.axis("off")

name_x = 0.02
col_x = np.linspace(0.44, 0.86, 2)
y0, dy = 0.95, 0.115

ax.text(name_x, y0, "Class", ha="left", va="center", fontsize=15,
        fontweight="bold", color=INK, transform=ax.transAxes)
ax.text(name_x, y0 - 0.05, "(cells: P / R / mAP@0.5)", ha="left", va="center",
        fontsize=10.5, color=MUTED, transform=ax.transAxes)
for j in range(2):
    ax.text(col_x[j], y0 + 0.035, COL_RECIPE[j], ha="center", va="center", fontsize=10.5,
            fontweight="normal", color=INK, transform=ax.transAxes, linespacing=1.3)
    ax.text(col_x[j], y0 - 0.045, COL_TAG[j], ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK, transform=ax.transAxes)
ax.plot([0.005, 0.995], [y0 - dy * 0.62] * 2, color=INK, lw=2.0,
        transform=ax.transAxes, clip_on=False)

ROWS = CLASSES + ["OVERALL"]
row_gaps = [dy * 1.35] + [dy] * (len(ROWS) - 1)
row_y = []
cursor = y0
for g in row_gaps:
    cursor -= g
    row_y.append(cursor)

for i, cls in enumerate(ROWS):
    y = row_y[i]
    is_overall = cls == "OVERALL"
    if is_overall:
        ax.plot([0.005, 0.995], [y + dy * 0.5] * 2, color=INK, lw=1.4,
                transform=ax.transAxes, clip_on=False)
    rname = "Overall (mAP@0.5)" if is_overall else cls.replace("_", " ")
    ax.text(name_x, y, rname, ha="left", va="center", fontsize=13,
            fontweight="bold" if is_overall else "normal", color=INK,
            transform=ax.transAxes)

    for j, c in enumerate(CONDS):
        if is_overall:
            pr, rc, ap = SUMMARY[c][0]["P"], SUMMARY[c][0]["R"], SUMMARY[c][0]["mAP50"]
        else:
            pc = SUMMARY[c][1][cls]
            pr, rc, ap = pc["P"], pc["R"], pc["AP"]
        ax.text(col_x[j], y, f"{pr:.2f} / {rc:.2f} / {ap:.2f}",
                ha="center", va="center", fontsize=12.5,
                color=INK2, transform=ax.transAxes)

    if not is_overall and i < len(CLASSES) - 1:
        ax.plot([0.005, 0.995], [y - dy * 0.5] * 2, color=GRID, lw=1.0,
                transform=ax.transAxes, clip_on=False)

fig.subplots_adjust(top=0.96, bottom=0.03, left=0.02, right=0.98)
fig.savefig(OUT / "fig_v11_perclass_prmap_table.png", facecolor="#fcfcfb", bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "fig_v11_perclass_prmap_table.png")
