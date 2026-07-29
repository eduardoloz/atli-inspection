"""Poster-themed (red background, white caption) versions of the per-class
P/R/mAP@0.5 breakdown for the 1280px and 640px YOLOv11n eduardo-CV champions.
Same data/source as generate_v11_perclass_prmap.py; only the chrome differs
to match the poster deck's Figure 7/8 slide style."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
DATA = json.load(open(Path(__file__).parent.parent / "eval_eduardo_results.json"))

CLASSES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
           "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
LABELS = ["Birdnest", "Broken\nInsulator", "Defective\nDamper", "Flashover\nInsulator",
          "Normal\nDamper", "Normal\nInsulators", "Self-Exploded\nInsulator"]

C_AP = "#4472C4"    # darker blue
C_R = "#ED7D31"     # orange
C_P = "#70AD47"     # green

POSTER_RED = "#9D2235"   # UNLV scarlet
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


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
    overall = {k: (np.mean(out[k]), np.std(out[k])) for k in ["P", "R", "mAP50"]}
    per_class = {c: {m: np.mean(out[c][m]) for m in ["AP", "P", "R"]} for c in CLASSES}
    return overall, per_class


def make_chart(key, panel_title, caption, fname):
    overall, per_class = summarize(key)

    ap_vals = [per_class[c]["AP"] for c in CLASSES]
    r_vals = [per_class[c]["R"] for c in CLASSES]
    p_vals = [per_class[c]["P"] for c in CLASSES]

    fig = plt.figure(figsize=(11, 8.6), dpi=200)
    fig.patch.set_facecolor(POSTER_RED)

    # white inset panel holding the chart, poster-slide style
    panel = plt.Rectangle((0.02, 0.14), 0.96, 0.80, transform=fig.transFigure,
                           facecolor="#fcfcfb", edgecolor="none", zorder=0)
    fig.patches.append(panel)
    ax = fig.add_axes([0.13, 0.16, 0.81, 0.74])
    ax.set_facecolor("none")
    ax.set_zorder(5)

    y = np.arange(len(CLASSES))
    bar_h, gap = 0.24, 0.02
    group_span = 3 * bar_h + 2 * gap

    series = [("AP@0.5", C_AP, ap_vals), ("Recall", C_R, r_vals), ("Precision", C_P, p_vals)]
    for si, (label, color, vals) in enumerate(series):
        offs = -group_span / 2 + bar_h / 2 + si * (bar_h + gap)
        ax.barh(y + offs, vals, height=bar_h, color=color, label=label,
                edgecolor="#fcfcfb", linewidth=1, zorder=3)
        for ci, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", xy=(v + 0.008, ci + offs), va="center",
                        fontsize=8.5, color=INK2, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(LABELS, fontsize=10.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)

    ax.set_title(panel_title, fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3,
              fontsize=10, frameon=False, labelcolor=INK2)

    fig.text(0.05, 0.005, caption, fontsize=15, fontweight="bold", color="white",
              ha="left", va="bottom")

    fig.savefig(OUT / fname, facecolor=POSTER_RED, bbox_inches=None)
    plt.close(fig)
    print("saved", OUT / fname, "| overall mAP", f"{overall['mAP50'][0]:.3f} ± {overall['mAP50'][1]:.3f}",
          "P", f"{overall['P'][0]:.3f}", "R", f"{overall['R'][0]:.3f}")


make_chart(
    "mix15_1280",
    "YOLOv11n per-class P/R/AP@0.5 — 1280px champion (mix15_1280)",
    "Figure 9 - Per-class Precision/Recall/AP@0.5, 1280px CV champion",
    "fig_v11_perclass_prmap_1280_poster.png",
)

make_chart(
    "blurmix_640",
    "YOLOv11n per-class P/R/AP@0.5 — 640px deployment champion (blurmix_640)",
    "Figure 10 - Per-class Precision/Recall/AP@0.5, 640px deployment champion",
    "fig_v11_perclass_prmap_640_poster.png",
)
