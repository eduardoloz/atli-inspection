"""Per-class Precision/Recall/mAP@0.5 breakdown for the best YOLOv11n eduardo-CV
recipe at each deployment resolution: mix15_1280 (1280px champion) and
blurmix_640 (640px deployment champion). Source: eval_eduardo_results.json,
5-fold CV means."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Helvetica Neue"

OUT = Path(__file__).parent
DATA = json.load(open(Path(__file__).parent.parent / "eval_eduardo_results.json"))

CLASSES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
           "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
LABELS = ["Birdnest", "Broken\nInsulator", "Defective\nDamper", "Flashover\nInsulator",
          "Normal\nDamper", "Normal\nInsulators", "Self-Exploded\nInsulator"]

C_AP = "#4472C4"    # darker blue
C_R = "#ED7D31"     # orange
C_P = "#70AD47"     # green

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


def make_chart(key, title, subtitle, fname):
    overall, per_class = summarize(key)

    ap_vals = [per_class[c]["AP"] for c in CLASSES]
    r_vals = [per_class[c]["R"] for c in CLASSES]
    p_vals = [per_class[c]["P"] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(7.7, 7.5), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

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

    X_MIN = 0.5
    ax.set_xlim(X_MIN, 1.0)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)

    # explicit break marks on the bottom spine — the axis does not start at 0
    d = 0.014
    kwargs = dict(transform=ax.transAxes, color=INK, clip_on=False, linewidth=1.3, zorder=6)
    for x0 in (-0.006, 0.010):
        ax.plot((x0 - d / 2, x0 + d / 2), (-0.016, 0.016), **kwargs)

    wrapped_title = textwrap.fill(title, width=34)
    ax.set_title(wrapped_title, fontsize=19, fontweight="bold", color=INK, loc="left", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=3,
              fontsize=10, frameon=False, labelcolor=INK2)

    wrapped = textwrap.fill(subtitle, width=82)
    fig.text(0.01, 0.01, wrapped, fontsize=8.5, color=MUTED, linespacing=1.4)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT / fname, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / fname, "| overall mAP", f"{overall['mAP50'][0]:.3f} ± {overall['mAP50'][1]:.3f}",
          "P", f"{overall['P'][0]:.3f}", "R", f"{overall['R'][0]:.3f}")


make_chart(
    "mix15_1280",
    "YOLOv11n per-class P/R/mAP@0.5 — 1280px champion (mix15_1280)",
    "5-fold eduardo-CV means, OBB task. Recipe: COCO-init 2-stage TL, OBB, deg15 rotation aug, "
    "all-defect oversample, mixup=0.15, imgsz 1280. Overall mAP@0.5 0.804 ± 0.019, P 0.856, R 0.747. "
    "Note: x-axis starts at 0.5 (break shown) to emphasize class differences — not zero-based.",
    "fig_v11_perclass_prmap_1280.png",
)

make_chart(
    "blurmix_640",
    "YOLOv11n per-class P/R/mAP@0.5 — 640px deployment champion (blurmix_640)",
    "5-fold eduardo-CV means, OBB task. Recipe: mixup=0.15 base + MotionBlur/GaussianBlur aug, imgsz 640. "
    "Overall mAP@0.5 0.757 ± 0.017, P 0.792, R 0.712 (clean; blurred-test mAP 0.710). "
    "Note: x-axis starts at 0.5 (break shown) to emphasize class differences — not zero-based.",
    "fig_v11_perclass_prmap_640.png",
)
