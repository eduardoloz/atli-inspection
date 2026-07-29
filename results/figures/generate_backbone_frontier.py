"""Backbone efficiency frontier: stock YOLOv11n/v8n vs lightweight-backbone
grafts (Ghost/DWS/FasterNet), all on the same OBB + deg15 2-stage-TL recipe
(eduardo-CV phases 14-15). Two figures:
  fig_backbone_frontier.png  - params (M) vs mAP@0.5, w/ CV std error bars
  fig_backbone_params.png    - plain param-count comparison, sorted
Sources: results/eval_eduardo_results.json (mAP means/std, 5-fold CV),
results/model_flops.json + CLAUDE.md phase-14 log (v8-graft params, not
separately collected in model_flops.json)."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Helvetica Neue"

OUT = Path(__file__).parent
DATA = json.load(open(Path(__file__).parent.parent / "eval_eduardo_results.json"))
FLOPS = json.load(open(Path(__file__).parent.parent / "model_flops.json"))

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_V11 = "#4472C4"   # blue
C_V8 = "#ED7D31"    # orange

# (eval key, model family, backbone label, params_M)
MODELS = [
    ("deg15",         "v11", "Stock",     FLOPS["v11_deg15_obb"]["params_M"]),
    ("fnet_deg15",    "v11", "FasterNet", FLOPS["v11_fnet_obb"]["params_M"]),
    ("dws_deg15",     "v11", "DWS",       FLOPS["v11_dws_obb"]["params_M"]),
    ("ghost_deg15",   "v11", "Ghost",     FLOPS["v11_ghost_obb"]["params_M"]),
    ("v8deg15",       "v8",  "Stock",     FLOPS["v8_deg15_obb"]["params_M"]),
    ("v8fnet_deg15",  "v8",  "FasterNet", 2.79),   # train/models_graft/yolov8n-fnet-obb.yaml
    ("v8dws_deg15",   "v8",  "DWS",       2.84),   # train/models_graft/yolov8n-dws-obb.yaml
    ("v8ghost_deg15", "v8",  "Ghost",     2.52),   # train/models_graft/yolov8n-ghost-obb.yaml
]

rows = []
for key, fam, backbone, params in MODELS:
    maps = [f["mAP50"] for f in DATA[key]["folds"]]
    rows.append(dict(key=key, fam=fam, backbone=backbone, params=params,
                      mean=np.mean(maps), std=np.std(maps)))


def family_series(fam):
    fr = [r for r in rows if r["fam"] == fam]
    fr.sort(key=lambda r: r["params"])
    return fr


fig, ax = plt.subplots(figsize=(7.7, 6.6), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

for fam, color, label in [("v11", C_V11, "YOLOv11n-OBB"), ("v8", C_V8, "YOLOv8n-OBB")]:
    fr = family_series(fam)
    xs = [r["params"] for r in fr]
    ys = [r["mean"] for r in fr]
    es = [r["std"] for r in fr]
    ax.plot(xs, ys, color=color, linewidth=1.6, zorder=2, alpha=0.55)
    ax.errorbar(xs, ys, yerr=es, fmt="o", color=color, ecolor=color, elinewidth=1.3,
                capsize=3.5, markersize=9, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                label=label, zorder=3)
    for r in fr:
        dx = 0.045 if r["backbone"] != "Stock" else 0.045
        va = "bottom"
        ax.annotate(r["backbone"], xy=(r["params"] + dx, r["mean"] + r["std"] + 0.012),
                    fontsize=9, color=INK2, ha="left", va=va, zorder=4)

ax.set_xlabel("Parameters (M)", fontsize=11, color=INK2)
ax.set_ylabel("mAP@0.5 (5-fold CV mean ± std)", fontsize=11, color=INK2)
ax.set_xlim(1.9, 3.35)
ax.set_ylim(0.55, 0.87)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
ax.tick_params(colors=MUTED)

title = textwrap.fill("Backbone efficiency frontier — YOLOv11n vs YOLOv8n, stock vs lightweight grafts", width=42)
ax.set_title(title, fontsize=17, fontweight="bold", color=INK, loc="left", pad=14)
ax.legend(loc="lower right", fontsize=10, frameon=False, labelcolor=INK2)

subtitle = ("5-fold eduardo-CV means ± std, OBB task, deg15 2-stage-TL recipe (eduardo-CV phases 14-15). "
            "Ghost/DWS/FasterNet backbones grafted in place of the stock backbone, same head/neck + recipe. "
            "Every backbone graft underperforms its stock counterpart on both axes — the stock YOLOv11n-OBB "
            "backbone dominates the whole frontier.")
wrapped = textwrap.fill(subtitle, width=82)
fig.text(0.01, 0.01, wrapped, fontsize=8.5, color=MUTED, linespacing=1.4)

plt.tight_layout(rect=[0, 0.14, 1, 1])
fig.savefig(OUT / "fig_backbone_frontier.png", facecolor="#fcfcfb", bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "fig_backbone_frontier.png")


# ---------------------------------------------------------------------------
# Companion figure: plain parameter-count comparison, sorted ascending
# ---------------------------------------------------------------------------
rows_sorted = sorted(rows, key=lambda r: r["params"])
labels = [f'{"YOLOv11n" if r["fam"] == "v11" else "YOLOv8n"} — {r["backbone"]}' for r in rows_sorted]
colors = [C_V11 if r["fam"] == "v11" else C_V8 for r in rows_sorted]
params = [r["params"] for r in rows_sorted]

fig2, ax2 = plt.subplots(figsize=(7.7, 6.6), dpi=200)
fig2.patch.set_facecolor("#fcfcfb")
ax2.set_facecolor("#fcfcfb")

y = np.arange(len(rows_sorted))
ax2.barh(y, params, height=0.62, color=colors, edgecolor="#fcfcfb", linewidth=1, zorder=3)
for yi, p in zip(y, params):
    ax2.annotate(f"{p:.2f}M", xy=(p + 0.03, yi), va="center", fontsize=9.5, color=INK2, zorder=4)

ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=10, color=INK)
ax2.invert_yaxis()
ax2.set_xlim(0, 3.6)
ax2.set_xlabel("Parameters (M)", fontsize=11, color=INK2)
ax2.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax2.spines[spine].set_visible(False)
ax2.spines["bottom"].set_color("#c3c2b7")
ax2.tick_params(colors=MUTED)

title2 = textwrap.fill("Parameter count — stock vs lightweight-backbone graft variants", width=34)
ax2.set_title(title2, fontsize=19, fontweight="bold", color=INK, loc="left", pad=14)

from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=C_V11, label="YOLOv11n-OBB family"),
                     Patch(color=C_V8, label="YOLOv8n-OBB family")],
           loc="upper right", fontsize=10, frameon=False, labelcolor=INK2)

subtitle2 = ("Params from eval/collect_model_flops.py (stock + v11 grafts) and train/models_graft/"
             "yolov8n-{ghost,dws,fnet}-obb.yaml (v8 grafts, phase 14). All backbone grafts save "
             "0.2-0.9M params vs. their stock counterpart but cost 0.06-0.17 mAP — see frontier figure.")
wrapped2 = textwrap.fill(subtitle2, width=82)
fig2.text(0.01, 0.01, wrapped2, fontsize=8.5, color=MUTED, linespacing=1.4)

plt.tight_layout(rect=[0, 0.11, 1, 1])
fig2.savefig(OUT / "fig_backbone_params.png", facecolor="#fcfcfb", bbox_inches="tight")
plt.close(fig2)
print("saved", OUT / "fig_backbone_params.png")
