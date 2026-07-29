"""Accuracy vs. edge throughput across backbones (YOLOv11n/v8n/v5n x
stock/Ghost/DWS/FasterNet grafts, all at 1280px). Rebuild of
paper/poster/figures/accuracy_vs_throughput_all_models.png, which had no
source script and — worse — mixed two incompatible speed metrics on one
axis: most points were PyTorch fp32 batch-1, but both Ghost grafts and the
v11n FasterNet graft were plotted at their TensorRT-fp16-engine speed
instead (1.3-2.5x faster than fp32 for the same model). This version uses
ONE consistent metric everywhere: PyTorch fp32, batch-1, on-device Jetson
Orin Nano, every model at its 1280px eval resolution (per
hardware_testing/jetson_fps_results.json). The original also had a P2-head
(640px) point, dropped here since it's a different eval resolution, not a
backbone comparison at fixed resolution.

mAP@0.5 source: results/eval_eduardo_results.json (974-image eduardo
group-aware 5-fold CV means) -- same numbers as the phase-14/15 backbone
frontier in CLAUDE.md.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Helvetica Neue"

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/poster/figures"

RES_ED = json.load(open(ROOT / "results/eval_eduardo_results.json"))
JETSON = json.load(open(ROOT / "hardware_testing/jetson_fps_results.json"))

BLUE, ORANGE, GREEN = "#2166ac", "#d0700e", "#1a9850"
INK, INK2, MUTED, GRID = "#1a1a1a", "#4d4c49", "#898781", "#e3e2dc"

FAMILY_COLOR = {"YOLOv11n": BLUE, "YOLOv8n": ORANGE, "YOLOv5n": GREEN}
BACKBONE_MARKER = {"Stock": "o", "Ghost": "s", "DWS": "^", "FasterNet": "D"}


def ed_map(cond):
    return float(np.mean([f["mAP50"] for f in RES_ED[cond]["folds"]]))


def fps(model_key, res):
    return JETSON[model_key][f"pt_{res}_fp32_conf0.25"]["fps"]


# (family, backbone, eduardo-CV key, jetson-json key, eval resolution)
POINTS = [
    ("YOLOv11n", "Stock", "deg15", "v11n_obb", 1280),
    ("YOLOv8n", "Stock", "v8deg15", "v8n_obb_champ", 1280),
    ("YOLOv5n", "Stock", "v5champ", "v5n", 1280),
    ("YOLOv11n", "Ghost", "ghost_deg15", "v11_ghost", 1280),
    ("YOLOv8n", "Ghost", "v8ghost_deg15", "v8_ghost", 1280),
    ("YOLOv11n", "DWS", "dws_deg15", "v11_dws", 1280),
    ("YOLOv8n", "DWS", "v8dws_deg15", "v8_dws", 1280),
    ("YOLOv11n", "FasterNet", "fnet_deg15", "v11_fnet", 1280),
    ("YOLOv8n", "FasterNet", "v8fnet_deg15", "v8_fnet", 1280),
]

fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
fig.subplots_adjust(top=0.90, bottom=0.13, left=0.13, right=0.97)

for family, backbone, cv_key, jkey, res in POINTS:
    x, y = fps(jkey, res), ed_map(cv_key)
    ax.scatter([x], [y], s=170, color=FAMILY_COLOR[family], marker=BACKBONE_MARKER[backbone],
               edgecolors="white", linewidths=1.3, zorder=3)

# legends: color = model family, marker (grey) = backbone
family_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                              markeredgecolor="white", markersize=11, label=f)
                  for f, c in FAMILY_COLOR.items()]
backbone_handles = [plt.Line2D([0], [0], marker=m, color="none", markerfacecolor=MUTED,
                                markeredgecolor="white", markersize=10, label=b)
                    for b, m in BACKBONE_MARKER.items()]
leg1 = ax.legend(handles=family_handles, title="Model family", loc="upper right",
                  frameon=True, edgecolor="#c9c8c0", facecolor="white", framealpha=0.95,
                  fontsize=10, title_fontsize=10.5, labelspacing=0.4,
                  handletextpad=0.6, borderaxespad=0.3, bbox_to_anchor=(0.72, 1.02))
ax.add_artist(leg1)
ax.legend(handles=backbone_handles, title="Backbone", loc="upper right",
          frameon=True, edgecolor="#c9c8c0", facecolor="white", framealpha=0.95,
          fontsize=10, title_fontsize=10.5, labelspacing=0.4,
          handletextpad=0.6, borderaxespad=0.3, bbox_to_anchor=(1.0, 1.02))

ax.set_xlabel("Jetson Orin Nano fps (batch-1, PyTorch fp32, measured)")
ax.set_ylabel("mAP@0.5 (5-fold CV mean)")
ax.set_xlim(9.0, 13.2)
ax.set_ylim(0.60, 0.82)
ax.grid(axis="y", color=GRID)
ax.grid(False, axis="x")
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=INK2)

fig.text(0.13, 0.915, "Accuracy vs. edge throughput across backbones (1280px)",
          ha="left", va="bottom", fontsize=17, fontweight="bold", color=INK)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"accuracy_vs_throughput_all_models.{ext}", facecolor="white", dpi=300)
plt.close(fig)
print("saved", OUT / "accuracy_vs_throughput_all_models.{png,pdf}")
