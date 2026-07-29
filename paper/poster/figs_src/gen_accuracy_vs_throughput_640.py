"""Accuracy vs. edge throughput across backbones at 640px -- the deployment
resolution (see paper/poster/poster_content.md Sec 4.6: native 640px training
+ blur-aug is the current deployment recipe). Sibling of
gen_accuracy_vs_throughput.py (which covers the 1280px backbone frontier);
kept as a separate script/output rather than a resolution parameter on that
one so both figures stay independently regenerable and diffable.

Coverage gap, stated honestly rather than papered over: the v8n-graft family
(Ghost/DWS/FasterNet) was only ever benchmarked at 1280px in this project
(phase 14) -- there is no 640px mAP for v8-Ghost/DWS/FasterNet in
eval_eduardo_results.json, so this figure has 6 points, not 9. Re-using the
1280px numbers and mislabeling them as 640px would be wrong; omitting the v8
grafts is the honest choice until/unless those runs get queued.

fps: PyTorch fp32, batch-1, on-device Jetson Orin Nano, pt_640_fp32_conf0.25
(hardware_testing/jetson_fps_results.json) -- exists for all 9 architectures.
mAP@0.5: results/eval_eduardo_results.json 640px-native keys (974-image
eduardo group-aware 5-fold CV means): deg15_640, v8deg15_640, v5champ_640,
ghost_640, dws_640, fnet_640.
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
# v8n-graft rows omitted -- no 640px mAP exists for them (see module docstring).
POINTS = [
    ("YOLOv11n", "Stock", "deg15_640", "v11n_obb", 640),
    ("YOLOv8n", "Stock", "v8deg15_640", "v8n_obb_champ", 640),
    ("YOLOv5n", "Stock", "v5champ_640", "v5n", 640),
    ("YOLOv11n", "Ghost", "ghost_640", "v11_ghost", 640),
    ("YOLOv11n", "DWS", "dws_640", "v11_dws", 640),
    ("YOLOv11n", "FasterNet", "fnet_640", "v11_fnet", 640),
]

fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
fig.subplots_adjust(top=0.86, bottom=0.13, left=0.13, right=0.97)

for family, backbone, cv_key, jkey, res in POINTS:
    x, y = fps(jkey, res), ed_map(cv_key)
    ax.scatter([x], [y], s=170, color=FAMILY_COLOR[family], marker=BACKBONE_MARKER[backbone],
               edgecolors="white", linewidths=1.3, zorder=3)

# legends: color = model family, marker (grey) = backbone -- only the
# families/backbones actually plotted (v8n has Stock only at this resolution)
family_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                              markeredgecolor="white", markersize=11, label=f)
                  for f, c in FAMILY_COLOR.items()]
backbones_present = sorted({b for _, b, *_ in POINTS}, key=list(BACKBONE_MARKER).index)
backbone_handles = [plt.Line2D([0], [0], marker=BACKBONE_MARKER[b], color="none",
                                markerfacecolor=MUTED, markeredgecolor="white",
                                markersize=10, label=b) for b in backbones_present]
leg1 = ax.legend(handles=family_handles, title="Model family", loc="upper right",
                  frameon=False, fontsize=10, title_fontsize=10.5, labelspacing=0.4,
                  handletextpad=0.6, borderaxespad=0.3, bbox_to_anchor=(0.72, 1.0))
ax.add_artist(leg1)
ax.legend(handles=backbone_handles, title="Backbone", loc="upper right",
          frameon=False, fontsize=10, title_fontsize=10.5, labelspacing=0.4,
          handletextpad=0.6, borderaxespad=0.3, bbox_to_anchor=(1.0, 1.0))

ax.set_xlabel("Jetson Orin Nano fps (batch-1, PyTorch fp32, measured)")
ax.set_ylabel("mAP@0.5 (5-fold CV mean)")
ax.set_xlim(19.0, 32.0)
ax.set_ylim(0.55, 0.78)
ax.grid(axis="y", color=GRID)
ax.grid(False, axis="x")
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=INK2)

fig.text(0.13, 0.945, "Accuracy vs. edge throughput across backbones (640px)",
          ha="left", va="bottom", fontsize=17, fontweight="bold", color=INK)
fig.text(0.13, 0.005, "v8n-graft family not yet benchmarked at 640px (1280px only) -- omitted rather than mislabeled.",
          ha="left", va="bottom", fontsize=8.5, style="italic", color=MUTED)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"accuracy_vs_throughput_all_models_640.{ext}", facecolor="white", dpi=300)
plt.close(fig)
print("saved", OUT / "accuracy_vs_throughput_all_models_640.{png,pdf}")
