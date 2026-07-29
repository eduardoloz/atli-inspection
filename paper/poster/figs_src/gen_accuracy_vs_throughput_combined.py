"""Accuracy vs. edge throughput across backbones -- COMBINED 1280px + 640px
view. Merges the two sibling figures (gen_accuracy_vs_throughput.py, 1280px
frontier; gen_accuracy_vs_throughput_640.py, 640px deployment resolution)
onto one chart: every model is plotted at its measured Jetson Orin Nano fps
for the SAME resolution it was trained/evaluated at (PyTorch fp32, batch-1,
pt_{res}_fp32_conf0.25 from hardware_testing/jetson_fps_results.json).

Encoding: color = model family, marker shape = backbone (as in the
siblings), and marker FILL = training resolution -- solid filled markers are
1280px-trained models, hollow (white-filled, color-edged) markers are
640px-trained models.

Recipe rule (2026-07-29, consistency fix vs poster_fig_resolution_frontier):
every point is the BEST recipe found for that backbone x resolution --
stock v11n/v8n use the mixup/blur champion rungs (mix15_1280 0.804 /
blurmix_640 0.757 / v8mix15_1280 0.798), matching the numbers quoted in the
resolution-frontier figure; grafts use whichever of deg15 / blurmix scored
higher (recipe gains do not rescue grafts -- phases 15/18/22). The v8-graft
640px cells trained 2026-07-29 (phase 22), completing all 9 points in both
clusters.

mAP@0.5: results/eval_eduardo_results.json (974-image eduardo group-aware
5-fold CV means).
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
BACKBONE_MARKER = {"Original model": "o", "Ghost": "s", "DWS": "^", "FasterNet": "D"}


def ed_map(cond):
    return float(np.mean([f["mAP50"] for f in RES_ED[cond]["folds"]]))


def fps(model_key, res):
    return JETSON[model_key][f"pt_{res}_fp32_conf0.25"]["fps"]


# (family, backbone, eduardo-CV key, jetson-json key, train/eval resolution)
# key rule: best recipe per backbone x resolution (see module docstring)
POINTS = [
    # 1280px-trained (filled)
    ("YOLOv11n", "Original model", "mix15_1280", "v11n_obb", 1280),      # 0.804 champion
    ("YOLOv8n", "Original model", "v8mix15_1280", "v8n_obb_champ", 1280),  # 0.798 (p20)
    ("YOLOv5n", "Original model", "v5champ", "v5n", 1280),
    ("YOLOv11n", "Ghost", "ghost_deg15", "v11_ghost", 1280),
    ("YOLOv8n", "Ghost", "v8ghost_deg15", "v8_ghost", 1280),
    ("YOLOv11n", "DWS", "dws_deg15", "v11_dws", 1280),
    ("YOLOv8n", "DWS", "v8dws_deg15", "v8_dws", 1280),
    ("YOLOv11n", "FasterNet", "fnet_deg15", "v11_fnet", 1280),
    ("YOLOv8n", "FasterNet", "v8fnet_deg15", "v8_fnet", 1280),
    # 640px-trained (hollow); v8 grafts from phase 22 (2026-07-29)
    ("YOLOv11n", "Original model", "blurmix_640", "v11n_obb", 640),      # 0.757 640-champion
    ("YOLOv8n", "Original model", "v8deg15_640", "v8n_obb_champ", 640),  # best/only v8@640
    ("YOLOv5n", "Original model", "v5champ_640", "v5n", 640),
    ("YOLOv11n", "Ghost", "ghost_640", "v11_ghost", 640),           # deg15 > blurmix
    ("YOLOv8n", "Ghost", "v8ghost_640", "v8_ghost", 640),           # deg15 > blurmix
    ("YOLOv11n", "DWS", "dws_blurmix640", "v11_dws", 640),          # blurmix > deg15
    ("YOLOv8n", "DWS", "v8dws_blurmix640", "v8_dws", 640),          # blurmix > deg15
    ("YOLOv11n", "FasterNet", "fnet_blurmix640", "v11_fnet", 640),  # blurmix > deg15
    ("YOLOv8n", "FasterNet", "v8fnet_blurmix640", "v8_fnet", 640),  # blurmix > deg15
]

# broken x-axis: left panel = 1280px cluster, right panel = 640px cluster,
# the empty 13-21 fps band is cut out
fig, (axL, axR) = plt.subplots(
    1, 2, sharey=True, figsize=(9.2, 5.6), dpi=200,
    gridspec_kw={"width_ratios": [4, 9.5], "wspace": 0.05})
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.11, right=0.97)

for family, backbone, cv_key, jkey, res in POINTS:
    x, y = fps(jkey, res), ed_map(cv_key)
    ax = axL if res == 1280 else axR
    if res == 1280:  # solid fill
        ax.scatter([x], [y], s=170, color=FAMILY_COLOR[family],
                   marker=BACKBONE_MARKER[backbone],
                   edgecolors="white", linewidths=1.3, zorder=3)
    else:  # hollow: white fill, colored edge
        ax.scatter([x], [y], s=170, facecolors="white",
                   marker=BACKBONE_MARKER[backbone],
                   edgecolors=FAMILY_COLOR[family], linewidths=1.8, zorder=3)

# legends: color = model family, marker (grey) = backbone,
# fill = training resolution (filled @1280, hollow @640)
family_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                              markeredgecolor="white", markersize=11, label=f)
                  for f, c in FAMILY_COLOR.items()]
backbone_handles = [plt.Line2D([0], [0], marker=m, color="none", markerfacecolor=MUTED,
                                markeredgecolor="white", markersize=10, label=b)
                    for b, m in BACKBONE_MARKER.items()]
fill_handles = [
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED,
               markeredgecolor="white", markersize=11, label="filled = trained @1280"),
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=MUTED, markeredgewidth=1.8, markersize=11,
               label="hollow = trained @640"),
]
LEG_KW = dict(loc="upper right", frameon=True, edgecolor="#c9c8c0", facecolor="white",
              framealpha=0.95, fontsize=10, title_fontsize=10.5, labelspacing=0.4,
              handletextpad=0.6, borderaxespad=0.3)
fig.legend(handles=family_handles, title="Model family",
           bbox_to_anchor=(0.54, 0.905), **LEG_KW)
fig.legend(handles=backbone_handles, title="Backbone",
           bbox_to_anchor=(0.70, 0.905), **LEG_KW)
fig.legend(handles=fill_handles, title="Training resolution",
           bbox_to_anchor=(0.97, 0.905), **LEG_KW)

fig.text(0.54, 0.055,
         "Jetson Orin Nano fps (batch-1, PyTorch fp32, measured; each model at its training resolution)",
         ha="center", va="center", fontsize=10, color=INK)
axL.set_ylabel("mAP@0.5 (5-fold CV mean)")
axL.set_xlim(9.0, 13.0)
axR.set_xlim(21.5, 31.0)
axL.set_ylim(0.52, 0.87)  # extra top headroom so the legends clear all points
# y grid + tick labels every 0.025 (line and number at each 0.05 halfway too)
axL.set_yticks(np.arange(0.525, 0.8501, 0.025))
axL.set_yticklabels([f"{t:.3f}" for t in np.arange(0.525, 0.8501, 0.025)])
for ax in (axL, axR):
    ax.grid(axis="y", color=GRID)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2)
# cut the adjoining spines and draw slanted break marks on the bottom axis
axL.spines["right"].set_visible(False)
axR.spines["left"].set_visible(False)
axR.tick_params(left=False)
break_kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=11, linestyle="none",
                color="#c3c2b7", mec="#c3c2b7", mew=1.4, clip_on=False)
axL.plot([1], [0], transform=axL.transAxes, **break_kw)
axR.plot([0], [0], transform=axR.transAxes, **break_kw)

fig.text(0.11, 0.915, "Accuracy vs. edge throughput across backbones (1280px & 640px)",
          ha="left", va="bottom", fontsize=17, fontweight="bold", color=INK)
# (footnote removed per user 2026-07-29; recipe rule documented in the module docstring)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"accuracy_vs_throughput_combined.{ext}", facecolor="white", dpi=300)
plt.close(fig)
print("saved", OUT / "accuracy_vs_throughput_combined.{png,pdf}")
