"""fps vs mAP@0.5 efficiency frontier across all 10 architectures benchmarked
on real Jetson Orin Nano hardware (results/jetson_orin_nano_fps_benchmark.md).
PyTorch fp32 fps used throughout (not TensorRT fp16) since 3 of 10
architectures either hard-fail TensorRT export (DWS) or show anomalous
TensorRT-slower-than-PyTorch behavior at these resolutions (P2, v8-FasterNet)
-- PyTorch numbers are complete and consistent across every architecture.

mAP@0.5 = 5-fold eduardo-CV mean +/- std, from results/eval_eduardo_results.json.
fps = batch-1 end-to-end (pre+forward+post) from hardware_testing/jetson_fps_results.json,
measured at each architecture's own eval resolution: 1280 for the 8-way
backbone-graft frontier (deg15 recipe), 640 for v5n and P2 (their CV numbers
are 640-only) -- these two are annotated separately since they are not
directly comparable to the 1280 points on the fps axis (smaller input =
faster for ANY architecture, independent of backbone efficiency)."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Helvetica Neue"

OUT = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent
EVAL = json.load(open(RESULTS_DIR / "eval_eduardo_results.json"))
FLOPS = json.load(open(RESULTS_DIR / "model_flops.json"))
FPS = json.load(open(RESULTS_DIR.parent / "hardware_testing" / "jetson_fps_results.json"))

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_V11 = "#4472C4"   # blue
C_V8 = "#ED7D31"    # orange
C_OTHER = "#70AD47"  # green, for the off-resolution (640) points

import numpy as np


def cv_map(key):
    maps = [f["mAP50"] for f in EVAL[key]["folds"]]
    return float(np.mean(maps)), float(np.std(maps))


def pt_fps(arch_key, res):
    return FPS[arch_key][f"pt_{res}_fp32_conf0.25"]["fps"]


# (label, family, eval_key, fps_key, resolution, params_M)
POINTS = [
    ("YOLOv11n-OBB\n(champion)", "v11", "deg15", "v11n_obb", 1280, FLOPS["v11_deg15_obb"]["params_M"]),
    ("v11-FasterNet", "v11", "fnet_deg15", "v11_fnet", 1280, FLOPS["v11_fnet_obb"]["params_M"]),
    ("v11-DWS", "v11", "dws_deg15", "v11_dws", 1280, FLOPS["v11_dws_obb"]["params_M"]),
    ("v11-Ghost", "v11", "ghost_deg15", "v11_ghost", 1280, FLOPS["v11_ghost_obb"]["params_M"]),
    ("YOLOv8n-OBB\n(true champion)", "v8", "v8deg15", "v8n_obb_champ", 1280, FLOPS["v8_deg15_obb"]["params_M"]),
    ("v8-FasterNet", "v8", "v8fnet_deg15", "v8_fnet", 1280, 2.79),
    ("v8-DWS", "v8", "v8dws_deg15", "v8_dws", 1280, 2.84),
    ("v8-Ghost", "v8", "v8ghost_deg15", "v8_ghost", 1280, 2.52),
    ("YOLOv5n\n(detect-only, @640)", "other", "v5champ_640", "v5n", 640, FLOPS["v5_champ_det"]["params_M"]),
    ("YOLOv11n-P2\n(@640)", "other", "p2_640", "v11_p2", 640, FLOPS["v11_p2_obb"]["params_M"]),
]

rows = []
for label, fam, ek, fk, res, params in POINTS:
    m, s = cv_map(ek)
    f = pt_fps(fk, res)
    rows.append(dict(label=label, fam=fam, mean=m, std=s, fps=f, res=res, params=params))

fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

fam_style = {
    "v11": (C_V11, "o", "YOLOv11n-OBB family (@1280)"),
    "v8": (C_V8, "o", "YOLOv8n-OBB family (@1280)"),
    "other": (C_OTHER, "^", "Off-resolution refs (@640 — not directly comparable on fps axis)"),
}
seen_fam = set()
# manual (dx, dy, ha, va) offsets per label, tuned to avoid collisions in the
# dense @1280 cluster -- v11 labels pushed left, v8 labels pushed right
LABEL_OFFSETS = {
    "YOLOv11n-OBB\n(champion)":      (-1.6, 0.020, "right", "bottom"),
    "v11-FasterNet":                 (-1.6, 0.006, "right", "center"),
    "v11-DWS":                       (-1.6, 0.000, "right", "center"),
    "v11-Ghost":                     (-1.6, 0.000, "right", "center"),
    "YOLOv8n-OBB\n(true champion)":  (1.6, 0.018, "left", "bottom"),
    "v8-FasterNet":                  (2.0, 0.010, "left", "center"),
    "v8-DWS":                        (1.6, -0.008, "left", "center"),
    "v8-Ghost":                      (1.6, 0.000, "left", "center"),
    "YOLOv5n\n(detect-only, @640)":  (0, 0.028, "center", "bottom"),
    "YOLOv11n-P2\n(@640)":           (0, 0.028, "center", "bottom"),
}
for r in rows:
    color, marker, flabel = fam_style[r["fam"]]
    lbl = flabel if r["fam"] not in seen_fam else None
    seen_fam.add(r["fam"])
    size = 200 + (r["params"] - 2.0) * 260
    ax.scatter(r["fps"], r["mean"], s=size, color=color, marker=marker,
               edgecolor="#fcfcfb", linewidth=1.4, alpha=0.88, zorder=3, label=lbl)
    ax.errorbar(r["fps"], r["mean"], yerr=r["std"], fmt="none", ecolor=color,
                elinewidth=1.2, capsize=3.5, zorder=2, alpha=0.7)
    dx, dy, ha, va = LABEL_OFFSETS[r["label"]]
    ax.annotate(r["label"], xy=(r["fps"], r["mean"]), xytext=(r["fps"] + dx, r["mean"] + dy),
                fontsize=8.3, color=INK2, ha=ha, va=va, zorder=4, linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED, alpha=0.6,
                                shrinkA=4, shrinkB=6) if (dx, dy) != (0, 0.028) or r["fam"] == "other" else None)

ax.set_xlabel("PyTorch fp32 fps, batch-1 end-to-end (Jetson Orin Nano)", fontsize=11, color=INK2)
ax.set_ylabel("mAP@0.5 (5-fold eduardo-CV mean ± std)", fontsize=11, color=INK2)
ax.set_xlim(5, 35)
ax.set_ylim(0.58, 0.85)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
ax.tick_params(colors=MUTED)

title = textwrap.fill("fps vs accuracy — real Jetson Orin Nano measurements across the full backbone frontier", width=52)
ax.set_title(title, fontsize=16, fontweight="bold", color=INK, loc="left", pad=14)
ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK2, markerscale=0.7)

subtitle = ("Marker size ~ parameter count. TensorRT fp16 fps omitted from this view: DWS hard-fails "
            "TensorRT export entirely, and P2/v8-FasterNet show anomalous TensorRT-slower-than-PyTorch "
            "results at these resolutions (see jetson_orin_nano_fps_benchmark.md) -- PyTorch fps is the "
            "complete, consistent axis across all 10 architectures. YOLOv8n-OBB dominates the 1280 "
            "frontier (faster AND more accurate than every graft); stock backbones beat every lightweight "
            "graft on both axes. v5n/P2 (green triangles, @640) trade resolution for fps and are not a "
            "fair same-resolution comparison against the @1280 points.")
wrapped = textwrap.fill(subtitle, width=92)
fig.text(0.01, 0.01, wrapped, fontsize=8.3, color=MUTED, linespacing=1.4)

plt.tight_layout(rect=[0, 0.13, 1, 1])
fig.savefig(OUT / "fig_jetson_fps_frontier.png", facecolor="#fcfcfb", bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "fig_jetson_fps_frontier.png")
