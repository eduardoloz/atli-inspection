"""Figure 11 — fps vs mAP@0.5 frontier on real Jetson Orin Nano hardware,
with the marker FILL encoding train/deploy resolution (user request
2026-07-29): same axes/legend/colors as fig_jetson_fps_frontier, but every
architecture appears (up to) twice —
  solid fill   = trained AND deployed @1280
  hatched fill = trained AND deployed @640
Both points of an architecture share its family color/marker; marker size ~
parameter count, as before. PyTorch fp32 fps throughout (same rationale as
the original figure: DWS hard-fails TensorRT export, P2/v8-FasterNet show
anomalous TensorRT results — PyTorch is the one complete, consistent axis).

mAP@0.5 = 5-fold eduardo-CV mean ± std (eval_eduardo_results.json):
@1280 keys = deg15-recipe conditions; @640 keys = their deg15@640 twins.
v8-graft @640 keys (v8ghost_640/v8dws_640/v8fnet_640) come from phase 22 —
points are skipped with a console note until those evals land, so re-running
this script after phase 22 completes the figure automatically."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Helvetica Neue"

OUT = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent
EVAL = json.load(open(RESULTS_DIR / "eval_eduardo_results.json"))
FLOPS = json.load(open(RESULTS_DIR / "model_flops.json"))
FPS = json.load(open(RESULTS_DIR.parent / "hardware_testing" / "jetson_fps_results.json"))

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_V11 = "#4472C4"   # blue
C_V8 = "#ED7D31"    # orange
C_OTHER = "#70AD47"  # green


def cv_map(key):
    maps = [f["mAP50"] for f in EVAL[key]["folds"]]
    return float(np.mean(maps)), float(np.std(maps))


def pt_fps(arch_key, res):
    return FPS[arch_key][f"pt_{res}_fp32_conf0.25"]["fps"]


P_V8FNET, P_V8DWS, P_V8GHOST = 2.79, 2.84, 2.52  # from p14 (not in model_flops.json)

# (label, family, eval_key, fps_key, resolution, params_M)
# resolution doubles as the fill code: 1280 = solid, 640 = hatched.
POINTS = [
    # --- trained & deployed @1280 (solid) ---
    ("YOLOv11n-OBB", "v11", "deg15", "v11n_obb", 1280, FLOPS["v11_deg15_obb"]["params_M"]),
    ("v11-FasterNet", "v11", "fnet_deg15", "v11_fnet", 1280, FLOPS["v11_fnet_obb"]["params_M"]),
    ("v11-DWS", "v11", "dws_deg15", "v11_dws", 1280, FLOPS["v11_dws_obb"]["params_M"]),
    ("v11-Ghost", "v11", "ghost_deg15", "v11_ghost", 1280, FLOPS["v11_ghost_obb"]["params_M"]),
    ("YOLOv8n-OBB", "v8", "v8deg15", "v8n_obb_champ", 1280, FLOPS["v8_deg15_obb"]["params_M"]),
    ("v8-FasterNet", "v8", "v8fnet_deg15", "v8_fnet", 1280, P_V8FNET),
    ("v8-DWS", "v8", "v8dws_deg15", "v8_dws", 1280, P_V8DWS),
    ("v8-Ghost", "v8", "v8ghost_deg15", "v8_ghost", 1280, P_V8GHOST),
    # --- trained & deployed @640 (hatched) ---
    ("YOLOv11n-OBB @640", "v11", "deg15_640", "v11n_obb", 640, FLOPS["v11_deg15_obb"]["params_M"]),
    ("v11-FasterNet @640", "v11", "fnet_640", "v11_fnet", 640, FLOPS["v11_fnet_obb"]["params_M"]),
    ("v11-DWS @640", "v11", "dws_640", "v11_dws", 640, FLOPS["v11_dws_obb"]["params_M"]),
    ("v11-Ghost @640", "v11", "ghost_640", "v11_ghost", 640, FLOPS["v11_ghost_obb"]["params_M"]),
    ("YOLOv8n-OBB @640", "v8", "v8deg15_640", "v8n_obb_champ", 640, FLOPS["v8_deg15_obb"]["params_M"]),
    ("v8-FasterNet @640", "v8", "v8fnet_640", "v8_fnet", 640, P_V8FNET),
    ("v8-DWS @640", "v8", "v8dws_640", "v8_dws", 640, P_V8DWS),
    ("v8-Ghost @640", "v8", "v8ghost_640", "v8_ghost", 640, P_V8GHOST),
    ("YOLOv5n (detect-only)", "other", "v5champ_640", "v5n", 640, FLOPS["v5_champ_det"]["params_M"]),
    ("YOLOv11n-P2", "other", "p2_640", "v11_p2", 640, FLOPS["v11_p2_obb"]["params_M"]),
]

rows = []
for label, fam, ek, fk, res, params in POINTS:
    if ek not in EVAL:
        print(f"[pending] {label}: eval key '{ek}' not in results yet — skipped "
              f"(re-run after phase 22 eval)")
        continue
    m, s = cv_map(ek)
    rows.append(dict(label=label, fam=fam, mean=m, std=s,
                     fps=pt_fps(fk, res), res=res, params=params))

fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

fam_style = {
    "v11": (C_V11, "o", "YOLOv11n-OBB family"),
    "v8": (C_V8, "o", "YOLOv8n-OBB family"),
    "other": (C_OTHER, "^", "Off-family refs (v5n / P2)"),
}
# label offsets tuned per point: 1280 cluster (fps 9-12) pushed outward as in
# the original; 640 cluster (fps 19-31) spread above/below
LABEL_OFFSETS = {
    "YOLOv11n-OBB":          (-1.4, 0.016, "right", "bottom"),
    "v11-FasterNet":         (-1.4, 0.006, "right", "center"),
    "v11-DWS":               (-1.4, 0.000, "right", "center"),
    "v11-Ghost":             (-1.4, -0.006, "right", "center"),
    "YOLOv8n-OBB":           (1.4, 0.016, "left", "bottom"),
    "v8-FasterNet":          (1.6, 0.008, "left", "center"),
    "v8-DWS":                (1.4, -0.008, "left", "center"),
    "v8-Ghost":              (1.4, 0.000, "left", "center"),
    "YOLOv11n-OBB @640":     (-1.6, 0.014, "right", "bottom"),
    "v11-FasterNet @640":    (-1.8, -0.006, "right", "center"),
    "v11-DWS @640":          (-1.8, -0.016, "right", "center"),
    "v11-Ghost @640":        (-1.8, 0.006, "right", "center"),
    "YOLOv8n-OBB @640":      (1.6, 0.012, "left", "bottom"),
    "v8-FasterNet @640":     (1.8, 0.004, "left", "center"),
    "v8-DWS @640":           (1.8, -0.010, "left", "center"),
    "v8-Ghost @640":         (1.8, -0.020, "left", "center"),
    "YOLOv5n (detect-only)": (0, 0.024, "center", "bottom"),
    "YOLOv11n-P2":           (0, -0.030, "center", "top"),
}
seen_fam = set()
for r in rows:
    color, marker, flabel = fam_style[r["fam"]]
    lbl = flabel if r["fam"] not in seen_fam else None
    seen_fam.add(r["fam"])
    size = 200 + (r["params"] - 2.0) * 260
    if r["res"] == 1280:  # solid fill
        ax.scatter(r["fps"], r["mean"], s=size, color=color, marker=marker,
                   edgecolor="#fcfcfb", linewidth=1.4, alpha=0.88, zorder=3, label=lbl)
    else:  # hatched fill, family-colored edge
        ax.scatter(r["fps"], r["mean"], s=size, facecolor="#fcfcfb", marker=marker,
                   edgecolor=color, linewidth=1.6, hatch="//////", alpha=0.88,
                   zorder=3, label=lbl)
    ax.errorbar(r["fps"], r["mean"], yerr=r["std"], fmt="none", ecolor=color,
                elinewidth=1.2, capsize=3.5, zorder=2, alpha=0.7)
    dx, dy, ha, va = LABEL_OFFSETS[r["label"]]
    ax.annotate(r["label"], xy=(r["fps"], r["mean"]), xytext=(r["fps"] + dx, r["mean"] + dy),
                fontsize=8.0, color=INK2, ha=ha, va=va, zorder=4, linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED, alpha=0.6,
                                shrinkA=4, shrinkB=6))

# fill-pattern legend entries (grey, family-neutral)
ax.scatter([], [], s=170, color="#9b9a94", marker="s", edgecolor="#fcfcfb",
           linewidth=1.2, label="trained & deployed @1280")
ax.scatter([], [], s=170, facecolor="#fcfcfb", marker="s", edgecolor="#9b9a94",
           linewidth=1.4, hatch="//////", label="trained & deployed @640")

ax.set_xlabel("PyTorch fp32 fps, batch-1 end-to-end (Jetson Orin Nano)", fontsize=11, color=INK2)
ax.set_ylabel("mAP@0.5 (5-fold eduardo-CV mean ± std)", fontsize=11, color=INK2)
ax.set_xlim(5, 35)
ax.set_ylim(0.49, 0.85)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
ax.tick_params(colors=MUTED)

title = textwrap.fill("fps vs accuracy on Jetson Orin Nano — fill shows train/deploy resolution", width=52)
ax.set_title(title, fontsize=16, fontweight="bold", color=INK, loc="left", pad=14)
ax.legend(loc="lower left", fontsize=9, frameon=False, labelcolor=INK2, markerscale=0.7)

subtitle = ("Marker size ~ parameter count; solid = trained & deployed @1280, hatched = trained & "
            "deployed @640 (deg15 recipe at both resolutions). PyTorch fp32 fps (TensorRT omitted: "
            "DWS hard-fails export; P2/v8-FasterNet anomalous — see jetson_orin_nano_fps_benchmark.md). "
            "Dropping 1280-to-640 roughly doubles fps for every architecture but costs 0.07-0.15 mAP, "
            "and the graft family loses MORE accuracy at 640 than at 1280 (phase 15) — stock "
            "backbones dominate both fill groups. v5n/P2 exist only as @640 points.")
wrapped = textwrap.fill(subtitle, width=92)
fig.text(0.01, 0.01, wrapped, fontsize=8.3, color=MUTED, linespacing=1.4)

plt.tight_layout(rect=[0, 0.13, 1, 1])
fig.savefig(OUT / "fig_jetson_fps_frontier_trainres.png", facecolor="#fcfcfb", bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "fig_jetson_fps_frontier_trainres.png")
