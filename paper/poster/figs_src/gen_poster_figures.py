#!/usr/bin/env python3
"""Generate poster-quality figures (300-dpi PNG + vector PDF) summarizing every
method the project tried, into paper/poster/figures/.

Style: dataviz-skill compliant — palette validated with the skill's
validate_palette.js on a white surface (blue #2166ac / green #1a9850 /
orange #d0700e all PASS; #d0700e is the repo orange #e08214 darkened to clear
3:1 contrast on white; gray #666666 is chrome/de-emphasis ink, never a series).
Thin marks, hairline solid grids, top/right spines off, direct labels in ink
(text never wears the series color), legends for >=2 series.

Sources of every number (all transcribed, none invented):
  Fig poster_fig_cv_main   : paper/main.tex Table tab:cv (proper 5-fold CV,
                             mean +- std over folds; DD recall row).
  Fig poster_fig_frontier  : paper/main.tex Table tab:sota (identical folds).
  Fig poster_fig_external  : paper/main.tex Table tab:external + Sec. sec:external
                             (baseline 0.641+-0.036 n=11; mixing ladder
                             0.58-0.65 n=8; scale-matched 0.63-0.65 n=3;
                             champ 0.699+-0.027 n=11; in-distribution control
                             ~0.91 n=4; in-domain pretrain -7.5 mAP).
  Fig poster_fig_ablation  : paper/main.tex Sec. sec:ablate —
                             mAP deltas: TTA +0.002; close_mosaic=20 0.767 vs
                             deploy 0.766 (@768 clean); stage-1 300ep 0.686 vs
                             baseline 0.691; no scale-aug 0.724 vs champ 0.747;
                             mosaic off -0.030 (@768); rotation 45deg -0.037
                             (@768); freeze-10 0.696 vs 0.747; freeze-20 0.448
                             vs 0.747.  DD deltas: close_mosaic=20 0.708 vs
                             0.664 (@768); oversample+scale @640 0.661 vs champ
                             0.699; no oversampling -0.07 (@768); rotation
                             10deg -0.10 (@768); x6 oversample 0.635 @640 vs
                             0.751 @1280.  Copy-paste "hurt the defective
                             class" has no logged seed-averaged delta -> figure
                             footnote only.
  Fig poster_fig_perclass  : paper/figures_src/gen_paper_figures.py Fig 2 data
                             arrays (= results/cplid_before_after.md, clean
                             decontaminated benchmark, 3 seeds).
  Fig poster_fig_deploy    : paper/main.tex Table tab:edge +
                             paper/figures_src/gen_paper_figures.py deploy
                             arrays (768 retrain 0.766+-0.009 / DD 0.664+-0.042;
                             TRT FP16 engine 0.768 / DD 0.732; floors 0.745 /
                             0.59 provisional).
  Fig poster_fig_blur      : results/eval_blur_robustness.json (blurred means
                             0.466+-0.050 and 0.678+-0.038) +
                             results/eduardo_cv_full_tables.md blur table
                             (clean 0.793 vs 0.790), 5-fold eduardo CV.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ palette
BLUE = "#2166ac"    # series: baseline / heavyweight reference
GREEN = "#1a9850"   # series: our recipe ("champ")
ORANGE = "#d0700e"  # series: third condition / negative effects
GRAY = "#666666"    # chrome & de-emphasis only (never a series)
INK = "#1a1a1a"     # primary text
INK2 = "#4d4c49"    # secondary text
MUTED = "#898781"   # muted text / axis
GRID = "#e3e2dc"    # hairline grid
SURF = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12.5,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "figure.facecolor": SURF,
    "savefig.facecolor": SURF,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "grid.linestyle": "-",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def title(ax_or_fig, main, sub, x=0.01, y=1.0):
    """Left-aligned takeaway title + secondary-ink subtitle on a figure."""
    ax_or_fig.text(x, y, main, ha="left", va="bottom", fontsize=17,
                   fontweight="bold", color=INK, transform=ax_or_fig.transFigure
                   if hasattr(ax_or_fig, "transFigure") else None)
    ax_or_fig.text(x, y - 0.055, sub, ha="left", va="bottom", fontsize=13,
                   color=INK2, transform=ax_or_fig.transFigure
                   if hasattr(ax_or_fig, "transFigure") else None)


# =====================================================================
# Fig 1 — proper 5-fold CV headline (Table tab:cv)
# =====================================================================
models = ["YOLOv5n", "YOLOv8n", "YOLOv11n"]
cv = {   # condition: (mAP, mAP_std, DD_AP, DD_std) per model v5/v8/v11
    "base":  ([0.716, 0.725, 0.722], [0.028, 0.021, 0.019],
              [0.695, 0.735, 0.743], [0.076, 0.044, 0.074]),
    "champ": ([0.758, 0.771, 0.785], [0.018, 0.023, 0.023],
              [0.789, 0.799, 0.812], [0.079, 0.063, 0.080]),
    "osall": ([0.776, 0.773, 0.772], [0.035, 0.022, 0.024],
              [0.811, 0.804, 0.809], [0.073, 0.058, 0.071]),
}
cond_colors = {"base": BLUE, "champ": GREEN, "osall": ORANGE}
cond_names = {"base": "base (640 px)",
              "champ": "champ (1280 px + 3× oversample + scale 0.9)",
              "osall": "osall (all-defect 3× oversample)"}

# OBB CV (results/cv_obb_results.md summary table; v8n/v11n only — no v5n OBB
# runs). Caveat: OBB folds come from a different annotation export than the
# detection CV -> compare within task; the det-aligned same-fold control shows
# task mode is a wash overall (det 0.802 vs OBB 0.808 mAP).
cv_obb = {  # condition: (mAP, mAP_std, DD_AP, DD_std) per model v8/v11
    "base":  ([0.766, 0.779], [0.026, 0.011], [0.758, 0.798], [0.043, 0.024]),
    "champ": ([0.801, 0.808], [0.021, 0.014], [0.743, 0.750], [0.047, 0.058]),
    "osall": ([0.799, 0.817], [0.016, 0.017], [0.743, 0.782], [0.068, 0.066]),
}

fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.2))
fig.subplots_adjust(top=0.815, bottom=0.06, left=0.07, right=0.985,
                    wspace=0.18, hspace=0.38)
w = 0.24


def cv_panel(ax, data, model_names, mi, ylab, with_legend=False):
    x = np.arange(len(model_names))
    for k, (cond, dat) in enumerate(data.items()):
        vals, errs = dat[mi], dat[mi + 1]
        xs = x + (k - 1) * (w + 0.03)
        ax.bar(xs, vals, width=w, color=cond_colors[cond],
               label=cond_names[cond] if with_legend else None, zorder=3)
        ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK2,
                    elinewidth=1.4, capsize=4, capthick=1.4, zorder=4)
        for xi_, v in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (xi_, v), xytext=(0, -20),
                        textcoords="offset points", ha="center", fontsize=11,
                        color="white", fontweight="bold", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_xlim(-0.65, len(model_names) - 0.35)
    ax.set_ylim(0.55, 0.92)
    ax.set_ylabel(ylab)
    ax.grid(axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


cv_panel(axes[0, 0], cv, models, 0, "mAP@0.5", with_legend=True)
cv_panel(axes[0, 1], cv, models, 2, "Defective Damper AP@0.5")
cv_panel(axes[1, 0], cv_obb, ["YOLOv8n", "YOLOv11n"], 0, "mAP@0.5 (OBB)")
cv_panel(axes[1, 1], cv_obb, ["YOLOv8n", "YOLOv11n"], 2, "Defective Damper AP@0.5 (OBB)")
axes[0, 0].set_title("Detection (axis-aligned boxes) — overall", fontsize=14,
                     color=INK2, loc="left", pad=8)
axes[0, 1].set_title("Detection — rare class", fontsize=14, color=INK2,
                     loc="left", pad=8)
axes[1, 0].set_title("OBB (rotated boxes) — overall", fontsize=14, color=INK2,
                     loc="left", pad=8)
axes[1, 1].set_title("OBB — rare class", fontsize=14, color=INK2,
                     loc="left", pad=8)
axes[0, 0].annotate("best (detection)", (2, 0.785 + 0.028), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=12.5, color=INK, fontweight="bold")
axes[1, 0].annotate("best (OBB)", (1 + w + 0.03, 0.817 + 0.022), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=12.5, color=INK, fontweight="bold")
axes[1, 1].annotate("baseline is best\nunder OBB", (1 - w - 0.03, 0.798 + 0.029),
                    xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=11.5, color=INK2)
fig.legend(loc="upper left", bbox_to_anchor=(0.07, 0.885), ncol=3,
           frameon=False, handlelength=1.2, handleheight=1.0,
           columnspacing=1.6)
title(fig, "The native data-centric recipe lifts every model — in detection and OBB",
      "Mean ± std over 5 disjoint folds, 1,343-image merged pool · no external data, no new labels · detection best = champ v11n; OBB best = osall v11n\n"
      "OBB folds use a separate annotation export — compare within task (same-fold control: task mode is a wash, det 0.802 vs OBB 0.808 mAP)",
      x=0.07, y=0.925)
save(fig, "poster_fig_cv_main")

# =====================================================================
# Fig 2 — accuracy vs parameters frontier (Table tab:sota)
# =====================================================================
fig, ax = plt.subplots(figsize=(10.5, 6.2))
fig.subplots_adjust(top=0.82, bottom=0.12, left=0.09, right=0.97)
pts = [  # name, params(M), mAP, std, color, marker, size
    ("DINO-4scale",   47.0, 0.785, 0.032, BLUE,  "o", 130),
    ("RTMDet-Tiny",    4.8, 0.747, 0.023, BLUE,  "o", 130),
    ("Dynamic R-CNN", 41.0, 0.718, 0.023, BLUE,  "o", 130),
    ("YOLOv11n base", 2.6, 0.722, 0.019, GRAY,  "o", 130),
    ("YOLOv11n + our recipe", 2.6, 0.785, 0.023, GREEN, "*", 550),
]
for name, p, m, s, c, mk, sz in pts:
    ax.errorbar(p, m, yerr=s, fmt="none", ecolor=c, elinewidth=1.6,
                capsize=5, capthick=1.6, alpha=0.55, zorder=2)
    ax.scatter(p, m, c=c, marker=mk, s=sz, zorder=3,
               edgecolors="white", linewidths=1.5)
lab = dict(fontsize=13, color=INK)
ax.annotate("DINO-4scale\n47M · 0.785 ± 0.032", (47, 0.785),
            xytext=(16, -8), textcoords="offset points", va="center", **lab)
ax.annotate("RTMDet-Tiny\n4.8M · 0.747 ± 0.023", (4.8, 0.747),
            xytext=(14, -18), textcoords="offset points", **lab)
ax.annotate("Dynamic R-CNN\n41M · 0.718 ± 0.023", (41, 0.718),
            xytext=(-16, -8), textcoords="offset points", ha="right",
            va="center", **lab)
ax.annotate("YOLOv11n base\n2.6M · 0.722 ± 0.019", (2.6, 0.722),
            xytext=(16, -14), textcoords="offset points", **lab)
ax.annotate("YOLOv11n + our recipe\n2.6M · 0.785 ± 0.023", (2.6, 0.785),
            xytext=(0, 26), textcoords="offset points", ha="center",
            fontsize=13.5, fontweight="bold", color=INK, zorder=6,
            bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5))
# the 18x arrow
ax.annotate("", xy=(2.9, 0.795), xytext=(42, 0.795),
            arrowprops=dict(arrowstyle="<->", lw=1.6, color=INK2))
ax.text(11.1, 0.7985, "same mAP@0.5, ~18× fewer parameters", ha="center",
        fontsize=13.5, color=INK2, fontstyle="italic")
ax.set_xscale("log")
ax.set_xlim(1.7, 100)
ax.set_ylim(0.66, 0.83)
ax.set_xticks([2, 5, 10, 20, 50, 100])
ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
ax.set_xlabel("Parameters (millions, log scale)")
ax.set_ylabel("mAP@0.5 (identical 5-fold CV)")
ax.set_axisbelow(True)
title(fig, "A 2.6M-parameter nano model ties the 47M-parameter DINO-4scale on identical folds",
      "MMDetection heavyweights at stock configurations vs. YOLOv11n with the data-centric recipe · error bars = fold std · rare-class AP also comparable (0.812 vs 0.789)",
      x=0.06, y=0.90)
save(fig, "poster_fig_frontier")

# =====================================================================
# Fig 3 — external data negative result (Table tab:external)
# =====================================================================
fig, ax = plt.subplots(figsize=(11.5, 5.8))
fig.subplots_adjust(top=0.78, bottom=0.14, left=0.30, right=0.97)
rows = [  # label, kind, lo, hi/mean, err, n, color
    ("In-distribution control\n(external test set)", "pt", None, 0.91, None, 4, GRAY),
    ("Native champ recipe (ours)\n1280 px + oversample + scale-aug", "pt", None, 0.699, 0.027, 11, GREEN),
    ("Native baseline (640 px)", "pt", None, 0.641, 0.036, 11, BLUE),
    ("Scale-matched external mixing", "rng", 0.63, 0.65, None, 3, ORANGE),
    ("External mixing, dose ladder\n(ratios 1:1 to 11.7:1)", "rng", 0.58, 0.65, None, 8, ORANGE),
]
ys = np.arange(len(rows))[::-1]
ax.axvline(0.641, color=MUTED, lw=1.2, ls=":", zorder=1)
ax.text(0.641, len(rows) - 0.45, " baseline mean", fontsize=11.5, color=MUTED,
        ha="left", va="bottom")
for y, (labl, kind, lo, v, err, n, c) in zip(ys, rows):
    if kind == "rng":
        ax.plot([lo, v], [y, y], lw=7, color=c, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.annotate(f"{lo:.2f}–{v:.2f}", ((lo + v) / 2, y), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=12.5, color=INK)
    else:
        if err:
            ax.errorbar(v, y, xerr=err, fmt="none", ecolor=c, elinewidth=1.8,
                        capsize=5, capthick=1.8, zorder=3)
        ax.scatter(v, y, s=210, color=c, zorder=4, edgecolors="white", linewidths=1.5)
        txt = f"≈{v:.2f}" if err is None else f"{v:.3f} ± {err:.3f}"
        ax.annotate(txt, (v, y), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=12.5, color=INK,
                    fontweight="bold" if c == GREEN else "normal")
    ax.annotate(f"n = {n}", (1.005, y), xycoords=("axes fraction", "data"),
                fontsize=11.5, color=MUTED, va="center", annotation_clip=False)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=13)
ax.set_xlim(0.52, 0.97)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("Defective Damper AP@0.5 (fixed split, seed-averaged)")
ax.grid(axis="x")
ax.grid(False, axis="y")
ax.set_axisbelow(True)
ax.annotate("external labels ARE learnable —\nthe transfer is what fails",
            (0.91, ys[0]), xytext=(-30, -46), textcoords="offset points",
            ha="center", fontsize=12, color=INK2, fontstyle="italic",
            arrowprops=dict(arrowstyle="-", lw=1.0, color=MUTED,
                            connectionstyle="arc3,rad=0.25", shrinkB=8))
fig.text(0.30, 0.015, "Also negative: pretrain-then-finetune (no run beat baseline) and "
         "in-domain source pretraining (−7.5 mAP@0.5, genuine negative transfer).",
         fontsize=11.5, color=INK2, ha="left")
title(fig, "External damper data never beat the native baseline — the native recipe did",
      "Every external-data route (2,175 pHash-vetted community images) at every dose · failure mode = recall drop from label-style / domain mismatch",
      x=0.06, y=0.885)
save(fig, "poster_fig_external")

# =====================================================================
# Fig 4 — ablation deltas (Sec. sec:ablate)
# =====================================================================
abl_map = [  # label, delta
    ("Test-time aug (vs champ) — rejected", +0.002),
    ("close_mosaic = 20  (@768)", +0.001),
    ("Stage-1 300 epochs (vs 150, @640)", -0.005),
    ("Remove scale-down aug (vs champ)", -0.023),
    ("Disable mosaic  (@768)", -0.030),
    ("Rotation 45°  (@768)", -0.037),
    ("Freeze 10 backbone layers (vs champ)", -0.051),
    ("Freeze 20 layers — collapse (vs champ)", -0.299),
]
abl_dd = [
    ("close_mosaic = 20  (@768)", +0.044),
    ("Train at 640 px, not 1280 (vs champ)", -0.038),
    ("Remove 3× oversampling  (@768)", -0.070),
    ("Rotation 10°  (@768)", -0.100),
    ("6× oversample @640 (vs 6× @1280)", -0.116),
]

fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
fig.subplots_adjust(top=0.77, bottom=0.16, left=0.235, right=0.985, wspace=0.95)
for ax, data, xlab in [(axes[0], abl_map, "Δ mAP@0.5 (points)"),
                       (axes[1], abl_dd, "Δ Defective Damper AP (points)")]:
    ys = np.arange(len(data))[::-1]
    vals = [d[1] for d in data]
    span = max(abs(v) for v in vals)
    colors = [GREEN if v > 0 else ORANGE for v in vals]
    ax.barh(ys, vals, height=0.55, color=colors, zorder=3)
    ax.axvline(0, color=INK2, lw=1.2, zorder=4)
    for y, v in zip(ys, vals):
        if abs(v) > 0.55 * span:  # long bar: label inside the bar end
            ax.annotate(f"{v:+.3f}", (v, y), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=12, color="white", fontweight="bold", zorder=5)
        else:
            ax.annotate(f"{v:+.3f}", (v, y),
                        xytext=(6 if v > 0 else -6, 0), textcoords="offset points",
                        ha="left" if v > 0 else "right", va="center",
                        fontsize=12, color=INK, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=12.5)
    ax.set_xlabel(xlab)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
axes[0].set_xlim(-0.32, 0.05)
axes[1].set_xlim(-0.13, 0.08)
fig.text(0.235, 0.02, "Copy-paste augmentation also hurt the rare class (paste artifacts; no seed-averaged delta logged). "
         "Scale-jitter sweet spot: 0.85–0.9.", fontsize=11.5, color=INK2, ha="left")
title(fig, "What moves the rare class: resolution + sampling help; freezing and rotation hurt",
      "Ablations of the recipe · each bar states its reference: 'vs champ' = fixed split @1280 vs the full recipe · '@768' = clean benchmark vs the deployment recipe, 3 seeds/arm",
      x=0.05, y=0.885)
save(fig, "poster_fig_ablation")

# =====================================================================
# Fig 5 — per-class AP on the clean benchmark (gen_paper_figures.py Fig 2)
# =====================================================================
classes = ["Birdnest", "Broken\nInsulator", "Defective\nDamper", "Flashover\nInsulator",
           "Normal\nDamper", "Normal\nInsulators", "Self-Expl.\nInsulator", "overall\n(mAP@0.5)"]
base = [0.959, 0.667, 0.614, 0.590, 0.743, 0.826, 0.755, 0.736]
base_e = [0.010, 0.023, 0.082, 0.011, 0.023, 0.015, 0.016, 0.015]
champ = [0.984, 0.699, 0.622, 0.660, 0.813, 0.838, 0.871, 0.784]
champ_e = [0.004, 0.029, 0.073, 0.024, 0.019, 0.013, 0.027, 0.011]

x = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(13.0, 5.6))
fig.subplots_adjust(top=0.80, bottom=0.15, left=0.07, right=0.98)
for xi_, b, c in zip(x, base, champ):
    ax.plot([xi_ - 0.14, xi_ + 0.14], [b, c], color=GRAY, lw=1.2, alpha=0.6, zorder=2)
ax.errorbar(x - 0.14, base, yerr=base_e, fmt="o", ms=11, capsize=4.5,
            lw=0, elinewidth=1.6, capthick=1.6, color=BLUE,
            markeredgecolor="white", markeredgewidth=1.5,
            label="base (640 px)", zorder=3)
ax.errorbar(x + 0.14, champ, yerr=champ_e, fmt="D", ms=10, capsize=4.5,
            lw=0, elinewidth=1.6, capthick=1.6, color=GREEN,
            markeredgecolor="white", markeredgewidth=1.5,
            label="champ (1280 px, 3× oversampling, scale 0.9)", zorder=3)
for xi_, b, be_, c, ce_ in zip(x, base, base_e, champ, champ_e):
    d = c - b
    ax.annotate(f"{d:+.3f}", (xi_, max(b + be_, c + ce_)), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=12,
                color=INK, fontweight="bold" if d >= 0.05 else "normal")
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=12.5)
ax.set_ylim(0.46, 1.06)
ax.set_ylabel("AP@0.5 (3 seeds ± std)")
ax.axvline(6.5, color=MUTED, lw=1.0)
ax.grid(axis="y")
ax.grid(False, axis="x")
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=True, edgecolor="none", facecolor="white",
          framealpha=1.0, fontsize=12.5)
ax.annotate("directional only:\n12 test instances", (2, 0.497), ha="center",
            va="top", fontsize=11, color=MUTED, fontstyle="italic")
title(fig, "On the decontaminated benchmark the recipe improves every class (mAP@0.5 0.736 to 0.784)",
      "797-image clean benchmark (all 249 CPLID duplicates removed) · largest gain: Self-Exploded Insulator +0.116 · the headline margin survives decontamination",
      x=0.06, y=0.895)
save(fig, "poster_fig_perclass")

# =====================================================================
# Fig 6 — deployment / resolution ladder (Table tab:edge)
# =====================================================================
cfgs = ["640\nnative", "768 infer\n(1280-trained)", "768\nnative retrain",
        "1024 infer\n(1280-trained)", "1280\nnative champ"]
fps = ["22.4 fps*", "20.0 fps*", "20.0 fps*", "14.9 fps*", "10.4 fps*"]
map_ = [0.736, 0.754, 0.766, 0.783, 0.784]
dd = [0.614, 0.593, 0.664, 0.634, 0.622]
xi = np.arange(len(cfgs))
fig, ax = plt.subplots(figsize=(11.0, 6.2))
fig.subplots_adjust(top=0.80, bottom=0.17, left=0.08, right=0.97)
ax.plot(xi, map_, "-o", color=BLUE, label="mAP@0.5", ms=11, lw=2.2,
        markeredgecolor="white", markeredgewidth=1.5, zorder=3)
ax.plot(xi, dd, "-o", color=GREEN, label="Defective Damper AP", ms=11, lw=2.2,
        markeredgecolor="white", markeredgewidth=1.5, zorder=3)
ax.errorbar([2], [0.766], yerr=[0.009], fmt="none", ecolor=BLUE, elinewidth=1.6,
            capsize=5, capthick=1.6, zorder=4)
ax.errorbar([2], [0.664], yerr=[0.042], fmt="none", ecolor=GREEN, elinewidth=1.6,
            capsize=5, capthick=1.6, zorder=4)
ax.axhline(0.745, color=BLUE, ls="--", lw=1.2, alpha=0.55, zorder=1)
ax.axhline(0.59, color=GREEN, ls="--", lw=1.2, alpha=0.55, zorder=1)
ax.annotate("mAP@0.5 floor 0.745", (4.02, 0.7375), fontsize=11.5, color=INK2)
ax.annotate("DD floor 0.59 (provisional)", (1.45, 0.5715), fontsize=11.5, color=INK2)
ax.scatter([2], [0.766], s=430, facecolors="none", edgecolors=INK, linewidths=1.8, zorder=5)
# TRT engine point
ax.scatter([2], [0.768], marker="s", s=0)  # anchor only
ax.annotate("deployed TensorRT FP16 engine @768:\nmAP@0.5 0.768 / DD 0.732  (−0.009 export cost, 8.1 MB)",
            xy=(2, 0.766), xytext=(0.03, 0.97), textcoords="axes fraction",
            fontsize=12.5, va="top", color=INK,
            arrowprops=dict(arrowstyle="-", lw=1.1, color=GRAY, shrinkB=16))
for i, (m, d) in enumerate(zip(map_, dd)):
    ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points",
                xytext=(0, 13) if i != 3 else (10, 10), ha="center", fontsize=12, color=INK)
    ax.annotate(f"{d:.3f}", (i, d), textcoords="offset points",
                xytext=(0, -22) if i != 2 else (30, 4),
                ha="center" if i != 2 else "left", fontsize=12, color=INK)
ax.set_xticks(xi)
ax.set_xticklabels([f"{c}\n{f}" for c, f in zip(cfgs, fps)], fontsize=12)
ax.set_ylim(0.52, 0.90)
ax.set_ylabel("score on clean test split")
ax.legend(loc="lower right", frameon=False, fontsize=12.5)
ax.grid(axis="y")
ax.grid(False, axis="x")
ax.set_axisbelow(True)
fig.text(0.08, 0.02, "* fps measured on-device on a Jetson Orin Nano (PyTorch fp32, batch-1, "
         "end-to-end incl. pre/post; hardware_testing/jetson_fps_results.json) — real, not a "
         "projection. Not comparable to the original (Maxwell) Jetson Nano.",
         fontsize=11.5, color=INK2, ha="left")
title(fig, "768 px native retrain (circled) is the deployment point — FP16 export costs only −0.009 mAP@0.5",
      "Resolution ladder, YOLOv11n champ recipe on the clean benchmark · retraining natively at 768 beats downscaled 1280 inference at 2.8× less compute",
      x=0.06, y=0.895)
save(fig, "poster_fig_deploy")

# =====================================================================
# Fig 7 — blur robustness (eval_blur_robustness.json + eduardo tables)
# =====================================================================
fig, ax = plt.subplots(figsize=(10.5, 6.0))
fig.subplots_adjust(top=0.79, bottom=0.12, left=0.09, right=0.96)
xpts = [0, 1]
std_line = [0.793, 0.466]
blur_line = [0.790, 0.678]
ax.plot(xpts, std_line, "-o", color=BLUE, lw=2.4, ms=12,
        markeredgecolor="white", markeredgewidth=1.5, zorder=3,
        label="standard recipe")
ax.plot(xpts, blur_line, "-o", color=GREEN, lw=2.4, ms=12,
        markeredgecolor="white", markeredgewidth=1.5, zorder=3,
        label="+ blur augmentation")
ax.errorbar([1], [0.466], yerr=[0.050], fmt="none", ecolor=BLUE,
            elinewidth=1.7, capsize=5, capthick=1.7, zorder=4)
ax.errorbar([1], [0.678], yerr=[0.038], fmt="none", ecolor=GREEN,
            elinewidth=1.7, capsize=5, capthick=1.7, zorder=4)
# direct labels (ink text beside colored line ends)
ax.annotate("0.793", (0, 0.793), xytext=(-10, 10), textcoords="offset points",
            ha="right", fontsize=13, color=INK)
ax.annotate("0.790", (0, 0.790), xytext=(-10, -20), textcoords="offset points",
            ha="right", fontsize=13, color=INK)
ax.annotate("+ blur augmentation\n0.678 ± 0.038", (1, 0.678),
            xytext=(14, 0), textcoords="offset points", va="center",
            fontsize=13.5, color=INK, fontweight="bold")
ax.annotate("standard recipe\n0.466 ± 0.050", (1, 0.466),
            xytext=(14, 0), textcoords="offset points", va="center",
            fontsize=13.5, color=INK)
ax.annotate("", xy=(1.42, 0.666), xytext=(1.42, 0.478),
            arrowprops=dict(arrowstyle="->", lw=1.8, color=INK2))
ax.text(1.46, 0.572, "+0.21", fontsize=15, color=INK, fontweight="bold", va="center")
ax.text(0.5, 0.815, "clean accuracy: a tie (−0.003)", ha="center",
        fontsize=12.5, color=INK2, fontstyle="italic")
ax.set_xticks(xpts)
ax.set_xticklabels(["clean test images", "7 px motion-blurred test"], fontsize=13.5)
ax.set_xlim(-0.25, 1.65)
ax.set_ylim(0.38, 0.88)
ax.set_ylabel("mAP@0.5 (5-fold CV mean)")
ax.legend(loc="lower left", frameon=False, fontsize=12.5)
ax.grid(axis="y")
ax.grid(False, axis="x")
ax.set_axisbelow(True)
title(fig, "Blur-augmented training buys +0.21 mAP@0.5 under motion blur at zero clean cost",
      "Train-time MotionBlur (p=0.3) + GaussianBlur (p=0.2) · 974-image eduardo-merged 5-fold CV, OBB champ recipe · deployment-recipe candidate",
      x=0.06, y=0.89)
save(fig, "poster_fig_blur")

# =====================================================================
# Fig 8 — poster_fig_deploy_perclass: full per-class AP across the ladder
# Sources:
#   640 native / 768 retrain / 1280 native: results/evals/results_all_conditions.csv
#     rows Bnc_v11, HR768nc, HROaugnc_v11 (3 seeds each, clean no-CPLID test split).
#     NOTE: HR768nc here is the 3-seed eval (mAP 0.769 / DD 0.670); the paper's
#     4-seed numbers are 0.766 / 0.664 — within noise, flagged in the footnote.
#   768 infer / 1024 infer: figs_src/res_ladder_infer_lo.csv (extracted from
#     branch opt/orchestrator, results/optimization/res_ladder_infer_lo.csv),
#     1280-trained champion evaluated at lower input size, seeds 0-2;
#     columns ap50_c0..c6 are the 7 classes in alphabetical order (dd_ap50 ==
#     ap50_c2 confirms Defective Damper = c2).
# =====================================================================
import csv

CLASS_NAMES = ["Birdnest", "Broken Insulator", "Defective Damper",
               "Flashover Insulator", "Normal Damper", "Normal Insulators",
               "Self-Exploded Insulator"]

ladder = np.full((5, 8), np.nan)      # 5 configs x (7 classes + mAP)
ladder_sd = np.full((5, 8), np.nan)

# native rows: per-class means/stds from results_all_conditions.csv
ladder[0, :7] = [0.959, 0.6673, 0.6143, 0.5897, 0.7433, 0.8257, 0.7547]   # Bnc_v11
ladder_sd[0, :7] = [0.0104, 0.0235, 0.082, 0.011, 0.0235, 0.0154, 0.0159]
ladder[0, 7], ladder_sd[0, 7] = 0.7363, 0.0153
ladder[2, :7] = [0.979, 0.6587, 0.6703, 0.6227, 0.8093, 0.8387, 0.802]    # HR768nc
ladder_sd[2, :7] = [0.0131, 0.0349, 0.0488, 0.0249, 0.009, 0.0225, 0.0288]
ladder[2, 7], ladder_sd[2, 7] = 0.7687, 0.0091
ladder[4, :7] = [0.984, 0.6987, 0.6223, 0.66, 0.8133, 0.838, 0.8713]      # HROaugnc_v11
ladder_sd[4, :7] = [0.0044, 0.029, 0.0731, 0.0241, 0.0188, 0.0132, 0.0265]
ladder[4, 7], ladder_sd[4, 7] = 0.7837, 0.011

# infer rows: seed-average res_ladder_infer_lo.csv at imgsz 768 and 1024
rows_by_sz = {768: [], 1024: []}
with open(Path(__file__).resolve().parent / "res_ladder_infer_lo.csv") as f:
    for r in csv.DictReader(f):
        sz = int(r["imgsz"])
        if sz in rows_by_sz:
            rows_by_sz[sz].append(
                [float(r[f"ap50_c{i}"]) for i in range(7)] + [float(r["map50"])])
for cfg_i, sz in ((1, 768), (3, 1024)):
    arr = np.array(rows_by_sz[sz])
    assert arr.shape[0] == 3, f"expected 3 seeds at {sz}"
    ladder[cfg_i] = arr.mean(axis=0)
    ladder_sd[cfg_i] = arr.std(axis=0, ddof=1)

CFG_LABELS = ["640\nnative", "768 infer\n(1280-tr.)", "768\nretrain",
              "1024 infer\n(1280-tr.)", "1280\nnative"]
DEPLOY_I = 2  # 768 native retrain

# single shared-axis chart: 7 class lines + overall mAP, one axes.
# 7-color categorical palette validated with the dataviz-skill script
# (all four checks PASS on white; worst adjacent-pair CVD dE 16.2), plus
# distinct per-line markers and ink end-labels as secondary encoding.
CLASS_STYLE = {  # class index: (color, marker)
    0: ("#2166ac", "o"),   # Birdnest
    1: ("#d0700e", "s"),   # Broken Insulator
    2: ("#1a9850", "D"),   # Defective Damper
    3: ("#b2182b", "^"),   # Flashover Insulator
    4: ("#762a83", "v"),   # Normal Damper
    5: ("#0891b2", "P"),   # Normal Insulators
    6: ("#c51b7d", "X"),   # Self-Exploded Insulator
}
# Table rendering (poster_fig_deploy_table): per-class P / R / AP@0.5 per
# input-resolution configuration, seed-averaged (3 seeds). Bold cell = best AP
# per row; asterisk = selected deployment configuration. Source:
# figs_src/ladder_perclass.json (yolo val re-evals 2026-07-23, clean noCPLID
# test split; overall mAP means reproduce results_all_conditions.csv exactly).
import json as _json

LPC = _json.load(open(Path(__file__).resolve().parent / "ladder_perclass.json"))
LPC_CFGS = ["B640", "infer768", "retrain768", "infer1024", "native1280"]
LPC_ROWS = ["Birdnest", "Self-Exploded_Insulator", "Normal_Insulators",
            "Normal_Damper", "Broken_Insulator", "Flashover_Insulator",
            "Defective_Damper"]
TBL_COLS = ["640\nnative", "768 infer\n(1280-tr.)", "768\nretrain",
            "1024 infer\n(1280-tr.)", "1280\nnative"]
fig, ax = plt.subplots(figsize=(15.2, 7.6))
ax.axis("off")
name_x = 0.02
col_x = np.linspace(0.34, 0.955, 5)
y0, dy = 0.925, 0.094
ax.text(name_x, y0, "Class", ha="left", va="center", fontsize=16,
        fontweight="bold", color=INK, transform=ax.transAxes)
ax.text(name_x, y0 - 0.045, "(cells: P / R / AP@0.5)", ha="left", va="center",
        fontsize=11.5, color=MUTED, transform=ax.transAxes)
for j, c in enumerate(TBL_COLS):
    ax.text(col_x[j], y0, c, ha="center", va="center", fontsize=15,
            fontweight="bold", color=INK, transform=ax.transAxes,
            linespacing=1.15)
ax.plot([0.01, 0.99], [y0 - dy * 0.68] * 2, color=INK, lw=2.2,
        transform=ax.transAxes, clip_on=False)
for i, cls in enumerate(LPC_ROWS + ["OVERALL"]):
    y = y0 - (i + 1) * dy
    is_overall = cls == "OVERALL"
    if is_overall:
        ax.plot([0.01, 0.99], [y + dy * 0.5] * 2, color=INK, lw=1.6,
                transform=ax.transAxes, clip_on=False)
    rname = "OVERALL (mAP@0.5)" if is_overall else cls.replace("_", " ")
    ax.text(name_x, y, rname, ha="left", va="center", fontsize=14.5,
            fontweight="bold" if is_overall else "normal",
            color=INK, transform=ax.transAxes)
    aps = [LPC[c][cls][2] for c in LPC_CFGS]
    best_j = int(np.argmax(aps))
    for j, c in enumerate(LPC_CFGS):
        pr, rc, ap = LPC[c][cls]
        bold = j == best_j
        ax.text(col_x[j], y, f"{pr:.2f} / {rc:.2f} / {ap:.2f}",
                ha="center", va="center", fontsize=13.5,
                transform=ax.transAxes,
                fontweight="bold" if bold else "normal",
                color=INK if bold else INK2)
    if not is_overall and i < len(LPC_ROWS) - 1:
        ax.plot([0.01, 0.99], [y - dy * 0.5] * 2, color=GRID, lw=1.0,
                transform=ax.transAxes, clip_on=False)
fig.text(0.5, 0.96, "YOLOv11n per-class precision / recall / AP@0.5 "
         "by input resolution (clean benchmark, 3 seeds)",
         ha="center", va="bottom", fontsize=21, fontweight="bold", color=INK)
fig.text(0.02, 0.012,
         "All models trained with the champion recipe (3\u00d7 defective-damper "
         "oversampling + scale-0.9 augmentation, two-stage transfer learning), "
         "except 640 native = baseline recipe.\nBold: best AP per row.  "
         "Deployment point 768 retrain: retraining natively at 768 px recovers "
         "the rare-class (Defective Damper) AP that inference-time downscaling loses.",
         ha="left", va="bottom", fontsize=12, style="italic", color=INK2)
fig.subplots_adjust(top=0.88, bottom=0.13, left=0.01, right=0.99)
save(fig, "poster_fig_deploy_table")

# =====================================================================
# Fig 9 — poster_fig_pr: precision vs recall per model x condition
# Source: results/cv_proper_results.md "Summary Table (5-fold means)" —
#   P/R columns for base/champ/osall x v5n/v8n/v11n (same runs as
#   poster_fig_cv_main / paper Table tab:cv).
# =====================================================================
PR = {  # condition: {model: (precision, recall)}
    "base":  {"v5n": (0.816, 0.663), "v8n": (0.828, 0.670), "v11n": (0.793, 0.692)},
    "champ": {"v5n": (0.828, 0.700), "v8n": (0.809, 0.729), "v11n": (0.840, 0.745)},
    "osall": {"v5n": (0.852, 0.722), "v8n": (0.857, 0.736), "v11n": (0.835, 0.741)},
}
PR_OBB = {  # results/cv_obb_results.md (v8n/v11n only; separate annotation export)
    "base":  {"v8n": (0.810, 0.748), "v11n": (0.826, 0.734)},
    "champ": {"v8n": (0.835, 0.774), "v11n": (0.842, 0.777)},
    "osall": {"v8n": (0.835, 0.776), "v11n": (0.853, 0.786)},
}
COND_COLOR = {"base": BLUE, "champ": GREEN, "osall": ORANGE}
COND_LABEL = {"base": "base (640 px)",
              "champ": "champ (1280 px + 3× oversample + scale 0.9)",
              "osall": "osall (all-defect 3× oversample)"}
MODEL_MARK = {"v5n": "o", "v8n": "s", "v11n": "D"}

fig, ax = plt.subplots(figsize=(10.5, 8.0))
# base -> champ movement arrows per model (chrome, behind the marks)
for m in MODEL_MARK:
    p0, r0 = PR["base"][m]
    p1, r1 = PR["champ"][m]
    ax.annotate("", xy=(r1, p1), xytext=(r0, p0),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#b9b8b0",
                                shrinkA=10, shrinkB=11), zorder=1)
for cond, pts in PR.items():
    for m, (p, r) in pts.items():
        ax.scatter(r, p, s=200 if (cond == "champ" and m == "v11n") else 150,
                   marker=MODEL_MARK[m], color=COND_COLOR[cond],
                   edgecolors=SURF, linewidths=2, zorder=3)
for cond, pts in PR_OBB.items():
    for m, (p, r) in pts.items():
        ax.scatter(r, p, s=150, marker=MODEL_MARK[m], facecolors="none",
                   edgecolors=COND_COLOR[cond], linewidths=2.4, zorder=3)
ax.annotate("hollow = OBB (rotated boxes):\nhigher recall at equal-or-better precision",
            xy=(0.786, 0.853), xytext=(0.757, 0.869), fontsize=12, color=INK2,
            arrowprops=dict(arrowstyle="-", lw=0.9, color=GRAY, shrinkB=10))
# direct model labels (ink, not series color); offsets tuned per point
lab_off = {("base", "v5n"): (-8, 10), ("base", "v8n"): (8, 8), ("base", "v11n"): (0, -20),
           ("champ", "v5n"): (0, 12), ("champ", "v8n"): (0, -20), ("champ", "v11n"): (12, 10),
           ("osall", "v5n"): (-14, 10), ("osall", "v8n"): (10, 8), ("osall", "v11n"): (12, -14)}
for cond, pts in PR.items():
    for m, (p, r) in pts.items():
        dx, dy = lab_off[(cond, m)]
        ax.annotate(m, (r, p), textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=12.5, color=INK2)
ax.annotate("best (detection):\nchamp v11n  P 0.840 / R 0.745", xy=(0.745, 0.840),
            xytext=(0.7555, 0.8265), fontsize=12, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.9, color=GRAY, shrinkB=9))
ax.annotate("arrows: base to champ, per model (detection)", xy=(0.6495, 0.8695),
            fontsize=12, color=MUTED, style="italic")
handles = [plt.Line2D([], [], ls="", marker="o", ms=11, color=COND_COLOR[c],
                      label=COND_LABEL[c]) for c in ["base", "champ", "osall"]]
handles += [plt.Line2D([], [], ls="", marker=MODEL_MARK[m], ms=9,
                       markerfacecolor="#b9b8b0", markeredgecolor="#b9b8b0",
                       label=f"YOLO{m}") for m in MODEL_MARK]
handles += [plt.Line2D([], [], ls="", marker="o", ms=9, markerfacecolor="#b9b8b0",
                       markeredgecolor="#b9b8b0", label="detection (filled)"),
            plt.Line2D([], [], ls="", marker="o", ms=9, markerfacecolor="none",
                       markeredgecolor="#b9b8b0", markeredgewidth=2.2,
                       label="OBB (hollow)")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11.5, ncol=2)
ax.set_xlabel("Recall (all classes, 5-fold CV mean)")
ax.set_ylabel("Precision (all classes, 5-fold CV mean)")
ax.set_xlim(0.645, 0.812)
ax.set_ylim(0.775, 0.88)
ax.set_axisbelow(True)
title(fig, "The recipe buys recall without paying precision — and OBB buys more",
      "Precision vs recall, proper 5-fold CV (1,343-image merged pool) · recall rises for every model under champ/osall; precision holds or improves\n"
      "(v8n champ -0.02 is the only dip) · hollow OBB points use a separate annotation export (v8n/v11n only) — read within task",
      x=0.07, y=0.925)
fig.subplots_adjust(top=0.82, bottom=0.09, left=0.09, right=0.97)
save(fig, "poster_fig_pr")

# =====================================================================
# Fig 10 — poster_fig_model_table: YOLO nano model complexity
# Source: ultralytics "Model summary" lines from the eduardo-CV training logs
# (~/atli/logs_eduardo/EDU_{v5base,v8base,base_v11}_f0.log, unfused training
# graphs; OBB heads add ~0.07M params: v8n-obb 3.08M/8.4G, v11n-obb 2.66M/6.7G).
MODEL_ROWS = [  # (model, layers, params, gflops)
    ("YOLOv5n", "154", "2.51 M", "7.2"),
    ("YOLOv8n", "130", "3.01 M", "8.2"),
    ("YOLOv11n", "182", "2.59 M", "6.4"),
]
COLS = ["Model", "Layers", "Parameters", "FLOPs (G)"]
fig, ax = plt.subplots(figsize=(8.4, 2.35))
ax.axis("off")
col_x = [0.04, 0.40, 0.66, 0.96]
col_ha = ["left", "center", "center", "right"]
y0, dy = 0.86, 0.30
for j, c in enumerate(COLS):
    ax.text(col_x[j], y0, c, ha=col_ha[j], va="center", fontsize=17,
            fontweight="bold", color=INK, transform=ax.transAxes)
ax.plot([0.02, 0.98], [y0 - dy / 2] * 2, color=INK, lw=2.2,
        transform=ax.transAxes, clip_on=False)
for i, row in enumerate(MODEL_ROWS):
    y = y0 - (i + 1) * dy
    for j, v in enumerate(row):
        ax.text(col_x[j], y, v, ha=col_ha[j], va="center", fontsize=17,
                color=INK if j == 0 else INK2, transform=ax.transAxes,
                fontweight="bold" if j == 0 else "normal")
    if i < len(MODEL_ROWS) - 1:
        ax.plot([0.02, 0.98], [y - dy / 2] * 2, color=GRID, lw=1.2,
                transform=ax.transAxes, clip_on=False)
fig.text(0.5, 0.935, "YOLO nano variants: model complexity",
         ha="center", va="bottom", fontsize=19, fontweight="bold", color=INK)
fig.text(0.03, 0.02,
         "Layers/parameters/FLOPs from Ultralytics training summaries "
         "(detection heads, 640 px; OBB heads add ~0.07 M parameters).",
         ha="left", va="bottom", fontsize=11.5, style="italic", color=INK2)
fig.subplots_adjust(top=0.80, bottom=0.16, left=0.02, right=0.98)
save(fig, "poster_fig_model_table")

# =====================================================================
# Fig 11 — poster_fig_main_pr_map: baseline vs champion, P/R/mAP, 640 vs 1280
# Source: results/eval_eduardo_results.json (974-img group-aware 5-fold CV,
# per-fold all-class P / R / mAP50). Champion recipes per model: v11n = OBB
# champ+osall+deg15, v8n = OBB champ+osall, v5n = det champ+osall; baselines
# are detection @640. Baseline @1280 was not run.
import json

RES = json.load(open(Path(__file__).resolve().parents[3]
                     / "results/eval_eduardo_results.json"))

def cv_stats(cond):
    arr = np.array([[f["P"], f["R"], f["mAP50"]] for f in RES[cond]["folds"]])
    return arr.mean(axis=0), arr.std(axis=0, ddof=1)

PRM_MODELS = [  # (label, color, {condition: json key})
    ("YOLOv5n", BLUE, {"base": "v5base", "c640": "v5champ_640", "c1280": "v5champ"}),
    ("YOLOv8n", ORANGE, {"base": "v8base", "c640": "v8deg15_640", "c1280": "v8obbchamp"}),
    ("YOLOv11n", GREEN, {"base": "baseline", "c640": "deg15_640", "c1280": "deg15"}),
]
COND_ORDER = ["base", "c640", "c1280"]
COND_STYLE = {  # marker, filled?, size, legend label
    "base": ("o", False, 13, "baseline (640 px)"),
    "c640": ("o", True, 13, "champion @640 px"),
    "c1280": ("D", True, 14, "champion @1280 px"),
}
prm = {(m[0], c): cv_stats(m[2][c]) for m in PRM_MODELS for c in COND_ORDER}

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(16.5, 7.6), gridspec_kw={"width_ratios": [1.15, 1]})

# left panel: P vs R with per-model base -> champ640 -> champ1280 trajectories
rr = np.linspace(0.5, 0.95, 200)
for f1 in (0.70, 0.75, 0.80):
    with np.errstate(divide="ignore"):
        pp = f1 * rr / (2 * rr - f1)
    m = (pp > 0.72) & (pp < 0.90)
    axL.plot(rr[m], pp[m], color=GRID, lw=1.4, zorder=1)
    if m.any():
        axL.annotate(f"F1={f1:.2f}", (rr[m][-1], pp[m][-1]),
                     textcoords="offset points", xytext=(4, -2), fontsize=11,
                     color=MUTED, va="top")
for name, col, keys in PRM_MODELS:
    pts = [prm[(name, c)] for c in COND_ORDER]
    axL.plot([p[0][1] for p in pts], [p[0][0] for p in pts], "-", color=col,
             lw=2.0, alpha=0.45, zorder=2)
    for c, (mean, sd) in zip(COND_ORDER, pts):
        mk, filled, ms, _ = COND_STYLE[c]
        axL.errorbar(mean[1], mean[0], xerr=sd[1], yerr=sd[0], fmt="none",
                     ecolor=col, elinewidth=1.4, capsize=3, alpha=0.5, zorder=3)
        axL.plot(mean[1], mean[0], mk, color=col, ms=ms,
                 markerfacecolor=col if filled else SURF,
                 markeredgecolor=col, markeredgewidth=2.2, zorder=4)
    end = prm[(name, "c1280")][0]
    axL.annotate(name, (end[1], end[0]), textcoords="offset points",
                 xytext=(12, 8), fontsize=16, fontweight="bold", color=col)
axL.set_xlabel("Recall (all classes)", fontsize=18)
axL.set_ylabel("Precision (all classes)", fontsize=18)
axL.tick_params(labelsize=15)
axL.set_xlim(0.555, 0.79)
axL.set_ylim(0.715, 0.875)
axL.grid(False)
axL.set_axisbelow(True)
handles = [plt.Line2D([], [], marker=mk, color=INK2, lw=0, ms=ms - 2,
                      markerfacecolor=INK2 if filled else SURF,
                      markeredgecolor=INK2, markeredgewidth=2.0, label=lab)
           for mk, filled, ms, lab in COND_STYLE.values()]
axL.legend(handles=handles, loc="lower right", fontsize=14.5, frameon=False)

# right panel: mAP@0.5 dot plot, same encoding
yb = {name: i for i, (name, _, _) in enumerate(PRM_MODELS[::-1])}
DY = {"base": 0.24, "c640": 0.0, "c1280": -0.24}
for name, col, keys in PRM_MODELS:
    for c in COND_ORDER:
        (mean, sd), y = prm[(name, c)], yb[name] + DY[c]
        mk, filled, ms, _ = COND_STYLE[c]
        axR.errorbar(mean[2], y, xerr=sd[2], fmt="none", ecolor=col,
                     elinewidth=1.6, capsize=3.5, alpha=0.6, zorder=2)
        axR.plot(mean[2], y, mk, color=col, ms=ms,
                 markerfacecolor=col if filled else SURF,
                 markeredgecolor=col, markeredgewidth=2.2, zorder=3)
        axR.annotate(f"{mean[2]:.3f}", (mean[2] + sd[2] + 0.006, y),
                     fontsize=13, color=INK2, va="center")
axR.set_yticks(list(yb.values()))
axR.set_yticklabels(list(yb.keys()), fontsize=17, fontweight="bold")
axR.set_ylim(-0.6, 2.6)
axR.set_xlim(0.60, 0.86)
axR.set_xlabel("mAP@0.5", fontsize=18)
axR.tick_params(axis="x", labelsize=15)
axR.grid(axis="x")
axR.grid(False, axis="y")
axR.set_axisbelow(True)

fig.text(0.5, 0.955, "Baseline vs champion recipe: P / R / mAP@0.5 at 640 and 1280 px "
         "(5-fold CV, all three nano models)",
         ha="center", va="bottom", fontsize=21, fontweight="bold", color=INK)
fig.text(0.02, 0.008,
         "Champion recipes: v11n = OBB champ+osall+deg15, v8n = OBB champ+osall, "
         "v5n = det champ+osall; baselines are detection @640 px. "
         "Error bars: ±1 std over 5 folds. Baseline @1280 was not run.",
         ha="left", va="bottom", fontsize=12.5, style="italic", color=INK2)
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.07, right=0.98, wspace=0.22)
save(fig, "poster_fig_main_pr_map")

# =====================================================================
# Fig 12 — poster_fig_cv_ladder_table: per-class P/R/AP@0.5 under 5-fold CV
# by input resolution. Source: figs_src/cv_ladder_perclass.json (yolo val
# re-evals 2026-07-23 on the 974-img eduardo group-aware 5-fold CV; fold means;
# base640/champ640/infer640/champ1280 overall mAPs reproduce
# results/eval_eduardo_results.json exactly; infer768/1024 are new evals of the
# 1280-trained fold champions at lower input size).
CVL = _json.load(open(Path(__file__).resolve().parent
                      / "cv_ladder_perclass.json"))
CVL_CFGS = ["base640", "champ640", "infer640", "infer768", "infer1024",
            "champ1280"]
CVL_COLS = ["Baseline\n(640)", "Champion\n(640)", "Infer 640\n(1280-tr.)",
            "Infer 768\n(1280-tr.)", "Infer 1024\n(1280-tr.)",
            "Champion\n(1280)"]
CVL_ROWS = ["Birdnest", "Self-Exploded_Insulator", "Normal_Insulators",
            "Normal_Damper", "Broken_Insulator", "Flashover_Insulator",
            "Defective_Damper"]
fig, ax = plt.subplots(figsize=(16.8, 5.4))
ax.axis("off")
name_x = 0.015
col_x = np.linspace(0.315, 0.96, 6)
y0, dy = 0.93, 0.102
ax.text(name_x, y0, "Class", ha="left", va="center", fontsize=16,
        fontweight="bold", color=INK, transform=ax.transAxes)
ax.text(name_x, y0 - 0.045, "(cells: P / R / AP@0.5)", ha="left", va="center",
        fontsize=11.5, color=MUTED, transform=ax.transAxes)
for j, c in enumerate(CVL_COLS):
    ax.text(col_x[j], y0 + 0.012, c, ha="center", va="center", fontsize=14.5,
            fontweight="bold", color=INK, transform=ax.transAxes,
            linespacing=1.15)
ax.plot([0.005, 0.995], [y0 - dy * 0.68] * 2, color=INK, lw=2.2,
        transform=ax.transAxes, clip_on=False)
for i, cls in enumerate(CVL_ROWS + ["OVERALL"]):
    y = y0 - (i + 1) * dy
    is_overall = cls == "OVERALL"
    if is_overall:
        ax.plot([0.005, 0.995], [y + dy * 0.5] * 2, color=INK, lw=1.6,
                transform=ax.transAxes, clip_on=False)
    rname = "OVERALL (mAP@0.5)" if is_overall else cls.replace("_", " ")
    ax.text(name_x, y, rname, ha="left", va="center", fontsize=14,
            fontweight="bold" if is_overall else "normal",
            color=INK, transform=ax.transAxes)
    aps = [CVL[c][cls]["mean"][2] for c in CVL_CFGS]
    best_j = int(np.argmax(aps))
    for j, c in enumerate(CVL_CFGS):
        pr, rc, ap = CVL[c][cls]["mean"]
        bold = j == best_j
        ax.text(col_x[j], y, f"{pr:.2f} / {rc:.2f} / {ap:.2f}",
                ha="center", va="center", fontsize=13,
                transform=ax.transAxes,
                fontweight="bold" if bold else "normal",
                color=INK if bold else INK2)
    if not is_overall and i < len(CVL_ROWS) - 1:
        ax.plot([0.005, 0.995], [y - dy * 0.5] * 2, color=GRID, lw=1.0,
                transform=ax.transAxes, clip_on=False)
fig.text(0.5, 0.955, "YOLOv11n per-class precision / recall / AP@0.5 "
         "by input resolution, 5-fold cross-validation",
         ha="center", va="bottom", fontsize=24, fontweight="bold", color=INK)
fig.text(0.015, 0.045,
         "974-image ATLI+eduardos pool, fold means (test folds disjoint; 24-36 "
         "Defective Damper test instances per fold). Champion = OBB + all-defect "
         "3\u00d7 oversample + 15\u00b0 rotation; Baseline = detection, no "
         "oversampling.\nInfer columns evaluate the 1280-trained Champion at "
         "smaller input sizes. Bold: best AP per row.",
         ha="left", va="bottom", fontsize=11.5, style="italic", color=INK2)
fig.subplots_adjust(top=0.955, bottom=0.175, left=0.005, right=0.995)
save(fig, "poster_fig_cv_ladder_table")

# =====================================================================
# Fig 12b — poster_fig_resolution_frontier: the 1280-trained champion run at
# lower inference resolutions, accuracy (CV mAP@0.5, same CVL data as Fig 12)
# vs. real measured speed (hardware_testing/jetson_fps_results.json, v11n_obb,
# Jetson Orin Nano, PyTorch fp32, batch-1 — not projected). Two native-640
# reference points (same speed as the 640-inferred champion) show what
# training natively at 640 buys back vs. just downscaling 1280-trained
# inference.
JETSON_FPS = _json.load(open(Path(__file__).resolve().parents[3]
                             / "hardware_testing/jetson_fps_results.json"))["v11n_obb"]

def jfps(res):
    return JETSON_FPS[f"pt_{res}_fp32_conf0.25"]["fps"]

INFER_LADDER = [(640, "mixinfer640"), (768, "mixinfer768"),
                (1024, "mixinfer1024"), (1280, "mixchamp1280")]
REF_POINTS = [(640, "blurmix640", "640-trained deployment champion"),
              (640, "base640", "640 native baseline")]

fig, ax = plt.subplots(figsize=(5.8, 5.8))
fig.subplots_adjust(top=0.88, bottom=0.13, left=0.15, right=0.96)

lad_x = [jfps(r) for r, _ in INFER_LADDER]
lad_y = [CVL[c]["OVERALL"]["mean"][2] for _, c in INFER_LADDER]
ax.plot(lad_x, lad_y, "-o", color=GREEN, lw=2.4, ms=13, zorder=3,
        markeredgecolor="white", markeredgewidth=1.6,
        label="1280-trained champion, inferred at lower res")
LAD_LABEL_XY = {1280: ((12, 22), "left"), 640: ((-44, -26), "center")}
for (res, c), fx, fy in zip(INFER_LADDER, lad_x, lad_y):
    tag = f"{res} infer" if res != 1280 else "1280 native"
    xytext, ha = LAD_LABEL_XY.get(res, ((0, 16), "center"))
    ax.annotate(f"{tag}\n{fy:.3f} mAP\n{fx:.1f} fps", (fx, fy),
                xytext=xytext, textcoords="offset points", ha=ha,
                fontsize=10.5, color=INK, linespacing=1.25)

ref_colors = [BLUE, GRAY]
ref_dx = [0.45, 0.45]  # small jitter so the champ640 diamond doesn't sit on top of the green 640-infer point
ref_label_xy = [(14, -18), (14, 0)]
for (res, c, lab), col, dx, lxy in zip(REF_POINTS, ref_colors, ref_dx, ref_label_xy):
    fx, fy = jfps(res) + dx, CVL[c]["OVERALL"]["mean"][2]
    ax.scatter([fx], [fy], s=170, color=col, marker="D", zorder=4,
               edgecolors="white", linewidths=1.4, label=lab)
    ax.annotate(f"{fy:.3f}", (fx, fy), xytext=lxy, textcoords="offset points",
                va="center", fontsize=10.5, color=INK)

ax.set_xlabel("Jetson Orin Nano fps (measured)")
ax.set_ylabel("mAP@0.5 (5-fold CV mean)")
ax.set_xlim(9.6, 26.6)
ax.set_xticks(range(10, 27, 2))
ax.set_ylim(0.65, 0.855)
ax.legend(loc="lower left", frameon=False, fontsize=11)
ax.grid(axis="y")
ax.grid(False, axis="x")
ax.set_axisbelow(True)
fig.text(0.07, 0.945, "Downscaling Best YOLOv11n Model: mAP vs. Speed",
         ha="left", va="bottom", fontsize=17, fontweight="bold", color=INK)
save(fig, "poster_fig_resolution_frontier")

# =====================================================================
# Fig 13 — poster_fig_recipe_ladder: YOLO nano models across training recipes
# (poster-styled rebuild of results/figures/yolo_comparison, canonical
# condition names). Source: results/eval_eduardo_results.json fold means.
RES_ED = _json.load(open(Path(__file__).resolve().parents[3]
                         / "results/eval_eduardo_results.json"))
# YOLOv5n-OBB (community yolov5_obb fork — Ultralytics ships no v5-OBB): same
# folds/recipe as the v8/v11 "+ OBB (1280)" rung (champ+osall, 150+100, 1280)
# but scored by DOTA_devkit Task1 rotated eval, not Ultralytics OBBMetrics —
# same metric family, different evaluator (see results/yolov5_obb_results.md).
RES_V5OBB = _json.load(open(Path(__file__).resolve().parents[3]
                            / "results/eval_v5obb_results.json"))
RES_ED["v5obbchamp"] = {"folds": [{"mAP50": f["task1_mAP50"]}
                                  for f in RES_V5OBB["champion"]["folds"]]}

def ed_map(cond):
    a = np.array([f["mAP50"] for f in RES_ED[cond]["folds"]])
    return a.mean(), a.std(ddof=1)

RL_CONDS = [  # (legend label, color, {model: json key})
    ("Baseline (640)", "#2166ac",
     {"YOLOv5n": "v5base", "YOLOv8n": "v8base", "YOLOv11n": "baseline"}),
    ("Baseline + oversample x3 + 1280 res", "#d0700e",
     {"YOLOv5n": "v5champ", "YOLOv8n": "v8champ", "YOLOv11n": "champosall"}),
    ("+ OBB (1280)", "#1a9850",
     {"YOLOv5n": "v5obbchamp", "YOLOv8n": "v8obbchamp", "YOLOv11n": "obbref"}),
    ("+ deg15 rotation (1280)", "#c51b7d",
     {"YOLOv8n": "v8deg15", "YOLOv11n": "deg15"}),
    ("+ blur-aug (1280)", "#0891b2",
     {"YOLOv8n": "v8blurdeg15", "YOLOv11n": "blurdeg15"}),
    ("+ mixup (1280) \u2014 CV champion", "#762a83",
     {"YOLOv8n": "v8mix15_1280", "YOLOv11n": "mix15_1280"}),
]
RL_MODELS = ["YOLOv5n", "YOLOv8n", "YOLOv11n"]
n_cond = len(RL_CONDS)
step = 0.86 / n_cond
bw = step * 0.82

# YOLOv5n only has 3 of the 6 conditions (its OBB rung comes from the
# community yolov5_obb fork; the fork has no deg15/blur/mixup equivalents), so
# its group only reserves 3 bar-slots instead of 6 -- no dead space held open
# for the missing bars. Groups are packed left-to-right with a fixed
# inter-group gap instead of evenly-spaced fixed-width slots.
model_entries = {m: [(k, lab, col, keys[m]) for k, (lab, col, keys) in enumerate(RL_CONDS)
                      if m in keys] for m in RL_MODELS}
GROUP_GAP = 0.22
group_starts, group_centers = {}, {}
cursor = 0.0
for m in RL_MODELS:
    w = len(model_entries[m]) * step
    group_starts[m] = cursor
    group_centers[m] = cursor + w / 2
    cursor += w + GROUP_GAP

fig, ax = plt.subplots(figsize=(8.6, 6.3))
legend_done = set()
for m in RL_MODELS:
    for k_pos, (k, lab, col, key) in enumerate(model_entries[m]):
        mean, sd = ed_map(key)
        xp = group_starts[m] + (k_pos + 0.5) * step
        # v5n's OBB bar comes from a different fork + evaluator -- hatch it
        # and star its label so it isn't read as bit-identical to v8/v11.
        # It also never provides the legend entry: its hatch would land on
        # the "+ OBB (1280)" swatch and imply every OBB bar is fork-sourced
        # (v5n is drawn first, so it would win the label).
        fork_bar = key == "v5obbchamp"
        ax.bar(xp, mean, width=bw, color=col, zorder=3,
               hatch="//" if fork_bar else None,
               edgecolor="white" if fork_bar else None, linewidth=0,
               label=None if fork_bar or lab in legend_done else lab)
        if not fork_bar:
            legend_done.add(lab)
        ax.errorbar(xp, mean, yerr=sd, fmt="none", ecolor=INK2,
                    elinewidth=1.3, capsize=3.5, zorder=4)
        # Sit right above this bar's own error-bar cap (no cross-bar
        # cascading push-up: that staircase drifted later labels far from
        # their bars and clipped the tallest one off the top of the axes).
        # annotation_clip=False as a belt-and-braces guard against that
        # clipping recurring if a label ever lands right at the ylim edge.
        ly = mean + sd + 0.006
        ax.annotate(f"{mean:.3f}*" if fork_bar else f"{mean:.3f}", (xp, ly),
                    ha="center", fontsize=9.5, color=INK, zorder=5,
                    annotation_clip=False)
# Footnote for the fork-sourced v5n OBB bar (replaces the old "no v5 OBB
# variant" gap note now that the rung exists).
fig.text(0.5, 0.012,
         "* v5n-OBB via community yolov5_obb fork (Ultralytics has no v5-OBB); "
         "rotated Task1 mAP@0.5, DOTA_devkit eval — later rungs N/A for the fork.",
         ha="center", fontsize=8.5, color=MUTED, style="italic")
ax.set_xticks([group_centers[m] for m in RL_MODELS])
ax.set_xticklabels(RL_MODELS, fontsize=18)
ax.set_xlim(-0.12, cursor - GROUP_GAP + 0.12)
ax.set_ylim(0.55, 0.88)
ax.set_ylabel("mAP@0.5 (5-fold CV mean \u00b1 std)", fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax.grid(axis="y")
ax.grid(False, axis="x")
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=11.5, frameon=False,
          handlelength=1.6, labelspacing=0.35, borderaxespad=0.2)
fig.text(0.5, 0.945, "YOLO nano models across training recipes",
         ha="center", va="bottom", fontsize=24, fontweight="bold", color=INK)
fig.subplots_adjust(top=0.90, bottom=0.09, left=0.11, right=0.995)
save(fig, "poster_fig_recipe_ladder")

# =====================================================================
# Fig 14 — poster_fig_perclass_v11: per-class AP@0.5 across the LEVER
# progression that builds YOLOv11n's current best recipe (not a resolution
# ladder) — baseline, then each training lever in the order it was adopted,
# ending with the 640-px deployment drop. Source: results/
# eval_eduardo_results.json (974-img eduardo pool, 5-fold CV means). Same
# lever order as poster_fig_recipe_ladder / poster_fig_worst_to_best (1280
# panel), plus one extra final rung: dropping the finished 1280 recipe
# (mixup + blur-aug) down to native 640 training (blurmix_640, the actual
# deployment recipe) — a resolution change, not a new lever, marked with a
# dashed separator.
PC2_CFGS = ["baseline", "champosall", "obbref", "deg15", "blurdeg15",
            "mix15_1280", "blurmix_640"]
# Labels match the poster_fig_recipe_ladder legend text verbatim (wrapped for
# tick display) so the two graphs read as the same progression.
PC2_LABELS = ["Baseline\n(640)", "Baseline +\noversample x3 +\n1280 res",
              "+ OBB\n(1280)", "+ deg15\nrotation (1280)",
              "+ blur-aug\n(1280)", "+ mixup (1280)\n— CV champion",
              "640px deploy\n(blurmix_640)"]
PC2_CLASSES = ["Birdnest", "Broken_Insulator", "Defective_Damper",
               "Flashover_Insulator", "Normal_Damper", "Normal_Insulators",
               "Self-Exploded_Insulator"]


def ed_class_ap(cond, cls):
    a = np.array([f[f"{cls}_AP"] for f in RES_ED[cond]["folds"]])
    return float(a.mean())


xi2 = np.arange(len(PC2_CFGS))
fig, ax = plt.subplots(figsize=(9.6, 7.2))
for ci, cls in enumerate(PC2_CLASSES):
    col, mk = CLASS_STYLE[ci]
    ys = [ed_class_ap(c, cls) for c in PC2_CFGS]
    ax.plot(xi2, ys, "-", color=col, marker=mk, ms=8, lw=2.8,
            label=cls.replace("_", " "), zorder=3)
overall = [ed_map(c)[0] for c in PC2_CFGS]
ax.plot(xi2, overall, "-o", color=INK, ms=9, lw=3.8,
        label="Overall mAP@0.5", zorder=4)
ax.scatter([xi2[-1]], [overall[-1]], s=200, facecolors="none",
           edgecolors=INK, linewidths=1.6, zorder=5)
ax.axvline(5.5, color=MUTED, lw=1.0, ls="--", zorder=2)
ax.annotate("640px deploy", (5.6, 0.965), ha="left", fontsize=11,
            color=MUTED, style="italic")
ax.set_xticks(xi2)
ax.set_xticklabels(PC2_LABELS, fontsize=11)
ax.set_ylim(0.45, 1.0)
ax.set_xlim(-0.35, len(PC2_CFGS) - 1 + 0.35)
ax.set_xlabel("Recipe lever progression (YOLOv11n, 5-fold CV)", fontsize=14,
              labelpad=12)
ax.set_ylabel("mAP@0.5 (5-fold CV mean)", fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax.grid(axis="y")
ax.grid(False, axis="x")
ax.set_axisbelow(True)
ax.legend(loc="lower right", bbox_to_anchor=(0.80, 0.02), fontsize=11,
          frameon=True, framealpha=0.88, facecolor="white", edgecolor="none",
          ncol=2, handlelength=1.2, labelspacing=0.3, columnspacing=1.0,
          borderaxespad=0.6)
fig.text(0.5, 0.955,
         "Per-class mAP@0.5 across the recipe's lever progression (YOLOv11n)",
         ha="center", va="bottom", fontsize=19, fontweight="bold", color=INK)
fig.subplots_adjust(top=0.88, bottom=0.16, left=0.09, right=0.98)
save(fig, "poster_fig_perclass_v11")


# =====================================================================
# Fig 15 — poster_fig_worst_to_best: YOLOv11n training strategies, worst to
# best, split into a 1280-native panel (left) and a 640-native panel
# (right). Source: results/eval_eduardo_results.json fold means. Sequential
# grey-to-ink shading encodes rank (lightest = worst, ink = best) instead of
# reusing per-recipe hues, since the point here is pure ordering.
def _shade(t, lo="#f6d9d3", hi="#8b1a10"):  # poster red ramp (light rose -> deep crimson)
    lo_rgb = np.array([int(lo[i:i + 2], 16) for i in (1, 3, 5)])
    hi_rgb = np.array([int(hi[i:i + 2], 16) for i in (1, 3, 5)])
    c = (lo_rgb + t * (hi_rgb - lo_rgb)).astype(int)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


WB_1280 = [("+ oversample x3\n+ 1280 res", "champosall"), ("+ OBB\n(1280)", "obbref"),
           ("+ deg15\nrotation", "deg15"), ("+ blur-aug", "blurdeg15"),
           ("+ mixup\n(CV champion)", "mix15_1280")]
WB_640 = [("Baseline", "baseline"), ("+ deg15\nrotation", "deg15_640"),
          ("+ mixup", "mixup640"), ("+ mixup +\nblur-aug (deploy)", "blurmix_640")]

fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
for ax_, (title, panel) in zip(axes, [("1280 native", WB_1280), ("640 native", WB_640)]):
    n = len(panel)
    xs = np.arange(n)
    for i, (lab, key) in enumerate(panel):
        mean, sd = ed_map(key)
        col = _shade(i / (n - 1))
        ax_.bar(xs[i], mean, width=0.62, color=col, zorder=3)
        ax_.errorbar(xs[i], mean, yerr=sd, fmt="none", ecolor=INK2,
                     elinewidth=1.3, capsize=3.5, zorder=4)
        ax_.annotate(f"{mean:.3f}", (xs[i], mean + sd + 0.006), ha="center",
                     fontsize=12, color=INK, zorder=5)
    ax_.set_xticks(xs)
    ax_.set_xticklabels([lab for lab, _ in panel], fontsize=12)
    ax_.set_ylim(0.60, 0.85)
    ax_.set_title(title, fontsize=17, fontweight="bold", color=INK, pad=10)
    ax_.grid(axis="y")
    ax_.grid(False, axis="x")
    ax_.set_axisbelow(True)
    ax_.tick_params(axis="y", labelsize=13)
axes[0].set_ylabel("mAP@0.5 (5-fold CV mean \u00b1 std)", fontsize=15)
fig.text(0.5, 0.945, "Training strategies, worst to best (YOLOv11n)",
         ha="center", va="bottom", fontsize=24, fontweight="bold", color=INK)
fig.text(0.02, 0.008,
         "974-image eduardo pool, group-aware 5-fold CV means. Left = 1280-px native\n"
         "training; right = 640-px native training (current deployment resolution).\n"
         "Both ladders end the same way: rotation/scale augmentation, then mixup \u2014 at\n"
         "640 mixup also composes with blur-aug for robustness at no clean-accuracy cost.",
         ha="left", va="bottom", fontsize=11, style="italic", color=INK2, linespacing=1.4)
fig.subplots_adjust(top=0.88, bottom=0.24, left=0.07, right=0.98, wspace=0.15)
save(fig, "poster_fig_worst_to_best")

# =====================================================================
# Fig 16 — poster_fig_strategy_summary: every strategy tested, one table.
# Consolidates results already scattered across poster_fig_recipe_ladder /
# poster_fig_worst_to_best (the winning ladder), poster_fig_external (every
# external-data route, all failed), and poster_fig_ablation (recipe-lever
# ablations) into a single at-a-glance reference, plus items not charted
# anywhere else (backbone grafts, 640 recall levers). Numbers: results/
# eval_eduardo_results.json fold means; external-data and ablation rows are
# transcribed from poster_fig_external.png / poster_fig_ablation.png (same
# source data, paper/main.tex Tables tab:external / sec:ablate).
GREEN2 = "#1a9850"
RED2 = "#b2182b"
LEFT_COL = [
    ("header", "The winning ladder (adopted)"),
    ("row", "Baseline (det, 640px)", "0.668", ""),
    ("row", "+ oversample + 1280px", "0.749", ""),
    ("row", "+ OBB task mode", "0.759", ""),
    ("row", "+ deg15 rotation", "0.793", ""),
    ("row", "+ blur-aug", "0.790", "ties clean; +0.21 blur-test mAP@0.5"),
    ("row", "+ mixup — CV CHAMPION", "0.804", ""),
    ("row", "640px deploy (mixup+blur-aug)", "0.757", "current deployment recipe"),
    ("header", "Geometric augmentation — no gain beyond deg15"),
    ("row", "Rotation 25°", "0.781", "worse than deg15"),
    ("row", "Rotation 30°", "0.784", "worse than deg15"),
    ("row", "Shear", "0.779", "worse than deg15"),
    ("row", "deg15 + shear 10°", "0.788", "no gain over deg15 alone"),
    ("row", "CPLID images restored to train", "0.792", "no gain over deg15"),
    ("header", "Lighter backbones — all below stock v11n/v8n"),
    ("row", "Ghost graft (v11n / v8n)", "0.68 / 0.62", "vs stock 0.79 / 0.77"),
    ("row", "DWS graft (v11n / v8n)", "0.70 / 0.69", "vs stock 0.79 / 0.77"),
    ("row", "FasterNet/PConv graft (v11n / v8n)", "0.72 / 0.70", "vs stock 0.79 / 0.77"),
    ("row", "P2 stride-4 head (640px)", "0.682", "fresh-head init debt"),
]
RIGHT_COL = [
    ("header", "External data — every route failed"),
    ("row", "Community mixing (1:1–11.7:1)", "0.58–0.65", "DD AP; recall drop"),
    ("row", "Scale-matched curation", "0.63–0.65", "DD AP; recall drop"),
    ("row", "Pretrain-then-finetune (external)", "below baseline", "no run beat it"),
    ("row", "In-domain source pretraining", "−7.5 mAP@0.5", "genuine negative transfer"),
    ("row", "(control) in-distribution eval", "≈0.91", "DD AP; labels ARE learnable"),
    ("header", "Recall-focused levers @640 — none beat mixup alone (0.757)"),
    ("row", "×6 defect oversample + mixup", "0.747", "worse"),
    ("row", "cls-loss gain 1.0 + mixup", "0.748", "flat"),
    ("row", "combo (×6 + cls-gain) + mixup", "0.746", "worse"),
    ("header", "Training-schedule ablations (Δ vs champion)"),
    ("row", "Freeze 10 backbone layers", "−0.051", "hurts"),
    ("row", "Freeze 20 backbone layers", "−0.299", "collapses"),
    ("row", "300-epoch stage-1 (vs 150)", "−0.005", "overfits"),
    ("row", "Disable mosaic", "−0.030", "hurts"),
    ("row", "Remove scale-down aug", "−0.023", "hurts"),
    ("row", "Test-time augmentation (TTA)", "+0.002", "marginal, rejected"),
    ("row", "close_mosaic=20 @768", "+0.001", "DD +0.044, adopt-candidate"),
]


def draw_column(ax, entries, x0, x1):
    y = 0.975
    dy_header = 0.061
    dy_row = 0.0475
    name_x = x0 + 0.01
    val_x = x1 - 0.145
    note_x = x1 - 0.005
    for kind, *vals in entries:
        if kind == "header":
            ax.add_patch(plt.Rectangle((x0, y - 0.036), x1 - x0, 0.036,
                                        transform=ax.transAxes, color="#f3d0c9",
                                        zorder=1, clip_on=False))
            ax.text(name_x, y - 0.018, vals[0], ha="left", va="center",
                    fontsize=12.5, fontweight="bold", color=INK,
                    transform=ax.transAxes, zorder=2)
            y -= dy_header
        else:
            name, val, note = vals
            ax.text(name_x, y - 0.015, name, ha="left", va="center", fontsize=11,
                     color=INK, transform=ax.transAxes)
            ax.text(val_x, y - 0.015, val, ha="left", va="center", fontsize=11,
                     fontweight="bold", color=INK, transform=ax.transAxes)
            if note:
                ax.text(note_x, y - 0.015, note, ha="right", va="center",
                         fontsize=9, style="italic", color=INK2,
                         transform=ax.transAxes)
            ax.plot([x0, x1], [y - dy_row + 0.008] * 2, color=GRID, lw=0.8,
                     transform=ax.transAxes, clip_on=False)
            y -= dy_row


fig, ax = plt.subplots(figsize=(16, 10.6))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
draw_column(ax, LEFT_COL, 0.02, 0.49)
draw_column(ax, RIGHT_COL, 0.51, 0.98)
ax.plot([0.5, 0.5], [0.02, 0.97], color=GRID, lw=1.2, transform=ax.transAxes)
fig.text(0.5, 0.985, "Every strategy tested (YOLOv11n unless noted)",
         ha="center", va="top", fontsize=24, fontweight="bold", color=INK)
fig.subplots_adjust(top=0.945, bottom=0.02, left=0.01, right=0.99)
save(fig, "poster_fig_strategy_summary")

print("\nAll figures written to", OUT)
