#!/usr/bin/env python3
"""Generate publication-quality figures for the PI writeup.
Run locally: python results/figures/generate_writeup_figures.py
Outputs PNGs to results/figures/writeup/
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUT = Path(__file__).parent / "writeup"
OUT.mkdir(exist_ok=True)

# ============================================================
# Color palette
# ============================================================
C_BASE = '#5B9BD5'    # blue
C_HROAUG = '#ED7D31'  # orange
C_OSALL = '#70AD47'   # green
C_OLD = '#A5A5A5'     # gray
C_NEW = '#4472C4'     # darker blue

# ============================================================
# Figure 1: Cross-model mAP comparison (grouped bar)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

models = ['YOLOv5n', 'YOLOv8n', 'YOLOv11n']
baseline = [75.5, 76.5, 76.0]
hroaug   = [79.3, 79.9, 80.2]
osall    = [79.8, 79.5, 80.6]

x = np.arange(len(models))
w = 0.25

b1 = ax.bar(x - w, baseline, w, label='Baseline (640px)', color=C_BASE, edgecolor='white', linewidth=0.5)
b2 = ax.bar(x,     hroaug,  w, label='HROaug (DD 3× OS)', color=C_HROAUG, edgecolor='white', linewidth=0.5)
b3 = ax.bar(x + w, osall,   w, label='OSall (All 3× OS)',  color=C_OSALL, edgecolor='white', linewidth=0.5)

ax.set_ylabel('mAP@0.5 (%)')
ax.set_title('5-Fold Cross-Validation: mAP@0.5 by Model and Training Recipe')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(72, 82.5)
ax.axhline(y=78.9, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Paper best (v5n, single split)')
ax.legend(loc='upper left')

# Value labels
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
fig.savefig(OUT / "fig1_crossmodel_map.png")
plt.close()
print("Fig 1: Cross-model mAP comparison")

# ============================================================
# Figure 2: Per-class AP improvement (baseline → OSall, v11)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

classes = ['Flashover\nInsulator', 'Self-Exploded\nInsulator', 'Normal\nDamper',
           'Defective\nDamper', 'Broken\nInsulator', 'Birdnest', 'Normal\nInsulators']
baseline_ap = [61.4, 85.5, 73.3, 72.7, 68.0, 80.1, 75.8]
osall_ap    = [72.4, 92.4, 78.1, 77.1, 71.8, 80.4, 77.3]  # using CV means

# Sort by improvement
delta = [o - b for o, b in zip(osall_ap, baseline_ap)]
order = np.argsort(delta)[::-1]
classes = [classes[i] for i in order]
baseline_ap = [baseline_ap[i] for i in order]
osall_ap = [osall_ap[i] for i in order]
delta = [delta[i] for i in order]

x = np.arange(len(classes))
w = 0.35

b1 = ax.barh(x + w/2, baseline_ap, w, label='Baseline', color=C_BASE, edgecolor='white')
b2 = ax.barh(x - w/2, osall_ap,    w, label='OSall (champion)', color=C_OSALL, edgecolor='white')

# Delta annotations
for i, (b, o, d) in enumerate(zip(baseline_ap, osall_ap, delta)):
    ax.annotate(f'+{d:.1f}%', xy=(o + 0.5, i - w/2), va='center', fontsize=9,
                fontweight='bold', color='#2E7D32')

ax.set_xlabel('AP@0.5 (%)')
ax.set_title('Per-Class AP: Baseline vs OSall Champion (YOLOv11n, 5-fold CV)')
ax.set_yticks(x)
ax.set_yticklabels(classes)
ax.set_xlim(55, 100)
ax.legend(loc='lower right')
ax.invert_yaxis()

plt.tight_layout()
fig.savefig(OUT / "fig2_perclass_improvement.png")
plt.close()
print("Fig 2: Per-class AP improvement")

# ============================================================
# Figure 3: Defect class recall comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

defect_classes = ['Broken\nInsulator', 'Defective\nDamper', 'Flashover\nInsulator', 'Self-Exploded\nInsulator']
recall_base   = [58.9, 68.0, 53.4, 78.6]
recall_hroaug = [60.0, 72.8, 64.3, 85.4]
recall_osall  = [63.0, 73.2, 67.9, 87.6]

x = np.arange(len(defect_classes))
w = 0.25

b1 = ax.bar(x - w, recall_base,   w, label='Baseline', color=C_BASE, edgecolor='white', linewidth=0.5)
b2 = ax.bar(x,     recall_hroaug, w, label='HROaug',   color=C_HROAUG, edgecolor='white', linewidth=0.5)
b3 = ax.bar(x + w, recall_osall,  w, label='OSall',    color=C_OSALL, edgecolor='white', linewidth=0.5)

ax.set_ylabel('Recall (%)')
ax.set_title('Defect Class Recall — YOLOv11n, 5-fold CV\n(higher = fewer missed defects)')
ax.set_xticks(x)
ax.set_xticklabels(defect_classes)
ax.set_ylim(40, 95)
ax.legend()

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "fig3_defect_recall.png")
plt.close()
print("Fig 3: Defect class recall")

# ============================================================
# Figure 4: Precision vs Recall (overall, by condition)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5.5))

# (P, R, label, color, marker)
points = [
    # Baselines
    (81.0, 71.0, 'v5n base', C_BASE, 'o'),
    (80.8, 73.0, 'v8n base', C_BASE, 's'),
    (82.3, 71.0, 'v11n base', C_BASE, '^'),
    # HROaug
    (81.7, 75.1, 'v5n HROaug', C_HROAUG, 'o'),
    (84.7, 74.1, 'v8n HROaug', C_HROAUG, 's'),
    (84.0, 75.5, 'v11n HROaug', C_HROAUG, '^'),
    # OSall
    (84.5, 75.6, 'v5n OSall', C_OSALL, 'o'),
    (86.0, 74.6, 'v8n OSall', C_OSALL, 's'),
    (84.7, 77.0, 'v11n OSall', C_OSALL, '^'),
]

for p, r, label, color, marker in points:
    ax.scatter(r, p, c=color, marker=marker, s=120, edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(label, (r, p), xytext=(5, 5), textcoords='offset points', fontsize=8)

# Draw arrows showing progression for v11n
ax.annotate('', xy=(75.5, 84.0), xytext=(71.0, 82.3),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
ax.annotate('', xy=(77.0, 84.7), xytext=(75.5, 84.0),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# Legend patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor=C_BASE, label='Baseline'),
    Patch(facecolor=C_HROAUG, label='HROaug'),
    Patch(facecolor=C_OSALL, label='OSall'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='v5n'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='v8n'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='v11n'),
]
ax.legend(handles=legend_elements, loc='lower left')

ax.set_xlabel('Recall (%)')
ax.set_ylabel('Precision (%)')
ax.set_title('Precision vs Recall — 5-fold CV, All Models × Conditions')
ax.set_xlim(69, 79)
ax.set_ylim(79, 88)

plt.tight_layout()
fig.savefig(OUT / "fig4_precision_recall.png")
plt.close()
print("Fig 4: Precision vs Recall scatter")

# ============================================================
# Figure 5: Annotation tightening A/B (insulator classes only)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

ins_classes = ['Normal\nInsulators', 'Broken\nInsulator', 'Flashover\nInsulator', 'Self-Exploded\nInsulator']

# AP
old_ap  = [76.9, 63.9, 65.1, 87.9]
new_ap  = [80.3, 70.1, 74.8, 90.4]
# Recall
old_r   = [73.9, 49.8, 58.8, 78.7]
new_r   = [79.2, 61.6, 71.9, 86.2]
# Precision
old_p   = [76.5, 85.8, 80.0, 96.5]
new_p   = [79.4, 76.5, 82.4, 93.7]

datasets = [
    (old_ap, new_ap, 'AP@0.5 (%)', 'AP@0.5'),
    (old_r, new_r, 'Recall (%)', 'Recall'),
    (old_p, new_p, 'Precision (%)', 'Precision'),
]

x = np.arange(len(ins_classes))
w = 0.35

for ax_i, (old_vals, new_vals, ylabel, title) in zip(axes, datasets):
    b1 = ax_i.bar(x - w/2, old_vals, w, label='Original (v4)', color=C_OLD, edgecolor='white')
    b2 = ax_i.bar(x + w/2, new_vals, w, label='Tightened (v5)', color=C_NEW, edgecolor='white')

    for i, (o, n) in enumerate(zip(old_vals, new_vals)):
        d = n - o
        sign = '+' if d >= 0 else ''
        color = '#2E7D32' if d >= 0 else '#C62828'
        ax_i.annotate(f'{sign}{d:.1f}', xy=(x[i] + w/2, n + 0.5), ha='center',
                      fontsize=8, fontweight='bold', color=color)

    ax_i.set_ylabel(ylabel if ax_i == axes[0] else '')
    ax_i.set_title(title)
    ax_i.set_xticks(x)
    ax_i.set_xticklabels(ins_classes, fontsize=9)
    ax_i.set_ylim(40, 102)
    if ax_i == axes[0]:
        ax_i.legend(loc='upper left', fontsize=9)

fig.suptitle('Annotation Tightening A/B: Insulator Classes (OSall v11, matched seeds)', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig5_annotation_ab.png")
plt.close()
print("Fig 5: Annotation A/B comparison")

# ============================================================
# Figure 6: Recipe component breakdown (waterfall-style)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

steps = ['Paper\nbaseline\n(reported)', 'Our\nbaseline\n(CV)', '+ Hi-res\n1280px', '+ DD 3×\noversample', '+ Scale\naug 0.9', '+ All-class\noversample']
values = [78.9, 76.0, None, None, 80.2, 80.6]
# Approximate breakdown: baseline 76.0, HROaug components together = 80.2, OSall = 80.6
# We don't have individual component isolation, so show baseline → HROaug → OSall
bar_vals = [78.9, 76.0, 80.2, 80.2, 80.2, 80.6]
colors = ['#FF6B6B', C_BASE, C_HROAUG, C_HROAUG, C_HROAUG, C_OSALL]

# Simpler: show the progression
labels = ['Paper\n(single split)', 'Baseline\n(5-fold CV)', 'HROaug\n(DD oversample)', 'OSall\n(all-class OS)']
vals = [78.9, 76.0, 80.2, 80.6]
bar_colors = ['#FF6B6B', C_BASE, C_HROAUG, C_OSALL]

bars = ax.bar(labels, vals, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.6)

for bar, v in zip(bars, vals):
    ax.annotate(f'{v:.1f}%', xy=(bar.get_x() + bar.get_width()/2, v),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

# Delta annotations
ax.annotate('', xy=(2, 80.2), xytext=(1, 76.0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('+4.2%', xy=(1.5, 78.3), ha='center', fontsize=10, fontweight='bold', color='#2E7D32')

ax.annotate('', xy=(3, 80.6), xytext=(2, 80.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('+0.4%', xy=(2.5, 80.6), ha='center', fontsize=10, fontweight='bold', color='#2E7D32')

ax.set_ylabel('mAP@0.5 (%)')
ax.set_title('Progression of Improvements — YOLOv11n')
ax.set_ylim(73, 83)
ax.axhline(y=78.9, color='red', linestyle=':', alpha=0.4)

plt.tight_layout()
fig.savefig(OUT / "fig6_progression.png")
plt.close()
print("Fig 6: Improvement progression")

print(f"\nAll figures saved to {OUT}/")
