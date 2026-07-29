"""Class-instance counts across the data pipeline: base ATLI pool -> merged
pool (+ additional/eduardo photos) -> train-split defect oversampling (x3).
Recreated (larger fonts, no heading) from the original one-off render at
results/figures/oversample_table.svg/png/pdf -- pipeline-stage counts are
unchanged, only chrome differs. No separate data source file; counts are
transcribed from that original render."""
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Helvetica Neue"

OUT = Path(__file__).parent

INK, MUTED, LIGHT, RED, PINK_BG = "#1d1d1f", "#52514e", "#b9b7b4", "#c8102e", "#fbeaec"


def btext(ax, x, y, s, *, fontsize, color, bold=False, **kw):
    """ax.text with a faux-bold stroke: this system's 'Helvetica Neue' is a
    single-face .ttc that matplotlib can't select a real bold weight from
    (fontweight='bold' silently renders as regular), so bold is faked by
    outlining the glyph in its own fill color."""
    effects = [pe.withStroke(linewidth=fontsize * 0.045, foreground=color)] if bold else None
    ax.text(x, y, s, fontsize=fontsize, color=color, path_effects=effects, **kw)

# (class, base ATLI pool, + additional photos (merged pool), training oversampling, is_defect)
ROWS = [
    ("Broken Insulator", 194, 290, 618, True),
    ("Defective Damper", 110, 193, 401, True),
    ("Flashover Insulator", 434, 434, 910, True),
    ("Self-Exploded Insulator", 304, 305, 622, True),
    ("Normal Damper", 1184, 1713, 2321, False),
    ("Normal Insulators", 856, 1458, 2171, False),
    ("Birdnest", 235, 234, 173, False),
]
COL_MAIN = ["base", "+ additional photos", "training oversampling"]
COL_SUB = ["(ATLI pool)", "(merged pool)", "(train split, ×3 defects)"]

fig, ax = plt.subplots(figsize=(10.6, 4.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

name_x = 0.015
col_x = [0.44, 0.68, 0.965]
y0, dy = 0.90, 0.125

btext(ax, name_x, y0 - 0.02, "Class", ha="left", va="center", fontsize=19,
      color=MUTED, bold=True, transform=ax.transAxes)
for j in range(3):
    btext(ax, col_x[j], y0 + 0.015, COL_MAIN[j], ha="right", va="center", fontsize=19,
          color=MUTED, bold=True, transform=ax.transAxes)
    ax.text(col_x[j], y0 - 0.05, COL_SUB[j], ha="right", va="center", fontsize=13.5,
            color=LIGHT, transform=ax.transAxes)

header_rule_y = y0 - dy * 0.62
ax.plot([0.015, 0.985], [header_rule_y] * 2, color=LIGHT, lw=1.3,
        transform=ax.transAxes, clip_on=False)

row_top = header_rule_y
for i, (cls, base, merged, os3, is_defect) in enumerate(ROWS):
    y_top = row_top - i * dy
    y = y_top - dy / 2
    if is_defect:
        ax.add_patch(plt.Rectangle((0.015, y_top - dy), 0.97, dy, transform=ax.transAxes,
                                    facecolor=PINK_BG, edgecolor="none", zorder=1))
    color = RED if is_defect else INK
    btext(ax, name_x, y, cls, ha="left", va="center", fontsize=15,
          color=color, bold=is_defect, transform=ax.transAxes, zorder=2)
    ax.text(col_x[0], y, f"{base:,}", ha="right", va="center", fontsize=15,
            color=INK, transform=ax.transAxes, zorder=2)
    ax.text(col_x[1], y, f"{merged:,}", ha="right", va="center", fontsize=15,
            color=INK, transform=ax.transAxes, zorder=2)
    btext(ax, col_x[2], y, f"{os3:,}", ha="right", va="center", fontsize=15,
          color=color, bold=is_defect, transform=ax.transAxes, zorder=2)

bottom_rule_y = row_top - len(ROWS) * dy
ax.plot([0.015, 0.985], [bottom_rule_y] * 2, color=LIGHT, lw=1.3,
        transform=ax.transAxes, clip_on=False)

fig.subplots_adjust(top=0.98, bottom=0.02, left=0.01, right=0.99)
for ext in ("svg", "png", "pdf"):
    fig.savefig(OUT / f"oversample_table.{ext}", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "oversample_table.{svg,png,pdf}")
