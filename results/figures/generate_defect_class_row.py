"""Poster motivation row: one example crop per defective class with the class
name underneath. Crops in defect_row_assets/ come from ATLI_target_tightNI_noCPLID
(square crops around the largest labeled defect box, 2.2x margin, 600x600 px);
picker/cropper history in the repo conversation of 2026-07-28."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams["font.family"] = "Helvetica Neue"

OUT = Path(__file__).parent
ASSETS = OUT / "defect_row_assets"

PICKS = [
    ("Birdnest_0.png", "Birdnest"),
    ("Broken_Insulator_1.png", "Broken Insulator"),
    ("Defective_Damper_0.png", "Defective Damper"),
    ("Flashover_Insulator_0.png", "Flashover Insulator"),
    ("Self-Exploded_Insulator_0.png", "Self-Exploded Insulator"),
]

INK = "#0b0b0b"

fig, axes = plt.subplots(1, len(PICKS), figsize=(15, 3.6), dpi=200)
fig.patch.set_facecolor("white")

for ax, (fname, label) in zip(axes, PICKS):
    ax.imshow(mpimg.imread(ASSETS / fname))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#d5d4cd")
        s.set_linewidth(1.2)
    ax.set_xlabel(label, fontsize=15, fontweight="bold", color=INK, labelpad=8)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.16, wspace=0.06)
fig.savefig(OUT / "fig_defect_class_row.png", dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.08)
print("wrote", OUT / "fig_defect_class_row.png")
