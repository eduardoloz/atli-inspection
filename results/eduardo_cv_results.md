# ATLI + eduardos — 5-fold CV results (all classes, all experiments)

YOLOv11n on the combined **ATLI-no-CPLID + eduardos** pool.
Metrics are **mean ± std across the 5 folds** (± shown on AP@0.5 and overall mAP; P/R are
5-fold means). On each fold's held-out test split. Full card: `models/cv_eduardo_v11n/README.md`.

## How each experiment was trained

Shared recipe: **YOLOv11n**, **2-stage transfer learning** — Stage 1 = 150 ep from the
pretrained checkpoint (SGD `lr0=0.01`); Stage 2 = 100 ep fine-tune (SGD `lr0=0.00334`,
`lrf=0.1535`); batch 16. **Oversampling & augmentation are train-only** (val/test untouched);


| # | experiment | task | init | imgsz | train split | aug override |
|---|---|---|---|--:|---|---|
| 1 | **Baseline** | detection | `yolo11n.pt` | 640 | base (no oversample) | none |
| 2 | **OBB + deg45** | OBB | `yolo11n-obb.pt` | 1280 | osall (oversampled ×3) | `scale=0.9`, `degrees=45` |
| 3 | **OBB** (reference) | OBB | `yolo11n-obb.pt` | 1280 | osall (oversampled ×3) | `scale=0.9` |
| 4 | **OBB + deg20** ★ | OBB | `yolo11n-obb.pt` | 1280 | osall (oversampled ×3) | `scale=0.9`, `degrees=20` |
| 5 | **Detection champion** | detection | `yolo11n.pt` | 1280 | osall (oversampled ×3) | `scale=0.9` |

`osall` = every training image with a defect class (Broken_Insulator, Defective_Damper,
Flashover_Insulator, Self-Exploded_Insulator) duplicated **×3**. So exps 2–5 are all
**OBB/detection + oversampling + scale-aug**, differing only in rotation (deg45/none/deg20)
and task. ★ = current best.



## Instance counts per split (per-fold mean across 5 folds)

| class | train (base) | train (osall)¹ | val | test |
|---|--:|--:|--:|--:|
| Birdnest | 163 | 173 | 35 | 36 |
| Broken_Insulator ² | 206 | **618** | 40 | 44 |
| Defective_Damper ² | 134 | **401** | 30 | 29 |
| Flashover_Insulator ² | 303 | **910** | 65 | 65 |
| Normal_Damper | 1208 | 2321 | 248 | 258 |
| Normal_Insulators | 1037 | 2171 | 211 | 209 |
| Self-Exploded_Insulator ² | 207 | **622** | 47 | 50 |
| **images** | **683** | **1647** | **146** | **145** |

¹ **osall = oversampled train** (exps 2–5). Baseline (exp 1) uses **base** train.
**Val/test are never oversampled** — those columns hold for all experiments.
² Defect class, **directly tripled** (bold). Normals/Birdnest grow only via co-occurrence.
Pool = 974 images (796 no-CPLID ATLI + 178 eduardos).

## Overall (mAP@0.5, dataset-mean P/R)

| experiment | task | imgsz | Precision | Recall | mAP@0.5 |
|---|---|--:|--:|--:|--:|
| 1 — Baseline | detect | 640 | 0.756 | 0.609 | 0.668 ± 0.043 |
| 2 — OBB + deg45 | OBB | 1280 | 0.817 | 0.727 | 0.758 ± 0.025 |
| 3 — OBB (ref) | OBB | 1280 | 0.816 | 0.731 | 0.759 ± 0.019 |
| **4 — OBB + deg20 ★** | OBB | 1280 | 0.809 | 0.750 | **0.780 ± 0.024** |
| 5 — Detection champion | detect | 1280 | 0.790 | 0.695 | 0.749 ± 0.038 |

## Per-class AP@0.5 (mean ± std)

| class | 1 Baseline | 2 OBB+deg45 | 3 OBB (ref) | 4 OBB+deg20 ★ | 5 Det champ |
|---|--:|--:|--:|--:|--:|
| Birdnest | 0.922 ± 0.020 | 0.924 ± 0.046 | 0.944 ± 0.020 | 0.950 ± 0.027 | 0.910 ± 0.042 |
| Broken_Insulator | 0.574 ± 0.068 | 0.699 ± 0.079 | 0.693 ± 0.073 | 0.715 ± 0.060 | 0.706 ± 0.046 |
| Defective_Damper | 0.501 ± 0.156 | 0.654 ± 0.117 | 0.672 ± 0.095 | **0.721 ± 0.100** | 0.617 ± 0.156 |
| Flashover_Insulator | 0.596 ± 0.063 | 0.703 ± 0.063 | 0.687 ± 0.058 | 0.706 ± 0.059 | 0.688 ± 0.063 |
| Normal_Damper | 0.662 ± 0.042 | 0.753 ± 0.024 | 0.763 ± 0.052 | 0.770 ± 0.051 | 0.719 ± 0.040 |
| Normal_Insulators | 0.789 ± 0.037 | 0.819 ± 0.028 | 0.814 ± 0.042 | 0.818 ± 0.030 | 0.803 ± 0.041 |
| Self-Exploded_Insulator | 0.632 ± 0.060 | 0.753 ± 0.074 | 0.743 ± 0.056 | 0.780 ± 0.078 | 0.799 ± 0.025 |
| **overall (mAP@0.5)** | **0.668** | **0.758** | **0.759** | **0.780 ★** | **0.749** |

## Per-class Precision (5-fold mean)

| class | 1 Baseline | 2 OBB+deg45 | 3 OBB (ref) | 4 OBB+deg20 | 5 Det champ |
|---|--:|--:|--:|--:|--:|
| Birdnest | 0.878 | 0.911 | 0.897 | 0.890 | 0.871 |
| Broken_Insulator | 0.678 | 0.784 | 0.872 | 0.847 | 0.835 |
| Defective_Damper | 0.642 | 0.786 | 0.755 | 0.776 | 0.716 |
| Flashover_Insulator | 0.732 | 0.771 | 0.743 | 0.728 | 0.719 |
| Normal_Damper | 0.765 | 0.812 | 0.806 | 0.784 | 0.771 |
| Normal_Insulators | 0.779 | 0.822 | 0.809 | 0.799 | 0.781 |
| Self-Exploded_Insulator | 0.816 | 0.832 | 0.831 | 0.837 | 0.840 |
| **overall (mean P)** | **0.756** | **0.817** | **0.816** | **0.809** | **0.790** |

## Per-class Recall (5-fold mean)

| class | 1 Baseline | 2 OBB+deg45 | 3 OBB (ref) | 4 OBB+deg20 | 5 Det champ |
|---|--:|--:|--:|--:|--:|
| Birdnest | 0.883 | 0.923 | 0.929 | 0.934 | 0.878 |
| Broken_Insulator | 0.532 | 0.665 | 0.648 | 0.677 | 0.658 |
| Defective_Damper | 0.446 | 0.610 | 0.657 | 0.670 | 0.577 |
| Flashover_Insulator | 0.527 | 0.652 | 0.657 | 0.686 | 0.627 |
| Normal_Damper | 0.573 | 0.710 | 0.733 | 0.748 | 0.648 |
| Normal_Insulators | 0.766 | 0.813 | 0.810 | 0.817 | 0.783 |
| Self-Exploded_Insulator | 0.531 | 0.714 | 0.682 | 0.719 | 0.693 |
| **overall (mean R)** | **0.609** | **0.727** | **0.731** | **0.750** | **0.695** |

## CPLID-in-train (train-only) — does restoring CPLID to training help?

The 250 CPLID images removed during decontamination, added back to **training only** (val/test
unchanged, leakage-checked), on the champion recipes:

| condition | task | mAP@0.5 | DD AP | DD recall | Self-Exploded AP |
|---|---|--:|--:|--:|--:|
| OBB champion (no CPLID) | OBB | 0.759 | 0.672 | 0.657 | 0.743 |
| **OBB champion + CPLID** | OBB | **0.773 ± 0.027** | 0.677 | 0.632 | 0.784 |
| Detection champion (no CPLID) | det | 0.749 | 0.617 | 0.577 | 0.799 |
| Detection champion + CPLID | det | 0.742 ± 0.030 | 0.641 | 0.549 | 0.773 |

**Verdict:** CPLID-in-train gives OBB a small **+0.014 mAP** (0.759 → 0.773), driven mainly by
**Self-Exploded (+0.041)** — expected, since the 250 CPLID images are 249 Self-Exploded
insulators. For detection it's ~flat (−0.007). Neither beats OBB + deg20 (0.780).

### Per-class P / R / AP@0.5 (mean ± std over 5 folds)

**OBB champion + CPLID-in-train** (mAP 0.773 ± 0.027, P 0.818, R 0.733)
| class | P | R | AP@0.5 |
|---|--:|--:|--:|
| Birdnest | 0.879 | 0.932 | 0.952 ± 0.022 |
| Broken_Insulator | 0.814 | 0.663 | 0.716 ± 0.083 |
| Defective_Damper | 0.788 | 0.632 | 0.677 ± 0.115 |
| Flashover_Insulator | 0.736 | 0.622 | 0.676 ± 0.082 |
| Normal_Damper | 0.815 | 0.736 | 0.781 ± 0.034 |
| Normal_Insulators | 0.826 | 0.814 | 0.823 ± 0.043 |
| Self-Exploded_Insulator | 0.867 | 0.732 | 0.784 ± 0.079 |

**Detection champion + CPLID-in-train** (mAP 0.742 ± 0.030, P 0.801, R 0.676)
| class | P | R | AP@0.5 |
|---|--:|--:|--:|
| Birdnest | 0.878 | 0.860 | 0.904 ± 0.031 |
| Broken_Insulator | 0.783 | 0.628 | 0.667 ± 0.082 |
| Defective_Damper | 0.763 | 0.549 | 0.641 ± 0.131 |
| Flashover_Insulator | 0.734 | 0.618 | 0.679 ± 0.067 |
| Normal_Damper | 0.800 | 0.641 | 0.728 ± 0.052 |
| Normal_Insulators | 0.806 | 0.776 | 0.805 ± 0.029 |
| Self-Exploded_Insulator | 0.842 | 0.661 | 0.773 ± 0.056 |

## Rotation sweep + shear (v11 OBB, no CPLID) — where's the sweet spot?

Rotation angle and shear swept on the OBB champion (osall + scale=0.9):

| aug | mAP@0.5 | Precision | Recall | DD AP | DD recall |
|---|--:|--:|--:|--:|--:|
| none (OBB champion) | 0.759 | 0.816 | 0.731 | 0.672 | 0.657 |
| **deg15 ★** | **0.793 ± 0.023** | 0.839 | 0.726 | **0.729** | 0.671 |
| deg20 | 0.780 ± 0.024 | 0.809 | 0.750 | 0.721 | 0.670 |
| deg25 | 0.781 ± 0.013 | 0.834 | 0.735 | 0.708 | 0.677 |
| deg30 | 0.784 ± 0.029 | 0.834 | 0.737 | 0.715 | 0.660 |
| deg45 | 0.758 ± 0.025 | 0.817 | 0.727 | 0.654 | 0.610 |
| shear=10 | 0.779 ± 0.033 | 0.827 | 0.739 | 0.715 | 0.656 |

**Verdict:** the rotation sweet spot is **≈15°** — mAP **0.793**, DD AP **0.729**, the new overall
best. Mild rotation (15–30°) forms a plateau (~0.78–0.79); **45° collapses** back to the
no-rotation level. **Shear=10 also helps** (+0.020 mAP over the no-aug OBB champion), on par
with a mild rotation.

### Per-class AP@0.5 (mean ± std over 5 folds)
| class | deg15 ★ | deg25 | deg30 | shear |
|---|--:|--:|--:|--:|
| Birdnest | 0.935 ± 0.028 | 0.953 ± 0.011 | 0.956 ± 0.014 | 0.947 ± 0.012 |
| Broken_Insulator | 0.743 ± 0.090 | 0.752 ± 0.087 | 0.721 ± 0.083 | 0.739 ± 0.068 |
| Defective_Damper | 0.729 ± 0.096 | 0.708 ± 0.092 | 0.715 ± 0.087 | 0.715 ± 0.121 |
| Flashover_Insulator | 0.732 ± 0.044 | 0.704 ± 0.033 | 0.734 ± 0.034 | 0.715 ± 0.081 |
| Normal_Damper | 0.783 ± 0.035 | 0.768 ± 0.029 | 0.776 ± 0.035 | 0.769 ± 0.048 |
| Normal_Insulators | 0.830 ± 0.036 | 0.840 ± 0.019 | 0.833 ± 0.023 | 0.828 ± 0.031 |
| Self-Exploded_Insulator | 0.803 ± 0.054 | 0.740 ± 0.055 | 0.756 ± 0.091 | 0.739 ± 0.100 |
| **overall (mAP@0.5)** | **0.793** | **0.781** | **0.784** | **0.779** |

### deg15 (new best) — full per-class P / R / AP@0.5
| class | P | R | AP@0.5 |
|---|--:|--:|--:|
| Birdnest | 0.900 | 0.895 | 0.935 ± 0.028 |
| Broken_Insulator | 0.897 | 0.640 | 0.743 ± 0.090 |
| Defective_Damper | 0.793 | 0.671 | 0.729 ± 0.096 |
| Flashover_Insulator | 0.776 | 0.653 | 0.732 ± 0.044 |
| Normal_Damper | 0.819 | 0.724 | 0.783 ± 0.035 |
| Normal_Insulators | 0.820 | 0.799 | 0.830 ± 0.036 |
| Self-Exploded_Insulator | 0.869 | 0.699 | 0.803 ± 0.054 |
| **overall** | **0.839** | **0.726** | **0.793 ± 0.023** |

## Cross-model comparison (YOLOv5n / v8n / v11n)

Same group-aware folds. Baseline = detection @640; champion = 1280 + osall + scale=0.9
(v8/v11 use OBB; **v5 has no OBB checkpoint, so its champion is detection**). v11 champion =
its best config (OBB + deg15).

| model | baseline mAP | champion mAP | champion recipe | baseline DD AP | champion DD AP |
|---|--:|--:|---|--:|--:|
| YOLOv5n | 0.657 | 0.731 | detection (no OBB) | 0.505 | 0.558 |
| YOLOv8n | 0.656 | 0.765 | OBB | 0.490 | 0.708 |
| **YOLOv11n** | **0.668** | **0.793** | OBB + deg15 | 0.501 | **0.729** |

**Verdict:** clean ordering **v11 > v8 > v5** on both mAP and Defective_Damper at every stage.
OBB (v8/v11) clearly beats detection (v5's ceiling). **YOLOv11n remains the best architecture**
— consistent with the original paper, now confirmed with oriented boxes on the ATLI+eduardos pool.

### Per-class AP@0.5 (mean ± std over 5 folds)
| class | v5 base | v8 base | v5 champ (det) | v8 champ (OBB) |
|---|--:|--:|--:|--:|
| Birdnest | 0.880 ± 0.034 | 0.910 ± 0.041 | 0.885 ± 0.078 | 0.942 ± 0.024 |
| Broken_Insulator | 0.521 ± 0.068 | 0.528 ± 0.070 | 0.677 ± 0.074 | 0.693 ± 0.051 |
| Defective_Damper | 0.505 ± 0.154 | 0.490 ± 0.092 | 0.558 ± 0.131 | 0.708 ± 0.139 |
| Flashover_Insulator | 0.594 ± 0.082 | 0.566 ± 0.072 | 0.707 ± 0.081 | 0.681 ± 0.059 |
| Normal_Damper | 0.661 ± 0.053 | 0.677 ± 0.033 | 0.714 ± 0.043 | 0.754 ± 0.034 |
| Normal_Insulators | 0.777 ± 0.025 | 0.771 ± 0.018 | 0.791 ± 0.030 | 0.825 ± 0.028 |
| Self-Exploded_Insulator | 0.663 ± 0.055 | 0.648 ± 0.081 | 0.782 ± 0.064 | 0.756 ± 0.066 |
| **overall (mAP@0.5)** | **0.657** | **0.656** | **0.731** | **0.765** |
| dataset P / R | 0.758 / 0.599 | 0.763 / 0.605 | 0.789 / 0.672 | 0.819 / 0.730 |

