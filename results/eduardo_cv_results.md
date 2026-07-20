# ATLI + eduardos — 5-fold CV results (all classes, all experiments)

YOLOv11n on the combined **ATLI-no-CPLID + eduardos** pool.
Metrics are **mean ± std across the 5 folds** (± shown on AP@0.5 and overall mAP; P/R are
5-fold means). On each fold's held-out test split. Full card: `models/cv_eduardo_v11n/README.md`.

## How each experiment was trained

Shared recipe: **YOLOv11n**, **2-stage transfer learning** — Stage 1 = 150 ep from the
pretrained checkpoint (SGD `lr0=0.01`); Stage 2 = 100 ep fine-tune (SGD `lr0=0.00334`,
`lrf=0.1535`); batch 16. **Oversampling & augmentation are train-only** (val/test untouched);
group-aware folds keep every duplicate cluster within one split (no near-dup spans train/test).

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

