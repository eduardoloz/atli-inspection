# ATLI + eduardos — 5-fold CV results (all classes, all experiments)

YOLOv11n on the combined **ATLI-no-CPLID + eduardos** pool, group-aware 5-fold CV.
All metrics are **mean ± std across the 5 folds**, on each fold's held-out test split.
Full model card: `models/cv_eduardo_v11n/README.md`. Raw: `results/eval_eduardo_results.json`.

## How each experiment was trained

Shared recipe (every run): **YOLOv11n**, **2-stage transfer learning** —
Stage 1 = 150 epochs from the pretrained checkpoint (SGD, `lr0=0.01`); Stage 2 = 100 epochs
fine-tune from stage-1 `best.pt` (SGD, `lr0=0.00334`, `lrf=0.1535`); batch 16.
**Oversampling and augmentation are applied to the training split only** — val/test are always
the original images. Group-aware 5-fold CV (70/15/15), eduardos' 21 duplicate clusters kept
whole within a fold (0 clusters span a split).

| # | experiment | task | init checkpoint | imgsz | train split | aug override |
|---|---|---|---|--:|---|---|
| 1 | **Baseline** | detection | `yolo11n.pt` (COCO) | 640 | base (no oversample) | none |
| 2 | **OBB + deg45** | OBB | `yolo11n-obb.pt` (DOTA) | 1280 | osall (oversampled) | `scale=0.9`, `degrees=45` |
| 3 | **OBB** (reference) | OBB | `yolo11n-obb.pt` (DOTA) | 1280 | osall (oversampled) | `scale=0.9` |
| 4 | **Detection champion** | detection | `yolo11n.pt` (COCO) | 1280 | osall (oversampled) | `scale=0.9` |

`osall` = every training image containing a defect class (Broken_Insulator, Defective_Damper,
Flashover_Insulator, Self-Exploded_Insulator) is duplicated **×3**.

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

¹ **osall = oversampled train** (experiments 2–4). Baseline (exp 1) uses the **base** train column.
**Val/test are never oversampled** — those two columns hold for all four experiments.
² Defect class, **directly tripled** (bold). Normals/Birdnest grow in osall only via co-occurrence
on the duplicated defect images. Pool total = 974 images (796 no-CPLID ATLI + 178 eduardos).

## Overall (mAP@0.5, dataset-mean P/R)

| experiment | task | imgsz | Precision | Recall | mAP@0.5 |
|---|---|--:|--:|--:|--:|
| 1 — Baseline | detect | 640 | 0.756 | 0.609 | 0.668 ± 0.043 |
| 2 — OBB + deg45 | OBB | 1280 | 0.817 | 0.727 | 0.758 ± 0.025 |
| **3 — OBB (ref)** | OBB | 1280 | 0.816 | 0.731 | **0.759 ± 0.019** |
| 4 — Detection champion | detect | 1280 | 0.790 | 0.695 | 0.749 ± 0.038 |

## Per-class AP@0.5

| class | 1 Baseline | 2 OBB+deg45 | 3 OBB (ref) | 4 Det champ |
|---|--:|--:|--:|--:|
| Birdnest | 0.922 ± 0.020 | 0.924 ± 0.046 | 0.944 ± 0.020 | 0.910 ± 0.042 |
| Broken_Insulator | 0.574 ± 0.068 | 0.699 ± 0.079 | 0.693 ± 0.073 | 0.706 ± 0.046 |
| Defective_Damper | 0.501 ± 0.156 | 0.654 ± 0.117 | **0.672 ± 0.095** | 0.617 ± 0.156 |
| Flashover_Insulator | 0.596 ± 0.063 | 0.703 ± 0.063 | 0.687 ± 0.058 | 0.688 ± 0.063 |
| Normal_Damper | 0.662 ± 0.042 | 0.753 ± 0.024 | 0.763 ± 0.052 | 0.719 ± 0.040 |
| Normal_Insulators | 0.789 ± 0.037 | 0.819 ± 0.028 | 0.814 ± 0.042 | 0.803 ± 0.041 |
| Self-Exploded_Insulator | 0.632 ± 0.060 | 0.753 ± 0.074 | 0.743 ± 0.056 | 0.799 ± 0.025 |
| **overall (mAP@0.5)** | **0.668** | **0.758** | **0.759** | **0.749** |

## Per-class Precision

| class | 1 Baseline | 2 OBB+deg45 | 3 OBB (ref) | 4 Det champ |
|---|--:|--:|--:|--:|
| Birdnest | 0.878 ± 0.054 | 0.911 ± 0.026 | 0.897 ± 0.040 | 0.871 ± 0.023 |
| Broken_Insulator | 0.678 ± 0.059 | 0.784 ± 0.022 | 0.872 ± 0.066 | 0.835 ± 0.073 |
| Defective_Damper | 0.642 ± 0.173 | 0.786 ± 0.107 | 0.755 ± 0.110 | 0.716 ± 0.143 |
| Flashover_Insulator | 0.732 ± 0.063 | 0.771 ± 0.096 | 0.743 ± 0.057 | 0.719 ± 0.080 |
| Normal_Damper | 0.765 ± 0.057 | 0.812 ± 0.035 | 0.806 ± 0.044 | 0.771 ± 0.070 |
| Normal_Insulators | 0.779 ± 0.043 | 0.822 ± 0.039 | 0.809 ± 0.051 | 0.781 ± 0.030 |
| Self-Exploded_Insulator | 0.816 ± 0.068 | 0.832 ± 0.069 | 0.831 ± 0.064 | 0.840 ± 0.055 |
| **overall (mean P)** | **0.756** | **0.817** | **0.816** | **0.790** |

## Per-class Recall

| class | 1 Baseline | 2 OBB+deg45 | 3 OBB (ref) | 4 Det champ |
|---|--:|--:|--:|--:|
| Birdnest | 0.883 ± 0.033 | 0.923 ± 0.032 | 0.929 ± 0.014 | 0.878 ± 0.043 |
| Broken_Insulator | 0.532 ± 0.067 | 0.665 ± 0.063 | 0.648 ± 0.098 | 0.658 ± 0.084 |
| Defective_Damper | 0.446 ± 0.122 | 0.610 ± 0.136 | **0.657 ± 0.090** | 0.577 ± 0.142 |
| Flashover_Insulator | 0.527 ± 0.065 | 0.652 ± 0.049 | 0.657 ± 0.070 | 0.627 ± 0.085 |
| Normal_Damper | 0.573 ± 0.029 | 0.710 ± 0.035 | 0.733 ± 0.049 | 0.648 ± 0.038 |
| Normal_Insulators | 0.766 ± 0.016 | 0.813 ± 0.016 | 0.810 ± 0.028 | 0.783 ± 0.046 |
| Self-Exploded_Insulator | 0.531 ± 0.070 | 0.714 ± 0.032 | 0.682 ± 0.060 | 0.693 ± 0.022 |
| **overall (mean R)** | **0.609** | **0.727** | **0.731** | **0.695** |

## Takeaways
- All three 1280 + oversampled champions beat the baseline by **+0.08–0.09 mAP**; the biggest
  per-class jump is **Defective_Damper (0.501 → up to 0.672 AP, +0.17)** and Self-Exploded.
- **OBB (exp 3) is the best model** overall (mAP 0.759) and on Defective_Damper (AP 0.672,
  recall 0.657); OBB edges the detection champion (0.759 vs 0.749).
- **`degrees=45` did not help** — exp 2 (with rotation) trails exp 3 (without) on DD AP and
  recall, consistent with rotation being unhelpful for these mostly axis-aligned defect boxes.
- Defective_Damper is now scored on **24–36 test instances/fold** (vs 12 in the old single
  split), so these numbers are substantially more stable — though DD still carries the widest
  fold-to-fold std, as expected for the rarest class.
- New pool = fresh benchmark (eduardos in val/test); **not** directly comparable to the prior
  clean-split 0.784 / 0.802 numbers.
