# cv_eduardo_v11n — ATLI(no-CPLID) + eduardos, group-aware 5-fold CV

YOLOv11n benchmark on the **combined ATLI-no-CPLID + eduardos-annotated-photos** pool,
4 conditions × 5 folds. Detection and OBB use the **same** group-aware folds so the tasks are
directly comparable. **Completed 2026-07-20.**

## Model

**YOLOv11n** (Ultralytics nano, ~2.6 M params — the project's edge-deployment champion).
- Detection conditions start from **`yolo11n.pt`** (COCO-pretrained).
- OBB conditions start from **`yolo11n-obb.pt`** (DOTAv1-pretrained).

## How training is done

**2-stage transfer learning**, per fold (`train/run_config_ext.sh` / `run_config_obb.sh`):
1. **Stage 1 (transfer):** 150 epochs from the pretrained checkpoint — SGD, `lr0=0.01`.
2. **Stage 2 (fine-tune):** 100 epochs from stage-1 `best.pt` — SGD, `lr0=0.00334`, `lrf=0.1535`.
3. **Evaluate** on the fold's held-out test split.

Batch 16. Online aug = Ultralytics defaults + the per-condition overrides below.
**Oversampling and augmentation touch the TRAIN split only**; val/test are original images.
All numbers are **mean ± std over the 5 folds**.

| # | condition | task | init | imgsz | train split | aug override |
|---|---|---|---|--:|---|---|
| 1 | Baseline | detect | `yolo11n.pt` | 640 | base (no oversample) | none |
| 2 | Champion + all-defect OS + deg45 | OBB | `yolo11n-obb.pt` | 1280 | **osall (oversampled)** | `scale=0.9`, `degrees=45` |
| 3 | Champion + all-defect OS (OBB ref) | OBB | `yolo11n-obb.pt` | 1280 | **osall (oversampled)** | `scale=0.9` |
| 4 | Champion + all-defect OS | detect | `yolo11n.pt` | 1280 | **osall (oversampled)** | `scale=0.9` |

## Dataset & split — instance counts per split

Pool = **974 images** (796 `atli_target-minus-the-cplid` v2 + 178 `eduardos-annotated-photos`
v6). Leakage-checked disjoint (0 cross-duplicates). **Group-aware 5-fold CV**, 70/15/15
multilabel-stratified, eduardos' 21 duplicate clusters kept whole within a fold (0 clusters
span a split). Counts are **per-fold means across the 5 folds**.

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

¹ **osall = oversampled train split** (used by conditions 2–4): every train image containing a
defect class is duplicated **×3**. Condition 1 (baseline) uses the **base** train column.
**Val and test are never oversampled** — those two columns apply to all four conditions.
² **Defect class, directly oversampled ×3** (bold = the tripled counts). Normal_Damper,
Normal_Insulators, and Birdnest grow in osall only because they co-occur on the duplicated
defect images.

Fold image ranges: train 673–691 · val 140–155 · test 141–148. **Defective_Damper test
instances/fold: 24–36 (mean 29)** — ≈2.4× the prior clean single-split's 12, so DD is measured
far more stably here.

## Annotation geometry (rectangle vs polygon)

Pooled instances by annotation type (polygon → oriented box under OBB; rectangle → axis-aligned):

| class | rectangle | polygon (OBB-capable) |
|---|--:|--:|
| Birdnest | 234 | 0 |
| Broken_Insulator | 290 | 0 |
| Defective_Damper | 179 | 14 |
| Flashover_Insulator | 434 | 0 |
| Normal_Damper | 1499 | 214 |
| Normal_Insulators | 1199 | 259 |
| Self-Exploded_Insulator | 305 | 0 |

Only Normal_Damper, Normal_Insulators, and 14 Defective_Damper carry real orientation; every
other defect box is axis-aligned (angle-0 under OBB).

## Results — Precision / Recall / AP@0.5 per class (mean ± std over 5 folds)

### Summary (overall mAP@0.5 + Defective_Damper)
| condition | task | mAP@0.5 | DD AP@0.5 | DD recall |
|---|---|--:|--:|--:|
| 1 — Baseline (640) | detect | 0.668 ± 0.043 | 0.501 ± 0.156 | 0.446 |
| 2 — Champ+OS+deg45 (1280) | OBB | 0.758 ± 0.025 | 0.654 ± 0.117 | 0.610 |
| **3 — Champ+OS ref (1280)** | OBB | **0.759 ± 0.019** | **0.672 ± 0.095** | **0.657** |
| 4 — Champ+OS (1280) | detect | 0.749 ± 0.038 | 0.617 ± 0.156 | 0.577 |

**Findings.** (1) All three champion conditions beat the baseline by **+0.08–0.09 mAP** — the
eduardos merge + 1280 + all-defect ×3 oversampling is a large, consistent win. (2) **OBB edges
detection**: the two OBB arms (0.758/0.759) top the detection champion (0.749). (3) **Best
Defective_Damper = OBB reference, AP 0.672 (+0.17 over baseline's 0.501), recall 0.657.**
(4) **deg45 did *not* help here** — the rotation arm (2) trails the no-rotation OBB reference (3)
on both DD AP (0.654 vs 0.672) and DD recall (0.610 vs 0.657), consistent with rotation being a
net-neutral-to-negative augmentation for these mostly axis-aligned defect boxes.

### Condition 1 — Baseline · detect · 640 · base train
| class | P | R | AP@0.5 |
|---|--:|--:|--:|
| Birdnest | 0.878 ± 0.054 | 0.883 ± 0.033 | 0.922 ± 0.020 |
| Broken_Insulator | 0.678 ± 0.059 | 0.532 ± 0.067 | 0.574 ± 0.068 |
| Defective_Damper | 0.642 ± 0.173 | 0.446 ± 0.122 | 0.501 ± 0.156 |
| Flashover_Insulator | 0.732 ± 0.063 | 0.527 ± 0.065 | 0.596 ± 0.063 |
| Normal_Damper | 0.765 ± 0.057 | 0.573 ± 0.029 | 0.662 ± 0.042 |
| Normal_Insulators | 0.779 ± 0.043 | 0.766 ± 0.016 | 0.789 ± 0.037 |
| Self-Exploded_Insulator | 0.816 ± 0.068 | 0.531 ± 0.070 | 0.632 ± 0.060 |
| **overall (mAP@0.5)** | 0.756 | 0.609 | **0.668 ± 0.043** |

### Condition 2 — Champion + all-defect OS + deg45 · OBB · 1280 · osall train
| class | P | R | AP@0.5 |
|---|--:|--:|--:|
| Birdnest | 0.911 ± 0.026 | 0.923 ± 0.032 | 0.924 ± 0.046 |
| Broken_Insulator | 0.784 ± 0.022 | 0.665 ± 0.063 | 0.699 ± 0.079 |
| Defective_Damper | 0.786 ± 0.107 | 0.610 ± 0.136 | 0.654 ± 0.117 |
| Flashover_Insulator | 0.771 ± 0.096 | 0.652 ± 0.049 | 0.703 ± 0.063 |
| Normal_Damper | 0.812 ± 0.035 | 0.710 ± 0.035 | 0.753 ± 0.024 |
| Normal_Insulators | 0.822 ± 0.039 | 0.813 ± 0.016 | 0.819 ± 0.028 |
| Self-Exploded_Insulator | 0.832 ± 0.069 | 0.714 ± 0.032 | 0.753 ± 0.074 |
| **overall (mAP@0.5)** | 0.817 | 0.727 | **0.758 ± 0.025** |

### Condition 3 — Champion + all-defect OS (OBB reference) · OBB · 1280 · osall train
| class | P | R | AP@0.5 |
|---|--:|--:|--:|
| Birdnest | 0.897 ± 0.040 | 0.929 ± 0.014 | 0.944 ± 0.020 |
| Broken_Insulator | 0.872 ± 0.066 | 0.648 ± 0.098 | 0.693 ± 0.073 |
| Defective_Damper | 0.755 ± 0.110 | 0.657 ± 0.090 | 0.672 ± 0.095 |
| Flashover_Insulator | 0.743 ± 0.057 | 0.657 ± 0.070 | 0.687 ± 0.058 |
| Normal_Damper | 0.806 ± 0.044 | 0.733 ± 0.049 | 0.763 ± 0.052 |
| Normal_Insulators | 0.809 ± 0.051 | 0.810 ± 0.028 | 0.814 ± 0.042 |
| Self-Exploded_Insulator | 0.831 ± 0.064 | 0.682 ± 0.060 | 0.743 ± 0.056 |
| **overall (mAP@0.5)** | 0.816 | 0.731 | **0.759 ± 0.019** |

### Condition 4 — Champion + all-defect OS · detect · 1280 · osall train
| class | P | R | AP@0.5 |
|---|--:|--:|--:|
| Birdnest | 0.871 ± 0.023 | 0.878 ± 0.043 | 0.910 ± 0.042 |
| Broken_Insulator | 0.835 ± 0.073 | 0.658 ± 0.084 | 0.706 ± 0.046 |
| Defective_Damper | 0.716 ± 0.143 | 0.577 ± 0.142 | 0.617 ± 0.156 |
| Flashover_Insulator | 0.719 ± 0.080 | 0.627 ± 0.085 | 0.688 ± 0.063 |
| Normal_Damper | 0.771 ± 0.070 | 0.648 ± 0.038 | 0.719 ± 0.040 |
| Normal_Insulators | 0.781 ± 0.030 | 0.783 ± 0.046 | 0.803 ± 0.041 |
| Self-Exploded_Insulator | 0.840 ± 0.055 | 0.693 ± 0.022 | 0.799 ± 0.025 |
| **overall (mAP@0.5)** | 0.790 | 0.695 | **0.749 ± 0.038** |

## Provenance
- Builder: `data/build_cv_eduardo.py` → `~/atli/CV_eduardo_det`, `~/atli/CV_eduardo_obb`.
- Sweep: `train/sweep_eduardo.sh` (GPUs 3–7, 4 conditions × 5 folds = 20 runs, 2026-07-20).
- Eval: `eval/eval_cv_eduardo.py` → per-fold P/R/AP → mean ± std (`~/atli/eval_eduardo_results.json`).
- Roboflow snapshots: `atli_target-minus-the-cplid` v2, `eduardos-annotated-photos` v6.
- Leakage: eduardos ↔ ATLI pHash cross-check = 0 duplicates; folds group-aware.
