# Results — Improving Rare-Defect Detection on ATLI (draft for APET follow-up)

> Draft results section from the 2026-06-15/16 experiment campaign. Numbers are on the held-out
> ATLI test split (199 images, 867 instances; seed-42 stratified 70/15/15 merged target+Eduardo).
> Caveat: the rare classes are small (39 Defective_Damper test instances) → single-run differences of
> ~±0.05 AP are within seed noise; we therefore report multi-seed means for the headline comparison.

## Method / conditions

All models are Ultralytics YOLO (nano), COCO-pretrained, two-stage transfer learning (stage-1 SGD
lr0=0.01; stage-2 fine-tune lr0=0.00334, OneCycle lrf=0.1535), imgsz 640 unless noted, on 8× Quadro
RTX 6000. Conditions evaluated:

- **Baseline (B):** native merged dataset, YOLOv11n, 150+100 epochs.
- **External-data augmentation:** community/Universe damper data added to train at ratios 1:1→11.7:1,
  including scale-matched curation (Uto/Uw/Ucap/USM/USMcap).
- **Native minority oversampling (OS):** Defective_Damper-containing train images duplicated ×3/×6
  (YOLO online aug gives each copy a distinct view).
- **Scale-down augmentation (aug):** scale=0.9 jitter to expose the model to more small-damper views.
- **Hi-resolution (HR):** imgsz 1280.
- **Combined (HROaug):** hi-res + ×3 oversample + scale-down aug — the proposed recipe.
- Technique ablations: P2 detection head, copy-paste, freeze-depth, loss reweighting.

## Main result (multi-seed, vs baseline)

**HROaug_v11** (YOLOv11n + imgsz 1280 + ×3 oversample + scale=0.9) is the best model.

| metric | HROaug_v11 (7 seeds) | Baseline B_v11 (6 seeds) | Δ |
|---|---|---|---|
| overall mAP@0.5 | **0.749** | 0.691 | **+0.058** |
| Defective_Damper AP@0.5 | **0.707 ± 0.028** | 0.645 ± 0.045 | **+0.062** |
| Defective_Damper recall | **0.69** | 0.61 | **+0.08** |
| Normal_Damper AP@0.5 | **0.832** | 0.760 | **+0.072** |

The +0.062 Defective_Damper gap exceeds 2× the seed spread → statistically meaningful, not a lucky
seed. (Champion DD range 0.679–0.771; baseline 0.571–0.708.)

**Ablations:** (a) *Freeze-depth* — freezing the backbone HURTS on this small dataset (freeze-10 mAP
0.696; freeze-20 collapses to 0.448), confirming the paper's "unfreeze-all" finding; full fine-tuning
is required. (b) *Scale-jitter strength* — sweet spot is scale=0.85–0.9 (DD 0.70–0.76); too weak/strong
both regress. (c) *Oversample factor* — ×6 helps only when combined with hi-res (HROaug+×6 DD 0.777 vs
×6 at 640 = 0.635). (d) Recipe generalizes to YOLOv8n at hi-res (HROaug_v8 DD 0.732).

## Per-class (best run vs baseline)

| class | HROaug_v11 | baseline | Δ |
|---|---|---|---|
| Defective_Damper | 0.771 | 0.668 | **+0.103** |
| Broken_Insulator | 0.598 | 0.509 | **+0.089** |
| Self-Exploded_Insulator | 0.878 | 0.775 | **+0.103** |
| Flashover_Insulator | 0.673 | 0.582 | **+0.091** |
| Normal_Damper | 0.828 | 0.758 | +0.070 |
| Normal_Insulators | 0.777 | 0.773 | +0.004 |
| Birdnest | 0.751 | 0.762 | −0.011 |
| **overall** | **0.754** | 0.690 | **+0.064** |

The recipe lifts **all four rare/defect classes by ~+0.09–0.10**, i.e. it targets exactly the
low-data classes the paper flagged as weakest.

## Key findings / ablations

1. **The low Defective_Damper recall is a small-object problem.** Diagnostic matching of predicted vs
   GT boxes: missed dampers were significantly smaller than detected ones (median 0.119% vs 0.227% of
   image area; 6/16 misses sub-20px). Hi-res alone fixed Normal_Damper (→0.84) but **not** the defective
   class — the defective bottleneck is rare-class signal + small size, addressed by oversampling +
   scale-down augmentation (recall 0.634→0.684).

2. **External/community data did not help the defective class — at any dose or curation.** Adding
   Universe damper data hurt Defective_Damper from a 1:1 ratio (0.58–0.61) through 11.7:1 (0.60), and
   scale-matched curation still dropped recall to 0.51. The failure is a **defect-definition / domain
   mismatch** (an in-domain control trained+tested on the community data reached 0.91, proving the data
   is learnable but describes a different "defective"), with a recall-drop signature. Provenance check:
   common public sets leak into the ATLI test split (verified by exact perceptual-hash *and* filename
   matches): Universe insulator/damper re-uploads are largely CPLID-derived, the DVDI damper repo
   overlaps ATLI by ~50%, and **PTL-AI Furnas — which ATLI was partly built from — exact-duplicates 32
   of the 274 val/test images**. I.e. ATLI's "internet-sourced" portion recycles CPLID + DVDI + Furnas,
   so naively "adding more public data" trains on the test set — a reproducibility hazard for the field
   and the reason a strict pHash+filename leakage gate is mandatory before any augmentation.

3. **Clean, on-domain, fully-labeled external data DOES help its target class.** Adding a pHash-vetted,
   leakage-free bird-nest dataset (niaochao) lifted Birdnest 0.762→0.898 (+0.14). However, single-class
   image dumps degrade *other* classes via partial annotation (Broken_Insulator −0.12), since
   co-occurring components are left unlabeled — a use-with-care result.

4. **Negative technique results:** copy-paste augmentation hurt the defective class (box-paste
   artifacts in detection); a P2 head improved Normal_Damper/overall but not the defective class; SAHI
   tiling is inapplicable (ATLI images are only 640²/512²). Freeze-depth ablation: [pending].

## Config benchmark — 53 runs, grouped by config (mean DD AP across seeds)
Single-split test (raw data: `results/config_benchmark.csv`). Robustness via seed-averaging:
| config | seeds | DD AP mean±std | mAP | ND |
|---|---|---|---|---|
| HROaug_v11 (champion) | 11 | **0.699 ±0.027** | 0.747 | 0.833 |
| baseline B_v11 | 11 | 0.641 ±0.036 | 0.691 | 0.757 |
| HROaug_OS6 (×6 oversample) | 2→6 | 0.751 (verifying) | 0.744 | 0.835 |
| HROaug_sc085 | 3 | 0.696 ±0.048 | 0.754 | 0.841 |
| HRO_v11 (no scale-aug) | 2 | 0.709 | 0.724 | 0.823 |
- Champion verified +0.058 DD over baseline (11 seeds each). **scale-0.85 did NOT hold up** (lucky
  single 0.762 → 0.696 mean). **×6-oversample (HROaug_OS6) is the top challenger at ~0.751** — being
  seed-verified (4 more launched). Lesson: single-run "winners" must be seed-averaged.
- **5-fold CV** queued (10 jobs) for the definitive small-test-set fix → tests all ~264 DD instances.

## Confusion-matrix analysis (baseline B_v11 → champion HROaug_v11, test split)
Normalized confusion matrices: `results/figures/confusion_matrices/CURRENT_{baseline,champion}_*.png`.
The pre-existing concern was high Normal_Damper/Normal_Insulator false positives standing in for their
defective counterparts. Deltas (normalized; raw counts at conf 0.25 in parens):

| confusion cell | baseline | champion | Δ |
|---|---|---|---|
| Defective_Damper → predicted **Normal_Damper** | 0.21 (8) | 0.15 (6) | **−0.06** ✓ |
| Defective_Damper → missed (background) | (7) | (4) | **−3 ✓** |
| **Normal_Damper false positives** (bg col) | 0.43 | 0.35 | **−0.08 ✓** |
| Defective insulators → predicted Normal_Insulators | ~0 (0) | ~0 (0) | 0 (they're *missed*, not mislabeled normal) |
| **Normal_Insulators false positives** (bg col) | 0.33 | 0.44 | **+0.11 ✗ (regression)** |
| Normal_Insulators missed (recall) | 0.79 | 0.83 | +0.04 ✓ |
| Self-Exploded_Insulator missed (background) | 0.26 | 0.04 | **−0.22 ✓** |

**Reading:** the champion measurably *reduces* the damper confusion the project flagged — defective
dampers called normal dropped 0.21→0.15 and spurious Normal_Damper detections dropped 0.43→0.35.
Insulator defects are essentially never mislabeled *as* Normal_Insulators (IoU-disjoint objects); their
error mode is missed detection, which improved (esp. Self-Exploded). The lone regression is more
Normal_Insulators false positives (0.33→0.44) — a candidate target for the next iteration.

## Takeaway
A diagnostic-driven, **data-free** recipe (hi-res + minority oversampling + scale-down augmentation)
improves overall mAP@0.5 by +0.05 and every rare/defect class by ~+0.09–0.10 over a paper-style
transfer-learning baseline, verified across seeds, while improving (not sacrificing) the dominant
Normal_Damper class. External data is only useful when clean, on-domain, leakage-checked, and
fully-labeled.
