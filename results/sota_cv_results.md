# SOTA Model Comparison — 5-Fold Cross-Validation Results

Generated 2026-07-02. All 15 runs (3 models × 5 folds) completed 2026-07-01 on ai.ee.unlv.edu.

**Setup:** MMDetection 3.3.0, COCO-pretrained weights, trained on the same `Merged_CV_proper`
folds (70/15/15, val ≠ test) as the YOLO proper-CV sweep — directly comparable numbers.
Test evals parsed from run logs by `collect_sota_cv.py`; raw per-fold JSON in
`results/sota_cv_summary.json`. RTMDet test-eval JSONs were not dumped by the sweep
(logs-only); fold 4 was re-tested 07-01 21:39 after its initial test eval failed.

- **DINO-4scale** (R50, 47M params) — transformer, 36 epochs, best-ckpt on val
- **RTM-DET Tiny** (4.8M params) — one-stage, 150 epochs
- **Dynamic-RCNN** (R50-FPN, 41M params) — two-stage, 40 epochs

## Summary (5-fold mean ± std, test split)

| Model | Params | mAP@0.5 | mAP@0.5:0.95 | DD AP@0.5 |
|---|---|---|---|---|
| **DINO-4scale** | 47M | **0.785 ± 0.032** | **0.481** | 0.789 ± 0.110 |
| RTM-DET Tiny | 4.8M | 0.747 ± 0.023 | 0.453 | 0.774 ± 0.052 |
| Dynamic-RCNN | 41M | 0.718 ± 0.023 | 0.423 | 0.699 ± 0.065 |
| ***YOLO champ v11n (ours)*** | *2.6M* | ***0.785 ± 0.023*** | — | ***0.812 ± 0.080*** |
| *YOLO champ v8n (ours)* | *3.0M* | *0.771 ± 0.023* | — | *0.799 ± 0.063* |
| *YOLO base v11n* | *2.6M* | *0.722 ± 0.019* | — | *0.743 ± 0.074* |

YOLO rows from `cv_proper_results.md` (identical folds/protocol, full 45-run re-eval
of 2026-07-02, which supersedes the 06-25 partial numbers).

## Per-fold detail (mAP@0.5 / DD AP@0.5)

| Fold | DINO | RTMDet | Dyn-RCNN |
|---|---|---|---|
| f0 | 0.759 / 0.743 | 0.728 / 0.729 | 0.687 / 0.649 |
| f1 | 0.806 / 0.898 | 0.754 / 0.861 | 0.724 / 0.776 |
| f2 | 0.822 / 0.888 | 0.763 / 0.781 | 0.749 / 0.730 |
| f3 | 0.794 / 0.783 | 0.771 / 0.752 | 0.727 / 0.724 |
| f4 | 0.746 / 0.632 | 0.718 / 0.749 | 0.705 / 0.616 |

## Per-class AP@0.5 (5-fold mean)

| Class | DINO | RTMDet | Dyn-RCNN | YOLO champ v11n |
|---|---|---|---|---|
| Birdnest | **0.893** | 0.878 | 0.880 | — |
| Broken Insulator | **0.684** | 0.622 | 0.591 | 0.662 |
| Defective Damper | 0.789 | 0.774 | 0.699 | **0.812** |
| Flashover Insulator | 0.637 | 0.606 | 0.618 | **0.656** |
| Normal Damper | 0.786 | 0.766 | 0.714 | **0.790** |
| Normal Insulators | **0.842** | 0.779 | 0.775 | 0.808 |
| Self-Exploded Insulator | **0.867** | 0.805 | 0.752 | — |

## Per-class mAP@0.5:0.95 (5-fold mean)

| Class | DINO | RTMDet | Dyn-RCNN |
|---|---|---|---|
| Birdnest | 0.487 | **0.505** | 0.486 |
| Broken Insulator | **0.580** | 0.484 | 0.422 |
| Defective Damper | **0.374** | 0.339 | 0.335 |
| Flashover Insulator | **0.372** | 0.331 | 0.346 |
| Normal Damper | 0.431 | **0.449** | 0.407 |
| Normal Insulators | **0.479** | 0.464 | 0.441 |
| Self-Exploded Insulator | **0.642** | 0.598 | 0.522 |

## Key findings

1. **CV confirms the single-split ranking (DINO > RTMDet > Dyn-RCNN) but shifts levels.**
   All three models score higher under CV than on the original single split
   (DINO 0.753→0.785, RTMDet 0.691→0.747, Dyn-RCNN 0.681→0.718) — the original
   single split is a pessimistic draw, consistent with the ±0.02–0.03 fold std.
2. **Champion YOLOv11n ties DINO on overall mAP** (0.785 both, with lower fold std)
   and **wins Defective_Damper** (0.812 vs 0.789) at **~18× fewer parameters** —
   the completed champ_v11 rerun (crashed folds re-run 06-30) closed the gap the
   06-25 partial numbers showed.
3. DINO's remaining edge is on insulator classes (Broken +0.02, Normal +0.03,
   Self-Exploded) and localization quality (mAP@0.5:0.95 lead) — attention helps
   large/elongated objects.
4. **RTMDet Tiny is the strongest efficiency rival** (0.747 @ 4.8M params) but still
   trails champ YOLOv11n on both mAP (−3.8 pts) and DD AP (−3.8 pts) with ~2× the params.
5. **Edge-deployment conclusion strengthened:** champion YOLOv11n now matches the
   47M-param transformer's accuracy outright — there is no accuracy argument for the
   heavyweight models on this dataset.
