# OBB 5-Fold CV Results — FINAL (all folds complete)

Updated 2026-07-02. Full re-eval via `eval/eval_cv_obb.py` after the base_v11 fold
re-runs completed 07-01 (log: server `~/atli/eval_cv_obb_final.log`). OBB was trained
for **v8n and v11n only** (no v5 OBB runs). Datasets: `Merged_CV_obb` (labels from
Roboflow segmentation export → `cv2.minAreaRect`, built by `data/build_cv_obb.py`).

**⚠ Cross-task caveat:** the OBB folds were built from a *different Roboflow export*
than `Merged_CV_proper`, so OBB numbers are NOT directly comparable to the detection
CV numbers in `cv_proper_results.md`. The controlled comparison is the
detection-on-OBB-folds isolation study (`Merged_CV_det_aligned`, CVD_* runs):
champ v11 detection 0.802 mAP / 0.745 DD vs champ v11 OBB 0.808 / 0.750 —
**task mode is a wash overall**; OBB's tighter boxes mainly help Normal Insulators
(38% area reduction) and Normal Damper (4.5%).

## Summary (5-fold means ± std, test split, OBB metrics)

| Condition | Model | mAP@0.5 | P | R | DD AP | BI AP | FI AP | ND AP | NI AP |
|---|---|---|---|---|---|---|---|---|---|
| base | v8n | 0.766 ± 0.026 | 0.810 | 0.748 | 0.758 ± 0.043 | 0.498 | 0.850 | 0.776 | 0.754 |
| base | v11n | 0.779 ± 0.011 | 0.826 | 0.734 | **0.798 ± 0.024** | 0.540 | 0.844 | 0.802 | 0.744 |
| champ | v8n | 0.801 ± 0.021 | 0.835 | 0.774 | 0.743 ± 0.047 | 0.632 | 0.882 | 0.806 | 0.762 |
| champ | v11n | 0.808 ± 0.014 | 0.842 | 0.777 | 0.750 ± 0.058 | 0.650 | 0.879 | 0.812 | 0.766 |
| osall | v8n | 0.799 ± 0.016 | 0.835 | 0.776 | 0.743 ± 0.068 | 0.663 | 0.882 | 0.806 | 0.727 |
| **osall** | **v11n ★** | **0.817 ± 0.017** | 0.853 | 0.786 | 0.782 ± 0.066 | **0.697** | 0.878 | 0.811 | 0.749 |

### Defect recall

| Condition | Model | DD R | BI R | FI R | SE R |
|---|---|---|---|---|---|
| base | v8n / v11n | 0.749 / 0.734 | 0.471 / 0.484 | 0.843 / 0.816 | 0.770 / 0.755 |
| champ | v8n / v11n | 0.708 / 0.720 | 0.604 / 0.608 | 0.865 / 0.859 | 0.816 / 0.823 |
| osall | v8n / v11n | 0.731 / **0.743** | 0.609 / **0.646** | 0.866 / 0.852 | 0.830 / **0.844** |

## Observations

1. **osall v11n is the best OBB condition** (mAP 0.817), and under OBB the osall
   recipe beats champ (unlike detection, where champ v11 wins) — broader defect
   oversampling pays off more when boxes are rotated.
2. **Under OBB, the champ/osall recipes do NOT improve DD AP over baseline**
   (base v11 DD 0.798 ± 0.024 is the best and tightest) — dampers are compact,
   so hi-res + oversampling mostly adds noise in rotated-box regime; the gains
   shift to BI (+0.10–0.16) and SE/FI recall instead.
3. **FI AP ≈ 0.85–0.88 here vs ≈ 0.60–0.66 in `cv_proper_results.md` is a fold/export
   difference, not a task-mode gain** (see caveat above; the det-aligned control shows
   per-class parity except NI box tightness).

## Per-fold detail (mAP / DD AP)

| Fold | base v8 | base v11 | champ v8 | champ v11 | osall v8 | osall v11 |
|---|---|---|---|---|---|---|
| f0 | 0.766/0.734 | 0.785/0.771 | 0.804/0.749 | 0.812/0.720 | 0.798/0.745 | 0.809/0.757 |
| f1 | 0.783/0.742 | 0.790/0.783 | 0.817/0.746 | 0.808/0.701 | 0.820/0.742 | 0.839/0.810 |
| f2 | 0.802/0.783 | 0.786/0.804 | 0.827/0.764 | 0.828/0.816 | 0.798/0.787 | 0.811/0.786 |
| f3 | 0.748/0.828 | 0.771/0.840 | 0.786/0.799 | 0.806/0.825 | 0.807/0.823 | 0.835/0.878 |
| f4 | 0.729/0.705 | 0.762/0.793 | 0.771/0.658 | 0.785/0.690 | 0.771/0.621 | 0.792/0.677 |
