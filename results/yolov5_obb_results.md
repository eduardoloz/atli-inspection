# YOLOv5n-OBB results — rotated Task1-mAP@0.5 (auto-generated 2026-07-28)

Generated automatically by `~/atli/eval_v5obb_task1.py`, chained off `sweep_v5obb.sh`'s completion marker via `~/atli/run_v5obb_autoeval.sh` -- no manual step. See `results/yolov5_obb_setup.md` for the fork, environment, and data-conversion background.

## Read this before comparing numbers

This sweep's YOLOv5n-OBB numbers below are scored two different ways:

- **`Task1 mAP@0.5` (rotated, polygon IoU)** — computed with `DOTA_devkit`'s `dota_evaluation_task1.voc_eval` (the standard DOTA/OBB scorer: `polyiou.iou_poly` for true rotated-box IoU, `ovthresh=0.5`, **`use_07_metric=False`**, i.e. the modern all-point/continuous AP integration — deliberately overriding this script's own `main()`, which hardcodes the legacy VOC07 11-point method and would NOT match the convention used elsewhere in this project). **This is the number that is actually comparable in kind to the v8n-OBB/v11n-OBB rotated mAP@0.5 numbers elsewhere in the project** (same rotated-IoU concept, same 0.5 threshold, same all-point AP integration) — though it comes from a different evaluator implementation (DOTA_devkit/polyiou vs. Ultralytics' internal `OBBMetrics`), so treat cross-tool matches as "same metric family," not bit-identical.
- **`HBBmAP@.5` (native, axis-aligned)** — this fork's own built-in `val.py` metric, kept here **for reference only**. It scores the axis-aligned enclosing box of each rotated prediction/GT pair, not the rotated polygon itself, so it is systematically NOT comparable to any OBB number elsewhere in this project. Do not use it to judge this model against v8n-OBB/v11n-OBB.

**Same-pool reference anchor:** CV_eduardo_obb (same 5-fold pool as this v5n-OBB sweep), YOLOv11n OBB champ+osall (no deg45): mAP 0.759 +/- 0.019, DefDamper AP 0.672 +/- 0.095, DD recall 0.657 (source: CLAUDE.md 'ATLI+eduardos group-aware 5-fold CV' section).

**Older-pool historical reference:** Merged_CV_obb (an older, DIFFERENT 5-fold pool/export -- not directly stackable with the numbers above, historical context only), from `results/cv_obb_results.md`: base v8n mAP 0.766 +/- 0.026 (DD 0.758), base v11n mAP 0.779 +/- 0.011 (DD 0.798); champ v8n mAP 0.801 +/- 0.021 (DD 0.743), champ v11n mAP 0.808 +/- 0.014 (DD 0.750).

## Summary (5-fold mean +/- std, test split)

| Condition | Task1 mAP@0.5 (rotated) | native HBBmAP@.5 (reference only) |
|---|---|---|
| baseline | 0.593 +/- 0.038 | 0.647 +/- 0.044 |
| champion | 0.671 +/- 0.026 | 0.727 +/- 0.023 |

### Per-class Task1 AP@0.5 (rotated, mean +/- std)

| Class | baseline | champion |
|---|---|---|
| Birdnest | 0.898 +/- 0.047 | 0.858 +/- 0.072 |
| Broken_Insulator | 0.587 +/- 0.086 | 0.641 +/- 0.081 |
| Defective_Damper | 0.432 +/- 0.108 | 0.489 +/- 0.117 |
| Flashover_Insulator | 0.563 +/- 0.069 | 0.681 +/- 0.075 |
| Normal_Damper | 0.537 +/- 0.086 | 0.628 +/- 0.048 |
| Normal_Insulators | 0.582 +/- 0.052 | 0.646 +/- 0.057 |
| Self-Exploded_Insulator | 0.549 +/- 0.102 | 0.753 +/- 0.057 |

### Per-fold detail (Task1 mAP@0.5 / native HBBmAP@.5)

| Fold | baseline | champion |
|---|---|---|
| f0 | 0.564 / 0.606 | 0.650 / 0.718 |
| f1 | 0.563 / 0.607 | 0.695 / 0.730 |
| f2 | 0.561 / 0.623 | 0.638 / 0.697 |
| f3 | 0.622 / 0.688 | 0.665 / 0.723 |
| f4 | 0.654 / 0.710 | 0.707 / 0.768 |

## Pipeline notes

- Ground truth: each fold's `CV_eduardo_obb_v5fmt/fold{k}/test/labelTxt/*.txt` (already in DOTA format, built by `yolov5_obb_convert.py`).
- Predictions: `val.py --task test --save-json` per run -> `runs/{name}_task1eval/best_obb_predictions.json` -> `json2task1.py` -> `runs/{name}_task1eval/obb_predictions_Txt/Task1_<class>.txt` (this project's own converter, since `tools/TestJson2VocClassTxt.py` hardcodes the 15/16-class DOTA-v1/v1.5 taxonomy instead of this dataset's 7 ATLI classes).
- No image splitting/merging step was needed (`DOTA_devkit/ImgSplit_multi_process.py` + `ResultMerge_multi_process.py`, normally required for full-size DOTA aerial tiles) since every image in this dataset is already a single 640px tile.
- `polyiou` (the rotated polygon-IoU C extension `voc_eval` depends on) was built via `cd DOTA_devkit && python setup.py build_ext --inplace` against the pre-generated `polyiou_wrap.cxx` already checked into the repo — no working `swig` binary was needed on this server (`swig` was not installed and there is no root/sudo access here) since the wrapper didn't need to be regenerated from `polyiou.i`, only compiled. Sanity-checked against hand-computed cases before trusting it: identical unit squares -> IoU 1.0, disjoint squares -> IoU 0.0, two 2x2 squares offset by 1 (1x2 intersection / 6 union) -> IoU 0.3333 — all exact.

