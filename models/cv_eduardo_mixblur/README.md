# cv_eduardo_mixblur — mixup/blur recipe champions (phases 13–23)

- **Architecture:** YOLOv11n (2.66M params) and YOLOv8n (3.08M) | task: **obb**
- **eduardos-annotated-photos:** **included** (974-img pool = 796 ATLI-no-CPLID + 178 eduardos v6)
- **Dataset:** group-aware 5-fold CV (`data/build_cv_eduardo.py`), 70/15/15, eduardos'
  21 dup-clusters kept whole; `cplid_*` conditions add 250 CPLID images to TRAIN only
  (val/test always clean). Server: `~/atli/Merged_CV_eduardo*`.
- **Recipe:** 2-stage TL (150 ep `lr0=0.01` → 100 ep `lr0=0.00334`), batch 16,
  init `yolo11n-obb.pt` / `yolov8n-obb.pt`, ×3 all-defect oversampling (train only),
  `scale=0.9`; per-condition: `degrees=15|30`, `mixup=0.15`, blur-aug via the
  albumentations `blurpatch` sitecustomize (motion p=0.3 + Gaussian p=0.2).
  Drivers: `train/sweep_eduardo_p{13,17,19,20,23}.sh`.
- **Weights:** GitHub release **`weights-eduardo-cv-2026-09-11`** — 8 conditions × 5
  folds = 40 `best.pt` (stage-2 final; asset table + inference commands in the
  release's NOTES.md). Earlier conditions (deg15 champions, grafts, v5/v8 rungs):
  release `weights-eduardo-cv-2026-07-22`.
- **Headline (5-fold mean ± std, mAP@0.5 / DD AP):**
  ★ `blurmix30_1280` **0.820 ± 0.022 / 0.798** (overall champion; blurred-test 0.701) ·
  `cplidblurmix30` 0.813 / 0.769 · `cplidmix15` 0.811 / 0.807 (best DD recall 0.749) ·
  `mix15_1280` 0.804 / 0.760 · `v8mix15` 0.798 · `v8blurdeg15` 0.783 ·
  ★ `blurmix30_640` 0.764 / 0.731 (640 deployment champion; blurred 0.702) ·
  `blurmix_640` 0.757 / 0.680 (blurred 0.710)
- **Evidence:** `results/eval_eduardo_results.json` (per-fold), `results/config_benchmark.csv`,
  `results/eval_blur_robustness.json` (blurred-test), CLAUDE.md experiment log
  (phases 13, 17, 19, 20, 23).
- **Eval protocol:** score each fold's `best.pt` on that fold's test split only
  (`eval/eval_cv_eduardo.py`). Never export TensorRT engines for sharing
  (device-specific — see GIT.md).
