# champ_v11n_1280_cplid_restore

Best benchmark model: champion recipe + 249 formerly-CPLID images restored to **train only**
(the clean test/val splits contain zero CPLID, so there is no leakage — leakage is train∩eval
overlap, and this configuration has none).

- **Architecture:** YOLOv11n (2.58M params) | task: detect | 7 classes (order as in
  `models/README.md`)
- **Dataset:** `ATLI_noCPLID_plusCPLIDtrain_OS3` = clean train split + 249 restored CPLID
  images (a Self-Exploded_Insulator on every one, 277 co-occurring Normal_Dampers) + ×3 DD
  oversampling. Train: **888 imgs / 3,419 annotations** (Birdnest 178, Broken 142, DD 255,
  Flashover 304, Normal_Damper 1,322, Normal_Ins 740, Self-Exploded 478). Val/test identical
  to the clean splits (116/120 imgs) — directly comparable to `champ_v11n_1280_clean`.
- **Training:** identical recipe to the clean champion, only DATA differs:
  ```
  MODEL=yolo11n.pt EXTRA="scale=0.9 seed=<s>" DATA=~/atli/ATLI_noCPLID_plusCPLIDtrain_OS3/data.yaml \
    bash train/run_config_ext.sh CPRest1280_s<s> <gpu> 150 100 1280 16
  ```
  (2-stage TL: 150 ep lr0=0.01 → 100 ep lr0=0.00334/lrf=0.1535, SGD, batch 16, imgsz 1280)
- **Augmentation:** identical to `champ_v11n_1280_clean` (defaults + scale=0.9; verified from
  args.yaml). No synthetic data.
- **Seeds trained:** 0, 1, 2.
- **Test metrics** (identical clean 120-img test split; AP@0.5 mean ± std, 3 seeds):

  | class | AP@0.5 | Δ vs clean champion |
  |---|---|---|
  | Birdnest | 0.974 ± 0.009 | −0.010 |
  | Broken_Insulator | 0.681 ± 0.009 | −0.018 |
  | Defective_Damper | 0.693 ± 0.050 | +0.071 (directional; 12 test instances) |
  | Flashover_Insulator | 0.680 ± 0.017 | +0.020 |
  | Normal_Damper | 0.826 ± 0.031 | +0.013 |
  | Normal_Insulators | 0.857 ± 0.002 | +0.019 |
  | Self-Exploded_Insulator | 0.907 ± 0.026 | +0.036 |
  | **overall** | **0.802 ± 0.015** | **+0.018** |

  Gains land where the restored data lives (Self-Exploded, Normal_Insulators), as predicted.
- **Provenance:** server runs `~/atli/runs/CPRest1280_s{0,1,2}_s2/weights/best.pt`, trained
  2026-07-08/09; dataset built by the restore experiment (249 imgs identified via
  `target_join_mapping.json` is_cplid records); scripts as of commit `9a1ffbf`.
- **Known limitations:** the restore HURTS at deploy resolution — the same data at 768 px
  scores 0.747 vs 0.766 clean (restored images are close-ups that skew the object-scale
  distribution), so this model is for benchmark/1280 use only; do not port the dataset choice
  to the 768 deployment model. Same DD-sample-size and blur caveats as the clean champion.
