# champ_v11n_1280_clean

Clean-data champion: the reference benchmark model after CPLID decontamination.

- **Architecture:** YOLOv11n (2.58M params, 6.3 GFLOPs @640) | task: detect | 7 classes
  (0 Birdnest, 1 Broken_Insulator, 2 Defective_Damper, 3 Flashover_Insulator,
  4 Normal_Damper, 5 Normal_Insulators, 6 Self-Exploded_Insulator)
- **Dataset:** `ATLI_noCPLID_OS3` = clean `ATLI_target_tightNI_noCPLID` train split with ×3
  image-level oversampling of Defective_Damper images (561 → 633 train imgs, 2,344 → 2,810
  train annotations; DD 82 → 246). Val 116 imgs / 506 ann and test 120 imgs / 467 ann are the
  untouched clean splits (see `models/README.md` for the per-class table).
- **Training (2-stage transfer learning, COCO-init):**
  ```
  MODEL=yolo11n.pt EXTRA="scale=0.9 seed=<s>" DATA=~/atli/ATLI_noCPLID_OS3/data.yaml \
    bash train/run_config_ext.sh HROaugnc_v11_s<s> <gpu> 150 100 1280 16
  ```
  which expands to: stage 1 `yolo detect train imgsz=1280 batch=16 epochs=150 optimizer=SGD
  lr0=0.01 scale=0.9` from COCO weights, then stage 2 continues from stage-1 best with
  `epochs=100 lr0=0.00334 lrf=0.1535`, then `yolo detect val split=test`.
- **Augmentation (verified from run args.yaml; Ultralytics defaults except scale):**
  hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, degrees 0, translate 0.1, **scale 0.9 (override)**,
  shear 0, perspective 0, flipud 0, fliplr 0.5, mosaic 1.0, close_mosaic 10, mixup 0,
  copy_paste 0, erasing 0.4. All online; no synthetic data.
- **Seeds trained:** 0, 1, 2.
- **Test metrics** (clean 120-img test split; AP@0.5 mean ± std over 3 seeds, recall in
  parentheses):

  | class | AP@0.5 | recall |
  |---|---|---|
  | Birdnest | 0.984 ± 0.004 | 0.886 |
  | Broken_Insulator | 0.699 ± 0.029 | 0.797 |
  | Defective_Damper | 0.622 ± 0.073 | 0.656 |
  | Flashover_Insulator | 0.660 ± 0.024 | 0.805 |
  | Normal_Damper | 0.813 ± 0.019 | 0.893 |
  | Normal_Insulators | 0.838 ± 0.013 | 0.836 |
  | Self-Exploded_Insulator | 0.871 ± 0.027 | 0.960 |
  | **overall** | **0.784 ± 0.011** | 0.833 |

- **Provenance:** server runs `~/atli/runs/HROaugnc_v11_s{0,1,2}_s2/weights/best.pt`, trained
  2026-07-07; scripts as of repo commit `9a1ffbf`.
- **Known limitations:** DD test basis is 12 instances (±0.073 seed spread — directional);
  blur-fragile (−0.246 mAP under 7-px synthetic motion blur); not comparable to pre-purge
  benchmarks (different test set). For a higher-accuracy variant see
  `champ_v11n_1280_cplid_restore.md`; for edge deployment use `champ_v11n_768_deploy.md`
  (this 1280 model projects to only ~6 fps on a Jetson Nano).
