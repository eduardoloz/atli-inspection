# obb_champ_v11n_1280

Oriented-bounding-box (rotated box) variant of the champion, trained on the dataset carrying
the hand-drawn oriented Normal_Damper annotations. **Evaluated on its own 107-image test
split — NOT comparable to the detection cards' 120-image split; compare only within OBB.**

- **Architecture:** YOLOv11n-OBB | task: obb (rotated boxes, rotated-IoU matching) | 7 classes
- **Dataset:** `ATLI_noCPLID_OBB_OS3`, built from the `atli_target-minus-the-cplid` v2
  segmentation export via `data/build_nocplid_obb.py`: polygon annotations (82 hand-drawn
  oriented Normal_Dampers on 27 imgs + 259 polygon Normal_Insulators) become true rotated
  boxes via `cv2.minAreaRect`; plain boxes become axis-aligned corners; then ×3 DD
  oversampling. Train **652 imgs / 2,841 annotations** (Birdnest 187, Broken 146, DD 258,
  Flashover 288, Normal_Damper 1,043, Normal_Ins 710, Self-Exploded 209); val 113 imgs / 450;
  test 107 imgs / 465 (merged-split layout).
- **Training:**
  ```
  MODEL=yolo11n-obb.pt EXTRA="scale=0.9 seed=<s>" DATA=~/atli/ATLI_noCPLID_OBB_OS3/data.yaml \
    bash train/run_config_obb.sh OBBnc_champ_v11_s<s> <gpu> 150 100 1280 16
  ```
  (same 2-stage TL: 150 ep lr0=0.01 → 100 ep lr0=0.00334/lrf=0.1535, `yolo obb train`)
- **Augmentation:** identical set to the detection cards (defaults + scale=0.9, verified from
  args.yaml). No synthetic data.
- **Seeds trained:** 0, 1, 2.
- **Test metrics** (107-img OBB test split; AP@0.5 rotated-IoU, mean ± std, 3 seeds):

  | class | AP@0.5 | vs OBB baseline (640) |
  |---|---|---|
  | Birdnest | 0.950 ± 0.003 | +0.026 |
  | Broken_Insulator | 0.846 ± 0.049 | +0.155 |
  | Defective_Damper | 0.768 ± 0.033 (recall 0.826) | +0.115 |
  | Flashover_Insulator | 0.681 ± 0.025 | +0.052 |
  | Normal_Damper | 0.767 ± 0.020 | +0.011 |
  | Normal_Insulators | 0.724 ± 0.027 | +0.055 |
  | Self-Exploded_Insulator | 0.621 ± 0.068 | −0.057 |
  | **overall** | **0.765 ± 0.015** (recall 0.831) | **+0.051** |

- **Max-recall variant:** adding `degrees=45` rotation augmentation (legitimate in OBB mode —
  rotated boxes rotate exactly, no box inflation) yields **DefDamper recall 0.957** at
  DD AP 0.738 ± 0.012, costing overall mAP (0.730, paid by Self-Exploded/NI). Runs
  `OBBdeg45_s{0,1,2}` — the configuration of choice if missing defective dampers is the
  dominant cost.
- **Provenance:** server runs `~/atli/runs/OBBnc_champ_v11_s{0,1,2}_s2/weights/best.pt`,
  trained 2026-07-08 (rerun after the numpy env incident; postmortem in
  `results/pi_summary_2026-07-08.md`); scripts as of commit `9a1ffbf`.
- **Known limitations:** only 27 of ~570 ND-bearing images carry oriented ground truth, so
  Normal_Damper itself barely moves (+0.011) — extending oriented annotation is the lever;
  Self-Exploded regresses under the champion recipe in OBB mode; test split differs from the
  detection benchmark (no cross-task comparisons).
