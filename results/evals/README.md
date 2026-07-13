# Evaluation results — all conditions (through 2026-07-09)

Seed-averaged test-set metrics (AP@0.5, mean ± std) for every condition run on the clean
post-CPLID data. Raw per-class dump: `results_all_conditions.csv` (30 conditions; **mixes
test sets and eras — use the labeled tables below for anything comparable**).

Full recipes for each condition are in `../../CLAUDE.md` (experiment log) and the model cards
in `../../models/`.

## Clean detection benchmark (identical 120-image test set — directly comparable)

Baseline = YOLOv11n 640. Champion = 1280 + ×3 DD oversample + scale=0.9. All 3 seeds.

| condition | recipe | mAP@0.5 | Defective_Damper | Self-Exploded |
|---|---|---|---|---|
| **CPRest1280** | champion + 249 CPLID imgs to train only | **0.802 ± 0.015** | 0.693 ± 0.050 | 0.907 |
| **HROaugnc_v11** | champion (clean train) | 0.784 ± 0.011 | 0.622 ± 0.073 | 0.871 |
| HR768nc | champion @768 (deployment model) | 0.769 ± 0.009 | 0.670 ± 0.049 | 0.802 |
| AUGcm20 | champion @768 + close_mosaic=20 | 0.767 ± 0.003 | 0.708 ± 0.051 | 0.810 |
| AUGnoOSdeg10 | @768, no oversample + degrees=10 | 0.761 ± 0.012 | 0.618 ± 0.031 | 0.805 |
| AUGsc07 | @768, scale=0.7 | 0.760 ± 0.024 | 0.611 ± 0.087 | 0.830 |
| OSallHRnc_v11 | champion, oversample ALL defects ×3 | 0.760 ± 0.006 | 0.541 ± 0.020 | 0.860 |
| AUGosall768 | @768, oversample all defects | 0.757 ± 0.015 | 0.567 ± 0.113 | 0.875 |
| AUGnoOS | @768, no oversampling | 0.754 ± 0.021 | 0.595 ± 0.091 | 0.843 |
| AUGsc05 | @768, scale=0.5 (default) | 0.753 ± 0.004 | 0.639 ± 0.075 | 0.807 |
| CPRest768 | @768 + CPLID restored | 0.747 ± 0.012 | 0.606 ± 0.039 | 0.833 |
| AUGdeg10 | @768 + degrees=10 | 0.746 ± 0.014 | 0.566 ± 0.103 | 0.838 |
| AUGnomos | @768, mosaic off | 0.736 ± 0.014 | 0.574 ± 0.097 | 0.738 |
| AUGdeg45 | @768 + degrees=45 | 0.729 ± 0.021 | 0.658 ± 0.054 | 0.726 |
| **Bnc_v11** | baseline 640 | 0.736 ± 0.015 | 0.614 ± 0.082 | 0.755 |

Takeaways (also in CLAUDE.md): CPLID-restore at 1280 is the best benchmark (0.802); champion
margin over baseline holds on clean data (+0.048); at 768 the deployment model keeps 98% of
accuracy; close_mosaic=20 is the only ablation upgrade; oversampling all defects hurts DD vs
DD-only; rotation and mosaic-off both fail in detection. Caveat: only 12 DD test instances, so
DD numbers carry ±0.05–0.08 seed spread and are directional.

## OBB benchmark (107-image OBB test set — compare only within this block)

| condition | recipe | mAP@0.5 | Defective_Damper | DD recall |
|---|---|---|---|---|
| **OBBnc_champ_v11** | champion recipe, OBB | **0.765 ± 0.015** | 0.768 ± 0.033 | 0.826 |
| OBBdeg45 | OBB + degrees=45 (max recall) | 0.730 ± 0.007 | 0.738 ± 0.012 | **0.957** |
| OBBnc_B_v11 | baseline OBB 640 | 0.714 ± 0.018 | 0.653 ± 0.021 | — |

## Not comparable (older / different test sets — in the CSV, excluded above)

`ABnew_*`, `ABold_v11`, `HROaug_v5`, `OSall_v{5,8,11}`, `B_v{5,8}`, `TH2_*` were run on the
pre-purge merged test set or are single-seed/superseded. See CLAUDE.md history for context;
do not compare their numbers against the clean-data tables.
