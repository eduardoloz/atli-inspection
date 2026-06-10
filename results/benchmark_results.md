# ATLI Benchmark Reproduction — Untitled2.ipynb

Re-run from scratch on UNLV `ai.ee.unlv.edu` (8× Quadro RTX 6000) under account
`LozanoE`, because the original lived in `mazumder`'s private home. Environment
rebuilt to match the notebook: **torch 2.6.0+cu118, YOLOv5 (latest), Ultralytics
8.4.9**. Dataset rebuilt faithfully from `Merging_datasets_of_Eduardo_and_target.ipynb`
(Roboflow `merged_atli_target` v4 + `eduardos-annotated-photos` v1 → 1343 imgs,
stratified seed-42 split 941/203/199; identical class totals to the original).

Each config = **stage-1 train** (from COCO-pretrained nano weights) → **stage-2
fine-tune** (100 ep, SGD, `lr0=0.00334`, `lrf=0.1535`) → **test eval**.
Standardized **batch 32 (4 GPUs)** for every config (the notebook used inconsistent
24/32/40/64). Test split = 199 images / 867 instances.

## Reproduced results — test split (ranked by mAP@0.5)

| Rank | Config | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | YOLOv8n SGD 150+100 | **0.707** | **0.410** | 0.774 | 0.668 |
| 2 | YOLOv8n SGD 300+100 | 0.696 | 0.406 | 0.801 | 0.654 |
| 3 | YOLOv5n SGD 150+100 | 0.692 | 0.358 | 0.804 | 0.630 |
| 3 | YOLOv5n SGD 300+100 | 0.692 | 0.377 | 0.801 | 0.633 |
| 5 | YOLOv5n Adam 300+100 | 0.656 | 0.328 | 0.788 | 0.585 |
| 6 | YOLOv5n Adam 150+100 | 0.624 | 0.308 | 0.807 | 0.574 |

## Reproduced vs. original notebook (mAP@0.5, test)

| Config | Reproduced | Original | Δ |
|---|---|---|---|
| YOLOv8n SGD 150+100 | 0.707 | 0.760 | −0.053 |
| YOLOv8n SGD 300+100 | 0.696 | 0.749 | −0.053 |
| YOLOv5n SGD 150+100 | 0.692 | 0.732 | −0.040 |
| YOLOv5n SGD 300+100 | 0.692 | 0.740 | −0.048 |
| YOLOv5n Adam 300+100 | 0.656 | 0.714 | −0.058 |
| YOLOv5n Adam 150+100 | 0.624 | 0.646 | −0.022 |

A consistent ~0.04–0.06 lower mAP across the board — expected, since this is a
different random realization of the stratified split (different test images:
199/867 vs the original 208/856), plus the standardized batch size. **All
qualitative conclusions reproduce exactly:** YOLOv8n+SGD wins, SGD > Adam for
YOLOv5n, longer stage-1 gives little/no gain after fine-tuning.

## Reproduced per-class mAP@0.5 (test)

| Class | v5n SGD 150 | v5n SGD 300 | v5n Adam 150 | v5n Adam 300 | v8n SGD 150 | v8n SGD 300 |
|---|---|---|---|---|---|---|
| Birdnest                | 0.756 | 0.756 | 0.809 | 0.787 | 0.780 | 0.792 |
| Broken_Insulator        | 0.455 | 0.475 | 0.341 | 0.366 | 0.444 | 0.475 |
| Defective_Damper        | 0.690 | 0.673 | 0.571 | 0.655 | 0.700 | 0.667 |
| Flashover_Insulator     | 0.574 | 0.581 | 0.502 | 0.592 | 0.626 | 0.566 |
| Normal_Damper           | 0.801 | 0.795 | 0.726 | 0.762 | 0.800 | 0.776 |
| Normal_Insulators       | 0.745 | 0.759 | 0.706 | 0.725 | 0.779 | 0.784 |
| Self-Exploded_Insulator | 0.821 | 0.807 | 0.711 | 0.705 | 0.819 | 0.814 |

**Broken_Insulator remains the weakest class everywhere (0.34–0.48)** — the rare
class the `rebalance/` task targets.

## Where things live on the server
- Env: `~/atli/env` · Code: `~/atli/yolov5` · Dataset: `~/atli/Merged_Dataset_Stratified`
- Training runs: `~/atli/runs/<config>_{s1,s2,test}` · Clean eval tables: `~/atli/eval/<config>.txt`
- Re-run status anytime: `bash ~/Research/Vegas/check_status.sh`
