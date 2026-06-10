#!/usr/bin/env bash
# Re-run ONLY the two configs whose tests didn't run in sweep 1:
#   v8n_sgd_150 — stage2 best.pt exists -> test step only.
#   v5n_sgd_150 — stage2 crashed (OOM) -> re-run stage2 from stage1 best.pt + test.
# Runs on the IDLE GPUs 4,5,6,7 (GPU 2 is in use by another user). Output appended to
# ~/atli/rerun.log with the SAME markers parse_results.py greps for.
set -uo pipefail
ROOT="$HOME/atli"
YAML="$ROOT/Merged_Dataset_Stratified/merged_stratified.yaml"
PROJ="$ROOT/runs"
LOG="$ROOT/rerun.log"
GPUS=4,5,6,7
DEV0=4            # CUDA_VISIBLE_DEVICES=4 -> visible index 0 = physical GPU 4
PORT=29530
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"

echo "RERUN_START $(date)  GPUs=$GPUS" | tee -a "$LOG"

# ---- v8n_sgd_150: test only (weights already exist) ----
{
  echo "[$(date)] --- v8n_sgd_150 test ---"
  CUDA_VISIBLE_DEVICES=$DEV0 yolo detect val model="$PROJ/v8n_sgd_150_s2/weights/best.pt" \
     data="$YAML" split=test imgsz=640 batch=8 device=0 \
     project="$PROJ" name=v8n_sgd_150_test exist_ok=True
  echo "[$(date)] ### v8n_sgd_150 DONE"
} >> "$LOG" 2>&1

# ---- v5n_sgd_150: stage2 finetune (from stage1 best) + test ----
cd "$ROOT/yolov5"
{
  echo "[$(date)] --- v5n_sgd_150 stage2 (finetune, SGD, oscar hyp) RERUN GPUs=$GPUS ---"
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=4 --master_port=$PORT train.py \
    --img 640 --batch 32 --epochs 100 --workers 8 --optimizer SGD \
    --data "$YAML" --weights "$PROJ/v5n_sgd_150_s1/weights/best.pt" --hyp hyp.oscar_paper.yaml \
    --project "$PROJ" --name v5n_sgd_150_s2 --exist-ok
  echo "[$(date)] --- v5n_sgd_150 test ---"
  CUDA_VISIBLE_DEVICES=$DEV0 python val.py --data "$YAML" \
    --weights "$PROJ/v5n_sgd_150_s2/weights/best.pt" --img 640 --batch 8 --task test \
    --device 0 --project "$PROJ" --name v5n_sgd_150_test --exist-ok
  echo "[$(date)] ### v5n_sgd_150 DONE"
} >> "$LOG" 2>&1

echo "RERUN_DONE $(date)" | tee -a "$LOG"
