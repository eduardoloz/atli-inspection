#!/usr/bin/env bash
# Single-GPU RT-DETR (transformer) config: stage1 train -> stage2 finetune -> test.
#   args: NAME GPU [EP1=150] [EP2=100]
# Uses DATA env var for the dataset yaml.
set -euo pipefail
NAME="$1"; GPU="$2"; EP1="${3:-150}"; EP2="${4:-100}"
ROOT="$HOME/atli"
YAML="${DATA:-$ROOT/Merged_Dataset_Stratified/merged_stratified.yaml}"
PROJ="$ROOT/runs"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
mkdir -p "$PROJ"
BATCH=4  # RT-DETR is much larger, needs smaller batch

echo "[$(date)] ### $NAME  (RT-DETR-l  s1=${EP1}ep s2=${EP2}ep)  GPU=$GPU batch=$BATCH"

echo "[$(date)] --- $NAME stage1 ---"
yolo detect train model=rtdetr-l.pt data="$YAML" imgsz=640 \
  batch=$BATCH epochs=$EP1 optimizer=SGD lr0=0.01 device=$GPU \
  project="$PROJ" name=${NAME}_s1 exist_ok=True

echo "[$(date)] --- $NAME stage2 (finetune ${EP2}ep) ---"
yolo detect train model="$PROJ/${NAME}_s1/weights/best.pt" \
  data="$YAML" imgsz=640 batch=$BATCH epochs=$EP2 workers=8 optimizer=SGD \
  lr0=0.00334 lrf=0.1535 device=$GPU \
  project="$PROJ" name=${NAME}_s2 exist_ok=True

echo "[$(date)] --- $NAME test ---"
yolo detect val model="$PROJ/${NAME}_s2/weights/best.pt" \
  data="$YAML" split=test imgsz=640 batch=$BATCH device=$GPU \
  project="$PROJ" name=${NAME}_test exist_ok=True

echo "[$(date)] ### $NAME DONE"
