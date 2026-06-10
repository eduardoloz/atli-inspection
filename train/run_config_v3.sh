#!/usr/bin/env bash
# Ultralytics single-GPU config with adjustable imgsz/batch/cls.
#   args: NAME FW(v8|v11|rtdetr) GPU EP1 EP2 IMGSZ BATCH [CLS=0.5]
# DATA env var = dataset yaml (required).
set -euo pipefail
NAME="$1"; FW="$2"; GPU="$3"; EP1="$4"; EP2="$5"; IMGSZ="$6"; BATCH="$7"; CLS="${8:-0.5}"
ROOT="$HOME/atli"
YAML="${DATA:?DATA env var required}"
PROJ="$ROOT/runs"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
mkdir -p "$PROJ"

case "$FW" in
  v8)     WEIGHTS="yolov8n.pt" ;;
  v11)    WEIGHTS="yolo11n.pt" ;;
  rtdetr) WEIGHTS="rtdetr-l.pt" ;;
  *)      echo "Unknown framework: $FW"; exit 1 ;;
esac

echo "[$(date)] ### $NAME  ($FW imgsz=$IMGSZ batch=$BATCH cls=$CLS s1=${EP1}ep s2=${EP2}ep)  GPU=$GPU"

echo "[$(date)] --- $NAME stage1 ---"
yolo detect train model="$WEIGHTS" data="$YAML" imgsz=$IMGSZ batch=$BATCH epochs=$EP1 \
  optimizer=SGD lr0=0.01 cls=$CLS device=$GPU project="$PROJ" name=${NAME}_s1 exist_ok=True

echo "[$(date)] --- $NAME stage2 (finetune ${EP2}ep) ---"
yolo detect train model="$PROJ/${NAME}_s1/weights/best.pt" data="$YAML" imgsz=$IMGSZ \
  batch=$BATCH epochs=$EP2 workers=8 optimizer=SGD lr0=0.00334 lrf=0.1535 cls=$CLS \
  device=$GPU project="$PROJ" name=${NAME}_s2 exist_ok=True

echo "[$(date)] --- $NAME test ---"
yolo detect val model="$PROJ/${NAME}_s2/weights/best.pt" data="$YAML" split=test imgsz=$IMGSZ \
  batch=$BATCH device=$GPU project="$PROJ" name=${NAME}_test exist_ok=True

echo "[$(date)] ### $NAME DONE"
