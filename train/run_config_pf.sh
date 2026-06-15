#!/usr/bin/env bash
# Pretrain->finetune (domain-adaptation) driver: stage 1 and stage 2 use DIFFERENT data.
#   args: NAME FW GPU EP1 EP2 IMGSZ BATCH DATA1 DATA2 [CLS=0.5]
#   s1: COCO-init, train on DATA1 (universe-rich source)  -> learn dampers broadly
#   s2: finetune s1-best on DATA2 (native-only target)    -> specialize, low LR
#   test: eval on DATA2 test split (native 199-img benchmark)
set -euo pipefail
NAME="$1"; FW="$2"; GPU="$3"; EP1="$4"; EP2="$5"; IMGSZ="$6"; BATCH="$7"
DATA1="$8"; DATA2="$9"; CLS="${10:-0.5}"
ROOT="$HOME/atli"; PROJ="$ROOT/runs"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
mkdir -p "$PROJ"
case "$FW" in
  v8)  WEIGHTS="yolov8n.pt" ;;
  v11) WEIGHTS="yolo11n.pt" ;;
  *)   echo "Unknown framework: $FW"; exit 1 ;;
esac

echo "[$(date)] ### $NAME  s1=DATA1($(basename $DATA1)) ${EP1}ep -> s2=DATA2($(basename $DATA2)) ${EP2}ep  GPU=$GPU"

echo "[$(date)] --- $NAME stage1 (source pretrain) ---"
yolo detect train model="$WEIGHTS" data="$DATA1" imgsz=$IMGSZ batch=$BATCH epochs=$EP1 \
  optimizer=SGD lr0=0.01 cls=$CLS device=$GPU project="$PROJ" name=${NAME}_s1 exist_ok=True

echo "[$(date)] --- $NAME stage2 (native finetune) ---"
yolo detect train model="$PROJ/${NAME}_s1/weights/best.pt" data="$DATA2" imgsz=$IMGSZ \
  batch=$BATCH epochs=$EP2 workers=8 optimizer=SGD lr0=0.00334 lrf=0.1535 cls=$CLS \
  device=$GPU project="$PROJ" name=${NAME}_s2 exist_ok=True

echo "[$(date)] --- $NAME test ---"
yolo detect val model="$PROJ/${NAME}_s2/weights/best.pt" data="$DATA2" split=test imgsz=$IMGSZ \
  batch=$BATCH device=$GPU project="$PROJ" name=${NAME}_test exist_ok=True
echo "[$(date)] ### $NAME DONE"
