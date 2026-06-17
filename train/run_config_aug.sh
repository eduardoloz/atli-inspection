#!/usr/bin/env bash
# v3 recipe + adjustable scale-down augmentation. args: NAME FW GPU EP1 EP2 IMGSZ BATCH SCALE
set -euo pipefail
NAME="$1";FW="$2";GPU="$3";EP1="$4";EP2="$5";IMGSZ="$6";BATCH="$7";SCALE="$8"
ROOT="$HOME/atli";YAML="${DATA:?}";PROJ="$ROOT/runs"
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
case "$FW" in v8) W="yolov8n.pt";; v11) W="yolo11n.pt";; esac
echo "[$(date)] ### $NAME scale=$SCALE"
yolo detect train model="$W" data="$YAML" imgsz=$IMGSZ batch=$BATCH epochs=$EP1 optimizer=SGD lr0=0.01 scale=$SCALE device=$GPU project="$PROJ" name=${NAME}_s1 exist_ok=True
yolo detect train model="$PROJ/${NAME}_s1/weights/best.pt" data="$YAML" imgsz=$IMGSZ batch=$BATCH epochs=$EP2 workers=8 optimizer=SGD lr0=0.00334 lrf=0.1535 scale=$SCALE device=$GPU project="$PROJ" name=${NAME}_s2 exist_ok=True
yolo detect val model="$PROJ/${NAME}_s2/weights/best.pt" data="$YAML" split=test imgsz=$IMGSZ batch=$BATCH device=$GPU project="$PROJ" name=${NAME}_test exist_ok=True
echo "[$(date)] ### $NAME DONE"
