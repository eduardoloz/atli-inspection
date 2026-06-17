#!/usr/bin/env bash
# Flexible runner. args: NAME GPU EP1 EP2 IMGSZ BATCH ; env: MODEL, PRETRAINED(opt), EXTRA(opt), DATA
set -euo pipefail
NAME="$1";GPU="$2";EP1="$3";EP2="$4";IMGSZ="$5";BATCH="$6"
ROOT="$HOME/atli";YAML="${DATA:?}";PROJ="$ROOT/runs";MODEL="${MODEL:?}";PRE="${PRETRAINED:-}";EXTRA="${EXTRA:-}"
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
PREARG=""; [ -n "$PRE" ] && PREARG="pretrained=$PRE"
echo "[$(date)] ### $NAME model=$MODEL pre=$PRE extra=[$EXTRA] data=$(basename $YAML)"
yolo detect train model="$MODEL" data="$YAML" imgsz=$IMGSZ batch=$BATCH epochs=$EP1 optimizer=SGD lr0=0.01 $PREARG $EXTRA device=$GPU project="$PROJ" name=${NAME}_s1 exist_ok=True
yolo detect train model="$PROJ/${NAME}_s1/weights/best.pt" data="$YAML" imgsz=$IMGSZ batch=$BATCH epochs=$EP2 workers=8 optimizer=SGD lr0=0.00334 lrf=0.1535 $EXTRA device=$GPU project="$PROJ" name=${NAME}_s2 exist_ok=True
yolo detect val model="$PROJ/${NAME}_s2/weights/best.pt" data="$YAML" split=test imgsz=$IMGSZ batch=$BATCH device=$GPU project="$PROJ" name=${NAME}_test exist_ok=True
echo "[$(date)] ### $NAME DONE"
