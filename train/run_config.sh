#!/usr/bin/env bash
# Reproduce one Untitled2.ipynb benchmark config: stage1 train -> stage2 finetune -> test.
#   args: NAME FRAMEWORK(v5|v8) OPT(SGD|Adam) EP1 GPUS(csv) MASTER_PORT [EP2=100]
set -euo pipefail
NAME="$1"; FW="$2"; OPT="$3"; EP1="$4"; GPUS="$5"; PORT="$6"; EP2="${7:-100}"
ROOT="$HOME/atli"
YAML="${DATA:-$ROOT/Merged_Dataset_Stratified/merged_stratified.yaml}"
PROJ="$ROOT/runs"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
mkdir -p "$PROJ"
N=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
BATCH=$((8 * N))                       # 8 imgs/GPU, matches the paper/notebook
DEV0=$(echo "$GPUS" | cut -d, -f1)
cd "$ROOT/yolov5"

echo "[$(date)] ### $NAME  ($FW $OPT  s1=${EP1}ep s2=${EP2}ep)  GPUs=$GPUS N=$N batch=$BATCH"

if [ "$FW" = "v5" ]; then
  echo "[$(date)] --- $NAME stage1 ---"
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N --master_port=$PORT train.py \
    --img 640 --batch $BATCH --epochs $EP1 --optimizer $OPT \
    --data "$YAML" --weights yolov5n.pt \
    --project "$PROJ" --name ${NAME}_s1 --exist-ok
  echo "[$(date)] --- $NAME stage2 (finetune, SGD, oscar hyp) ---"
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N --master_port=$PORT train.py \
    --img 640 --batch $BATCH --epochs $EP2 --workers 8 --optimizer SGD \
    --data "$YAML" --weights "$PROJ/${NAME}_s1/weights/best.pt" --hyp hyp.oscar_paper.yaml \
    --project "$PROJ" --name ${NAME}_s2 --exist-ok
  echo "[$(date)] --- $NAME test ---"
  CUDA_VISIBLE_DEVICES=$DEV0 python val.py --data "$YAML" \
    --weights "$PROJ/${NAME}_s2/weights/best.pt" --img 640 --batch 8 --task test \
    --device 0 --project "$PROJ" --name ${NAME}_test --exist-ok
else
  echo "[$(date)] --- $NAME stage1 ---"
  yolo detect train model=yolov8n.pt data="$YAML" imgsz=640 batch=$BATCH epochs=$EP1 \
    optimizer=SGD lr0=0.01 device=$GPUS project="$PROJ" name=${NAME}_s1 exist_ok=True
  echo "[$(date)] --- $NAME stage2 (finetune) ---"
  yolo detect train model="$PROJ/${NAME}_s1/weights/best.pt" data="$YAML" imgsz=640 \
    batch=$BATCH epochs=$EP2 workers=8 optimizer=SGD lr0=0.00334 lrf=0.1535 device=$GPUS \
    project="$PROJ" name=${NAME}_s2 exist_ok=True
  echo "[$(date)] --- $NAME test ---"
  yolo detect val model="$PROJ/${NAME}_s2/weights/best.pt" data="$YAML" split=test imgsz=640 \
    batch=8 device=$DEV0 project="$PROJ" name=${NAME}_test exist_ok=True
fi
echo "[$(date)] ### $NAME DONE"
