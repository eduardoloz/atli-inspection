#!/usr/bin/env bash
# Run one ATLI benchmark config: stage1 train -> stage2 finetune -> test.
#   args: NAME FRAMEWORK(v5|v8|v11) OPT(SGD|Adam) EP1 GPUS(csv) MASTER_PORT [EP2=100]
# Uses DATA env var for the dataset yaml (defaults to merged_stratified).
# Uses NCCL_TIMEOUT env var (seconds) for DDP timeout (default 7200 = 2hr).
set -euo pipefail
NAME="$1"; FW="$2"; OPT="$3"; EP1="$4"; GPUS="$5"; PORT="$6"; EP2="${7:-100}"
ROOT="$HOME/atli"
YAML="${DATA:-$ROOT/Merged_Dataset_Stratified/merged_stratified.yaml}"
PROJ="$ROOT/runs"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
mkdir -p "$PROJ"
N=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
BATCH=$((8 * N))                       # 8 imgs/GPU
DEV0=$(echo "$GPUS" | cut -d, -f1)

# Increase NCCL timeout to avoid the DDP crash we saw with v8
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_BLOCKING_WAIT=0

# Map framework to pretrained weights
case "$FW" in
  v5)  WEIGHTS="yolov5n.pt" ;;
  v8)  WEIGHTS="yolov8n.pt" ;;
  v11) WEIGHTS="yolo11n.pt" ;;
  *)   echo "Unknown framework: $FW"; exit 1 ;;
esac

echo "[$(date)] ### $NAME  ($FW $OPT  s1=${EP1}ep s2=${EP2}ep)  GPUs=$GPUS N=$N batch=$BATCH"

if [ "$FW" = "v5" ]; then
  cd "$ROOT/yolov5"
  echo "[$(date)] --- $NAME stage1 ---"
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N --master_port=$PORT train.py \
    --img 640 --batch $BATCH --epochs $EP1 --optimizer $OPT \
    --data "$YAML" --weights "$WEIGHTS" \
    --project "$PROJ" --name ${NAME}_s1 --exist-ok
  echo "[$(date)] --- $NAME stage2 (finetune ${EP2}ep) ---"
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N --master_port=$PORT train.py \
    --img 640 --batch $BATCH --epochs $EP2 --workers 8 --optimizer SGD \
    --data "$YAML" --weights "$PROJ/${NAME}_s1/weights/best.pt" --hyp hyp.oscar_paper.yaml \
    --project "$PROJ" --name ${NAME}_s2 --exist-ok
  echo "[$(date)] --- $NAME test ---"
  CUDA_VISIBLE_DEVICES=$DEV0 python val.py --data "$YAML" \
    --weights "$PROJ/${NAME}_s2/weights/best.pt" --img 640 --batch 8 --task test \
    --device 0 --project "$PROJ" --name ${NAME}_test --exist-ok
else
  # v8 and v11 both use the ultralytics CLI
  echo "[$(date)] --- $NAME stage1 ---"
  yolo detect train model="$WEIGHTS" data="$YAML" imgsz=640 batch=$BATCH epochs=$EP1 \
    optimizer=SGD lr0=0.01 device=$GPUS project="$PROJ" name=${NAME}_s1 exist_ok=True
  echo "[$(date)] --- $NAME stage2 (finetune ${EP2}ep) ---"
  yolo detect train model="$PROJ/${NAME}_s1/weights/best.pt" data="$YAML" imgsz=640 \
    batch=$BATCH epochs=$EP2 workers=8 optimizer=SGD lr0=0.00334 lrf=0.1535 device=$GPUS \
    project="$PROJ" name=${NAME}_s2 exist_ok=True
  echo "[$(date)] --- $NAME test ---"
  yolo detect val model="$PROJ/${NAME}_s2/weights/best.pt" data="$YAML" split=test imgsz=640 \
    batch=8 device=$DEV0 project="$PROJ" name=${NAME}_test exist_ok=True
fi
echo "[$(date)] ### $NAME DONE"
