#!/usr/bin/env bash
# Single-GPU config with class-weighted loss (cls parameter upweighted).
#   args: NAME FRAMEWORK(v5|v8|v11) OPT(SGD|Adam) EP1 GPU [EP2=100] [CLS_WT=3.0]
# Uses DATA env var for the dataset yaml.
set -euo pipefail
NAME="$1"; FW="$2"; OPT="$3"; EP1="$4"; GPU="$5"; EP2="${6:-100}"; CLS_WT="${7:-3.0}"
ROOT="$HOME/atli"
YAML="${DATA:-$ROOT/Merged_Dataset_Stratified/merged_stratified.yaml}"
PROJ="$ROOT/runs"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
mkdir -p "$PROJ"
BATCH=8

case "$FW" in
  v5)  WEIGHTS="yolov5n.pt" ;;
  v8)  WEIGHTS="yolov8n.pt" ;;
  v11) WEIGHTS="yolo11n.pt" ;;
  *)   echo "Unknown framework: $FW"; exit 1 ;;
esac

echo "[$(date)] ### $NAME  ($FW $OPT  s1=${EP1}ep s2=${EP2}ep cls=$CLS_WT)  GPU=$GPU"

if [ "$FW" = "v5" ]; then
  cd "$ROOT/yolov5"
  # YOLOv5: cls loss weight is set via --hyp or default hyp
  # Create a custom hyp file with boosted cls weight
  cat > "/tmp/hyp_cls${CLS_WT}.yaml" <<YAMLEOF
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: ${CLS_WT}
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
YAMLEOF

  echo "[$(date)] --- $NAME stage1 (cls=$CLS_WT) ---"
  CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --img 640 --batch $BATCH --epochs $EP1 --optimizer $OPT \
    --data "$YAML" --weights "$WEIGHTS" --hyp "/tmp/hyp_cls${CLS_WT}.yaml" \
    --project "$PROJ" --name ${NAME}_s1 --exist-ok --device 0

  # Stage 2: use oscar hyp but with boosted cls
  cat > "/tmp/hyp_oscar_cls${CLS_WT}.yaml" <<YAMLEOF
lr0: 0.00334
lrf: 0.1535
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: ${CLS_WT}
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
YAMLEOF

  echo "[$(date)] --- $NAME stage2 (finetune ${EP2}ep, cls=$CLS_WT) ---"
  CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --img 640 --batch $BATCH --epochs $EP2 --workers 8 --optimizer SGD \
    --data "$YAML" --weights "$PROJ/${NAME}_s1/weights/best.pt" \
    --hyp "/tmp/hyp_oscar_cls${CLS_WT}.yaml" \
    --project "$PROJ" --name ${NAME}_s2 --exist-ok --device 0

  echo "[$(date)] --- $NAME test ---"
  CUDA_VISIBLE_DEVICES=$GPU python val.py --data "$YAML" \
    --weights "$PROJ/${NAME}_s2/weights/best.pt" --img 640 --batch 8 --task test \
    --device 0 --project "$PROJ" --name ${NAME}_test --exist-ok
else
  echo "[$(date)] --- $NAME stage1 (cls=$CLS_WT) ---"
  yolo detect train model="$WEIGHTS" data="$YAML" imgsz=640 \
    batch=$BATCH epochs=$EP1 optimizer=SGD lr0=0.01 cls=$CLS_WT device=$GPU \
    project="$PROJ" name=${NAME}_s1 exist_ok=True

  echo "[$(date)] --- $NAME stage2 (finetune ${EP2}ep, cls=$CLS_WT) ---"
  yolo detect train model="$PROJ/${NAME}_s1/weights/best.pt" \
    data="$YAML" imgsz=640 batch=$BATCH epochs=$EP2 workers=8 optimizer=SGD \
    lr0=0.00334 lrf=0.1535 cls=$CLS_WT device=$GPU \
    project="$PROJ" name=${NAME}_s2 exist_ok=True

  echo "[$(date)] --- $NAME test ---"
  yolo detect val model="$PROJ/${NAME}_s2/weights/best.pt" \
    data="$YAML" split=test imgsz=640 batch=8 device=$GPU \
    project="$PROJ" name=${NAME}_test exist_ok=True
fi
echo "[$(date)] ### $NAME DONE"
