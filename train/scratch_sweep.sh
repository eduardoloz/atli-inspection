#!/usr/bin/env bash
# Scratch-vs-TL comparison on dataset A (target-only, shared Eduardo-free test set).
# Addresses the "scratch trained longer catches up to TL" critique with our own runs:
#   scratch — random init via model *.yaml (NOT *.pt): 150 / 300 / 600 ep, single stage
#   tl300   — TL 300+100 (stage-1 300 ep from COCO .pt + 100 ep fine-tune)
# Lanes: GPU4=v5 scratch, GPU5=v8 scratch, GPU6=v11 scratch, GPU7=TL 300+100 x3.
# GPU 1 is A_T_hr (RT-DETR); GPUs 0/2/3 left free for the hr_stack final eval.
set -uo pipefail
ROOT=$HOME/atli
YAML=$ROOT/abl_target_only.yaml
PROJ=$ROOT/runs
STATUS=$ROOT/scratch_status.txt
RC=~/run_config_single.sh
CLS='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
mkdir -p "$PROJ" "$ROOT/eval"
echo "SCRATCH_START $(date)" > "$STATUS"

train_scratch () {  # NAME FW EPOCHS GPU
  local NAME=$1 FW=$2 EP=$3 GPU=$4
  echo "[$(date)] ### $NAME ($FW scratch ${EP}ep) GPU=$GPU"
  if [ "$FW" = v5 ]; then
    cd "$ROOT/yolov5"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --img 640 --batch 8 --epochs "$EP" \
      --optimizer SGD --data "$YAML" --weights '' --cfg models/yolov5n.yaml \
      --project "$PROJ" --name "$NAME" --exist-ok --device 0
    CUDA_VISIBLE_DEVICES=$GPU python val.py --data "$YAML" \
      --weights "$PROJ/$NAME/weights/best.pt" --img 640 --batch 8 --task test \
      --device 0 --project "$ROOT/eval" --name "$NAME" --exist-ok 2>&1 \
      | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLS)[[:space:]]+[0-9]" > "$ROOT/eval/$NAME.txt"
  else
    local M=yolov8n.yaml; [ "$FW" = v11 ] && M=yolo11n.yaml
    yolo detect train model="$M" pretrained=False data="$YAML" imgsz=640 batch=8 \
      epochs="$EP" optimizer=SGD lr0=0.01 device="$GPU" \
      project="$PROJ" name="$NAME" exist_ok=True
    yolo detect val model="$PROJ/$NAME/weights/best.pt" data="$YAML" split=test \
      imgsz=640 batch=8 device="$GPU" project="$ROOT/eval" name="$NAME" exist_ok=True 2>&1 \
      | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLS)[[:space:]]+[0-9]" > "$ROOT/eval/$NAME.txt"
  fi
  echo "${NAME}_DONE $(date)" >> "$STATUS"
}

( train_scratch v5_scr150  v5  150 4
  train_scratch v5_scr300  v5  300 4
  train_scratch v5_scr600  v5  600 4 ) > "$ROOT/scratch_v5.log" 2>&1 &

( train_scratch v8_scr150  v8  150 5
  train_scratch v8_scr300  v8  300 5
  train_scratch v8_scr600  v8  600 5 ) > "$ROOT/scratch_v8.log" 2>&1 &

( train_scratch v11_scr150 v11 150 6
  train_scratch v11_scr300 v11 300 6
  train_scratch v11_scr600 v11 600 6 ) > "$ROOT/scratch_v11.log" 2>&1 &

( DATA=$YAML bash "$RC" v5_to300  v5  SGD 300 7 100
  echo "v5_to300_DONE $(date)"  >> "$STATUS"
  DATA=$YAML bash "$RC" v8_to300  v8  SGD 300 7 100
  echo "v8_to300_DONE $(date)"  >> "$STATUS"
  DATA=$YAML bash "$RC" v11_to300 v11 SGD 300 7 100
  echo "v11_to300_DONE $(date)" >> "$STATUS" ) > "$ROOT/tl300.log" 2>&1 &

wait
echo "TRAIN_ALL_DONE $(date)" >> "$STATUS"

# Scrape the TL 300+100 test evals into eval/*.txt like every other config
cd "$ROOT/yolov5"
for c in v5_to300 v8_to300 v11_to300; do
  w="$PROJ/${c}_s2/weights/best.pt"
  [ -f "$w" ] || { echo "MISSING $w"; continue; }
  case "$c" in
    v5_*)
      CUDA_VISIBLE_DEVICES=7 python val.py --data "$YAML" --weights "$w" --img 640 \
        --batch 8 --task test --device 0 --project "$ROOT/eval" --name "$c" --exist-ok 2>&1 ;;
    *)
      yolo detect val model="$w" data="$YAML" split=test imgsz=640 batch=8 device=7 \
        project="$ROOT/eval" name="$c" exist_ok=True 2>&1 ;;
  esac | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLS)[[:space:]]+[0-9]" > "$ROOT/eval/$c.txt"
done
echo "SCRATCH_SWEEP_DONE $(date)" >> "$STATUS"
