#!/usr/bin/env bash
# Re-evaluate the 4 ablation models on the SHARED Eduardo-free test set (clean numbers).
set -uo pipefail
ROOT=$HOME/atli
YAML=$ROOT/abl_target_only.yaml      # its test/ == the shared Eduardo-free test set
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
cd "$ROOT/yolov5"
mkdir -p "$ROOT/eval"
DEV=1
CLS='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'
for c in abl_v5_to abl_v5_tpe abl_v5_tpef abl_v8_to abl_v8_tpe abl_v8_tpef; do
  w="$ROOT/runs/${c}_s2/weights/best.pt"
  echo "=== EVAL $c ==="
  [ -f "$w" ] || { echo "MISSING $w"; continue; }
  case "$c" in
    abl_v5*) CUDA_VISIBLE_DEVICES=$DEV python val.py --data "$YAML" --weights "$w" --img 640 \
               --batch 8 --task test --device 0 --project "$ROOT/eval" --name "$c" --exist-ok 2>&1 ;;
    abl_v8_*) CUDA_VISIBLE_DEVICES=$DEV yolo detect val model="$w" data="$YAML" split=test imgsz=640 \
               batch=8 device=0 project="$ROOT/eval" name="$c" exist_ok=True 2>&1 ;;
  esac | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLS)[[:space:]]+[0-9]" > "$ROOT/eval/$c.txt"
  cat "$ROOT/eval/$c.txt"
done
echo ABL_EVAL_DONE
