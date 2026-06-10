#!/usr/bin/env bash
# Evaluate all sweep_v2 models on the shared Eduardo-free test set.
set -uo pipefail
ROOT=$HOME/atli
YAML=$ROOT/abl_target_only.yaml      # its test/ == shared Eduardo-free test set
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
cd "$ROOT/yolov5"
mkdir -p "$ROOT/eval"
DEV=1
CLS='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'

CONFIGS=(
  # 150+100 re-runs and new
  v8_tpe_r2 v8_tpef_r2
  v11_to v11_tpe v11_tpef
  # 150+200 longer fine-tune
  v5_to_ft200 v5_tpe_ft200 v5_tpef_ft200
  v8_to_ft200 v8_tpe_ft200 v8_tpef_ft200
  v11_to_ft200 v11_tpe_ft200 v11_tpef_ft200
)

for c in "${CONFIGS[@]}"; do
  w="$ROOT/runs/${c}_s2/weights/best.pt"
  echo "=== EVAL $c ==="
  [ -f "$w" ] || { echo "MISSING $w"; continue; }
  case "$c" in
    v5_*)
      CUDA_VISIBLE_DEVICES=$DEV python val.py --data "$YAML" --weights "$w" --img 640 \
        --batch 8 --task test --device 0 --project "$ROOT/eval" --name "$c" --exist-ok 2>&1 ;;
    v8_*|v11_*)
      CUDA_VISIBLE_DEVICES=$DEV yolo detect val model="$w" data="$YAML" split=test imgsz=640 \
        batch=8 device=0 project="$ROOT/eval" name="$c" exist_ok=True 2>&1 ;;
  esac | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLS)[[:space:]]+[0-9]" | tee "$ROOT/eval/$c.txt"
done
echo "SWEEP_V2_EVAL_DONE"
