#!/usr/bin/env bash
# Authoritative re-evaluation: run test-split val on every config's stage-2 best.pt
# and save the clean per-class metric table to ~/atli/eval/<config>.txt
set -uo pipefail
ROOT=$HOME/atli
YAML=$ROOT/Merged_Dataset_Stratified/merged_stratified.yaml
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
cd "$ROOT/yolov5"
mkdir -p "$ROOT/eval"
DEV=1   # GPU0 has another user's process parked on it; use GPU1
CLS='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'
for c in v5n_sgd_150 v5n_sgd_300 v5n_adam_150 v5n_adam_300 v8n_sgd_150 v8n_sgd_300; do
  w="$ROOT/runs/${c}_s2/weights/best.pt"
  echo "=== EVAL $c ==="
  [ -f "$w" ] || { echo "MISSING $w"; continue; }
  case "$c" in
    v5n_*) CUDA_VISIBLE_DEVICES=$DEV python val.py --data "$YAML" --weights "$w" --img 640 \
             --batch 8 --task test --device 0 --project "$ROOT/eval" --name "$c" --exist-ok 2>&1 ;;
    v8n_*) CUDA_VISIBLE_DEVICES=$DEV yolo detect val model="$w" data="$YAML" split=test imgsz=640 \
             batch=8 device=0 project="$ROOT/eval" name="$c" exist_ok=True 2>&1 ;;
  esac | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLS)[[:space:]]+[0-9]" > "$ROOT/eval/$c.txt"
  cat "$ROOT/eval/$c.txt"
done
echo EVAL_ALL_DONE
