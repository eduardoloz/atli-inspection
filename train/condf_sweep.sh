#!/usr/bin/env bash
# Condition F: Eduardo Full minus 4 hard-negative Defective_Damper images.
# Single-GPU: v5n, v8n, v11n @ 150+100.
set -uo pipefail
ROOT=$HOME/atli
RC=~/run_config_single.sh

echo "Building Condition F dataset..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
python3 ~/.atli_build_condition_f.py || { echo "BUILD FAILED"; exit 1; }

CF=$ROOT/abl_condition_f.yaml
echo "CONDF_START $(date)" > "$ROOT/condf_status.txt"

DATA=$CF bash "$RC" v5_condf  v5  SGD 150 0 100 > "$ROOT/condf_v5.log" 2>&1 &
DATA=$CF bash "$RC" v8_condf  v8  SGD 150 1 100 > "$ROOT/condf_v8.log" 2>&1 &
DATA=$CF bash "$RC" v11_condf v11 SGD 150 2 100 > "$ROOT/condf_v11.log" 2>&1 &
wait
echo "CONDF_DONE $(date)" >> "$ROOT/condf_status.txt"

# Eval on shared test set
echo "Running eval..."
YAML=$ROOT/abl_target_only.yaml
cd "$ROOT/yolov5"
CLS='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'
mkdir -p "$ROOT/eval"
for c in v5_condf v8_condf v11_condf; do
  w="$ROOT/runs/${c}_s2/weights/best.pt"
  echo "=== EVAL $c ==="
  [ -f "$w" ] || { echo "MISSING $w"; continue; }
  case "$c" in
    v5_*)
      CUDA_VISIBLE_DEVICES=0 python val.py --data "$YAML" --weights "$w" --img 640 \
        --batch 8 --task test --device 0 --project "$ROOT/eval" --name "$c" --exist-ok 2>&1 ;;
    *)
      CUDA_VISIBLE_DEVICES=0 yolo detect val model="$w" data="$YAML" split=test imgsz=640 \
        batch=8 device=0 project="$ROOT/eval" name="$c" exist_ok=True 2>&1 ;;
  esac | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLS)[[:space:]]+[0-9]" | tee "$ROOT/eval/$c.txt"
done
echo "CONDF_EVAL_DONE"
