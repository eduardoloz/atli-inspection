#!/usr/bin/env bash
# Condition G: Undersample Normal_Damper (ratio ~1:5)
# Condition H: Class-weighted loss (cls=3.0) on Condition C dataset
# Single-GPU, v5n + v8n + v11n each.
set -uo pipefail
ROOT=$HOME/atli
RC=~/run_config_single.sh
RC_CLS=~/run_config_clswt.sh
STATUS=$ROOT/condgh_status.txt

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"

# Build Condition G dataset
echo "Building Condition G (undersample)..."
python3 ~/.atli_build_condition_g.py || { echo "BUILD G FAILED"; exit 1; }

echo "CONDGH_START $(date)" > "$STATUS"

GF=$ROOT/abl_condition_g.yaml
CF=$ROOT/abl_target_plus_eduardo_full.yaml

# Wave 1: G (undersample) on GPUs 0-2, H (cls weight) on GPUs 5-7
# (GPUs 3-5 may still be running RT-DETR)

# Condition G: undersample Normal_Damper
DATA=$GF bash "$RC" v5_condg  v5  SGD 150 0 100 > "$ROOT/condg_v5.log" 2>&1 &
DATA=$GF bash "$RC" v8_condg  v8  SGD 150 1 100 > "$ROOT/condg_v8.log" 2>&1 &
DATA=$GF bash "$RC" v11_condg v11 SGD 150 2 100 > "$ROOT/condg_v11.log" 2>&1 &

# Condition H: class-weighted loss (cls=3.0) on full Eduardo (C dataset)
# GPUs 3-5 may have RT-DETR, use 6-7 and wait for G to free 0
DATA=$CF bash "$RC_CLS" v5_condh  v5  SGD 150 6 100 3.0 > "$ROOT/condh_v5.log" 2>&1 &
DATA=$CF bash "$RC_CLS" v8_condh  v8  SGD 150 7 100 3.0 > "$ROOT/condh_v8.log" 2>&1 &
# v11 H waits — will run after G finishes on GPU 0
wait
DATA=$CF bash "$RC_CLS" v11_condh v11 SGD 150 0 100 3.0 > "$ROOT/condh_v11.log" 2>&1
echo "CONDH_V11_DONE $(date)" >> "$STATUS"

wait
echo "CONDGH_DONE $(date)" >> "$STATUS"

# Eval all on shared test set
echo "Running eval..."
YAML=$ROOT/abl_target_only.yaml
cd "$ROOT/yolov5"
CLS='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'
mkdir -p "$ROOT/eval"
for c in v5_condg v8_condg v11_condg v5_condh v8_condh v11_condh; do
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
echo "CONDGH_EVAL_DONE"
