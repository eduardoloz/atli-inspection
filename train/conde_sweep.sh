#!/usr/bin/env bash
# Condition E: Eduardo Full, keep "can't tell" defective images, remove only bad ones.
# Single-GPU: v5n, v8n, v11n @ 150+100.
set -uo pipefail
ROOT=$HOME/atli
RC=~/run_config_single.sh

echo "Building Condition E dataset..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
python3 ~/.atli_build_condition_e.py || { echo "BUILD FAILED"; exit 1; }

CF=$ROOT/abl_target_plus_eduardo_full_keepcanttell.yaml
echo "CONDE_START $(date)" > "$ROOT/conde_status.txt"

DATA=$CF bash "$RC" v5_tpef_keepct  v5  SGD 150 0 100 > "$ROOT/conde_v5.log" 2>&1 &
DATA=$CF bash "$RC" v8_tpef_keepct  v8  SGD 150 1 100 > "$ROOT/conde_v8.log" 2>&1 &
DATA=$CF bash "$RC" v11_tpef_keepct v11 SGD 150 2 100 > "$ROOT/conde_v11.log" 2>&1 &
wait
echo "CONDE_DONE $(date)" >> "$ROOT/conde_status.txt"

# Eval on shared test set
echo "Running eval..."
YAML=$ROOT/abl_target_only.yaml
cd "$ROOT/yolov5"
CLS='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'
for c in v5_tpef_keepct v8_tpef_keepct v11_tpef_keepct; do
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
echo "CONDE_EVAL_DONE"
