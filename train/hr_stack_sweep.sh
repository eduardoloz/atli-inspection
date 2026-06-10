#!/usr/bin/env bash
# Defective_Damper recall push:
#   v11_A_hr  — v11n, dataset A, imgsz=1280 (resolution test)
#   A_T_hr    — RT-DETR-l, dataset A, imgsz=960
#   v8_stack  — v8n, dataset G (strip ND) + cls=3.0  (combine the two partial wins)
#   v11_stack — v11n, dataset G + cls=3.0
#   TTA evals — v11n A, v11n C, RT-DETR E_T with augment=True (no training)
set -uo pipefail
ROOT=$HOME/atli
TO=$ROOT/abl_target_only.yaml
GF=$ROOT/abl_condition_g.yaml
RC=~/run_config_v3.sh
STATUS=$ROOT/hr_stack_status.txt
CLSRE='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'

echo "HRSTACK_START $(date)" > "$STATUS"

DATA=$TO bash "$RC" v11_A_hr  v11    0 150 100 1280 4 0.5 > "$ROOT/hr_v11_A.log" 2>&1 &
DATA=$TO bash "$RC" A_T_hr    rtdetr 1 150 100 960  2 0.5 > "$ROOT/hr_AT.log"   2>&1 &
DATA=$GF bash "$RC" v8_stack  v8     2 150 100 640  8 3.0 > "$ROOT/stack_v8.log" 2>&1 &
DATA=$GF bash "$RC" v11_stack v11    3 150 100 640  8 3.0 > "$ROOT/stack_v11.log" 2>&1 &

# TTA evals on GPU 4 — quick, no training
(
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate "$ROOT/env"
  mkdir -p "$ROOT/eval"
  yolo detect val model="$ROOT/runs/v11_to_s2/weights/best.pt" data="$TO" split=test \
    imgsz=640 batch=8 device=4 augment=True project="$ROOT/eval" name=v11_A_tta exist_ok=True 2>&1 \
    | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLSRE)[[:space:]]+[0-9]" > "$ROOT/eval/v11_A_tta.txt"
  yolo detect val model="$ROOT/runs/v11_tpef_s2/weights/best.pt" data="$TO" split=test \
    imgsz=640 batch=8 device=4 augment=True project="$ROOT/eval" name=v11_C_tta exist_ok=True 2>&1 \
    | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLSRE)[[:space:]]+[0-9]" > "$ROOT/eval/v11_C_tta.txt"
  yolo detect val model="$ROOT/runs/E_T_s2/weights/best.pt" data="$TO" split=test \
    imgsz=640 batch=4 device=4 augment=True project="$ROOT/eval" name=E_T_tta exist_ok=True 2>&1 \
    | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLSRE)[[:space:]]+[0-9]" > "$ROOT/eval/E_T_tta.txt"
  echo "TTA_DONE $(date)" >> "$STATUS"
) > "$ROOT/tta.log" 2>&1 &

wait
echo "HRSTACK_DONE $(date)" >> "$STATUS"

# Final evals on shared test set at each model's native resolution
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
for spec in "v11_A_hr 1280 4" "A_T_hr 960 2" "v8_stack 640 8" "v11_stack 640 8"; do
  set -- $spec; c=$1; sz=$2; b=$3
  w="$ROOT/runs/${c}_s2/weights/best.pt"
  echo "=== EVAL $c (imgsz=$sz) ==="
  [ -f "$w" ] || { echo "MISSING $w"; continue; }
  yolo detect val model="$w" data="$TO" split=test imgsz=$sz batch=$b device=0 \
    project="$ROOT/eval" name="$c" exist_ok=True 2>&1 \
    | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLSRE)[[:space:]]+[0-9]" | tee "$ROOT/eval/$c.txt"
done
echo "HRSTACK_EVAL_DONE $(date)" >> "$STATUS"
