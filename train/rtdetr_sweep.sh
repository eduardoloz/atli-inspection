#!/usr/bin/env bash
# RT-DETR (transformer) sweep: A_T, C_T, E_T on conditions A, C, E.
# Single-GPU, batch=4 (RT-DETR is ~32M params, needs more VRAM).
# Runs 3 parallel jobs on GPUs 0, 1, 2.
set -uo pipefail
ROOT=$HOME/atli
RC=~/run_config_rtdetr.sh
TO=$ROOT/abl_target_only.yaml
CF=$ROOT/abl_target_plus_eduardo_full.yaml
EF=$ROOT/abl_target_plus_eduardo_full_keepcanttell.yaml
STATUS=$ROOT/rtdetr_status.txt

echo "RTDETR_START $(date)" > "$STATUS"

# A_T: Target only (same dataset as Condition A)
DATA=$TO bash "$RC" A_T 3 150 100 > "$ROOT/rtdetr_A_T.log" 2>&1 &

# C_T: Eduardo Full (same dataset as Condition C)
DATA=$CF bash "$RC" C_T 4 150 100 > "$ROOT/rtdetr_C_T.log" 2>&1 &

# E_T: Eduardo Full keep can't-tell (same dataset as Condition E)
DATA=$EF bash "$RC" E_T 5 150 100 > "$ROOT/rtdetr_E_T.log" 2>&1 &

wait
echo "RTDETR_DONE $(date)" >> "$STATUS"

# Eval all on shared test set
echo "Running eval..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
YAML=$ROOT/abl_target_only.yaml
CLS='all|Birdnest|Broken_Insulator|Defective_Damper|Flashover_Insulator|Normal_Damper|Normal_Insulators|Self-Exploded_Insulator'
mkdir -p "$ROOT/eval"
for c in A_T C_T E_T; do
  w="$ROOT/runs/${c}_s2/weights/best.pt"
  echo "=== EVAL $c ==="
  [ -f "$w" ] || { echo "MISSING $w"; continue; }
  yolo detect val model="$w" data="$YAML" split=test imgsz=640 \
    batch=4 device=0 project="$ROOT/eval" name="$c" exist_ok=True 2>&1 \
    | tr '\r' '\n' | grep -aE "^[[:space:]]*($CLS)[[:space:]]+[0-9]" | tee "$ROOT/eval/$c.txt"
done
echo "RTDETR_EVAL_DONE"
