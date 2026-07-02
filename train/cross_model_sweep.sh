#!/usr/bin/env bash
# Cross-model sweep: HROaug champion recipe + baselines on v5/v8/v11.
# 8 GPUs in parallel, single-GPU each.
#
# Already done:  B_v11 (11 seeds), HROaug_v11 (11 seeds),
#                HROaug_v8 (5 seeds), B_v8 (1 seed)
#
# This script runs:
#   GPU 0: B_v5_s1        — baseline 640, native, 150+100
#   GPU 1: B_v5_s2
#   GPU 2: B_v5_s3
#   GPU 3: HROaug_v5_s1   — champion: 1280, OS3, scale=0.9, 150+100
#   GPU 4: HROaug_v5_s2
#   GPU 5: HROaug_v5_s3
#   GPU 6: B_v8_s2         — extra v8 baseline seed
#   GPU 7: B_v8_s3         — extra v8 baseline seed
set -uo pipefail
ROOT="$HOME/atli"
NATIVE="$ROOT/Merged_Dataset_Stratified/merged_stratified.yaml"
OS3="$ROOT/Merged_Native_OS3/merged_native_os3.yaml"
V3="$ROOT/run_config_v3.sh"
EXT="$ROOT/run_config_ext.sh"
STATUS="$ROOT/cross_model_status.txt"

echo "CROSS_MODEL_SWEEP_START $(date)" > "$STATUS"

# --- Baselines: v5 ×3  (run_config_v3: NAME FW GPU EP1 EP2 IMGSZ BATCH) ---
( DATA=$NATIVE bash "$V3" B_v5_s1 v5 0 150 100 640 32 ) > "$ROOT/B_v5_s1.log" 2>&1 &
P0=$!
( DATA=$NATIVE bash "$V3" B_v5_s2 v5 1 150 100 640 32 ) > "$ROOT/B_v5_s2.log" 2>&1 &
P1=$!
( DATA=$NATIVE bash "$V3" B_v5_s3 v5 2 150 100 640 32 ) > "$ROOT/B_v5_s3.log" 2>&1 &
P2=$!

# --- HROaug champion: v5 ×3  (run_config_ext: NAME GPU EP1 EP2 IMGSZ BATCH) ---
( MODEL=yolov5nu.pt EXTRA="scale=0.9" DATA=$OS3 bash "$EXT" HROaug_v5_s1 3 150 100 1280 16 ) > "$ROOT/HROaug_v5_s1.log" 2>&1 &
P3=$!
( MODEL=yolov5nu.pt EXTRA="scale=0.9" DATA=$OS3 bash "$EXT" HROaug_v5_s2 4 150 100 1280 16 ) > "$ROOT/HROaug_v5_s2.log" 2>&1 &
P4=$!
( MODEL=yolov5nu.pt EXTRA="scale=0.9" DATA=$OS3 bash "$EXT" HROaug_v5_s3 5 150 100 1280 16 ) > "$ROOT/HROaug_v5_s3.log" 2>&1 &
P5=$!

# --- Extra v8 baseline seeds ---
( DATA=$NATIVE bash "$V3" B_v8_s2 v8 6 150 100 640 32 ) > "$ROOT/B_v8_s2.log" 2>&1 &
P6=$!
( DATA=$NATIVE bash "$V3" B_v8_s3 v8 7 150 100 640 32 ) > "$ROOT/B_v8_s3.log" 2>&1 &
P7=$!

echo "All 8 jobs launched: B_v5(×3) HROaug_v5(×3) B_v8(×2)  $(date)" >> "$STATUS"
wait $P0 $P1 $P2 $P3 $P4 $P5 $P6 $P7
echo "CROSS_MODEL_SWEEP_DONE $(date)" >> "$STATUS"
