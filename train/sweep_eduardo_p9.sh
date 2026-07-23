#!/usr/bin/env bash
# Phase 9: cross-model 640 resolution arms (companions to phase 8's v11n deg15@640).
#   A: EDU_v8deg15_640 — YOLOv8n-OBB, champ+osall+deg15 @640 (same recipe as v11 arm)
#   B: EDU_v5champ_640 — YOLOv5n det champ+osall @640 (no OBB checkpoint for v5;
#      no rotation — deg augs hurt detection mode in the ablation)
# Waits for phase 8. 2 conditions x 5 folds = 10 runs. GPUs 3-7. 2-stage TL (150+100).
set -uo pipefail
ROOT="$HOME/atli"
DET="$ROOT/CV_eduardo_det"; OBB="$ROOT/CV_eduardo_obb"
EXT="$ROOT/run_config_ext.sh"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p9_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

echo "P9_WAITING_FOR_P8 $(date)" > "$ST"
while ! grep -q SWEEP_P8_DONE "$ROOT/sweep_eduardo_p8_status.txt" 2>/dev/null; do sleep 120; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P9_START $(date)" >> "$ST"

run(){ local runner=$1 model=$2 name=$3 yaml=$4 gpu=$5 extra=$6
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu model=$model" >>"$ST"
  MODEL="$model" DATA="$yaml" EXTRA="$extra" bash "$runner" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P9A_v8deg15_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run "$OBBR" yolov8n-obb.pt "EDU_v8deg15_640_f$k" "$OBB/fold$k/osall.yaml" "${GPUS[$k]}" "scale=0.9 degrees=15" & done; wait

echo "P9B_v5champ_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run "$EXT" yolov5nu.pt "EDU_v5champ_640_f$k" "$DET/fold$k/osall.yaml" "${GPUS[$k]}" "scale=0.9" & done; wait

echo "SWEEP_P9_DONE $(date)" >>"$ST"
