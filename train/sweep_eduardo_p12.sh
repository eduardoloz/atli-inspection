#!/usr/bin/env bash
# Phase 12: fill the two runnable gaps in the YOLO-family comparison chart @1280.
#   A: EDU_v8champ  — YOLOv8n detection champion (det + osall, scale=0.9)
#   B: EDU_v8deg15  — YOLOv8n OBB + osall + deg15 (the "best recipe" tier)
# (v5n OBB tiers are impossible: Ultralytics has no YOLOv5 OBB variant.)
# Waits for phase 11. 2 conditions x 5 folds = 10 runs. GPUs 3-7. 150+100, batch 16.
set -uo pipefail
ROOT="$HOME/atli"
DET="$ROOT/CV_eduardo_det"; OBB="$ROOT/CV_eduardo_obb"
EXT="$ROOT/run_config_ext.sh"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p12_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

echo "P12_WAITING_FOR_P11 $(date)" > "$ST"
while ! grep -q SWEEP_P11_DONE "$ROOT/sweep_eduardo_p11_status.txt" 2>/dev/null; do sleep 120; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P12_START $(date)" >> "$ST"

run(){ local runner=$1 model=$2 name=$3 yaml=$4 gpu=$5 extra=$6
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu model=$model" >>"$ST"
  MODEL="$model" DATA="$yaml" EXTRA="$extra" bash "$runner" "$name" "$gpu" 150 100 1280 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P12A_v8champ_det $(date)" >>"$ST"
for k in 0 1 2 3 4; do run "$EXT" yolov8n.pt "EDU_v8champ_f$k" "$DET/fold$k/osall.yaml" "${GPUS[$k]}" "scale=0.9" & done; wait

echo "P12B_v8deg15_obb $(date)" >>"$ST"
for k in 0 1 2 3 4; do run "$OBBR" yolov8n-obb.pt "EDU_v8deg15_f$k" "$OBB/fold$k/osall.yaml" "${GPUS[$k]}" "scale=0.9 degrees=15" & done; wait

echo "SWEEP_P12_DONE $(date)" >>"$ST"
