#!/usr/bin/env bash
# Phase 8 (v11n OBB): resolution ablation — the deg15 champion recipe at imgsz 640.
# Isolates the hi-res lever on the eduardo CV: deg15 @1280 = 0.793 vs same recipe @640.
# 1 condition x 5 folds = 5 runs. GPUs 3-7. 2-stage TL (150+100), batch 16.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p8_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P8_START $(date)" > "$ST"

run_obb(){ local name=$1 fold=$2 gpu=$3
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu" >>"$ST"
  MODEL=yolo11n-obb.pt DATA="$OBB/fold$fold/osall.yaml" EXTRA="scale=0.9 degrees=15" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P8_deg15_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_deg15_640_v11_f$k" "$k" "${GPUS[$k]}" & done; wait

echo "SWEEP_P8_DONE $(date)" >>"$ST"
