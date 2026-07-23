#!/usr/bin/env bash
# Phase 7 (v11n OBB): does shear compose with the deg15 winner?
#   deg15shear10 — osall + degrees=15 + shear=10 (phase-3 arms combined:
#   deg15 was best mAP 0.793, shear10 alone 0.779; both beat no-geo-aug 0.759)
# 1 condition x 5 folds = 5 runs. GPUs 3-7. 2-stage TL (150+100), imgsz 1280, batch 16.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p7_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P7_START $(date)" > "$ST"
python -c "from ultralytics import YOLO; YOLO('yolo11n-obb.pt')" >> "$ST" 2>&1

run_obb(){ local name=$1 yaml=$2 gpu=$3 extra=$4; shift 4
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra] env=[$*]" >>"$ST"
  env "$@" MODEL=yolo11n-obb.pt DATA="$yaml" EXTRA="$extra" bash "$OBBR" "$name" "$gpu" 150 100 1280 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P7_deg15shear10 $(date)" >>"$ST"
for k in 0 1 2 3 4; do
  run_obb "EDU_deg15shear10_v11_f$k" "$OBB/fold$k/osall.yaml" "${GPUS[$k]}" "scale=0.9 degrees=15 shear=10" &
done; wait

echo "SWEEP_P7_DONE $(date)" >>"$ST"
