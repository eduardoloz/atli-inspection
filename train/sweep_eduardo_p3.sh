#!/usr/bin/env bash
# Phase 3 (v11n OBB, NO CPLID): rotation sweep {15,25,30} + shear, on the existing
# CV_eduardo_obb folds (osall). Waits for phase 2 to finish, then runs on GPUs 3-7.
# 4 conditions x 5 folds = 20 runs. 2-stage TL (150+100).
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p3_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

echo "P3_WAITING_FOR_P2 $(date)" > "$ST"
while ! grep -q SWEEP_P2_DONE "$ROOT/sweep_eduardo_p2_status.txt" 2>/dev/null; do sleep 120; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P3_START $(date)" >> "$ST"
python -c "from ultralytics import YOLO; YOLO('yolo11n-obb.pt')" >> "$ST" 2>&1

run_obb(){ local name=$1 yaml=$2 imgsz=$3 gpu=$4 extra=${5:-}
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra]" >>"$ST"
  MODEL=yolo11n-obb.pt DATA="$yaml" EXTRA="$extra" bash "$OBBR" "$name" "$gpu" 150 100 "$imgsz" 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

# rotation sweep first (15/25/30), then shear — all on OBB champion + osall + scale=0.9
for cfg in "deg15:degrees=15" "deg25:degrees=25" "deg30:degrees=30" "shear:shear=10"; do
  tag="${cfg%%:*}"; extra="${cfg#*:}"
  echo "P3_${tag} $(date)" >>"$ST"
  for k in 0 1 2 3 4; do
    run_obb "EDU_${tag}_v11_f$k" "$OBB/fold$k/osall.yaml" 1280 "${GPUS[$k]}" "scale=0.9 $extra" &
  done
  wait
done
echo "SWEEP_P3_DONE $(date)" >>"$ST"
