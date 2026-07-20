#!/usr/bin/env bash
# Phase 2 (v11n): deg20 + CPLID-in-train champions on the existing eduardos CV folds.
# 3 conditions x 5 folds = 15 runs. GPUs 3-7. 2-stage TL (150+100).
set -uo pipefail
ROOT="$HOME/atli"
DET="$ROOT/CV_eduardo_det"; OBB="$ROOT/CV_eduardo_obb"
EXT="$ROOT/run_config_ext.sh"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p2_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P2_START $(date)" > "$ST"
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-obb.pt')" >>"$ST" 2>&1

run_det(){ local name=$1 yaml=$2 imgsz=$3 gpu=$4 extra=${5:-}
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra]" >>"$ST"
  MODEL=yolo11n.pt DATA="$yaml" EXTRA="$extra" bash "$EXT" "$name" "$gpu" 150 100 "$imgsz" 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }
run_obb(){ local name=$1 yaml=$2 imgsz=$3 gpu=$4 extra=${5:-}
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra]" >>"$ST"
  MODEL=yolo11n-obb.pt DATA="$yaml" EXTRA="$extra" bash "$OBBR" "$name" "$gpu" 150 100 "$imgsz" 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

# P2A: OBB champion + osall + deg20 (milder rotation than the deg45 that hurt DD)
echo "P2A_OBBdeg20_1280 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_obbdeg20_v11_f$k" "$OBB/fold$k/osall.yaml" 1280 "${GPUS[$k]}" "scale=0.9 degrees=20" & done; wait

# P2B: OBB champion + osall + CPLID-in-train
echo "P2B_CPLID_OBBchamp_1280 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_cplidobb_v11_f$k" "$OBB/fold$k/osall_cplid.yaml" 1280 "${GPUS[$k]}" "scale=0.9" & done; wait

# P2C: detection champion + osall + CPLID-in-train
echo "P2C_CPLID_DETchamp_1280 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_det "EDU_cpliddet_v11_f$k" "$DET/fold$k/osall_cplid.yaml" 1280 "${GPUS[$k]}" "scale=0.9" & done; wait

echo "SWEEP_P2_DONE $(date)" >>"$ST"
