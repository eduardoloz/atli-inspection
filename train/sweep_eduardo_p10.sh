#!/usr/bin/env bash
# Phase 10: recall recovery at 640 px — three mechanisms on the deg15 recipe @640
# (reference: deg15@640 = 0.726 mAP / R 0.681; deg15@1280 = 0.793 / 0.726).
#   A: cm20_640 — close_mosaic=20: full-size objects for the last 20 epochs
#      (only winner of the 768 augmentation ablation; untested at 640 / this pool)
#   B: ms_640   — multi_scale=True: train across ±50% scales for scale-robust features
#   C: p2_640   — stride-4 P2 head (models_graft/yolo11n-p2-obb.yaml): 4x head
#      resolution for small objects; partial COCO init via pretrained=
# 3 conditions x 5 folds = 15 runs. GPUs 3-7. 2-stage TL (150+100), imgsz 640, batch 16.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p10_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P10_START $(date)" > "$ST"

run_obb(){ local name=$1 model=$2 fold=$3 gpu=$4 extra=$5
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra]" >>"$ST"
  MODEL="$model" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$extra" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P10A_cm20_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_cm20_640_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" "scale=0.9 degrees=15 close_mosaic=20" & done; wait

echo "P10B_ms_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_ms_640_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" "scale=0.9 degrees=15 multi_scale=True" & done; wait

echo "P10C_p2_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_p2_640_f$k" "$ROOT/models_graft/yolo11n-p2-obb.yaml" "$k" "${GPUS[$k]}" "scale=0.9 degrees=15 pretrained=$ROOT/yolo11n-obb.pt" & done; wait

echo "SWEEP_P10_DONE $(date)" >>"$ST"
