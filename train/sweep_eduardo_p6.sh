#!/usr/bin/env bash
# Phase 6 (lighter-backbone grafts, nano-class): winning recipe (OBB + osall + deg15
# @1280) held fixed, v11n backbone swapped for lighter grafts (train/models_graft/):
#   A: EDU_ghost_deg15 — GhostConv + C3Ghost backbone            (2.21M vs stock 2.70M)
#   B: EDU_dws_deg15   — MobileNet-style DWConv downsampling     (2.22M)
#   C: EDU_fnet_deg15  — FasterNet PConv blocks, FNET=1 + modpatch sitecustomize (2.50M)
# Partial COCO init via pretrained=yolo11n-obb.pt (head/neck transfer; grafted layers fresh).
# Waits for phase 5. 3 conditions x 5 folds = 15 runs. GPUs 3-7. 2-stage TL (150+100).
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p6_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)
RECIPE="scale=0.9 degrees=15 pretrained=$ROOT/yolo11n-obb.pt"
FNETENV=(FNET=1 PYTHONPATH="$ROOT/modpatch")

echo "P6_WAITING_FOR_P5 $(date)" > "$ST"
while ! grep -q SWEEP_P5_DONE "$ROOT/sweep_eduardo_p5_status.txt" 2>/dev/null; do sleep 120; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P6_START $(date)" >> "$ST"

run_obb(){ local name=$1 model=$2 fold=$3 gpu=$4; shift 4
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu model=$(basename $model) env=[$*]" >>"$ST"
  env "$@" MODEL="$model" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$RECIPE" \
    bash "$OBBR" "$name" "$gpu" 150 100 1280 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P6A_ghost_deg15 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_ghost_deg15_f$k" "$ROOT/models_graft/yolo11n-ghost-obb.yaml" "$k" "${GPUS[$k]}" & done; wait

echo "P6B_dws_deg15 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_dws_deg15_f$k" "$ROOT/models_graft/yolo11n-dws-obb.yaml" "$k" "${GPUS[$k]}" & done; wait

echo "P6C_fnet_deg15 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_fnet_deg15_f$k" "$ROOT/models_graft/yolo11n-fnet-obb.yaml" "$k" "${GPUS[$k]}" "${FNETENV[@]}" & done; wait

echo "SWEEP_P6_DONE $(date)" >>"$ST"
