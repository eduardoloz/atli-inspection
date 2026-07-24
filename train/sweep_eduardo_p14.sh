#!/usr/bin/env bash
# Phase 14: v8n backbone-graft twins of phase 6 — completes the low-param
# backbone grid across v8 and v11 (v11 grafts: ghost 0.682 / dws 0.697 /
# fnet 0.717 vs stock deg15 0.793; v8 stock OBB deg15 = 0.773).
# Winning recipe held fixed (OBB + osall + deg15 @1280, 2-stage TL 150+100,
# partial COCO init via pretrained=yolov8n-obb.pt — head/neck transfer only).
#   A: EDU_v8ghost_deg15 — GhostConv + C3Ghost backbone
#   B: EDU_v8dws_deg15   — MobileNet-style DWConv downsampling
#   C: EDU_v8fnet_deg15  — FasterNet PConv (FNET=1 + modpatch sitecustomize)
# 3 conditions x 5 folds = 15 runs. GPUs 0-2 (phase 13 owns 3-7) via per-GPU
# queues, 5 runs each. batch 16.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p14_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
MG="$ROOT/models_graft"
RECIPE="scale=0.9 degrees=15 pretrained=$ROOT/yolov8n-obb.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P14_START $(date)" > "$ST"

run_one(){ local name=$1 model=$2 fold=$3 gpu=$4 fnet=$5
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu model=$(basename $model) fnet=$fnet" >>"$ST"
  local envs=()
  [ "$fnet" = "1" ] && envs=(FNET=1 PYTHONPATH="$ROOT/modpatch")
  env "${envs[@]}" MODEL="$model" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$RECIPE" \
    bash "$OBBR" "$name" "$gpu" 150 100 1280 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

# item format: name|yaml|fold|fnet — round-robin across the 3 GPUs
Q0=("EDU_v8ghost_deg15_f0|$MG/yolov8n-ghost-obb.yaml|0|0"
    "EDU_v8ghost_deg15_f3|$MG/yolov8n-ghost-obb.yaml|3|0"
    "EDU_v8dws_deg15_f1|$MG/yolov8n-dws-obb.yaml|1|0"
    "EDU_v8dws_deg15_f4|$MG/yolov8n-dws-obb.yaml|4|0"
    "EDU_v8fnet_deg15_f2|$MG/yolov8n-fnet-obb.yaml|2|1")
Q1=("EDU_v8ghost_deg15_f1|$MG/yolov8n-ghost-obb.yaml|1|0"
    "EDU_v8ghost_deg15_f4|$MG/yolov8n-ghost-obb.yaml|4|0"
    "EDU_v8dws_deg15_f2|$MG/yolov8n-dws-obb.yaml|2|0"
    "EDU_v8fnet_deg15_f0|$MG/yolov8n-fnet-obb.yaml|0|1"
    "EDU_v8fnet_deg15_f3|$MG/yolov8n-fnet-obb.yaml|3|1")
Q2=("EDU_v8ghost_deg15_f2|$MG/yolov8n-ghost-obb.yaml|2|0"
    "EDU_v8dws_deg15_f0|$MG/yolov8n-dws-obb.yaml|0|0"
    "EDU_v8dws_deg15_f3|$MG/yolov8n-dws-obb.yaml|3|0"
    "EDU_v8fnet_deg15_f1|$MG/yolov8n-fnet-obb.yaml|1|1"
    "EDU_v8fnet_deg15_f4|$MG/yolov8n-fnet-obb.yaml|4|1")

worker(){ local gpu=$1; shift
  for item in "$@"; do
    IFS='|' read -r name yaml fold fnet <<<"$item"
    run_one "$name" "$yaml" "$fold" "$gpu" "$fnet"
  done; }

worker 0 "${Q0[@]}" &
worker 1 "${Q1[@]}" &
worker 2 "${Q2[@]}" &
wait
echo "SWEEP_P14_DONE $(date)" >> "$ST"
