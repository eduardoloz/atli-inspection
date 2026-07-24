#!/usr/bin/env bash
# Phase 15: v11n backbone grafts at 640 (deployment res) + batch-1 latency bench.
# (Grafts @1280 lost 0.08-0.11 mAP vs stock deg15 0.793; question: does the
#  accuracy cost shrink at 640, where stock deg15 = 0.726 ± 0.023?)
# Waits for phase 13 to release GPUs 3-7, then:
#   0: batch-1 latency benchmark on idle GPU 3 (eval/bench_latency.py)
#   A: EDU_ghost_640 — GhostConv + C3Ghost @640
#   B: EDU_dws_640   — DWConv downsampling @640
#   C: EDU_fnet_640  — FasterNet PConv @640 (FNET=1 + modpatch)
# Same recipe as p6 otherwise: OBB + osall + deg15, 2-stage TL 150+100,
# partial COCO init. 3 conditions x 5 folds = 15 runs. GPUs 3-7. batch 16.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p15_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
MG="$ROOT/models_graft"
GPUS=(3 4 5 6 7)
RECIPE="scale=0.9 degrees=15 pretrained=$ROOT/yolo11n-obb.pt"
FNETENV=(FNET=1 PYTHONPATH="$ROOT/modpatch")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "P15_WAITING_FOR_P13 $(date)" > "$ST"
while ! grep -q SWEEP_P13_DONE "$ROOT/sweep_eduardo_p13_status.txt" 2>/dev/null; do sleep 300; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P15_START $(date)" >> "$ST"

echo "P15_0_latency_bench $(date)" >>"$ST"
BENCH_DEV=3 python "$ROOT/bench_latency.py" >"$LOGS/bench_latency.log" 2>&1
env "${FNETENV[@]}" BENCH_DEV=3 python "$ROOT/bench_latency.py" >>"$LOGS/bench_latency.log" 2>&1
echo "P15_0_done $(date)" >>"$ST"

run_obb(){ local name=$1 model=$2 fold=$3 gpu=$4; shift 4
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu model=$(basename $model) env=[$*]" >>"$ST"
  env "$@" MODEL="$model" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$RECIPE" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P15A_ghost_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_ghost_640_f$k" "$MG/yolo11n-ghost-obb.yaml" "$k" "${GPUS[$k]}" & done; wait

echo "P15B_dws_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_dws_640_f$k" "$MG/yolo11n-dws-obb.yaml" "$k" "${GPUS[$k]}" & done; wait

echo "P15C_fnet_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_fnet_640_f$k" "$MG/yolo11n-fnet-obb.yaml" "$k" "${GPUS[$k]}" "${FNETENV[@]}" & done; wait

echo "SWEEP_P15_DONE $(date)" >> "$ST"
