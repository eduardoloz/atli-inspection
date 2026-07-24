#!/usr/bin/env bash
# Phase 13: exploit the phase-11 mixup win + map the 640..1280 resolution frontier.
# (reference: deg15@640 = 0.726/R 0.681; mixup=0.15@640 = 0.757/R 0.709 — first
#  working 640 lever, closes ~46% of the gap to deg15@1280 = 0.793/R 0.726.
#  val640 probe 0.721 ~= trained-at-640 => gap is test-time information loss;
#  prog-FT/shear/distill-proxy levers dead.)
#   A: mix010_640 — deg15 + mixup=0.10 @640   (dose-response, low side)
#   B: mix025_640 — deg15 + mixup=0.25 @640   (dose-response, high side)
#   C: mix15_1280 — deg15 + mixup=0.15 @1280  (champion-recipe upgrade test)
#   D: deg15_768  — deg15, no mixup @768      (frontier point, deployment res)
#   E: mix15_768  — deg15 + mixup=0.15 @768   (deployment-res mixup pair)
# 5 conditions x 5 folds = 25 runs. GPUs 3-7, one condition per wave. batch 16.
# val960/val768 eval-only probes live in eval_cv_eduardo.py (val960_hi/val768_hi).
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p13_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P13_START $(date)" > "$ST"

run_obb(){ local name=$1 model=$2 fold=$3 gpu=$4 imz=$5 extra=$6
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu imgsz=$imz extra=[$extra]" >>"$ST"
  MODEL="$model" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$extra" \
    bash "$OBBR" "$name" "$gpu" 150 100 "$imz" 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P13A_mix010_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_mix010_640_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" 640 "scale=0.9 degrees=15 mixup=0.10" & done; wait

echo "P13B_mix025_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_mix025_640_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" 640 "scale=0.9 degrees=15 mixup=0.25" & done; wait

echo "P13C_mix15_1280 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_mix15_1280_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" 1280 "scale=0.9 degrees=15 mixup=0.15" & done; wait

echo "P13D_deg15_768 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_deg15_768_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" 768 "scale=0.9 degrees=15" & done; wait

echo "P13E_mix15_768 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_mix15_768_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" 768 "scale=0.9 degrees=15 mixup=0.15" & done; wait

echo "SWEEP_P13_DONE $(date)" >>"$ST"
