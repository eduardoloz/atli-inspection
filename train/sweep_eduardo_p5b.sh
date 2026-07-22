#!/usr/bin/env bash
# Phase 5b: RERUN of the blurdeg15 arm — the phase-5 attempt died at `conda activate`
# because the blurpatch sitecustomize printed to stdout, which conda's activation
# evals (fixed: patch now prints to stderr). Waits for phase 6 to free the GPUs.
# 1 condition x 5 folds. GPUs 3-7. OBB champ + osall + deg15 + MotionBlur/GaussianBlur.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p5b_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)
BLURENV=(BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/blurpatch" NO_ALBUMENTATIONS_UPDATE=1)

echo "P5B_WAITING_FOR_P6 $(date)" > "$ST"
while ! grep -q SWEEP_P6_DONE "$ROOT/sweep_eduardo_p6_status.txt" 2>/dev/null; do sleep 120; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P5B_START $(date)" >> "$ST"

run_obb(){ local name=$1 fold=$2 gpu=$3
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu" >>"$ST"
  env "${BLURENV[@]}" MODEL=yolo11n-obb.pt DATA="$OBB/fold$fold/osall.yaml" \
    EXTRA="scale=0.9 degrees=15" bash "$OBBR" "$name" "$gpu" 150 100 1280 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P5B_blurdeg15_rerun $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_blurdeg15_v11_f$k" "$k" "${GPUS[$k]}" & done; wait

echo "SWEEP_P5B_DONE $(date)" >>"$ST"
