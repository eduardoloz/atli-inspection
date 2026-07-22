#!/usr/bin/env bash
# Phase 5 (v11n OBB): stack + robustness arms on the existing CV_eduardo_obb folds.
#   A: cpliddeg15 — best-recipe stack: osall + CPLID-in-train + degrees=15 (both winners combined)
#   B: blurdeg15  — osall + degrees=15 + train-time MotionBlur/GaussianBlur via
#      ~/atli/blurpatch/sitecustomize.py (BLUR_AUG=1) and isolated ~/atli/pylibs_blur.
# 2 conditions x 5 folds = 10 runs. GPUs 3-7. 2-stage TL (150+100), imgsz 1280, batch 16.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p5_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)
BLURENV=(BLUR_AUG=1 PYTHONPATH="$HOME/atli/pylibs_blur:$HOME/atli/blurpatch" NO_ALBUMENTATIONS_UPDATE=1)

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P5_START $(date)" > "$ST"
python -c "from ultralytics import YOLO; YOLO('yolo11n-obb.pt')" >> "$ST" 2>&1

run_obb(){ local name=$1 yaml=$2 gpu=$3 extra=$4; shift 4
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra] env=[$*]" >>"$ST"
  env "$@" MODEL=yolo11n-obb.pt DATA="$yaml" EXTRA="$extra" bash "$OBBR" "$name" "$gpu" 150 100 1280 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P5A_cpliddeg15 $(date)" >>"$ST"
for k in 0 1 2 3 4; do
  run_obb "EDU_cpliddeg15_v11_f$k" "$OBB/fold$k/osall_cplid.yaml" "${GPUS[$k]}" "scale=0.9 degrees=15" &
done; wait

echo "P5B_blurdeg15 $(date)" >>"$ST"
for k in 0 1 2 3 4; do
  run_obb "EDU_blurdeg15_v11_f$k" "$OBB/fold$k/osall.yaml" "${GPUS[$k]}" "scale=0.9 degrees=15" "${BLURENV[@]}" &
done; wait

echo "SWEEP_P5_DONE $(date)" >>"$ST"
