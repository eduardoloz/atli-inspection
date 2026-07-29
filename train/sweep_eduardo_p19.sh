#!/usr/bin/env bash
# Phase 19: does MORE rotation stack with the blurmix_640 champion recipe?
# User hypothesis: higher rotation angle helps DefDamper recall further (cf.
# the single-split OBB deg45 recall 0.957 finding) — but every OTHER stack-
# on-deg15 attempt at 640/1280 (CPLID, shear) failed to compose, and the
# eduardo-CV deg-sweep already showed deg15 as the sweet spot vs deg25/deg30
# at 1280 without mixup/blur. This tests whether swapping degrees=15->30
# INSIDE the blurmix_640 recipe (mixup=0.15 + blur-aug, osall, 640, OBB,
# 2-stage TL 150+100) helps or hurts vs blurmix_640 itself (clean mAP
# 0.757 ± 0.017, DD AP 0.680 ± 0.099, DD recall — see eval_eduardo_results).
# Stock v11n backbone only (not the phase-18 grafts) — cleanest apples-to-
# apples comparison against the champion.
#   A: EDU_blurmix30_640 — osall + scale=0.9 degrees=30 mixup=0.15 + BLUR_AUG
# Waits for phase 18 to release GPUs 4-7 (0-3 remain another user's job on
# this shared server — see phase 18). 1 condition x 5 folds, 4 GPU slots.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p19_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
MIX="scale=0.9 degrees=30 mixup=0.15"
BLURENV=(BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/blurpatch" NO_ALBUMENTATIONS_UPDATE=1)
GPUS=(4 5 6 7)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "P19_WAITING_FOR_P18 $(date)" > "$ST"
while ! grep -q SWEEP_P18_DONE "$ROOT/sweep_eduardo_p18_status.txt" 2>/dev/null; do sleep 300; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P19_START $(date)" >> "$ST"

run_one(){ local name=$1 fold=$2 gpu=$3
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$MIX +blur]" >>"$ST"
  env "${BLURENV[@]}" MODEL="yolo11n-obb.pt" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$MIX" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P19A_blurmix30_640 folds0-3 $(date)" >>"$ST"
for k in 0 1 2 3; do run_one "EDU_blurmix30_640_f$k" "$k" "${GPUS[$k]}" & done; wait
echo "P19A_blurmix30_640 fold4 $(date)" >>"$ST"
run_one "EDU_blurmix30_640_f4" "4" "${GPUS[0]}"

echo "P19_EVAL $(date)" >>"$ST"
EVAL_DEV=4 python "$ROOT/eval_cv_eduardo.py" >"$LOGS/eval_p19.log" 2>&1
EVAL_DEV=4 python "$ROOT/eval_blur_robustness.py" >"$LOGS/eval_p19_blur.log" 2>&1
echo "SWEEP_P19_DONE $(date)" >> "$ST"
