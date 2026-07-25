#!/usr/bin/env bash
# Phase 17: 640-px deployment recipe — blur-aug stacked on the mixup base.
# Context: 640 is the edge-deployment resolution (latency bench: resolution
# only pays on edge silicon). Best 640 base = mixup=0.15+deg15 (0.757 ± 0.016).
# The mixup-mechanism probe showed mixup adds no blur robustness (+0.03 vs
# blur-aug's +0.21), and the model is blur-fragile — so the deployment model
# needs blur-aug ON TOP of mixup. Composition is not guaranteed (CPLID/shear
# failed to stack on deg15) — this tests it at 640.
#   A: EDU_blurmix_640 — osall + scale=0.9 degrees=15 mixup=0.15 + BLUR_AUG
#      (MotionBlur p=0.3 + GaussianBlur p=0.2 via blurpatch sitecustomize)
# Success = clean mAP >= 0.75 (no composition cost) AND blurred-test >> 0.48.
# Waits for phase 15 to release GPUs 3-7. 1 condition x 5 folds, one per GPU.
# Post-sweep: eval_cv_eduardo.py (key blurmix_640) + eval_blur_robustness.py.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p17_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
MIX="scale=0.9 degrees=15 mixup=0.15"
BLURENV=(BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/blurpatch" NO_ALBUMENTATIONS_UPDATE=1)
GPUS=(3 4 5 6 7)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "P17_WAITING_FOR_P15 $(date)" > "$ST"
while ! grep -q SWEEP_P15_DONE "$ROOT/sweep_eduardo_p15_status.txt" 2>/dev/null; do sleep 300; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P17_START $(date)" >> "$ST"

run_one(){ local name=$1 fold=$2 gpu=$3
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$MIX +blur]" >>"$ST"
  env "${BLURENV[@]}" MODEL="yolo11n-obb.pt" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$MIX" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P17A_blurmix_640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_one "EDU_blurmix_640_f$k" "$k" "${GPUS[$k]}" & done; wait

echo "P17_EVAL $(date)" >>"$ST"
EVAL_DEV=3 python "$ROOT/eval_cv_eduardo.py" >"$LOGS/eval_p17.log" 2>&1
EVAL_DEV=3 python "$ROOT/eval_blur_robustness.py" >"$LOGS/eval_p17_blur.log" 2>&1
echo "SWEEP_P17_DONE $(date)" >> "$ST"
