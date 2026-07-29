#!/usr/bin/env bash
# Phase 18: v11n backbone grafts (ghost/dws/fnet) on the NEW 640 champion
# recipe (blurmix_640 = osall + deg15 + mixup=0.15 + blur-aug, phase 17,
# clean mAP 0.757 / blurred 0.710). Phase 15 already ran these three grafts
# at 640 but with the OLDER deg15-only recipe (ghost 0.577, dws 0.571,
# fnet 0.626 vs stock deg15_640 0.726) — this asks whether the mixup+blur
# recipe closes any of that gap for the lighter backbones, same as it did
# for the stock backbone.
#   A: EDU_ghost_blurmix640 — GhostConv + C3Ghost @640
#   B: EDU_dws_blurmix640   — DWConv downsampling @640
#   C: EDU_fnet_blurmix640  — FasterNet PConv @640 (FNET=1, combined
#      fnetblur_patch sitecustomize — see fnetblur_patch_sitecustomize.py)
# Same recipe as p17 (osall + scale=0.9 degrees=15 mixup=0.15 + BLUR_AUG),
# 2-stage TL 150+100, partial COCO init (pretrained=yolo11n-obb.pt for the
# head/neck since the backbone shape differs). 3 conditions x 5 folds = 15
# runs. GPUs 4-7 ONLY — 0-3 are occupied by another user's unrelated job
# (valdif1/MS-RAFT-3D) on this shared server, so only a 4-slot pool is
# available; jobs are dispatched from a queue via `wait -n` as slots free
# instead of assuming 5 parallel slots like p15/p17 did.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p18_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
MG="$ROOT/models_graft"
GPUS=(4 5 6 7)
MIX="scale=0.9 degrees=15 mixup=0.15 pretrained=$ROOT/yolo11n-obb.pt"
BLURENV=(BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/blurpatch" NO_ALBUMENTATIONS_UPDATE=1)
FNETBLURENV=(FNET=1 BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/fnetblur_patch" NO_ALBUMENTATIONS_UPDATE=1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P18_START $(date)" >> "$ST"

run_one(){ local name=$1 model=$2 fold=$3 gpu=$4; shift 4
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu model=$(basename "$model") env=[$*]" >>"$ST"
  env "$@" MODEL="$model" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$MIX" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

# Build the 15-job queue: (name, model, fold, extra-env...)
JOBS=()
for k in 0 1 2 3 4; do JOBS+=("EDU_ghost_blurmix640_f$k|$MG/yolo11n-ghost-obb.yaml|$k|"); done
for k in 0 1 2 3 4; do JOBS+=("EDU_dws_blurmix640_f$k|$MG/yolo11n-dws-obb.yaml|$k|"); done
for k in 0 1 2 3 4; do JOBS+=("EDU_fnet_blurmix640_f$k|$MG/yolo11n-fnet-obb.yaml|$k|fnet"); done

echo "P18_QUEUE ${#JOBS[@]} jobs, 4 GPU slots (4-7) $(date)" >>"$ST"
slot=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r name model fold envtag <<<"$job"
  gpu="${GPUS[$((slot % 4))]}"
  if [ "$envtag" = "fnet" ]; then
    run_one "$name" "$model" "$fold" "$gpu" "${FNETBLURENV[@]}" &
  else
    run_one "$name" "$model" "$fold" "$gpu" "${BLURENV[@]}" &
  fi
  slot=$((slot + 1))
  if [ $((slot % 4)) -eq 0 ]; then wait; fi
done
wait

echo "P18_EVAL $(date)" >>"$ST"
EVAL_DEV=4 python "$ROOT/eval_cv_eduardo.py" >"$LOGS/eval_p18.log" 2>&1
env FNET=1 PYTHONPATH="$ROOT/modpatch" EVAL_DEV=4 python "$ROOT/eval_cv_eduardo.py" >"$LOGS/eval_p18_fnet.log" 2>&1
echo "SWEEP_P18_DONE $(date)" >> "$ST"
