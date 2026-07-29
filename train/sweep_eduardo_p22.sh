#!/usr/bin/env bash
# Phase 22: v8n backbone grafts at 640 — the missing cells of the graft grid.
# Existing: v8 grafts @1280 deg15 (p14), v11 grafts @640 deg15 (p15), v11
# grafts @640 blurmix (p18). Missing: v8 grafts @640, on BOTH recipes:
#   A: EDU_v8ghost_640  / D: EDU_v8ghost_blurmix640  — GhostConv + C3Ghost
#   B: EDU_v8dws_640    / E: EDU_v8dws_blurmix640    — DWConv downsampling
#   C: EDU_v8fnet_640   / F: EDU_v8fnet_blurmix640   — FasterNet PConv
# deg15 recipe   = osall + scale=0.9 degrees=15            (v11 twins: 0.577/0.571/0.626)
# blurmix recipe = osall + scale=0.9 degrees=15 mixup=0.15 + BLUR_AUG
# Partial COCO init pretrained=yolov8n-obb.pt (head/neck only), OBB, 2-stage
# TL 150+100 @640 batch16. 6 conditions x 5 folds = 30 runs.
# GPUs 0-5 (server idle as of 2026-07-29; GPUs 6-7 reserved for the
# concurrent v5-universe sweep). Per-GPU queues of 5 jobs, round-robin.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p22_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
MG="$ROOT/models_graft"
GPUS=(0 1 2 3 4 5)
DEG15="scale=0.9 degrees=15 pretrained=$ROOT/yolov8n-obb.pt"
MIX="scale=0.9 degrees=15 mixup=0.15 pretrained=$ROOT/yolov8n-obb.pt"
FNETENV=(FNET=1 PYTHONPATH="$ROOT/modpatch")
BLURENV=(BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/blurpatch" NO_ALBUMENTATIONS_UPDATE=1)
FNETBLURENV=(FNET=1 BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/fnetblur_patch" NO_ALBUMENTATIONS_UPDATE=1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P22_START $(date)" >> "$ST"

run_one(){ local name=$1 model=$2 fold=$3 extra=$4 gpu=$5; shift 5
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu model=$(basename "$model") env=[$*]" >>"$ST"
  env "$@" MODEL="$model" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$extra" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  local rc=$?
  echo "[$(date)] DONE $name (exit $rc)" >>"$ST"
  # real completion check (the p16 lesson: exit code alone can lie)
  [ -f "$PROJ/${name}_s2/weights/best.pt" ] || echo "[$(date)] WARN $name missing s2 best.pt" >>"$ST"; }

# flat job list: name|yaml|fold|extra-kind (cond-major; round-robin over 6 GPUs)
JOBS=()
for k in 0 1 2 3 4; do JOBS+=("EDU_v8ghost_640_f$k|$MG/yolov8n-ghost-obb.yaml|$k|deg15|plain"); done
for k in 0 1 2 3 4; do JOBS+=("EDU_v8dws_640_f$k|$MG/yolov8n-dws-obb.yaml|$k|deg15|plain"); done
for k in 0 1 2 3 4; do JOBS+=("EDU_v8fnet_640_f$k|$MG/yolov8n-fnet-obb.yaml|$k|deg15|fnet"); done
for k in 0 1 2 3 4; do JOBS+=("EDU_v8ghost_blurmix640_f$k|$MG/yolov8n-ghost-obb.yaml|$k|mix|blur"); done
for k in 0 1 2 3 4; do JOBS+=("EDU_v8dws_blurmix640_f$k|$MG/yolov8n-dws-obb.yaml|$k|mix|blur"); done
for k in 0 1 2 3 4; do JOBS+=("EDU_v8fnet_blurmix640_f$k|$MG/yolov8n-fnet-obb.yaml|$k|mix|fnetblur"); done
echo "P22_QUEUE ${#JOBS[@]} jobs, 6 GPU slots (0-5) $(date)" >>"$ST"

worker(){ local gpu=$1; shift
  for item in "$@"; do
    IFS='|' read -r name yaml fold ekind envtag <<<"$item"
    local extra="$DEG15"; [ "$ekind" = "mix" ] && extra="$MIX"
    case "$envtag" in
      plain)    run_one "$name" "$yaml" "$fold" "$extra" "$gpu" ;;
      fnet)     run_one "$name" "$yaml" "$fold" "$extra" "$gpu" "${FNETENV[@]}" ;;
      blur)     run_one "$name" "$yaml" "$fold" "$extra" "$gpu" "${BLURENV[@]}" ;;
      fnetblur) run_one "$name" "$yaml" "$fold" "$extra" "$gpu" "${FNETBLURENV[@]}" ;;
    esac
  done; }

# GPU g takes jobs g, g+6, g+12, g+18, g+24
for g in 0 1 2 3 4 5; do
  Q=(); for i in $(seq $g 6 29); do Q+=("${JOBS[$i]}"); done
  worker "${GPUS[$g]}" "${Q[@]}" &
done
wait

echo "P22_EVAL $(date)" >>"$ST"
EVAL_DEV=0 python "$ROOT/eval_cv_eduardo.py" >"$LOGS/eval_p22.log" 2>&1
env FNET=1 PYTHONPATH="$ROOT/modpatch" EVAL_DEV=0 python "$ROOT/eval_cv_eduardo.py" >"$LOGS/eval_p22_fnet.log" 2>&1
echo "SWEEP_P22_DONE $(date)" >> "$ST"
