#!/usr/bin/env bash
# Phase 20: fill in the missing +blur-aug and +mixup rungs for YOLOv8n so its
# recipe ladder matches YOLOv11n's (v8n currently stops at "Champion: OBB +
# deg15 (1280)" = 0.773; v11n continues on to +blur-aug (0.790) and +mixup
# -- CV champion (0.804)). Same two levers, same OBB+osall+deg15 base, @1280,
# applied to v8n.
#   A: EDU_v8blurdeg15  -- YOLOv8n-obb, osall + scale=0.9 degrees=15 + BLUR_AUG
#   B: EDU_v8mix15_1280 -- YOLOv8n-obb, osall + scale=0.9 degrees=15 mixup=0.15
# (v5n-OBB is not attempted -- no viable maintained open-source fork; see
# CLAUDE.md / phase-12 note: "no Ultralytics YOLOv5 OBB variant".)
# Waits for phase 19 to release GPUs 4-7 (0-3 remain another user's job on
# this shared server -- see phase 18/19). 2 conditions x 5 folds = 10 runs,
# 4 GPU slots, queue-dispatched like phase 18.
# Post-sweep: "v8blurdeg15"/"v8mix15_1280" keys added to eval_cv_eduardo.py
# CONDS (phase 20 block), then EVAL_DEV=4 python eval_cv_eduardo.py.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p20_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(4 5 6 7)
MIX="scale=0.9 degrees=15 mixup=0.15"
DEG15="scale=0.9 degrees=15"
BLURENV=(BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/blurpatch" NO_ALBUMENTATIONS_UPDATE=1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "P20_WAITING_FOR_P19 $(date)" > "$ST"
while ! grep -q SWEEP_P19_DONE "$ROOT/sweep_eduardo_p19_status.txt" 2>/dev/null; do sleep 300; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P20_START $(date)" >> "$ST"

run_one(){ local name=$1 fold=$2 gpu=$3 extra=$4; shift 4
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra] env=[$*]" >>"$ST"
  env "$@" MODEL="yolov8n-obb.pt" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$extra" \
    bash "$OBBR" "$name" "$gpu" 150 100 1280 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

# Build the 10-job queue: (name, fold, kind)
JOBS=()
for k in 0 1 2 3 4; do JOBS+=("EDU_v8blurdeg15_f$k|$k|blur"); done
for k in 0 1 2 3 4; do JOBS+=("EDU_v8mix15_1280_f$k|$k|mix"); done

echo "P20_QUEUE ${#JOBS[@]} jobs, 4 GPU slots (4-7) $(date)" >>"$ST"
slot=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r name fold kind <<<"$job"
  gpu="${GPUS[$((slot % 4))]}"
  if [ "$kind" = "blur" ]; then
    run_one "$name" "$fold" "$gpu" "$DEG15" "${BLURENV[@]}" &
  else
    run_one "$name" "$fold" "$gpu" "$MIX" &
  fi
  slot=$((slot + 1))
  if [ $((slot % 4)) -eq 0 ]; then wait; fi
done
wait

echo "P20_EVAL $(date)" >>"$ST"
EVAL_DEV=4 python "$ROOT/eval_cv_eduardo.py" >"$LOGS/eval_p20.log" 2>&1
echo "SWEEP_P20_DONE $(date)" >> "$ST"
