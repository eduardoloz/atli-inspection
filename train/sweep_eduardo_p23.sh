#!/usr/bin/env bash
# Phase 23: CPLID in TRAIN ONLY for the two best v11n recipes (user request
# 2026-07-29, corrected: CPLID goes into training, test stays clean native).
# Precedent: cpliddeg15 (deg15 recipe + CPLID-in-train @1280) = 0.792 vs
# deg15 0.793 — no gain. Does CPLID-in-train compose with the newer recipes?
#   A: EDU_cplidmix15_1280      — mix15 recipe   (twin of mix15_1280, 0.804)
#   B: EDU_cplidblurmix30_1280  — blurmix30 recipe (twin of blurmix30_1280,
#      0.820 — current overall CV champion)
# Data: osall_cplid.yaml (trainosall_cplid = osall train + 250 CPLID; val and
# test are the untouched native fold splits). OBB, 2-stage TL 150+100 @1280
# batch16, stock yolo11n-obb.pt init. 2 conditions x 5 folds = 10 runs.
# Waits for phase 22 AND the v5-universe sweep to finish, then uses GPUs
# 0,3,5,6,7 only — mazumder's jupyter holds ~7GB on GPUs 1/2/4 and 1280
# training needs ~16GB, too tight on a 24GB card to share safely.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p23_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(0 3 5 6 7)
MIX15="scale=0.9 degrees=15 mixup=0.15"
BLURMIX30="scale=0.9 degrees=30 mixup=0.15"
BLURENV=(BLUR_AUG=1 PYTHONPATH="$ROOT/pylibs_blur:$ROOT/blurpatch" NO_ALBUMENTATIONS_UPDATE=1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "P23_WAITING_FOR_P22_AND_V5U $(date)" > "$ST"
while ! { grep -q SWEEP_P22_DONE "$ROOT/sweep_eduardo_p22_status.txt" 2>/dev/null \
       && grep -q SWEEP_V5U_DONE "$ROOT/sweep_v5_universe_status.txt" 2>/dev/null; }; do
  sleep 300
done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P23_START $(date)" >> "$ST"

run_one(){ local name=$1 fold=$2 extra=$3 gpu=$4; shift 4
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra] env=[$*]" >>"$ST"
  env "$@" MODEL="yolo11n-obb.pt" DATA="$OBB/fold$fold/osall_cplid.yaml" EXTRA="$extra" \
    bash "$OBBR" "$name" "$gpu" 150 100 1280 16 >"$LOGS/${name}.log" 2>&1
  local rc=$?
  echo "[$(date)] DONE $name (exit $rc)" >>"$ST"
  [ -f "$PROJ/${name}_s2/weights/best.pt" ] || echo "[$(date)] WARN $name missing s2 best.pt" >>"$ST"; }

echo "P23A_cplidmix15_1280 folds0-4 $(date)" >>"$ST"
for i in 0 1 2 3 4; do run_one "EDU_cplidmix15_1280_f$i" "$i" "$MIX15" "${GPUS[$i]}" & done; wait

echo "P23B_cplidblurmix30_1280 folds0-4 $(date)" >>"$ST"
for i in 0 1 2 3 4; do run_one "EDU_cplidblurmix30_1280_f$i" "$i" "$BLURMIX30" "${GPUS[$i]}" "${BLURENV[@]}" & done; wait

echo "P23_EVAL $(date)" >>"$ST"
EVAL_DEV=0 python "$ROOT/eval_cv_eduardo.py" >"$LOGS/eval_p23.log" 2>&1
echo "SWEEP_P23_DONE $(date)" >> "$ST"
