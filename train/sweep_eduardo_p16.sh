#!/usr/bin/env bash
# Phase 16: defect-recall levers at 640 on the mixup recipe (best 640 base:
# mixup=0.15+deg15 = 0.757 / R 0.709 / DD R 0.617).
# Hypotheses: (1) mixup's regularization makes x6 defect oversampling viable
# (x6 overfit at 640 WITHOUT mixup in the old pool); (2) doubling the cls-loss
# gain (0.5 -> 1.0) lifts minority-class recall; (3) they compose.
#   A: EDU_os6mix_640     — osall6 (defect x6) + mixup recipe
#   B: EDU_cls1mix_640    — osall  (x3) + mixup recipe + cls=1.0
#   C: EDU_os6cls1mix_640 — osall6 + mixup + cls=1.0
# Waits for phase 14 to release GPUs 0-2. 3 conditions x 5 folds = 15 runs,
# per-GPU queues (5 each). Requires build_osall6_eduardo.py to have run.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p16_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
MIX="scale=0.9 degrees=15 mixup=0.15"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "P16_WAITING_FOR_P14 $(date)" > "$ST"
while ! grep -q SWEEP_P14_DONE "$ROOT/sweep_eduardo_p14_status.txt" 2>/dev/null; do sleep 300; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P16_START $(date)" >> "$ST"

run_one(){ local name=$1 fold=$2 gpu=$3 yaml=$4 extra=$5
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu yaml=$yaml extra=[$extra]" >>"$ST"
  MODEL="yolo11n-obb.pt" DATA="$OBB/fold$fold/$yaml" EXTRA="$extra" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

# item: name|fold|yaml|extra — round-robin over GPUs 0-2
Q0=("EDU_os6mix_640_f0|0|osall6.yaml|$MIX"
    "EDU_os6mix_640_f3|3|osall6.yaml|$MIX"
    "EDU_cls1mix_640_f1|1|osall.yaml|$MIX cls=1.0"
    "EDU_cls1mix_640_f4|4|osall.yaml|$MIX cls=1.0"
    "EDU_os6cls1mix_640_f2|2|osall6.yaml|$MIX cls=1.0")
Q1=("EDU_os6mix_640_f1|1|osall6.yaml|$MIX"
    "EDU_os6mix_640_f4|4|osall6.yaml|$MIX"
    "EDU_cls1mix_640_f2|2|osall.yaml|$MIX cls=1.0"
    "EDU_os6cls1mix_640_f0|0|osall6.yaml|$MIX cls=1.0"
    "EDU_os6cls1mix_640_f3|3|osall6.yaml|$MIX cls=1.0")
Q2=("EDU_os6mix_640_f2|2|osall6.yaml|$MIX"
    "EDU_cls1mix_640_f0|0|osall.yaml|$MIX cls=1.0"
    "EDU_cls1mix_640_f3|3|osall.yaml|$MIX cls=1.0"
    "EDU_os6cls1mix_640_f1|1|osall6.yaml|$MIX cls=1.0"
    "EDU_os6cls1mix_640_f4|4|osall6.yaml|$MIX cls=1.0")

worker(){ local gpu=$1; shift
  for item in "$@"; do
    IFS='|' read -r name fold yaml extra <<<"$item"
    run_one "$name" "$fold" "$gpu" "$yaml" "$extra"
  done; }

worker 0 "${Q0[@]}" &
worker 1 "${Q1[@]}" &
worker 2 "${Q2[@]}" &
wait
echo "SWEEP_P16_DONE $(date)" >> "$ST"
