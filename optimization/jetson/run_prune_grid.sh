#!/bin/bash
# Prune+finetune grid: {1.5x, 1.75x, 2.0x} FLOPs speedup x 3 seeds @768.
# Base weights: native-768 champion HR768nc_v11_s{0,1,2}_s2 (seed i pruned
# model fine-tunes from seed i base — seeds stay independent).
# One seed's 3 runs per GPU, sequential. GPU0's queue waits until the GPU
# is actually idle (<100 MiB used) before starting.
#
# Usage (on the UNLV server):  ./run_prune_grid.sh <seed> <physical_gpu>
#   e.g.  nohup ./run_prune_grid.sh 0 1 > grid_s0.log 2>&1 &
#         nohup ./run_prune_grid.sh 1 2 > grid_s1.log 2>&1 &
#         nohup ./run_prune_grid.sh 2 0 > grid_s2.log 2>&1 &   # waits for GPU0
set -u

SEED=${1:?seed}
GPU=${2:?gpu}
LR0=${3:-0.00334}   # 0.01 for post-prune recovery ("hlr") runs
LRF=${4:-0.1535}    # pair 0.01/0.01 for the standard YOLO recovery schedule
SUFFIX=${5:-}       # e.g. "_hlr" to keep conditions separate
PY=~/atli/env_jetson/bin/python
BASE=~/atli/runs/HR768nc_v11_s${SEED}_s2/weights/best.pt
DATA=~/atli/ATLI_noCPLID_OS3/data.yaml
OUTDIR=~/atli/export_jetson
PROJ=~/atli/runs_prune
cd ~/atli/jetson_test

# wait until the GPU is idle (parity jobs etc.)
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [ "$USED" -lt 100 ] && break
  echo "$(date) GPU$GPU busy (${USED}MiB) — waiting 5 min"; sleep 300
done

# tag:keep-ratio pairs (keep = 1/speedup)
for CFG in "pr150:0.667" "pr175:0.571" "pr200:0.50"; do
  TAG=${CFG%%:*}; KEEP=${CFG##*:}
  NAME=${TAG}${SUFFIX}_s${SEED}
  echo "=== $(date) $NAME (keep ${KEEP} MACs, lr0=$LR0 lrf=$LRF) on GPU$GPU ==="
  # NB: pass the PHYSICAL gpu id via --device and do NOT set
  # CUDA_VISIBLE_DEVICES here: ultralytics select_device() overwrites
  # CUDA_VISIBLE_DEVICES with the --device string, so combining both
  # remaps the job onto the wrong physical GPU.
  $PY prune_v11n.py \
    --weights "$BASE" \
    --target-flops-ratio "$KEEP" --imgsz 768 \
    --out "$OUTDIR/${NAME}.pt" \
    --finetune --data "$DATA" --epochs 100 --batch 16 \
    --lr0 "$LR0" --lrf "$LRF" \
    --device "$GPU" --seed "$SEED" --project "$PROJ" --name "$NAME" \
    || echo "!!! $NAME FAILED"
done
echo "=== $(date) seed $SEED queue done ==="
