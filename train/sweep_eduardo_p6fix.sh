#!/usr/bin/env bash
# Phase 6 fix: rerun EDU_ghost_deg15_f4 — original crashed 2 min in with CUDA OOM
# during early-epoch val NMS (batch_probiou spike), masked as exit 0. Reruns with
# expandable_segments to resist fragmentation. Waits for the blur sweep (p5b).
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p6fix_status.txt"; LOGS="$ROOT/logs_eduardo"

echo "P6FIX_WAITING_FOR_P5B $(date)" > "$ST"
while ! grep -q SWEEP_P5B_DONE "$ROOT/sweep_eduardo_p5b_status.txt" 2>/dev/null; do sleep 120; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "P6FIX_START $(date)" >> "$ST"
rm -rf "$ROOT/runs/EDU_ghost_deg15_f4_s1"   # stale 2-min crash artifact
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL="$ROOT/models_graft/yolo11n-ghost-obb.yaml" DATA="$OBB/fold4/osall.yaml" \
  EXTRA="scale=0.9 degrees=15 pretrained=$ROOT/yolo11n-obb.pt" \
  bash "$OBBR" "EDU_ghost_deg15_f4" 3 150 100 1280 16 >"$LOGS/EDU_ghost_deg15_f4.log" 2>&1
echo "P6FIX_DONE (exit $?) $(date)" >> "$ST"
