#!/usr/bin/env bash
# Round 3 (2026-07-08): settle T2 (thesis-style stage-2 lr0=1e-4) with 2 extra seeds.
# Chained on GPU 6; mirrors ~/atli/th2_gpu6_lowlr_s12.sh on the server.
# Launch: nohup bash ~/atli/th2_gpu6_lowlr_s12.sh > ~/atli/TH2_gpu6_lowlr_s12.log 2>&1 &
set -uo pipefail
ROOT="$HOME/atli"; cd "$ROOT"
for S in 1 2; do
  echo "[$(date)] ### launching TH2_lowlr_v11_768_s$S (GPU6)"
  MODEL=yolo11n.pt EXTRA="scale=0.9 seed=$S" LR2=0.0001 \
  DATA="$ROOT/ATLI_noCPLID_OS3/data.yaml" \
  bash "$ROOT/run_config_ext2.sh" "TH2_lowlr_v11_768_s$S" 6 150 100 768 16
done
echo "[$(date)] ### GPU6 LOWLR S1-2 CHAIN DONE"
