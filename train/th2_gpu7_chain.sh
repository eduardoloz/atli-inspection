#!/usr/bin/env bash
# Thesis-round-2 GPU-7 chain: champ_lowlr pilot @768, then champion-@768 control (COCO-init).
# The control makes both 768 pilots interpretable (champion reference numbers exist only at 1280/640).
# Launch: nohup bash ~/atli/th2_gpu7_chain.sh > ~/atli/TH2_gpu7.log 2>&1 &
set -euo pipefail
ROOT="$HOME/atli"; cd "$ROOT"

echo "[$(date)] ### STEP 1: champ_lowlr pilot (stage-2 lr0=0.0001, thesis Table 4.1) @768"
MODEL=yolo11n.pt EXTRA="scale=0.9" LR2=0.0001 \
DATA="$ROOT/ATLI_noCPLID_OS3/data.yaml" \
bash "$ROOT/run_config_ext2.sh" TH2_lowlr_v11_768_s0 7 150 100 768 16

echo "[$(date)] ### STEP 2: champion control @768 (identical but stage-2 lr0=0.00334 default)"
MODEL=yolo11n.pt EXTRA="scale=0.9" \
DATA="$ROOT/ATLI_noCPLID_OS3/data.yaml" \
bash "$ROOT/run_config_ext2.sh" TH2_champ768_v11_s0 7 150 100 768 16

echo "[$(date)] ### GPU7 CHAIN DONE"
