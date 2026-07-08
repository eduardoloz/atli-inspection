#!/usr/bin/env bash
# Thesis-round-2 GPU-6 chain: build gated source -> source pretrain -> champion fine-tune @768.
# Launch: nohup bash ~/atli/th2_gpu6_chain.sh > ~/atli/TH2_gpu6.log 2>&1 &
set -euo pipefail
ROOT="$HOME/atli"; cd "$ROOT"
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"

echo "[$(date)] ### STEP 1: build leakage-gated pretraining source"
python "$ROOT/build_source_pretrain.py"

echo "[$(date)] ### STEP 2: source pretrain (COCO-init yolo11n -> ATLI_source_pretrain, 150ep@640)"
yolo detect train model=yolo11n.pt data="$ROOT/ATLI_source_pretrain/data.yaml" \
  imgsz=640 batch=16 epochs=150 optimizer=SGD lr0=0.01 device=6 \
  project="$ROOT/runs" name=TH2_srcpre_v11_s1 exist_ok=True

echo "[$(date)] ### STEP 3: champion recipe fine-tune from source weights @768 (1 seed)"
MODEL="$ROOT/runs/TH2_srcpre_v11_s1/weights/best.pt" \
EXTRA="scale=0.9" \
DATA="$ROOT/ATLI_noCPLID_OS3/data.yaml" \
bash "$ROOT/run_config_ext2.sh" TH2_srcTL_v11_768_s0 6 150 100 768 16

echo "[$(date)] ### GPU6 CHAIN DONE"
