#!/usr/bin/env bash
# Revised GPU-7 step 2 (orchestrator 2026-07-07): the seed-0 champion@768 control duplicates a
# run owned by another workstream, so the control becomes a seed=3 replicate instead.
# The original th2_gpu7_chain.sh parent was killed after its step 1 started (editing a running
# bash script in place is unsafe); this watcher waits for the lowlr runner to exit, then runs
# the seed-3 control. Launch: nohup bash ~/atli/th2_gpu7_control_s3.sh >> ~/atli/TH2_gpu7.log 2>&1 &
set -euo pipefail
ROOT="$HOME/atli"; cd "$ROOT"

echo "[$(date)] ### watcher: waiting for TH2_lowlr_v11_768_s0 runner to finish"
while pgrep -f "run_config_ext2.sh TH2_lowlr_v11_768_s0" >/dev/null; do sleep 60; done
if [ ! -d "$ROOT/runs/TH2_lowlr_v11_768_s0_test" ]; then
  echo "[$(date)] ### WARNING: TH2_lowlr_v11_768_s0_test missing — lowlr may have crashed; running control anyway"
fi

echo "[$(date)] ### STEP 2 (revised): champion control @768 seed=3 (TH2_champ768_v11_s3)"
MODEL=yolo11n.pt EXTRA="scale=0.9 seed=3" \
DATA="$ROOT/ATLI_noCPLID_OS3/data.yaml" \
bash "$ROOT/run_config_ext2.sh" TH2_champ768_v11_s3 7 150 100 768 16

echo "[$(date)] ### GPU7 CHAIN DONE"
