#!/usr/bin/env bash
# Condition C: target + FULL Eduardo (Defective_Insulators -> Broken_Insulator).
# Waits for the A/B ablation to free the GPUs, then runs v5n + v8n in parallel lanes.
set -uo pipefail
ROOT=$HOME/atli
CF=$ROOT/abl_target_plus_eduardo_full.yaml
while ! grep -q ABL_DONE "$ROOT/abl_status.txt" 2>/dev/null; do sleep 30; done
echo "CONDC_START $(date)" > "$ROOT/condc_status.txt"

( DATA=$CF bash ~/run_config.sh abl_v5_tpef v5 SGD 150 0,1,2,3 29510 100 ) > "$ROOT/condcA.log" 2>&1 &
A=$!
( DATA=$CF bash ~/run_config.sh abl_v8_tpef v8 SGD 150 4,5,6,7 29520 100 ) > "$ROOT/condcB.log" 2>&1 &
B=$!
wait $A $B
echo "CONDC_DONE $(date)" >> "$ROOT/condc_status.txt"
