#!/usr/bin/env bash
# Eduardo ablation: v5n + v8n, each trained on target-only (A) vs target+Eduardo (B),
# both evaluated on the SAME Eduardo-free test set. Two 4-GPU lanes.
set -uo pipefail
ROOT=$HOME/atli
TO=$ROOT/abl_target_only.yaml
TPE=$ROOT/abl_target_plus_eduardo.yaml
echo "ABL_START $(date)" > $ROOT/abl_status.txt

(
  DATA=$TO  bash ~/run_config.sh abl_v5_to v5 SGD 150 0,1,2,3 29510 100
  DATA=$TO  bash ~/run_config.sh abl_v8_to v8 SGD 150 0,1,2,3 29510 100
) > $ROOT/ablA.log 2>&1 &
A=$!

(
  DATA=$TPE bash ~/run_config.sh abl_v5_tpe v5 SGD 150 4,5,6,7 29520 100
  DATA=$TPE bash ~/run_config.sh abl_v8_tpe v8 SGD 150 4,5,6,7 29520 100
) > $ROOT/ablB.log 2>&1 &
B=$!

wait $A $B
echo "ABL_DONE $(date)" >> $ROOT/abl_status.txt
