#!/usr/bin/env bash
# Full reproduction sweep of Untitled2.ipynb's 6 benchmark configs.
# Two lanes of 4 GPUs (batch 32) running in parallel; 3 configs each, sequential.
#   Lane A (GPU 0-3): v5n_sgd_150, v5n_sgd_300, v8n_sgd_150     (~900 train-epochs)
#   Lane B (GPU 4-7): v5n_adam_150, v5n_adam_300, v8n_sgd_300   (~1050 train-epochs)
set -uo pipefail
echo "SWEEP_START $(date)" > ~/atli/sweep_status.txt

(
  bash ~/run_config.sh v5n_sgd_150  v5 SGD  150 0,1,2,3 29510 100
  bash ~/run_config.sh v5n_sgd_300  v5 SGD  300 0,1,2,3 29510 100
  bash ~/run_config.sh v8n_sgd_150  v8 SGD  150 0,1,2,3 29510 100
) > ~/atli/laneA.log 2>&1 &
LA=$!

(
  bash ~/run_config.sh v5n_adam_150 v5 Adam 150 4,5,6,7 29520 100
  bash ~/run_config.sh v5n_adam_300 v5 Adam 300 4,5,6,7 29520 100
  bash ~/run_config.sh v8n_sgd_300  v8 SGD  300 4,5,6,7 29520 100
) > ~/atli/laneB.log 2>&1 &
LB=$!

wait $LA $LB
echo "SWEEP_DONE $(date)" >> ~/atli/sweep_status.txt
