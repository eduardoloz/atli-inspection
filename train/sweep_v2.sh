#!/usr/bin/env bash
# Extended ablation sweep: v5n / v8n / v11n × A / B / C × fine-tune 200ep
# Also re-runs the crashed v8 B/C at 100ep fine-tune.
#
# Layout: 4 sequential waves, each using 2 parallel 4-GPU lanes.
# Estimated ~3-4 hr per wave on 8× RTX 6000.
#
# Existing completed runs (150+100):
#   v5: A(to), B(tpe), C(tpef)  — all done
#   v8: A(to) — done; B(tpe), C(tpef) — crashed (NCCL timeout)
#
# This sweep runs:
#   Wave 1: v8 B/C at 150+100 (re-run crashed) + v11 A at 150+100
#   Wave 2: v11 B/C at 150+100
#   Wave 3: v5 A/B at 150+200 (longer fine-tune)
#   Wave 4: v5 C + v8 A at 150+200
#   Wave 5: v8 B/C at 150+200
#   Wave 6: v11 A/B at 150+200
#   Wave 7: v11 C at 150+200 + eval
#
# To keep it simpler: run in 3 waves with 2 lanes each.
set -uo pipefail
ROOT=$HOME/atli
TO=$ROOT/abl_target_only.yaml
TPE=$ROOT/abl_target_plus_eduardo.yaml
CF=$ROOT/abl_target_plus_eduardo_full.yaml
RC=~/run_config_v2.sh
STATUS=$ROOT/sweep_v2_status.txt

echo "SWEEP_V2_START $(date)" > "$STATUS"

# ============================================================
# WAVE 1: Fix crashed v8 (B+C @ 100ep) + v11 A @ 100ep
# ============================================================
echo "WAVE1_START $(date)" >> "$STATUS"
(
  DATA=$TPE bash "$RC" v8_tpe_r2   v8  SGD 150 0,1,2,3 29510 100
  DATA=$CF  bash "$RC" v8_tpef_r2  v8  SGD 150 0,1,2,3 29510 100
) > "$ROOT/sweep_v2_w1a.log" 2>&1 &
W1A=$!
(
  DATA=$TO  bash "$RC" v11_to      v11 SGD 150 4,5,6,7 29520 100
  DATA=$TPE bash "$RC" v11_tpe     v11 SGD 150 4,5,6,7 29520 100
) > "$ROOT/sweep_v2_w1b.log" 2>&1 &
W1B=$!
wait $W1A $W1B
echo "WAVE1_DONE $(date)" >> "$STATUS"

# ============================================================
# WAVE 2: v11 C @ 100ep + start 200ep fine-tune runs
# ============================================================
echo "WAVE2_START $(date)" >> "$STATUS"
(
  DATA=$CF  bash "$RC" v11_tpef    v11 SGD 150 0,1,2,3 29510 100
  DATA=$TO  bash "$RC" v5_to_ft200 v5  SGD 150 0,1,2,3 29510 200
) > "$ROOT/sweep_v2_w2a.log" 2>&1 &
W2A=$!
(
  DATA=$TPE bash "$RC" v5_tpe_ft200  v5  SGD 150 4,5,6,7 29520 200
  DATA=$CF  bash "$RC" v5_tpef_ft200 v5  SGD 150 4,5,6,7 29520 200
) > "$ROOT/sweep_v2_w2b.log" 2>&1 &
W2B=$!
wait $W2A $W2B
echo "WAVE2_DONE $(date)" >> "$STATUS"

# ============================================================
# WAVE 3: v8 + v11 all conditions @ 200ep fine-tune
# ============================================================
echo "WAVE3_START $(date)" >> "$STATUS"
(
  DATA=$TO  bash "$RC" v8_to_ft200   v8  SGD 150 0,1,2,3 29510 200
  DATA=$TPE bash "$RC" v8_tpe_ft200  v8  SGD 150 0,1,2,3 29510 200
  DATA=$CF  bash "$RC" v8_tpef_ft200 v8  SGD 150 0,1,2,3 29510 200
) > "$ROOT/sweep_v2_w3a.log" 2>&1 &
W3A=$!
(
  DATA=$TO  bash "$RC" v11_to_ft200   v11 SGD 150 4,5,6,7 29520 200
  DATA=$TPE bash "$RC" v11_tpe_ft200  v11 SGD 150 4,5,6,7 29520 200
  DATA=$CF  bash "$RC" v11_tpef_ft200 v11 SGD 150 4,5,6,7 29520 200
) > "$ROOT/sweep_v2_w3b.log" 2>&1 &
W3B=$!
wait $W3A $W3B
echo "WAVE3_DONE $(date)" >> "$STATUS"

echo "SWEEP_V2_DONE $(date)" >> "$STATUS"
