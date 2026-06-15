#!/usr/bin/env bash
# Universe-damper benchmark sweep: 8 configs, one per GPU, launched in parallel.
#   datasets: A = Merged_Universe_Stratified (combined re-stratified split)
#             B = Merged_Universe_TrainOnly  (old val/test preserved; comparable)
#   recipes : 150+100 (standard two-stage)  and  300+100 (Soum's longer stage-1)
#   models  : YOLOv8n, YOLO11n   (run_config_v3.sh: s1 COCO-init SGD lr0=0.01,
#             s2 finetune lr0=0.00334 lrf=0.1535, then test-split eval)
# Per-epoch mAP lands in runs/<name>_s{1,2}/results.csv automatically.
set -uo pipefail
ROOT="$HOME/atli"
BENCH="$ROOT/universe_bench"
RUNCFG="$BENCH/run_config_v3.sh"
DATA_A="$ROOT/Merged_Universe_Stratified/merged_universe.yaml"
DATA_B="$ROOT/Merged_Universe_TrainOnly/merged_universe_trainonly.yaml"
mkdir -p "$BENCH/logs"

[ -f "$DATA_A" ] || { echo "missing $DATA_A — run build_dataset_universe.py"; exit 1; }
[ -f "$DATA_B" ] || { echo "missing $DATA_B — run build_dataset_universe.py"; exit 1; }

#      name              fw   gpu  ep1  ep2  data
CFGS=(
  "Ufull_v8_150p100      v8   0    150  100  $DATA_A"
  "Ufull_v8_300p100      v8   1    300  100  $DATA_A"
  "Ufull_v11_150p100     v11  2    150  100  $DATA_A"
  "Ufull_v11_300p100     v11  3    300  100  $DATA_A"
  "Uto_v8_150p100        v8   4    150  100  $DATA_B"
  "Uto_v8_300p100        v8   5    300  100  $DATA_B"
  "Uto_v11_150p100       v11  6    150  100  $DATA_B"
  "Uto_v11_300p100       v11  7    300  100  $DATA_B"
)

echo "[$(date)] launching ${#CFGS[@]} configs (imgsz=640 batch=32)"
for cfg in "${CFGS[@]}"; do
  read -r NAME FW GPU EP1 EP2 DATA_YAML <<<"$cfg"
  DATA="$DATA_YAML" nohup bash "$RUNCFG" "$NAME" "$FW" "$GPU" "$EP1" "$EP2" 640 32 \
    > "$BENCH/logs/${NAME}.log" 2>&1 &
  echo "  $NAME -> GPU $GPU (pid $!)"
done
echo "[$(date)] all launched. logs: $BENCH/logs/"
