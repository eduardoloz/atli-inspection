#!/usr/bin/env bash
# Full 5-fold CV sweep: baseline/HROaug/OSall × v5/v8/v11
# Runs in batches of 8 (one per GPU), sequential between batches.
# Already done: CVbase_v11 (5 folds), CVchamp_v11 (5 folds)
set -uo pipefail
ROOT="$HOME/atli"
CV="$ROOT/Merged_CV"
EXT="$ROOT/run_config_ext.sh"
ST="$ROOT/cv_sweep_status.txt"

run() {
    # args: NAME MODEL YAML IMGSZ GPU [EXTRA]
    local name=$1 model=$2 yaml=$3 imgsz=$4 gpu=$5 extra=${6:-}
    echo "[$(date)] Starting $name gpu=$gpu" >> "$ST"
    MODEL="$model" DATA="$yaml" EXTRA="$extra" bash "$EXT" "$name" "$gpu" 150 100 "$imgsz" 16 \
        > "$ROOT/${name}.log" 2>&1 &
}

echo "CV_SWEEP_START $(date)" > "$ST"

# ===== BATCH 1: baselines v5(5) + v8(3) = 8 GPUs, ~2hr =====
echo "BATCH1_START $(date)" >> "$ST"
run CVbase_v5_f0 yolov5nu.pt "$CV/fold0/base.yaml" 640 0
run CVbase_v5_f1 yolov5nu.pt "$CV/fold1/base.yaml" 640 1
run CVbase_v5_f2 yolov5nu.pt "$CV/fold2/base.yaml" 640 2
run CVbase_v5_f3 yolov5nu.pt "$CV/fold3/base.yaml" 640 3
run CVbase_v5_f4 yolov5nu.pt "$CV/fold4/base.yaml" 640 4
run CVbase_v8_f0 yolov8n.pt  "$CV/fold0/base.yaml" 640 5
run CVbase_v8_f1 yolov8n.pt  "$CV/fold1/base.yaml" 640 6
run CVbase_v8_f2 yolov8n.pt  "$CV/fold2/base.yaml" 640 7
wait
echo "BATCH1_DONE $(date)" >> "$ST"

# ===== BATCH 2: baselines v8(2) + HROaug v5(5) + v8(1) = 8 GPUs, ~5hr =====
echo "BATCH2_START $(date)" >> "$ST"
run CVbase_v8_f3  yolov8n.pt  "$CV/fold3/base.yaml"  640  0
run CVbase_v8_f4  yolov8n.pt  "$CV/fold4/base.yaml"  640  1
run CVchamp_v5_f0 yolov5nu.pt "$CV/fold0/champ.yaml" 1280 2 "scale=0.9"
run CVchamp_v5_f1 yolov5nu.pt "$CV/fold1/champ.yaml" 1280 3 "scale=0.9"
run CVchamp_v5_f2 yolov5nu.pt "$CV/fold2/champ.yaml" 1280 4 "scale=0.9"
run CVchamp_v5_f3 yolov5nu.pt "$CV/fold3/champ.yaml" 1280 5 "scale=0.9"
run CVchamp_v5_f4 yolov5nu.pt "$CV/fold4/champ.yaml" 1280 6 "scale=0.9"
run CVchamp_v8_f0 yolov8n.pt  "$CV/fold0/champ.yaml" 1280 7 "scale=0.9"
wait
echo "BATCH2_DONE $(date)" >> "$ST"

# ===== BATCH 3: HROaug v8(4) + OSall v11(4) = 8 GPUs, ~5hr =====
echo "BATCH3_START $(date)" >> "$ST"
run CVchamp_v8_f1  yolov8n.pt "$CV/fold1/champ.yaml" 1280 0 "scale=0.9"
run CVchamp_v8_f2  yolov8n.pt "$CV/fold2/champ.yaml" 1280 1 "scale=0.9"
run CVchamp_v8_f3  yolov8n.pt "$CV/fold3/champ.yaml" 1280 2 "scale=0.9"
run CVchamp_v8_f4  yolov8n.pt "$CV/fold4/champ.yaml" 1280 3 "scale=0.9"
run CVosall_v11_f0 yolo11n.pt "$CV/fold0/osall.yaml"  1280 4 "scale=0.9"
run CVosall_v11_f1 yolo11n.pt "$CV/fold1/osall.yaml"  1280 5 "scale=0.9"
run CVosall_v11_f2 yolo11n.pt "$CV/fold2/osall.yaml"  1280 6 "scale=0.9"
run CVosall_v11_f3 yolo11n.pt "$CV/fold3/osall.yaml"  1280 7 "scale=0.9"
wait
echo "BATCH3_DONE $(date)" >> "$ST"

# ===== BATCH 4: OSall v11(1) + v5(5) + v8(2) = 8 GPUs, ~5hr =====
echo "BATCH4_START $(date)" >> "$ST"
run CVosall_v11_f4 yolo11n.pt  "$CV/fold4/osall.yaml" 1280 0 "scale=0.9"
run CVosall_v5_f0  yolov5nu.pt "$CV/fold0/osall.yaml" 1280 1 "scale=0.9"
run CVosall_v5_f1  yolov5nu.pt "$CV/fold1/osall.yaml" 1280 2 "scale=0.9"
run CVosall_v5_f2  yolov5nu.pt "$CV/fold2/osall.yaml" 1280 3 "scale=0.9"
run CVosall_v5_f3  yolov5nu.pt "$CV/fold3/osall.yaml" 1280 4 "scale=0.9"
run CVosall_v5_f4  yolov5nu.pt "$CV/fold4/osall.yaml" 1280 5 "scale=0.9"
run CVosall_v8_f0  yolov8n.pt  "$CV/fold0/osall.yaml" 1280 6 "scale=0.9"
run CVosall_v8_f1  yolov8n.pt  "$CV/fold1/osall.yaml" 1280 7 "scale=0.9"
wait
echo "BATCH4_DONE $(date)" >> "$ST"

# ===== BATCH 5: OSall v8(3 remaining) = 3 GPUs, ~5hr =====
echo "BATCH5_START $(date)" >> "$ST"
run CVosall_v8_f2 yolov8n.pt "$CV/fold2/osall.yaml" 1280 0 "scale=0.9"
run CVosall_v8_f3 yolov8n.pt "$CV/fold3/osall.yaml" 1280 1 "scale=0.9"
run CVosall_v8_f4 yolov8n.pt "$CV/fold4/osall.yaml" 1280 2 "scale=0.9"
wait
echo "BATCH5_DONE $(date)" >> "$ST"

echo "CV_SWEEP_ALL_DONE $(date)" >> "$ST"
