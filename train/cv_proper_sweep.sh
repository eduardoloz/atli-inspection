#!/usr/bin/env bash
# 5-fold CV sweep with proper 70/15/15 splits (val != test).
# 9 conditions: baseline/HROaug/OSall x v5/v8/v11 = 45 runs.
# Runs in batches of 8 (one per GPU).
set -uo pipefail
ROOT="$HOME/atli"
CV="$ROOT/Merged_CV_proper"
EXT="$ROOT/run_config_ext.sh"
ST="$ROOT/cv_proper_status.txt"

run() {
    local name=$1 model=$2 yaml=$3 imgsz=$4 gpu=$5 extra=${6:-}
    echo "[$(date)] Starting $name gpu=$gpu" >> "$ST"
    MODEL="$model" DATA="$yaml" EXTRA="$extra" bash "$EXT" "$name" "$gpu" 150 100 "$imgsz" 16 \
        > "$ROOT/${name}.log" 2>&1 &
}

echo "CV_PROPER_SWEEP_START $(date)" > "$ST"

# ===== BATCH 1: baseline v5(5) + v8(3) = 8 GPUs =====
echo "BATCH1_START $(date)" >> "$ST"
run CVP_base_v5_f0 yolov5nu.pt "$CV/fold0/base.yaml" 640 0
run CVP_base_v5_f1 yolov5nu.pt "$CV/fold1/base.yaml" 640 1
run CVP_base_v5_f2 yolov5nu.pt "$CV/fold2/base.yaml" 640 2
run CVP_base_v5_f3 yolov5nu.pt "$CV/fold3/base.yaml" 640 3
run CVP_base_v5_f4 yolov5nu.pt "$CV/fold4/base.yaml" 640 4
run CVP_base_v8_f0 yolov8n.pt  "$CV/fold0/base.yaml" 640 5
run CVP_base_v8_f1 yolov8n.pt  "$CV/fold1/base.yaml" 640 6
run CVP_base_v8_f2 yolov8n.pt  "$CV/fold2/base.yaml" 640 7
wait
echo "BATCH1_DONE $(date)" >> "$ST"

# ===== BATCH 2: baseline v8(2) + v11(5) + champ v5(1) = 8 GPUs =====
echo "BATCH2_START $(date)" >> "$ST"
run CVP_base_v8_f3  yolov8n.pt  "$CV/fold3/base.yaml"  640  0
run CVP_base_v8_f4  yolov8n.pt  "$CV/fold4/base.yaml"  640  1
run CVP_base_v11_f0 yolo11n.pt  "$CV/fold0/base.yaml"  640  2
run CVP_base_v11_f1 yolo11n.pt  "$CV/fold1/base.yaml"  640  3
run CVP_base_v11_f2 yolo11n.pt  "$CV/fold2/base.yaml"  640  4
run CVP_base_v11_f3 yolo11n.pt  "$CV/fold3/base.yaml"  640  5
run CVP_base_v11_f4 yolo11n.pt  "$CV/fold4/base.yaml"  640  6
run CVP_champ_v5_f0 yolov5nu.pt "$CV/fold0/champ.yaml" 1280 7 "scale=0.9"
wait
echo "BATCH2_DONE $(date)" >> "$ST"

# ===== BATCH 3: champ v5(4) + v8(4) = 8 GPUs =====
echo "BATCH3_START $(date)" >> "$ST"
run CVP_champ_v5_f1 yolov5nu.pt "$CV/fold1/champ.yaml" 1280 0 "scale=0.9"
run CVP_champ_v5_f2 yolov5nu.pt "$CV/fold2/champ.yaml" 1280 1 "scale=0.9"
run CVP_champ_v5_f3 yolov5nu.pt "$CV/fold3/champ.yaml" 1280 2 "scale=0.9"
run CVP_champ_v5_f4 yolov5nu.pt "$CV/fold4/champ.yaml" 1280 3 "scale=0.9"
run CVP_champ_v8_f0 yolov8n.pt  "$CV/fold0/champ.yaml" 1280 4 "scale=0.9"
run CVP_champ_v8_f1 yolov8n.pt  "$CV/fold1/champ.yaml" 1280 5 "scale=0.9"
run CVP_champ_v8_f2 yolov8n.pt  "$CV/fold2/champ.yaml" 1280 6 "scale=0.9"
run CVP_champ_v8_f3 yolov8n.pt  "$CV/fold3/champ.yaml" 1280 7 "scale=0.9"
wait
echo "BATCH3_DONE $(date)" >> "$ST"

# ===== BATCH 4: champ v8(1) + v11(5) + osall v5(2) = 8 GPUs =====
echo "BATCH4_START $(date)" >> "$ST"
run CVP_champ_v8_f4  yolov8n.pt  "$CV/fold4/champ.yaml" 1280 0 "scale=0.9"
run CVP_champ_v11_f0 yolo11n.pt  "$CV/fold0/champ.yaml" 1280 1 "scale=0.9"
run CVP_champ_v11_f1 yolo11n.pt  "$CV/fold1/champ.yaml" 1280 2 "scale=0.9"
run CVP_champ_v11_f2 yolo11n.pt  "$CV/fold2/champ.yaml" 1280 3 "scale=0.9"
run CVP_champ_v11_f3 yolo11n.pt  "$CV/fold3/champ.yaml" 1280 4 "scale=0.9"
run CVP_champ_v11_f4 yolo11n.pt  "$CV/fold4/champ.yaml" 1280 5 "scale=0.9"
run CVP_osall_v5_f0  yolov5nu.pt "$CV/fold0/osall.yaml" 1280 6 "scale=0.9"
run CVP_osall_v5_f1  yolov5nu.pt "$CV/fold1/osall.yaml" 1280 7 "scale=0.9"
wait
echo "BATCH4_DONE $(date)" >> "$ST"

# ===== BATCH 5: osall v5(3) + v8(5) = 8 GPUs =====
echo "BATCH5_START $(date)" >> "$ST"
run CVP_osall_v5_f2 yolov5nu.pt "$CV/fold2/osall.yaml" 1280 0 "scale=0.9"
run CVP_osall_v5_f3 yolov5nu.pt "$CV/fold3/osall.yaml" 1280 1 "scale=0.9"
run CVP_osall_v5_f4 yolov5nu.pt "$CV/fold4/osall.yaml" 1280 2 "scale=0.9"
run CVP_osall_v8_f0 yolov8n.pt  "$CV/fold0/osall.yaml" 1280 3 "scale=0.9"
run CVP_osall_v8_f1 yolov8n.pt  "$CV/fold1/osall.yaml" 1280 4 "scale=0.9"
run CVP_osall_v8_f2 yolov8n.pt  "$CV/fold2/osall.yaml" 1280 5 "scale=0.9"
run CVP_osall_v8_f3 yolov8n.pt  "$CV/fold3/osall.yaml" 1280 6 "scale=0.9"
run CVP_osall_v8_f4 yolov8n.pt  "$CV/fold4/osall.yaml" 1280 7 "scale=0.9"
wait
echo "BATCH5_DONE $(date)" >> "$ST"

# ===== BATCH 6: osall v11(5) = 5 GPUs =====
echo "BATCH6_START $(date)" >> "$ST"
run CVP_osall_v11_f0 yolo11n.pt "$CV/fold0/osall.yaml" 1280 0 "scale=0.9"
run CVP_osall_v11_f1 yolo11n.pt "$CV/fold1/osall.yaml" 1280 1 "scale=0.9"
run CVP_osall_v11_f2 yolo11n.pt "$CV/fold2/osall.yaml" 1280 2 "scale=0.9"
run CVP_osall_v11_f3 yolo11n.pt "$CV/fold3/osall.yaml" 1280 3 "scale=0.9"
run CVP_osall_v11_f4 yolo11n.pt "$CV/fold4/osall.yaml" 1280 4 "scale=0.9"
wait
echo "BATCH6_DONE $(date)" >> "$ST"

echo "CV_PROPER_SWEEP_ALL_DONE $(date)" >> "$ST"
