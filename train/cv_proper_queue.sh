#!/usr/bin/env bash
# GPU-pooled job queue for proper 70/15/15 CV sweep.
# Each GPU runs jobs sequentially from a shared queue.
# When a GPU finishes a job, it immediately picks up the next one.
# Skips jobs that are already done (checks for DONE in log).
set -uo pipefail
ROOT="$HOME/atli"
CV="$ROOT/Merged_CV_proper"
EXT="$ROOT/run_config_ext.sh"
ST="$ROOT/cv_proper_status.txt"
QUEUE="$ROOT/cv_proper_queue.txt"
LOCK="$ROOT/cv_proper_queue.lock"

echo "CV_PROPER_QUEUE_START $(date)" >> "$ST"

# Build the full job list: NAME MODEL YAML IMGSZ EXTRA
# Sort: 640px jobs first (faster), then 1280px
cat > "$QUEUE" << 'EOF'
CVP_base_v5_f0 yolov5nu.pt fold0/base.yaml 640
CVP_base_v5_f1 yolov5nu.pt fold1/base.yaml 640
CVP_base_v5_f2 yolov5nu.pt fold2/base.yaml 640
CVP_base_v5_f3 yolov5nu.pt fold3/base.yaml 640
CVP_base_v5_f4 yolov5nu.pt fold4/base.yaml 640
CVP_base_v8_f0 yolov8n.pt fold0/base.yaml 640
CVP_base_v8_f1 yolov8n.pt fold1/base.yaml 640
CVP_base_v8_f2 yolov8n.pt fold2/base.yaml 640
CVP_base_v8_f3 yolov8n.pt fold3/base.yaml 640
CVP_base_v8_f4 yolov8n.pt fold4/base.yaml 640
CVP_base_v11_f0 yolo11n.pt fold0/base.yaml 640
CVP_base_v11_f1 yolo11n.pt fold1/base.yaml 640
CVP_base_v11_f2 yolo11n.pt fold2/base.yaml 640
CVP_base_v11_f3 yolo11n.pt fold3/base.yaml 640
CVP_base_v11_f4 yolo11n.pt fold4/base.yaml 640
CVP_champ_v5_f0 yolov5nu.pt fold0/champ.yaml 1280 scale=0.9
CVP_champ_v5_f1 yolov5nu.pt fold1/champ.yaml 1280 scale=0.9
CVP_champ_v5_f2 yolov5nu.pt fold2/champ.yaml 1280 scale=0.9
CVP_champ_v5_f3 yolov5nu.pt fold3/champ.yaml 1280 scale=0.9
CVP_champ_v5_f4 yolov5nu.pt fold4/champ.yaml 1280 scale=0.9
CVP_champ_v8_f0 yolov8n.pt fold0/champ.yaml 1280 scale=0.9
CVP_champ_v8_f1 yolov8n.pt fold1/champ.yaml 1280 scale=0.9
CVP_champ_v8_f2 yolov8n.pt fold2/champ.yaml 1280 scale=0.9
CVP_champ_v8_f3 yolov8n.pt fold3/champ.yaml 1280 scale=0.9
CVP_champ_v8_f4 yolov8n.pt fold4/champ.yaml 1280 scale=0.9
CVP_champ_v11_f0 yolo11n.pt fold0/champ.yaml 1280 scale=0.9
CVP_champ_v11_f1 yolo11n.pt fold1/champ.yaml 1280 scale=0.9
CVP_champ_v11_f2 yolo11n.pt fold2/champ.yaml 1280 scale=0.9
CVP_champ_v11_f3 yolo11n.pt fold3/champ.yaml 1280 scale=0.9
CVP_champ_v11_f4 yolo11n.pt fold4/champ.yaml 1280 scale=0.9
CVP_osall_v5_f0 yolov5nu.pt fold0/osall.yaml 1280 scale=0.9
CVP_osall_v5_f1 yolov5nu.pt fold1/osall.yaml 1280 scale=0.9
CVP_osall_v5_f2 yolov5nu.pt fold2/osall.yaml 1280 scale=0.9
CVP_osall_v5_f3 yolov5nu.pt fold3/osall.yaml 1280 scale=0.9
CVP_osall_v5_f4 yolov5nu.pt fold4/osall.yaml 1280 scale=0.9
CVP_osall_v8_f0 yolov8n.pt fold0/osall.yaml 1280 scale=0.9
CVP_osall_v8_f1 yolov8n.pt fold1/osall.yaml 1280 scale=0.9
CVP_osall_v8_f2 yolov8n.pt fold2/osall.yaml 1280 scale=0.9
CVP_osall_v8_f3 yolov8n.pt fold3/osall.yaml 1280 scale=0.9
CVP_osall_v8_f4 yolov8n.pt fold4/osall.yaml 1280 scale=0.9
CVP_osall_v11_f0 yolo11n.pt fold0/osall.yaml 1280 scale=0.9
CVP_osall_v11_f1 yolo11n.pt fold1/osall.yaml 1280 scale=0.9
CVP_osall_v11_f2 yolo11n.pt fold2/osall.yaml 1280 scale=0.9
CVP_osall_v11_f3 yolo11n.pt fold3/osall.yaml 1280 scale=0.9
CVP_osall_v11_f4 yolo11n.pt fold4/osall.yaml 1280 scale=0.9
EOF

# Claim the next available job from the queue (atomic via flock)
next_job() {
    (
        flock -x 200
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            local name=$(echo "$line" | awk '{print $1}')
            # Skip if already done
            if grep -q "DONE" "$ROOT/${name}.log" 2>/dev/null; then
                continue
            fi
            # Skip if currently running (check for active yolo process)
            if grep -q "CLAIMED_${name}" "$ROOT/cv_proper_claimed.txt" 2>/dev/null; then
                continue
            fi
            # Claim it
            echo "CLAIMED_${name}" >> "$ROOT/cv_proper_claimed.txt"
            echo "$line"
            break
        done < "$QUEUE"
    ) 200>"$LOCK"
}

# GPU worker: keep pulling jobs until queue is empty
gpu_worker() {
    local gpu=$1
    while true; do
        local job=$(next_job)
        [[ -z "$job" ]] && break

        local name=$(echo "$job" | awk '{print $1}')
        local model=$(echo "$job" | awk '{print $2}')
        local yaml=$(echo "$job" | awk '{print $3}')
        local imgsz=$(echo "$job" | awk '{print $4}')
        local extra=$(echo "$job" | awk '{print $5}')

        local batch=16
        [[ "$imgsz" == "640" ]] && batch=32

        echo "[$(date)] GPU $gpu: starting $name ($model $imgsz)" >> "$ST"
        MODEL="$model" DATA="$CV/$yaml" EXTRA="$extra" \
            bash "$EXT" "$name" "$gpu" 150 100 "$imgsz" "$batch" \
            > "$ROOT/${name}.log" 2>&1
        echo "[$(date)] GPU $gpu: finished $name" >> "$ST"
    done
    echo "[$(date)] GPU $gpu: queue empty, worker done" >> "$ST"
}

# Clear claimed list (fresh start, existing DONE logs will be skipped)
> "$ROOT/cv_proper_claimed.txt"

# Launch 8 GPU workers in parallel
for gpu in 0 1 2 3 4 5 6 7; do
    gpu_worker "$gpu" &
done

wait
echo "CV_PROPER_QUEUE_ALL_DONE $(date)" >> "$ST"
