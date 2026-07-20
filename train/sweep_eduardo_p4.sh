#!/usr/bin/env bash
# Phase 4: YOLOv5 + v8 baseline + champion, on the existing eduardos CV folds.
# v8 gets the real OBB champion; v5 has no OBB checkpoint so it gets the detection
# champion (its best available). Waits for phase 3, then runs on GPUs 3-7.
# 4 conditions x 5 folds = 20 runs. 2-stage TL (150+100).
set -uo pipefail
ROOT="$HOME/atli"
DET="$ROOT/CV_eduardo_det"; OBB="$ROOT/CV_eduardo_obb"
EXT="$ROOT/run_config_ext.sh"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_p4_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

echo "P4_WAITING_FOR_P3 $(date)" > "$ST"
while ! grep -q SWEEP_P3_DONE "$ROOT/sweep_eduardo_p3_status.txt" 2>/dev/null; do sleep 120; done

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P4_START $(date)" >> "$ST"
python -c "from ultralytics import YOLO; [YOLO(m) for m in ('yolov5nu.pt','yolov8n.pt','yolov8n-obb.pt')]" >> "$ST" 2>&1

run(){ local runner=$1 model=$2 name=$3 yaml=$4 imgsz=$5 gpu=$6 extra=${7:-}
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu model=$model extra=[$extra]" >>"$ST"
  MODEL=$model DATA="$yaml" EXTRA="$extra" bash "$runner" "$name" "$gpu" 150 100 "$imgsz" 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

# baselines (detection, 640) — fast
echo "P4A_baselines_det640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run "$EXT" yolov5nu.pt "EDU_v5base_f$k" "$DET/fold$k/base.yaml" 640 "${GPUS[$k]}" "" & done; wait
for k in 0 1 2 3 4; do run "$EXT" yolov8n.pt  "EDU_v8base_f$k" "$DET/fold$k/base.yaml" 640 "${GPUS[$k]}" "" & done; wait

# v5 champion (detection — no OBB checkpoint for v5), 1280
echo "P4B_v5champ_det1280 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run "$EXT" yolov5nu.pt "EDU_v5champ_f$k" "$DET/fold$k/osall.yaml" 1280 "${GPUS[$k]}" "scale=0.9" & done; wait

# v8 OBB champion, 1280
echo "P4C_v8obbchamp_1280 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run "$OBBR" yolov8n-obb.pt "EDU_v8obbchamp_f$k" "$OBB/fold$k/osall.yaml" 1280 "${GPUS[$k]}" "scale=0.9" & done; wait

echo "SWEEP_P4_DONE $(date)" >>"$ST"
