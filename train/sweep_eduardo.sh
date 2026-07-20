#!/usr/bin/env bash
# ATLI-no-CPLID + eduardos, group-aware 5-fold CV. 4 conditions x 5 folds = 20 runs.
# YOLOv11n, 2-stage TL (150+100). GPUs 3-7 only (0-2 in use by others).
# Order (user priority): baseline -> OBB+deg45 -> detect champ+osall -> OBB ref.
set -uo pipefail
ROOT="$HOME/atli"
DET="$ROOT/CV_eduardo_det"; OBB="$ROOT/CV_eduardo_obb"
EXT="$ROOT/run_config_ext.sh"; OBBR="$ROOT/run_config_obb.sh"
ST="$ROOT/sweep_eduardo_status.txt"
LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_EDUARDO_START $(date)" > "$ST"
# prefetch checkpoints once to avoid 5-way download races
python - <<'PY' >> "$ST" 2>&1
from ultralytics import YOLO
for m in ("yolo11n.pt", "yolo11n-obb.pt"):
    YOLO(m); print("prefetched", m)
PY

run_det(){ local name=$1 yaml=$2 imgsz=$3 gpu=$4 extra=${5:-}
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name (done)" >> "$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu imgsz=$imgsz extra=[$extra]" >> "$ST"
  MODEL=yolo11n.pt DATA="$yaml" EXTRA="$extra" bash "$EXT" "$name" "$gpu" 150 100 "$imgsz" 16 > "$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE  $name (exit $?)" >> "$ST"; }
run_obb(){ local name=$1 yaml=$2 imgsz=$3 gpu=$4 extra=${5:-}
  [ -d "$ROOT/runs/${name}_test" ] && { echo "[$(date)] SKIP $name (done)" >> "$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu imgsz=$imgsz extra=[$extra]" >> "$ST"
  MODEL=yolo11n-obb.pt DATA="$yaml" EXTRA="$extra" bash "$OBBR" "$name" "$gpu" 150 100 "$imgsz" 16 > "$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE  $name (exit $?)" >> "$ST"; }

echo "WAVE1_BASELINE_det640 $(date)" >> "$ST"
for k in 0 1 2 3 4; do run_det "EDU_base_v11_f$k"        "$DET/fold$k/base.yaml"  640  "${GPUS[$k]}" "" & done; wait

echo "WAVE2_OBB_osall_deg45_1280 $(date)" >> "$ST"
for k in 0 1 2 3 4; do run_obb "EDU_obbdeg45_v11_f$k"    "$OBB/fold$k/osall.yaml" 1280 "${GPUS[$k]}" "scale=0.9 degrees=45" & done; wait

echo "WAVE3_OBB_osall_ref_1280 $(date)" >> "$ST"
for k in 0 1 2 3 4; do run_obb "EDU_obbref_v11_f$k"      "$OBB/fold$k/osall.yaml" 1280 "${GPUS[$k]}" "scale=0.9" & done; wait

echo "WAVE4_DET_champ_osall_1280 $(date)" >> "$ST"
for k in 0 1 2 3 4; do run_det "EDU_champosall_v11_f$k"  "$DET/fold$k/osall.yaml" 1280 "${GPUS[$k]}" "scale=0.9" & done; wait

echo "SWEEP_EDUARDO_DONE $(date)" >> "$ST"
