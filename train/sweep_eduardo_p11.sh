#!/usr/bin/env bash
# Phase 11: 640-px shear pair + resolution-transfer levers
# (reference: deg15@640 = 0.726 mAP / R 0.681; deg15@1280 = 0.793 / 0.726;
#  @1280 shear did not compose: deg15+shear10 0.788, shear10 alone 0.779.)
#   Z: val640    — eval-only: 1280-trained EDU_deg15_v11 best.pt validated at
#                  imgsz=640 (splits the 640 gap into train-time vs test-time loss)
#   A: shear640  — deg15+shear10 @640 (composition test at deployment res)
#   B: shonly640 — shear10 alone @640 (mirror of the 1280 shear pair)
#   C: prog640   — 1280->640 progressive fine-tune: init from EDU_deg15_v11_f{k}_s2
#                  best.pt, one stage-2 (100 ep, low LR) at 640 — cheap distillation proxy
#   D: mixup640  — deg15 + mixup=0.15 @640 (untested aug lever on this pool)
# 4 train conditions x 5 folds = 20 runs + 5 evals. GPUs 3-7. imgsz 640, batch 16.
set -uo pipefail
ROOT="$HOME/atli"
OBB="$ROOT/CV_eduardo_obb"; OBBR="$ROOT/run_config_obb.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_eduardo_p11_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
GPUS=(3 4 5 6 7)

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_P11_START $(date)" > "$ST"

run_obb(){ local name=$1 model=$2 fold=$3 gpu=$4 extra=$5
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu extra=[$extra]" >>"$ST"
  MODEL="$model" DATA="$OBB/fold$fold/osall.yaml" EXTRA="$extra" \
    bash "$OBBR" "$name" "$gpu" 150 100 640 16 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"; }

echo "P11Z_val640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do
  name="EDU_deg15hi_val640_f$k"
  [ -d "$PROJ/${name}" ] && { echo "[$(date)] SKIP $name" >>"$ST"; continue; }
  yolo obb val model="$PROJ/EDU_deg15_v11_f${k}_s2/weights/best.pt" \
    data="$OBB/fold$k/osall.yaml" split=test imgsz=640 batch=16 \
    device="${GPUS[$k]}" project="$PROJ" name="$name" exist_ok=True \
    >"$LOGS/${name}.log" 2>&1 &
done; wait
echo "P11Z_done $(date)" >>"$ST"

echo "P11A_shear640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_shear640_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" "scale=0.9 degrees=15 shear=10" & done; wait

echo "P11B_shonly640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_shonly640_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" "scale=0.9 shear=10" & done; wait

echo "P11C_prog640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do
  name="EDU_prog640_f$k"; gpu="${GPUS[$k]}"
  if [ -d "$PROJ/${name}_test" ]; then echo "[$(date)] SKIP $name" >>"$ST"; else
  ( echo "[$(date)] START $name gpu=$gpu" >>"$ST"
    yolo obb train model="$PROJ/EDU_deg15_v11_f${k}_s2/weights/best.pt" \
      data="$OBB/fold$k/osall.yaml" imgsz=640 batch=16 epochs=100 workers=8 \
      optimizer=SGD lr0=0.00334 lrf=0.1535 scale=0.9 degrees=15 \
      device="$gpu" project="$PROJ" name="${name}_s2" exist_ok=True
    yolo obb val model="$PROJ/${name}_s2/weights/best.pt" \
      data="$OBB/fold$k/osall.yaml" split=test imgsz=640 batch=16 \
      device="$gpu" project="$PROJ" name="${name}_test" exist_ok=True
    echo "[$(date)] DONE $name (exit $?)" >>"$ST" ) >"$LOGS/${name}.log" 2>&1 &
  fi
done; wait

echo "P11D_mixup640 $(date)" >>"$ST"
for k in 0 1 2 3 4; do run_obb "EDU_mixup640_f$k" yolo11n-obb.pt "$k" "${GPUS[$k]}" "scale=0.9 degrees=15 mixup=0.15" & done; wait

echo "SWEEP_P11_DONE $(date)" >>"$ST"
