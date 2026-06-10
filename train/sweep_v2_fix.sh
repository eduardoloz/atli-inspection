#!/usr/bin/env bash
# Fixed sweep: single-GPU runs for all v8/v11 + v5 200ep configs.
# 8 GPUs = 8 parallel single-GPU jobs. No DDP.
set -uo pipefail
ROOT=$HOME/atli
TO=$ROOT/abl_target_only.yaml
TPE=$ROOT/abl_target_plus_eduardo.yaml
CF=$ROOT/abl_target_plus_eduardo_full.yaml
RC=~/run_config_single.sh
STATUS=$ROOT/sweep_v2_fix_status.txt

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"

echo "SWEEP_FIX_START $(date)" > "$STATUS"

# ============================================================
# Helper: run s2+test only (reuse existing s1 weights)
# ============================================================
run_s2_only() {
  local NAME="$1" FW="$2" GPU="$3" EP2="$4" YAML="$5"
  local W="$ROOT/runs/${NAME}_s1/weights/best.pt"
  [ -f "$W" ] || { echo "MISSING s1 for $NAME"; return 1; }
  echo "[$(date)] ### $NAME s2-only ($FW, finetune ${EP2}ep) GPU=$GPU"
  if [ "$FW" = "v5" ]; then
    cd "$ROOT/yolov5"
    echo "[$(date)] --- $NAME stage2 ---"
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --img 640 --batch 8 --epochs $EP2 --workers 8 --optimizer SGD \
      --data "$YAML" --weights "$W" --hyp hyp.oscar_paper.yaml \
      --project "$ROOT/runs" --name ${NAME}_s2 --exist-ok --device 0
    echo "[$(date)] --- $NAME test ---"
    CUDA_VISIBLE_DEVICES=$GPU python val.py --data "$YAML" \
      --weights "$ROOT/runs/${NAME}_s2/weights/best.pt" --img 640 --batch 8 --task test \
      --device 0 --project "$ROOT/runs" --name ${NAME}_test --exist-ok
  else
    echo "[$(date)] --- $NAME stage2 ---"
    yolo detect train \
      model="$W" data="$YAML" imgsz=640 batch=8 epochs=$EP2 workers=8 \
      optimizer=SGD lr0=0.00334 lrf=0.1535 device=$GPU \
      project="$ROOT/runs" name=${NAME}_s2 exist_ok=True
    echo "[$(date)] --- $NAME test ---"
    yolo detect val \
      model="$ROOT/runs/${NAME}_s2/weights/best.pt" data="$YAML" split=test imgsz=640 \
      batch=8 device=$GPU project="$ROOT/runs" name=${NAME}_test exist_ok=True
  fi
  echo "[$(date)] ### $NAME DONE"
}

# ============================================================
# WAVE 1: 8 parallel single-GPU jobs
# GPU 0: v8 B s2+test (s1 exists)
# GPU 1: v8 C s2+test (s1 exists)
# GPU 2: v11 A full
# GPU 3: v11 B full
# GPU 4: v11 C full
# GPU 5: v5 A 150+200
# GPU 6: v5 B 150+200
# GPU 7: v5 C 150+200
# ============================================================
echo "WAVE1_FIX_START $(date)" >> "$STATUS"

run_s2_only v8_tpe_r2  v8 0 100 "$TPE"  > "$ROOT/fix_v8_tpe.log" 2>&1 &
run_s2_only v8_tpef_r2 v8 1 100 "$CF"   > "$ROOT/fix_v8_tpef.log" 2>&1 &

DATA=$TO  bash "$RC" v11_to   v11 SGD 150 2 100  > "$ROOT/fix_v11_to.log" 2>&1 &
DATA=$TPE bash "$RC" v11_tpe  v11 SGD 150 3 100  > "$ROOT/fix_v11_tpe.log" 2>&1 &
DATA=$CF  bash "$RC" v11_tpef v11 SGD 150 4 100  > "$ROOT/fix_v11_tpef.log" 2>&1 &

DATA=$TO  bash "$RC" v5_to_ft200   v5 SGD 150 5 200  > "$ROOT/fix_v5_to_ft200.log" 2>&1 &
DATA=$TPE bash "$RC" v5_tpe_ft200  v5 SGD 150 6 200  > "$ROOT/fix_v5_tpe_ft200.log" 2>&1 &
DATA=$CF  bash "$RC" v5_tpef_ft200 v5 SGD 150 7 200  > "$ROOT/fix_v5_tpef_ft200.log" 2>&1 &

wait
echo "WAVE1_FIX_DONE $(date)" >> "$STATUS"

# ============================================================
# WAVE 2: 6 parallel single-GPU jobs (v8 + v11 @ 200ep)
# GPU 0: v8 A 150+200
# GPU 1: v8 B 150+200
# GPU 2: v8 C 150+200
# GPU 3: v11 A 150+200
# GPU 4: v11 B 150+200
# GPU 5: v11 C 150+200
# ============================================================
echo "WAVE2_FIX_START $(date)" >> "$STATUS"

DATA=$TO  bash "$RC" v8_to_ft200    v8  SGD 150 0 200  > "$ROOT/fix_v8_to_ft200.log" 2>&1 &
DATA=$TPE bash "$RC" v8_tpe_ft200   v8  SGD 150 1 200  > "$ROOT/fix_v8_tpe_ft200.log" 2>&1 &
DATA=$CF  bash "$RC" v8_tpef_ft200  v8  SGD 150 2 200  > "$ROOT/fix_v8_tpef_ft200.log" 2>&1 &
DATA=$TO  bash "$RC" v11_to_ft200   v11 SGD 150 3 200  > "$ROOT/fix_v11_to_ft200.log" 2>&1 &
DATA=$TPE bash "$RC" v11_tpe_ft200  v11 SGD 150 4 200  > "$ROOT/fix_v11_tpe_ft200.log" 2>&1 &
DATA=$CF  bash "$RC" v11_tpef_ft200 v11 SGD 150 5 200  > "$ROOT/fix_v11_tpef_ft200.log" 2>&1 &

wait
echo "WAVE2_FIX_DONE $(date)" >> "$STATUS"

echo "SWEEP_FIX_DONE $(date)" >> "$STATUS"
