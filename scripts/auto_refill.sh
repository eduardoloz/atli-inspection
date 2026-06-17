#!/usr/bin/env bash
# Auto-refiller: keeps queue non-empty with meaningful seeds (champion/baseline/2 candidate-leaders).
B="$HOME/atli/universe_bench"; QF="$B/queue.txt"; DISP="$B/queue.dispatched"; STOP="$B/STOP_QUEUE"
OS3=/mnt/sdb/home/LozanoE/atli/Merged_Native_OS3/merged_native_os3.yaml
OS6=/mnt/sdb/home/LozanoE/atli/Merged_Native_OS6/merged_native_os6.yaml
NAT=/mnt/sdb/home/LozanoE/atli/Merged_Dataset_Stratified/merged_stratified.yaml
CAP=80; n=30
echo "[$(date)] auto-refiller started (cap $CAP jobs)"
while true; do
  [ -f "$STOP" ] && { echo "[$(date)] stop"; break; }
  tot=$(grep -vc '^#' "$QF"); disp=$(grep -c . "$DISP"); pend=$((tot-disp))
  if [ "$pend" -lt 6 ] && [ "$tot" -lt "$CAP" ]; then
    n=$((n+1))
    {
      echo "HROaug_v11_a$n|yolo11n.pt|scale=0.9 seed=$n|$OS3|150|100|1280|16"
      echo "Bv11_a$n|yolo11n.pt|seed=$n|$NAT|150|100|640|32"
      echo "HROaug_sc085_v11_a$n|yolo11n.pt|scale=0.85 seed=$n|$OS3|150|100|1280|16"
      echo "HROaug_OS6_v11_a$n|yolo11n.pt|scale=0.9 seed=$n|$OS6|150|100|1280|16"
    } >> "$QF"
    echo "[$(date)] auto-refill n=$n (pending was $pend, total now $((tot+4)))"
  fi
  sleep 120
done
