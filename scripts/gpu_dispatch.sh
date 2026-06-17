#!/usr/bin/env bash
# GPU queue dispatcher. Free-detection by COMPUTE-PROCESS COUNT (robust to between-stage util dips).
# A GPU is free iff <=1 compute proc (the shared ~4.5GB baseline pid), confirmed 3x over 30s.
# Re-reads queue.txt every pass. Stop: touch ~/atli/universe_bench/STOP_QUEUE
B="$HOME/atli/universe_bench"; QF="$B/queue.txt"; DISP="$B/queue.dispatched"; RECENT="$B/queue.recent"; STOP="$B/STOP_QUEUE"
touch "$DISP" "$RECENT"; echo "[$(date)] hardened dispatcher started"
nproc_gpu(){ nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .; }
is_free(){ for k in 1 2 3; do [ "$(nproc_gpu "$1")" -gt 1 ] && return 1; [ "$k" -lt 3 ] && sleep 15; done; return 0; }
while true; do
  [ -f "$STOP" ] && { echo "[$(date)] STOP_QUEUE, exit"; rm -f "$STOP"; break; }
  for g in 0 1 2 3 4 5 6 7; do
    last=$(grep "^$g " "$RECENT" | tail -1 | awk '{print $2}'); now=$(date +%s)
    [ -n "$last" ] && [ $((now-last)) -lt 240 ] && continue
    [ "$(nproc_gpu "$g")" -gt 1 ] && continue          # quick reject
    is_free "$g" || continue                            # confirmed-free (30s)
    job=$(grep -v '^#' "$QF" 2>/dev/null | grep -vxF -f "$DISP" | head -1)
    [ -z "$job" ] && break
    echo "$job" >> "$DISP"; echo "$g $now" >> "$RECENT"
    IFS='|' read -r NAME MODEL EXTRA DATA EP1 EP2 IMZ BAT <<< "$job"
    MODEL="$MODEL" EXTRA="$EXTRA" DATA="$DATA" nohup bash "$B/run_config_ext.sh" "$NAME" $g $EP1 $EP2 $IMZ $BAT > "$B/logs/$NAME.log" 2>&1 &
    echo "[$(date)] dispatched $NAME -> GPU $g"
  done
  sleep 45
done
