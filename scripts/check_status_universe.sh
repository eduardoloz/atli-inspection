#!/usr/bin/env bash
# Status of the universe-damper benchmark sweep on ai.ee.unlv.edu (launched 2026-06-11).
# Usage: bash scripts/check_status_universe.sh
ssh LozanoE@ai.ee.unlv.edu '
echo "=== GPUs ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo; echo "=== runs (epoch + current val mAP50, from results.csv) ==="
for d in ~/atli/runs/U*_s[12]; do
  [ -f "$d/results.csv" ] || continue
  n=$(basename "$d")
  last=$(tail -1 "$d/results.csv")
  ep=$(echo "$last" | cut -d, -f1 | tr -d " ")
  col=$(head -1 "$d/results.csv" | tr "," "\n" | grep -n "metrics/mAP50(B)" | cut -d: -f1)
  map=$(echo "$last" | cut -d, -f"$col" | tr -d " ")
  done_marker=$(grep -c "### .* DONE" ~/atli/universe_bench/logs/"${n%_s[12]}".log 2>/dev/null || echo 0)
  printf "%-26s epoch %-6s val mAP50 %-8s %s\n" "$n" "$ep" "${map:0:6}" "$([ "$done_marker" -ge 1 ] && echo [ALL DONE])"
done
echo; echo "=== finished test evals ==="; ls ~/atli/runs/ | grep "^U.*_test$" || echo "(none yet)"
'
