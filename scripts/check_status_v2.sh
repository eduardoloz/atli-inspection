#!/usr/bin/env bash
# Check sweep_v2 status on the UNLV server.
#   Usage:  bash ~/Research/Vegas/check_status_v2.sh
SRV=${SRV:-${ATLI_SERVER:?set ATLI_SERVER (see .env) or SRV}}
ssh -o BatchMode=yes -o ConnectTimeout=10 "$SRV" 'bash -s' <<'EOF'
cd ~/atli 2>/dev/null || { echo "no ~/atli on server"; exit 0; }
echo "================== Sweep V2 status =================="
cat sweep_v2_status.txt 2>/dev/null || echo "(not started)"
echo
echo "Configs finished:"
for f in sweep_v2_w*.log; do
  [ -f "$f" ] || continue
  grep -ah "### .* DONE" "$f" 2>/dev/null | sed -E 's/.*### /  done: /; s/ DONE//'
done
echo
echo "Current activity:"
for f in sweep_v2_w*.log; do
  [ -f "$f" ] || continue
  wave=$(echo "$f" | sed 's/sweep_v2_//;s/\.log//')
  cfg=$(grep -aE '\] ### ' "$f" | tail -1 | sed -E 's/.*### ([^ ]+).*/\1/')
  phase=$(grep -aE '\] --- ' "$f" | tail -1 | sed -E 's/.*--- (.*) ---/\1/')
  ep=$(grep -aoE '[0-9]+/(149|299|99|199|150|300|100|200)' "$f" | tail -1)
  echo "  $wave:  ${cfg:-?}  |  ${phase:-starting}  |  epoch ${ep:-?}"
done
echo
echo "GPU util:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  /'
echo "======================================================="
EOF
