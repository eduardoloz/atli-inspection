#!/usr/bin/env bash
# Check the ATLI benchmark sweep status on the UNLV server from your own machine.
#   Usage:  bash ~/Research/Vegas/check_status.sh
SRV=${SRV:-${ATLI_SERVER:?set ATLI_SERVER (see .env) or SRV}}
ssh -o BatchMode=yes -o ConnectTimeout=10 "$SRV" 'bash -s' <<'EOF'
cd ~/atli 2>/dev/null || { echo "no ~/atli on server"; exit 0; }
echo "================== ATLI sweep status =================="
cat sweep_status.txt 2>/dev/null
echo
echo "Configs finished:"
grep -ah "### .* DONE" laneA.log laneB.log 2>/dev/null | sed -E 's/.*### /  done: /; s/ DONE//' || true
grep -qah "### .* DONE" laneA.log laneB.log 2>/dev/null || echo "  (none yet)"
echo
for L in A B; do
  log="lane${L}.log"; [ -f "$log" ] || continue
  cfg=$(grep -aE '\] ### ' "$log" | tail -1 | sed -E 's/.*### ([^ ]+).*/\1/')
  phase=$(grep -aE '\] --- ' "$log" | tail -1 | sed -E 's/.*--- (.*) ---/\1/')
  ep=$(grep -aoE '[0-9]+/(149|299|99|150|300|100)' "$log" | tail -1)
  echo "Lane $L:  ${cfg:-?}  |  ${phase:-starting}  |  epoch ${ep:-?}"
done
echo
echo "GPU util (idx, %util, mem):"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  /'
if grep -q SWEEP_DONE sweep_status.txt 2>/dev/null; then
  echo; echo "================== FINAL RESULTS =================="
  python3 ~/parse_results.py 2>/dev/null
fi
if [ -f abl_status.txt ]; then
  echo; echo "================== Eduardo ablation =================="
  cat abl_status.txt
  for L in A B; do
    log="abl${L}.log"; [ -f "$log" ] || continue
    cfg=$(grep -aE '\] ### ' "$log" | tail -1 | sed -E 's/.*### ([^ ]+).*/\1/')
    phase=$(grep -aE '\] --- ' "$log" | tail -1 | sed -E 's/.*--- (.*) ---/\1/')
    ep=$(grep -aoE '[0-9]+/(149|99)' "$log" | tail -1)
    echo "Lane $L:  ${cfg:-?}  |  ${phase:-starting}  |  epoch ${ep:-?}"
  done
fi
if [ -f condc_status.txt ]; then
  echo; echo "================== Condition C (defect-insulator -> Broken) =================="
  cat condc_status.txt
  for L in A B; do
    log="condc${L}.log"; [ -f "$log" ] || continue
    cfg=$(grep -aE '\] ### ' "$log" | tail -1 | sed -E 's/.*### ([^ ]+).*/\1/')
    phase=$(grep -aE '\] --- ' "$log" | tail -1 | sed -E 's/.*--- (.*) ---/\1/')
    ep=$(grep -aoE '[0-9]+/(149|99)' "$log" | tail -1)
    echo "Lane $L:  ${cfg:-?}  |  ${phase:-waiting for A/B}  |  epoch ${ep:-?}"
  done
fi
echo "======================================================="
EOF
