#!/usr/bin/env bash
# After the length-scale sweep, apply the Stage-B gate and run the pilot (or record that the gate failed).
set -u
cd "$(dirname "$0")"
while true; do
  ./.venv/bin/python analysis/restore_runs.py --out results/merged_r2 --prefix r6sspwidthqueue- >> results/r6_pilot_after.log 2>&1
  out=$(./.venv/bin/python analysis/ssp_pilot_launch.py 2>&1)
  echo "[pilot-after] $(date -u +%H:%M:%S) $(echo "$out" | tr '\n' ' ')"
  CAMPAIGN=$(echo "$out" | sed -n 's/^CAMPAIGN=//p')
  MANIFEST=$(echo "$out" | sed -n 's/^MANIFEST=//p')
  if [ -n "$CAMPAIGN" ] && [ -n "$MANIFEST" ]; then
    echo "[pilot-after] launching Stage B pilot $CAMPAIGN"
    exec ./.venv/bin/python orchestrator.py --campaign "$CAMPAIGN" --manifest "$MANIFEST" --launch 1 \
      --podrun-procs 6 --poll 20 --retry-wait 30 --stock-first --max-price 1.05 --min-cuda 13.0 \
      --keep-hours 0 --hard-cap 4.0 --soft-cap 3.5 --max-pods-created 20 --price 0.69
  fi
  case "$out" in
    *"gate fails"*) echo "[pilot-after] gate failed; nothing more to launch"; exit 0 ;;
  esac
  sleep 300
done
