#!/usr/bin/env bash
# Supervises the single round-5 hunter: one campaign, one pod at a time, every Go1 job in priority order (d122).
# It never launches anything not in this list and never creates pods itself (the round-4 supervisor bug was a
# state-driven extra-campaign loop, d111).
set -u
cd "$(dirname "$0")"
C=r5all-48c22931
M=manifests/r5all-48c22931.json
while true; do
  if ! pgrep -f "orchestrator.py --campaign $C" >/dev/null; then
    echo "[keeper] $(date -u +%H:%M:%S) (re)starting $C"
    PYTHONUNBUFFERED=1 setsid nohup .venv/bin/python orchestrator.py --campaign "$C" --manifest "$M" --launch 1 \
      --podrun-procs 6 --poll 20 --retry-wait 30 --stock-first --max-price 1.05 --min-cuda 13.2 \
      --hard-cap 12.0 --soft-cap 10.0 --max-pods-created 60 >> "results/orch_${C}.log" 2>&1 &
  fi
  if ! pgrep -f "orchestrator.py --campaign r5biasfin-f59d5faf" >/dev/null; then
    echo "[keeper] $(date -u +%H:%M:%S) (re)starting r5biasfin-f59d5faf"
    PYTHONUNBUFFERED=1 setsid nohup .venv/bin/python orchestrator.py --campaign r5biasfin-f59d5faf \
      --manifest manifests/r5biasfin-f59d5faf.json --launch 1 --podrun-procs 2 --poll 20 --retry-wait 30 \
      --stock-first --max-price 1.05 --min-cuda 13.2 --hard-cap 1.0 --soft-cap 0.8 --max-pods-created 20 \
      >> results/orch_r5biasfin-f59d5faf.log 2>&1 &
  fi
  if ! pgrep -f "orchestrator.py --campaign r5bias01fin-59a99670" >/dev/null; then
    echo "[keeper] $(date -u +%H:%M:%S) (re)starting r5bias01fin-59a99670"
    PYTHONUNBUFFERED=1 setsid nohup .venv/bin/python orchestrator.py --campaign r5bias01fin-59a99670 \
      --manifest manifests/r5bias01fin-59a99670.json --launch 1 --podrun-procs 4 --poll 20 --retry-wait 30 \
      --stock-first --max-price 1.05 --min-cuda 13.2 --hard-cap 1.5 --soft-cap 1.2 --max-pods-created 20 \
      >> results/orch_r5bias01fin-59a99670.log 2>&1 &
  fi
  sleep 90
done
