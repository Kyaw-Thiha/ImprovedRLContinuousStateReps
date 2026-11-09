#!/usr/bin/env bash
# Waits until every job of the round-6 G1 claim set has a DONE marker on R2, then starts the paused SSP width stage.
# The cut order (d126) puts the claim set first; this only enforces it without a human in the loop.
set -u
cd "$(dirname "$0")"
CLAIM=r6g1queue-0ede5636
while true; do
  n=$(./.venv/bin/python analysis/count_done.py --campaign "$CLAIM" 2>/dev/null || echo 0)
  echo "[width-after] $(date -u +%H:%M:%S) claim DONE=$n/60"
  if [ "${n:-0}" -ge 60 ]; then
    echo "[width-after] claim queue complete; running the width stage in this unit"
    # exec directly rather than `systemctl --user start`: a service cannot rely on the session bus, and this unit is
    # already supervised with Restart=on-failure.
    # Newest hosts first (d1003): CUDA-13 drivers live on the newer SKUs, so a 3090 or 4000 Ada is the most likely
    # capture to waste on the preflight. Same price ceiling, different order.
    exec ./.venv/bin/python orchestrator.py --campaign r6sspwidthqueue-31a9570f \
      --manifest manifests/r6sspwidthqueue-31a9570f.json --launch 1 \
      --gpus "NVIDIA GeForce RTX 5090:0.69:COMMUNITY,NVIDIA L40S:0.79:COMMUNITY,NVIDIA L40:0.69:COMMUNITY,NVIDIA RTX A6000:0.33:COMMUNITY,NVIDIA GeForce RTX 4090:0.34:COMMUNITY,NVIDIA RTX 4000 Ada Generation:0.20:COMMUNITY,NVIDIA RTX A5000:0.16:COMMUNITY,NVIDIA GeForce RTX 3090:0.22:COMMUNITY,NVIDIA A100 80GB PCIe:1.19:COMMUNITY" \
      --podrun-procs 6 --poll 20 --retry-wait 30 --stock-first --max-price 1.25 --min-cuda 13.0 \
      --keep-hours 0 --hard-cap 6.5 --soft-cap 6.0 --max-pods-created 30 --price 0.69
  fi
  sleep 300
done
