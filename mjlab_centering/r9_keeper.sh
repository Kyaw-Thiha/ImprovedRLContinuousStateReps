#!/bin/bash
# Round-9 keeper: keep a campaign's orchestrator alive until its queue drains or a TERMINATE marker appears.
# Only ever touches the campaign named on the command line and pods named `mjc-<campaign>`.
#
#   ./r9_keeper.sh [campaign] [manifest] [launch] [max_price]
#   ./r9_keeper.sh r9aleak-faa5c3d5 manifests/r9aleak-faa5c3d5.json 4 0.40
set -u
cd "$(dirname "$0")"
CAMP="${1:-r9aleak-faa5c3d5}"
MAN="${2:-manifests/${CAMP}.json}"
LAUNCH="${3:-4}"
MAXP="${4:-0.40}"
[ -f "$MAN" ] || { echo "[keeper] no manifest $MAN"; exit 1; }
while true; do
  if [ -f "results/${CAMP}.TERMINATE" ]; then echo "[keeper] TERMINATE marker; exiting"; break; fi
  if grep -q '"done": *[0-9]*' "results/${CAMP}_orch.log" 2>/dev/null && tail -60 "results/${CAMP}_orch.log" | grep -q "queue drained"; then
    echo "[keeper] orchestrator reports the queue drained; exiting"; break
  fi
  if ! pgrep -f "orchestrator.py --campaign ${CAMP}" >/dev/null; then
    echo "[keeper] $(date -u +%H:%M:%S) orchestrator not running; restarting"
    setsid nohup .venv/bin/python orchestrator.py --campaign "$CAMP" --manifest "$MAN" \
      --launch "$LAUNCH" --podrun-procs 1 --stock-first --poll 60 --retry-wait 30 \
      --soft-cap 13.5 --hard-cap 15 --max-price "$MAXP" \
      ${GPUS:+--gpus "$GPUS"} >> "results/${CAMP}_orch.log" 2>&1 < /dev/null &
  fi
  sleep 60
done
