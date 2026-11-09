#!/usr/bin/env bash
# Stop round 7 when its campaign drains, and delete any of its pods that linger.
set -u
cd "$(dirname "$0")"
CAMP=r7queue-9e67be96
[ -f results/r7_AUTOCLOSED ] && exit 0
n=$(./.venv/bin/python analysis/count_done.py --campaign "$CAMP" 2>/dev/null || echo 0)
if [ "${n:-0}" -ge 21 ]; then
  ./.venv/bin/python - <<'PY' || true
import sys; sys.path.insert(0,'src')
from mjlab_centering import pods
for p in pods.list_pods():
    if "r7queue" in str(p.get("name","")):
        try: pods.delete_pod(p["id"]); print("[r7-autoclose] deleted", p.get("name"))
        except Exception as e: print("[r7-autoclose] delete failed", e)
PY
  systemctl --user stop r7.service 2>/dev/null || true
  date -u +%Y-%m-%dT%H:%M:%SZ > results/r7_AUTOCLOSED
  echo "[r7-autoclose] campaign complete; service stopped"
else
  echo "[r7-autoclose] $n/21 done; waiting"
fi
