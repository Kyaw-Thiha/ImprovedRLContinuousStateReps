#!/usr/bin/env bash
# Unattended close-out: once the width+sweep campaign has 30 DONE and the Stage-B pilot is finished (or the gate failed,
# or the pilot was never launched because the gate failed), delete any remaining round-6 pod and stop the keepers.
set -u
cd "$(dirname "$0")"
WIDTH=r6sspwidthqueue-31a9570f
marker=results/r6_AUTOCLOSED
[ -f "$marker" ] && exit 0

n_width=$(./.venv/bin/python analysis/count_done.py --campaign "$WIDTH" 2>/dev/null || echo 0)
[ "${n_width:-0}" -ge 30 ] || { echo "[autoclose] width $n_width/30; waiting"; exit 0; }

# gate outcome, as the keeper logs it
gate_failed=0
grep -q "gate fails" results/r6_pilot_after.log 2>/dev/null && gate_failed=1

pilot_camp=$(sed -n 's/.*launching Stage B pilot \([a-zA-Z0-9-]*\).*/\1/p' results/r6_pilot_after.log 2>/dev/null | tail -1)
pilot_done=0
if [ -n "${pilot_camp:-}" ]; then
  pilot_total=$(./.venv/bin/python -c "import json;print(len(json.load(open('manifests/${pilot_camp}.json'))['jobs']))" 2>/dev/null || echo 20)
  pilot_done=$(./.venv/bin/python analysis/count_done.py --campaign "$pilot_camp" 2>/dev/null || echo 0)
  echo "[autoclose] pilot $pilot_camp $pilot_done/$pilot_total"
  if [ "${pilot_done:-0}" -lt "${pilot_total:-20}" ]; then echo "[autoclose] pilot still running; waiting"; exit 0; fi
elif [ "$gate_failed" -eq 1 ]; then
  echo "[autoclose] gate failed; nothing to run"
else
  echo "[autoclose] width done but the pilot has not launched yet; waiting"
  exit 0
fi

echo "[autoclose] round-6 work complete; deleting pods and stopping keepers"
./.venv/bin/python - <<'PY' || true
import sys; sys.path.insert(0, 'src')
from mjlab_centering import pods
for p in pods.list_pods():
    if any(k in str(p.get("name", "")) for k in ("r6sspwidthqueue", "r6sspp", "r6a-", "mjc-r6")):
        try:
            pods.delete_pod(p["id"]); print("[autoclose] deleted", p.get("name"))
        except Exception as e:
            print("[autoclose] delete failed for", p.get("name"), e)
PY
systemctl --user stop r6-width-after.service r6-pilot-after.service r6rawtop.service r6-watchdog.timer 2>/dev/null || true
systemctl --user disable r6-width-after.service r6-pilot-after.service r6rawtop.service 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ > "$marker"
echo "[autoclose] done; marker written"
