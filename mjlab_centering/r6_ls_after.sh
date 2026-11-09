#!/usr/bin/env bash
# After the width stage (r6a, 18 jobs) completes, build d138's length-scale sweep at the best width and run it.
set -u
cd "$(dirname "$0")"
WIDTH=r6a-ad66ecf2
COUNT='import sys; sys.path.insert(0,"src")
from mjlab_centering import r2
c=r2.client(); b=r2.bucket(); n=0; tok=None
while True:
    kw=dict(Bucket=b,Prefix=r2.key("'"$WIDTH"'/runs/"),MaxKeys=1000)
    if tok: kw["ContinuationToken"]=tok
    r=c.list_objects_v2(**kw)
    n+=sum(1 for o in r.get("Contents",[]) if o["Key"].endswith("/DONE"))
    tok=r.get("NextContinuationToken")
    if not tok: break
print(n)'
while true; do
  n=$(./.venv/bin/python -c "$COUNT" 2>/dev/null || echo 0)
  echo "[ls-after] $(date -u +%H:%M:%S) width DONE=$n/18"
  if [ "${n:-0}" -ge 18 ]; then
    # the launcher reads results/merged_r2; new runs live on R2 only, so restore just this campaign first
    ./.venv/bin/python analysis/restore_runs.py --out results/merged_r2 --prefix r6a- >> results/r6_ls_after.log 2>&1
    eval "$(./.venv/bin/python analysis/ssp_ls_launch.py 2>&1 | tee /tmp/ls_launch.out | sed -n 's/^\(BEST_WIDTH\|CAMPAIGN\|MANIFEST\|JOBS\|ITERS\|REFUSE\)=/\1=/p' | tr '\n' ' ')"
    cat /tmp/ls_launch.out
    if [ -n "${CAMPAIGN:-}" ] && [ -n "${MANIFEST:-}" ]; then
      echo "[ls-after] launching sweep $CAMPAIGN"
      exec ./.venv/bin/python orchestrator.py --campaign "$CAMPAIGN" --manifest "$MANIFEST" --launch 2 \
        --podrun-procs 6 --poll 20 --retry-wait 30 --stock-first --max-price 1.05 --min-cuda 13.0 \
        --hard-cap 3.0 --soft-cap 2.5 --max-pods-created 20 --price 0.69
    fi
  fi
  sleep 300
done
